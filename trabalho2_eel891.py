"""
Trabalho 2 - EEL891 - Analise Exploratoria de Dados, Pre-processamento
e Engenharia de Features.
Problema de regressao: estimar preco de venda de imoveis.
Metrica de avaliacao: RMSPE (Root Mean Square Percentage Error) -- por isso
o target dos modelos e log1p(preco): RMSPE em preco equivale, de forma
aproximada, a RMSE em log(preco).

EDA (secoes 1-12) + pre-processamento/feature engineering/tratamento de
outliers (secoes 13-23) + baselines de modelagem com 5-fold CV avaliados
por RMSPE, com e sem outliers (secoes 24-27) + otimizacao de
hiperparametros com Optuna para LightGBM/XGBoost/CatBoost (secoes 28-32) +
ensemble (voting e blend multi-seed) e geracao da submissao (secoes 33-38) +
iteracao/melhorias pos-Kaggle: encoding de bairro, features de interacao e
corte de outliers mais agressivo, com novas submissoes (secoes 39-42) +
segunda rodada: re-otimizacao Optuna no dataset v4, weighted blending e
features extras do texto 'diferenciais' (secoes 43-46) + terceira rodada:
stacking com meta-learner, LOO target encoding do bairro, features de
preco por m2 e binning de numericas (secoes 47-51).
"""

import json
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from xgboost import XGBRegressor

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 160)

sns.set_theme(style="whitegrid")

TREINO_PATH = "conjunto_de_treinamento.csv"
TESTE_PATH = "conjunto_de_teste.csv"

COLS_BINARIAS = [
    "churrasqueira", "estacionamento", "piscina", "playground", "quadra",
    "s_festas", "s_jogos", "s_ginastica", "sauna", "vista_mar",
]
COLS_CATEGORICAS = ["tipo", "bairro", "tipo_vendedor"]
COLS_NUMERICAS = ["quartos", "suites", "vagas", "area_util", "area_extra"]


def secao(titulo):
    print("\n" + "=" * 80)
    print(titulo)
    print("=" * 80)


# ---------------------------------------------------------------------------
# 1. Carga dos dados
# ---------------------------------------------------------------------------
secao("1. CARGA DOS DADOS")

treino = pd.read_csv(TREINO_PATH)
teste = pd.read_csv(TESTE_PATH)

print(f"Treino: {treino.shape[0]} linhas x {treino.shape[1]} colunas")
print(f"Teste : {teste.shape[0]} linhas x {teste.shape[1]} colunas")

print("\nDtypes (treino):")
print(treino.dtypes)

print("\nDtypes (teste):")
print(teste.dtypes)


# ---------------------------------------------------------------------------
# 2. Distribuicao do preco: preco e log(preco)
# ---------------------------------------------------------------------------
secao("2. DISTRIBUICAO DO PRECO")

log_preco = np.log(treino["preco"])

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

sns.histplot(treino["preco"], bins=60, ax=axes[0], color="#4C72B0")
axes[0].set_title("Distribuicao do preco")
axes[0].set_xlabel("preco (R$)")
axes[0].set_ylabel("frequencia")

sns.histplot(log_preco, bins=60, ax=axes[1], color="#DD8452")
axes[1].set_title("Distribuicao do log(preco)")
axes[1].set_xlabel("log(preco)")
axes[1].set_ylabel("frequencia")

fig.tight_layout()
fig.savefig("eda_preco.png", dpi=120)
plt.close(fig)
print("Figura salva em eda_preco.png")


# ---------------------------------------------------------------------------
# 3. Estatisticas descritivas do preco
# ---------------------------------------------------------------------------
secao("3. ESTATISTICAS DESCRITIVAS DO PRECO")

stats_preco = treino["preco"].describe(
    percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
)
print(stats_preco)
print(f"\nAssimetria (skew): {treino['preco'].skew():.3f}")
print(f"Curtose         : {treino['preco'].kurtosis():.3f}")


# ---------------------------------------------------------------------------
# 4. Nulos por coluna
# ---------------------------------------------------------------------------
secao("4. VALORES NULOS POR COLUNA")

nulos_treino = treino.isnull().sum()
nulos_treino_pct = (nulos_treino / len(treino) * 100).round(2)
tab_nulos_treino = pd.DataFrame(
    {"n_nulos": nulos_treino, "pct_nulos": nulos_treino_pct}
)
tab_nulos_treino = tab_nulos_treino[tab_nulos_treino["n_nulos"] > 0]
print("Treino:")
print(tab_nulos_treino if not tab_nulos_treino.empty else "  Nenhum valor nulo encontrado.")

nulos_teste = teste.isnull().sum()
nulos_teste_pct = (nulos_teste / len(teste) * 100).round(2)
tab_nulos_teste = pd.DataFrame(
    {"n_nulos": nulos_teste, "pct_nulos": nulos_teste_pct}
)
tab_nulos_teste = tab_nulos_teste[tab_nulos_teste["n_nulos"] > 0]
print("\nTeste:")
print(tab_nulos_teste if not tab_nulos_teste.empty else "  Nenhum valor nulo encontrado.")


# ---------------------------------------------------------------------------
# 5. Categoricas: value_counts e preco mediano por categoria
# ---------------------------------------------------------------------------
secao("5. VARIAVEIS CATEGORICAS")

for col in COLS_CATEGORICAS:
    print(f"\n--- {col} ---")
    vc = treino[col].value_counts()
    mediana_por_cat = treino.groupby(col)["preco"].median().sort_values(ascending=False)
    resumo = pd.DataFrame({"contagem": vc, "preco_mediano": mediana_por_cat}).sort_values(
        "contagem", ascending=False
    )
    if col == "bairro":
        print(resumo.head(15))
        print(f"... ({resumo.shape[0]} categorias no total, mostrando top 15)")
    else:
        print(resumo)


# ---------------------------------------------------------------------------
# 6. Cardinalidade do bairro
# ---------------------------------------------------------------------------
secao("6. CARDINALIDADE DO BAIRRO")

n_bairros = treino["bairro"].nunique()
n_bairros_teste = teste["bairro"].nunique()
bairros_treino = set(treino["bairro"].unique())
bairros_teste = set(teste["bairro"].unique())
so_no_teste = bairros_teste - bairros_treino

print(f"Bairros unicos no treino: {n_bairros}")
print(f"Bairros unicos no teste : {n_bairros_teste}")
print(f"Bairros presentes no teste mas ausentes no treino: {len(so_no_teste)}")
if so_no_teste:
    print(f"  -> {sorted(so_no_teste)}")

top15_bairros = treino["bairro"].value_counts().head(15)
print("\nTop 15 bairros por frequencia (treino):")
print(top15_bairros)


# ---------------------------------------------------------------------------
# 7. Correlacao das numericas com preco
# ---------------------------------------------------------------------------
secao("7. CORRELACAO DAS VARIAVEIS NUMERICAS COM O PRECO")

corr_cols = COLS_NUMERICAS + ["preco"]
corr = treino[corr_cols].corr()
print(corr["preco"].sort_values(ascending=False))

fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
ax.set_title("Correlacao entre numericas e preco")
fig.tight_layout()
fig.savefig("eda_correlacoes.png", dpi=120)
plt.close(fig)
print("\nFigura salva em eda_correlacoes.png")


# ---------------------------------------------------------------------------
# 8. Boxplot de log(preco) por tipo
# ---------------------------------------------------------------------------
secao("8. LOG(PRECO) POR TIPO DE IMOVEL")

treino["log_preco"] = log_preco

ordem_tipo = treino.groupby("tipo")["log_preco"].median().sort_values(ascending=False).index

fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(
    data=treino, x="tipo", y="log_preco", order=ordem_tipo,
    hue="tipo", legend=False, ax=ax, palette="Set2",
)
ax.set_title("log(preco) por tipo de imovel")
ax.set_xlabel("tipo")
ax.set_ylabel("log(preco)")
fig.tight_layout()
fig.savefig("eda_tipo.png", dpi=120)
plt.close(fig)
print("Figura salva em eda_tipo.png")


# ---------------------------------------------------------------------------
# 9. Boxplot de log(preco) por bairro (top 15 mais frequentes)
# ---------------------------------------------------------------------------
secao("9. LOG(PRECO) POR BAIRRO (TOP 15 MAIS FREQUENTES)")

top15_nomes = top15_bairros.index.tolist()
sub_top15 = treino[treino["bairro"].isin(top15_nomes)]
ordem_bairro = sub_top15.groupby("bairro")["log_preco"].median().sort_values(ascending=False).index

fig, ax = plt.subplots(figsize=(11, 6))
sns.boxplot(
    data=sub_top15, x="bairro", y="log_preco", order=ordem_bairro,
    hue="bairro", legend=False, ax=ax, palette="Set3",
)
ax.set_title("log(preco) por bairro (top 15 mais frequentes)")
ax.set_xlabel("bairro")
ax.set_ylabel("log(preco)")
ax.tick_params(axis="x", rotation=60)
fig.tight_layout()
fig.savefig("eda_bairro.png", dpi=120)
plt.close(fig)
print("Figura salva em eda_bairro.png")


# ---------------------------------------------------------------------------
# 10. Coluna 'diferenciais'
# ---------------------------------------------------------------------------
secao("10. ANALISE DA COLUNA 'diferenciais'")

n_nulos_dif = treino["diferenciais"].isnull().sum()
print(f"Nulos em 'diferenciais' (treino): {n_nulos_dif}")

print("\n10 exemplos de 'diferenciais':")
print(treino["diferenciais"].head(10).to_string())

# mapeamento coluna binaria -> lista de palavras/frases que a caracterizam no texto
# ("quadra" e "vista_mar" so batem 100% quando se usam as frases reais do texto,
# nao a palavra ingenua igual ao nome da coluna)
palavras_chave = {
    "churrasqueira": ["churrasqueira"],
    "estacionamento": ["estacionamento"],
    "piscina": ["piscina"],
    "playground": ["playground"],
    "quadra": ["quadra", "campo de futebol"],
    "s_festas": ["festas"],
    "s_jogos": ["jogos"],
    "s_ginastica": ["ginastica"],
    "sauna": ["sauna"],
    "vista_mar": ["frente para o mar"],
}

print("\nVerificando se as colunas binarias derivam do texto de 'diferenciais':")
print(f"{'coluna':15s} {'palavra(s) buscada(s)':28s} {'concordancia':>13s}  {'binaria=1':>10s}  {'contem_palavra':>15s}")
for col_bin, palavras in palavras_chave.items():
    contem_palavra = pd.Series(False, index=treino.index)
    for palavra in palavras:
        contem_palavra |= treino["diferenciais"].str.contains(palavra, case=False, na=False)
    binaria_1 = treino[col_bin] == 1
    concordancia = (contem_palavra == binaria_1).mean() * 100
    print(
        f"{col_bin:15s} {' | '.join(palavras):28s} {concordancia:12.1f}%  "
        f"{binaria_1.sum():10d}  {contem_palavra.sum():15d}"
    )


# ---------------------------------------------------------------------------
# 11. Taxa de cada amenidade e preco mediano presente vs ausente
# ---------------------------------------------------------------------------
secao("11. TAXA DE AMENIDADES E PRECO MEDIANO (PRESENTE VS AUSENTE)")

linhas = []
for col in COLS_BINARIAS:
    taxa = treino[col].mean() * 100
    preco_presente = treino.loc[treino[col] == 1, "preco"].median()
    preco_ausente = treino.loc[treino[col] == 0, "preco"].median()
    diferenca_pct = (preco_presente / preco_ausente - 1) * 100 if preco_ausente else np.nan
    linhas.append(
        {
            "amenidade": col,
            "taxa_presenca_%": round(taxa, 1),
            "preco_mediano_presente": preco_presente,
            "preco_mediano_ausente": preco_ausente,
            "diferenca_%": round(diferenca_pct, 1),
        }
    )

tab_amenidades = pd.DataFrame(linhas).sort_values("diferenca_%", ascending=False)
print(tab_amenidades.to_string(index=False))


# ---------------------------------------------------------------------------
# 12. Resumo final
# ---------------------------------------------------------------------------
secao("12. RESUMO FINAL - OBSERVACOES-CHAVE PARA O PRE-PROCESSAMENTO")

observacoes = [
    f"- Preco varia de R$ {treino['preco'].min():,.0f} a R$ {treino['preco'].max():,.0f}, "
    f"com forte assimetria (skew={treino['preco'].skew():.2f}) e curtose extrema "
    f"(kurtosis={treino['preco'].kurtosis():.0f}). O percentil 99 e R$ {stats_preco['99%']:,.0f}, "
    "bem abaixo do maximo -- ha poucos imoveis com preco muito acima do resto da distribuicao "
    "(possiveis outliers/erros de digitacao). Como a metrica e RMSPE (erro percentual), "
    "considerar treinar em log(preco) para estabilizar a variancia, e avaliar remover ou "
    "tratar separadamente esses poucos precos extremos antes de modelar.",

    "- Correlacao linear de quartos/suites/vagas/area_util/area_extra com preco e muito fraca "
    "(todas < 0.05, ver secao 7) -- isso e esperado dado o outlier extremo de preco distorcendo "
    "a correlacao de Pearson; vale recalcular a correlacao apos tratar outliers e/ou usar "
    "correlacao de Spearman, alem de explorar area_util e quartos como features nao-lineares "
    "(splines, binning) ou em interacao com bairro/tipo.",

    f"- 'bairro' tem alta cardinalidade ({n_bairros} categorias no treino). "
    "Encoding simples (one-hot) pode gerar muitas colunas esparsas; considerar "
    "target/mean encoding (com cuidado para evitar vazamento) ou agrupamento de bairros raros.",

    (f"- Existem {len(so_no_teste)} bairro(s) no teste que nao aparecem no treino "
     "(tratar como categoria 'desconhecido' ou usar fallback por regiao/cidade)."
     if so_no_teste else
     "- Todos os bairros do teste tambem aparecem no treino (nao ha categorias inéditas)."),

    "- Colunas binarias parecem derivadas do texto livre em 'diferenciais' "
    "(ver secao 10); 'diferenciais' pode ser descartada apos validar a extracao, "
    "ou mantida para engenharia de features adicionais (ex.: contagem de diferenciais).",

    "- area_util e area_extra sao boas candidatas a outliers/erros de digitacao "
    "(valores extremos podem distorcer o RMSPE); avaliar winsorization ou capping.",

    "- Verificar nulos reportados na secao 4 antes de definir estrategia de imputacao "
    "(mediana/moda por grupo, ex.: por tipo+bairro).",

    "- 'quartos', 'suites', 'vagas' possuem correlacao com preco (ver secao 7); "
    "poucas suites/vagas iguais a zero podem ser legitimas, nao necessariamente nulos disfarcados.",

    "- Tipo de imovel e bairro mostram medianas de preco bem diferentes (secoes 5, 8 e 9); "
    "sao fortes candidatos a features de alta importancia no modelo.",

    "- Amenidades com maior diferenca de preco mediano (presente vs ausente, secao 11) "
    "sao boas candidatas a manter como features; as com pouca diferenca podem ser combinadas "
    "ou descartadas para reduzir dimensionalidade.",
]

for obs in observacoes:
    print(obs)

print("\nEDA concluida. Iniciando pre-processamento e engenharia de features.")


# ---------------------------------------------------------------------------
# 13. Pre-processamento: target, ids e unificacao treino+teste
# ---------------------------------------------------------------------------
secao("13. PRE-PROCESSAMENTO - TARGET, IDS E UNIFICACAO DOS CONJUNTOS")

# 1. Separar target e ids
target = treino["preco"].copy()
test_ids = teste["Id"].copy()

# 2. log_preco = target real para treinar os modelos (RMSPE em preco ~= RMSE em log_preco)
treino_prep = treino.drop(columns=["log_preco"]).copy()  # remove coluna auxiliar criada na EDA (secao 8)
treino_prep["log_preco"] = np.log1p(treino_prep["preco"])
teste_prep = teste.copy()

# 3. Combinar treino + teste com flag is_train (preprocessamento uniforme)
treino_prep["is_train"] = 1
teste_prep["is_train"] = 0
dados = pd.concat([treino_prep, teste_prep], axis=0, ignore_index=True, sort=False)

print(f"Dados combinados: {dados.shape[0]} linhas x {dados.shape[1]} colunas")
print(f"  is_train=1 (treino): {(dados['is_train'] == 1).sum()}")
print(f"  is_train=0 (teste) : {(dados['is_train'] == 0).sum()}")

# 4. Remover Id e preco do combinado (ja preservados em test_ids / target / log_preco)
dados = dados.drop(columns=["Id", "preco"])
print("Colunas 'Id' e 'preco' removidas do DataFrame combinado.")


# ---------------------------------------------------------------------------
# 14. Tratamento da coluna 'diferenciais'
# ---------------------------------------------------------------------------
secao("14. TRATAMENTO DA COLUNA 'diferenciais'")

dados["n_palavras_diferenciais"] = dados["diferenciais"].fillna("").str.split().apply(len)
dados["tem_diferenciais"] = dados["diferenciais"].notnull().astype(int)

print("n_palavras_diferenciais - estatisticas:")
print(dados["n_palavras_diferenciais"].describe())
print("\ntem_diferenciais - contagem:")
print(dados["tem_diferenciais"].value_counts())

PALAVRAS_LUXO = [
    "gourmet", "designer", "marmore", "granito", "porcelanato", "reformado",
    "novo", "andar alto", "sol da manha", "varanda",
]


def extrair_features_texto_luxo(serie_diferenciais):
    """n_chars_diferenciais, n_keywords_luxo e tem_keyword_luxo a partir do texto livre.

    Definida aqui (antes de remover 'diferenciais') mas reutilizada mais tarde, na secao 45
    (Melhoria 6), diretamente sobre treino/teste originais -- evita ter que manter um DataFrame
    auxiliar alinhado por indice ao longo de todo o pre-processamento/tratamento de outliers.
    """
    texto = serie_diferenciais.fillna("")
    n_keywords = pd.Series(0, index=serie_diferenciais.index)
    for palavra in PALAVRAS_LUXO:
        n_keywords = n_keywords + texto.str.contains(palavra, case=False, na=False).astype(int)
    return pd.DataFrame(
        {
            "n_chars_diferenciais": texto.str.len(),
            "n_keywords_luxo": n_keywords,
            "tem_keyword_luxo": (n_keywords > 0).astype(int),
        }
    )


# Preview das features extras do texto (usadas na Melhoria 6, secao 45). Mantidas FORA de
# 'dados' de proposito -- assim X_train/X_test continuam sem elas ate a melhoria 6 testar
# explicitamente se ajudam, numa comparacao limpa "com vs sem".
preview_features_luxo = extrair_features_texto_luxo(dados["diferenciais"])
print("\nFeatures extras do texto (preview, reservadas para a Melhoria 6, secao 45):")
print(preview_features_luxo.describe())
if (preview_features_luxo["n_keywords_luxo"] == 0).all():
    print("\nNota: nenhuma das palavras-chave de luxo aparece no vocabulario de 'diferenciais' -- a")
    print("coluna e uma combinacao fixa das 10 amenidades (+ 'esquina'/'copa'), sem texto livre real.")

dados = dados.drop(columns=["diferenciais"])
print("\nColuna 'diferenciais' removida (as 10 binarias ja capturam a informacao, ver secao 10).")


# ---------------------------------------------------------------------------
# 15. Tratamento de nulos
# ---------------------------------------------------------------------------
secao("15. TRATAMENTO DE NULOS")

# 6. area_extra: nulo = imovel sem area extra
n_nulos_area_extra = dados["area_extra"].isnull().sum()
dados["area_extra"] = dados["area_extra"].fillna(0)
print(f"area_extra: {n_nulos_area_extra} nulo(s) preenchido(s) com 0")

# 7. Demais numericas: preencher com a mediana calculada no treino
outras_numericas = [c for c in COLS_NUMERICAS if c != "area_extra"]
for col in outras_numericas:
    n_nulos_col = dados[col].isnull().sum()
    if n_nulos_col > 0:
        mediana = dados.loc[dados["is_train"] == 1, col].median()
        dados[col] = dados[col].fillna(mediana)
        print(f"{col}: {n_nulos_col} nulo(s) preenchido(s) com a mediana do treino ({mediana})")
    else:
        print(f"{col}: nenhum nulo encontrado")

# 8. Verificar nulos restantes (log_preco tem NaN no teste por definicao -- alvo desconhecido)
nulos_restantes = dados.drop(columns=["log_preco"]).isnull().sum()
nulos_restantes = nulos_restantes[nulos_restantes > 0]
print("\nNulos restantes (exceto 'log_preco', que e NaN no teste por nao ter alvo conhecido):")
print(nulos_restantes if not nulos_restantes.empty else "  Nenhum. Zero NaN confirmado.")
assert nulos_restantes.empty, "Existem nulos nao tratados no DataFrame combinado!"


# ---------------------------------------------------------------------------
# 16. Feature engineering
# ---------------------------------------------------------------------------
secao("16. FEATURE ENGINEERING")

dados["area_total"] = dados["area_util"] + dados["area_extra"]
dados["log_area_util"] = np.log1p(dados["area_util"])
dados["log_area_total"] = np.log1p(dados["area_total"])
dados["area_por_quarto"] = dados["area_util"] / (dados["quartos"] + 1)
dados["tem_suite"] = (dados["suites"] > 0).astype(int)
dados["proporcao_suites"] = dados["suites"] / (dados["quartos"] + 1)
dados["total_amenidades"] = dados[COLS_BINARIAS].sum(axis=1)
dados["categoria_luxo"] = ((dados["piscina"] + dados["sauna"] + dados["vista_mar"]) >= 2).astype(int)
dados["tem_area_extra"] = (dados["area_extra"] > 0).astype(int)

novas_features = [
    "area_total", "log_area_util", "log_area_total", "area_por_quarto",
    "tem_suite", "proporcao_suites", "total_amenidades", "categoria_luxo", "tem_area_extra",
]
print("Novas features criadas (estatisticas descritivas):")
print(dados[novas_features].describe().T)


# ---------------------------------------------------------------------------
# 17. Encoding - 'tipo' (One-Hot Encoding)
# ---------------------------------------------------------------------------
secao("17. ENCODING - 'tipo' (One-Hot Encoding)")

colunas_antes = set(dados.columns)
dados = pd.get_dummies(dados, columns=["tipo"], prefix="tipo", dtype=int)
colunas_tipo = sorted(set(dados.columns) - colunas_antes)
print(f"Colunas geradas para 'tipo': {colunas_tipo}")


# ---------------------------------------------------------------------------
# 18. Encoding - 'tipo_vendedor' (binaria)
# ---------------------------------------------------------------------------
secao("18. ENCODING - 'tipo_vendedor' (binaria)")

dados["tipo_vendedor_imobiliaria"] = (dados["tipo_vendedor"] == "Imobiliaria").astype(int)
dados = dados.drop(columns=["tipo_vendedor"])
print("Criada 'tipo_vendedor_imobiliaria' (1=Imobiliaria, 0=Pessoa Fisica):")
print(dados["tipo_vendedor_imobiliaria"].value_counts())


# ---------------------------------------------------------------------------
# 19. Encoding - 'bairro' (Target Encoding por mediana, apenas com dados de treino)
# ---------------------------------------------------------------------------
secao("19. ENCODING - 'bairro' (Target Encoding pela mediana de log_preco)")

mediana_por_bairro = dados.loc[dados["is_train"] == 1].groupby("bairro")["log_preco"].median()
mediana_global = dados.loc[dados["is_train"] == 1, "log_preco"].median()

dados["bairro_target_enc"] = dados["bairro"].map(mediana_por_bairro)
n_linhas_bairro_desconhecido = dados["bairro_target_enc"].isnull().sum()
dados["bairro_target_enc"] = dados["bairro_target_enc"].fillna(mediana_global)

print(f"Mediana global de log_preco (fallback para bairro desconhecido): {mediana_global:.4f}")
print(f"Linhas com bairro sem correspondencia no treino (preenchidas com fallback): {n_linhas_bairro_desconhecido}")
print("\nTop 5 bairros por mediana de log_preco (treino):")
print(mediana_por_bairro.sort_values(ascending=False).head())

dados = dados.drop(columns=["bairro"])


# ---------------------------------------------------------------------------
# 20. Tratamento de outliers (somente no treino)
# ---------------------------------------------------------------------------
secao("20. TRATAMENTO DE OUTLIERS")

# 1. Os imoveis mais caros do treino e percentis extremos do preco
print("10 imoveis mais caros do treino:")
print(treino.nlargest(10, "preco")[["preco", "tipo", "bairro", "area_util"]].to_string(index=False))

percentis_extremos = treino["preco"].quantile([0.99, 0.995, 0.999])
print("\nPercentis extremos do preco (treino):")
print(percentis_extremos)

# Snapshot do treino ANTES da remocao, para comparar RMSPE com/sem outliers (secoes 25-27).
# Mesmas features do X_train final -- isola exatamente o efeito da remocao de outliers.
X_train_com_outliers = (
    dados.loc[dados["is_train"] == 1].drop(columns=["is_train", "log_preco"]).reset_index(drop=True)
)
y_train_com_outliers = dados.loc[dados["is_train"] == 1, "log_preco"].reset_index(drop=True)

# 2. Remover outliers do TREINO usando log_preco (IQR com fator 3, mais permissivo que o
# 1.5 classico para nao descartar imoveis de alto padrao legitimos)
log_preco_treino = dados.loc[dados["is_train"] == 1, "log_preco"]
q1 = log_preco_treino.quantile(0.25)
q3 = log_preco_treino.quantile(0.75)
iqr = q3 - q1
limite_inferior = q1 - 3 * iqr
limite_superior = q3 + 3 * iqr

print(f"\nQ1={q1:.4f}  Q3={q3:.4f}  IQR={iqr:.4f}")
print(f"Limite inferior: log_preco={limite_inferior:.4f}  (preco ~ R$ {np.expm1(limite_inferior):,.2f})")
print(f"Limite superior: log_preco={limite_superior:.4f}  (preco ~ R$ {np.expm1(limite_superior):,.2f})")

mask_outlier = (dados["is_train"] == 1) & (
    (dados["log_preco"] > limite_superior) | (dados["log_preco"] < limite_inferior)
)
idx_outliers = dados.index[mask_outlier]

print(f"\nOutliers detectados pelo criterio IQR (3x): {len(idx_outliers)}")
if len(idx_outliers) > 0:
    print(np.expm1(dados.loc[idx_outliers, "log_preco"]).sort_values(ascending=False))

# Fallback: se o IQR nao capturar os extremos conhecidos (R$340M, R$630M), usar cap fixo no p99.5
maior_outlier = np.expm1(dados.loc[idx_outliers, "log_preco"]).max() if len(idx_outliers) > 0 else 0
if maior_outlier < 100_000_000:
    print("\nIQR nao capturou os outliers extremos conhecidos -- aplicando cap fixo no percentil 99.5.")
    preco_treino_atual = np.expm1(log_preco_treino)
    p995 = preco_treino_atual.quantile(0.995)
    mask_outlier = (dados["is_train"] == 1) & (np.expm1(dados["log_preco"]) > p995)
    idx_outliers = dados.index[mask_outlier]
    print(f"Percentil 99.5 do preco (treino): R$ {p995:,.2f}")
    print(f"Outliers detectados pelo cap fixo: {len(idx_outliers)}")

shape_antes = dados.shape[0]
n_treino_antes = int((dados["is_train"] == 1).sum())
dados = dados.drop(index=idx_outliers)  # remove apenas linhas de treino (mask exige is_train == 1)
n_treino_depois = int((dados["is_train"] == 1).sum())

# Indices originais do treino (== indices de 'treino') que sobreviveram -- usados mais tarde
# (secao 39) para re-derivar a coluna 'bairro' bruta, ja removida de 'dados' na secao 19.
idx_treino_limpo = dados.index[dados["is_train"] == 1]

print(f"\nImoveis removidos do treino: {len(idx_outliers)}")
print(f"Shape combinado: {shape_antes} linhas -> {dados.shape[0]} linhas")
print(f"Treino: {n_treino_antes} -> {n_treino_depois} imoveis")
print(f"Teste (inalterado): {int((dados['is_train'] == 0).sum())} imoveis")


# ---------------------------------------------------------------------------
# 21. Separacao final: X_train, y_train, X_test (apos remocao de outliers)
# ---------------------------------------------------------------------------
secao("21. SEPARACAO FINAL: X_train, y_train, X_test")

y_train = dados.loc[dados["is_train"] == 1, "log_preco"].reset_index(drop=True)
X_train = dados.loc[dados["is_train"] == 1].drop(columns=["is_train", "log_preco"]).reset_index(drop=True)
X_test = dados.loc[dados["is_train"] == 0].drop(columns=["is_train", "log_preco"]).reset_index(drop=True)

print(f"X_train: {X_train.shape}")
print(f"y_train: {y_train.shape}")
print(f"X_test : {X_test.shape}")


# ---------------------------------------------------------------------------
# 22. Verificacao final do pre-processamento
# ---------------------------------------------------------------------------
secao("22. VERIFICACAO FINAL DO PRE-PROCESSAMENTO")

print(f"Numero de features finais: {X_train.shape[1]}")
print("\nLista de features:")
for c in X_train.columns:
    print(f"  - {c}")

nan_train = int(X_train.isnull().sum().sum())
nan_test = int(X_test.isnull().sum().sum())
nan_y = int(y_train.isnull().sum())
print(f"\nNaN em X_train: {nan_train}")
print(f"NaN em X_test : {nan_test}")
print(f"NaN em y_train: {nan_y}")
assert nan_train == 0 and nan_test == 0 and nan_y == 0, "Existem NaN remanescentes apos o pre-processamento!"
print("Confirmado: zero NaN em X_train, X_test e y_train.")


# ---------------------------------------------------------------------------
# 23. Salvar dados processados em pickle
# ---------------------------------------------------------------------------
secao("23. SALVANDO DADOS PROCESSADOS")

dados_processados = {
    "X_train": X_train,
    "y_train": y_train,
    "X_test": X_test,
    "test_ids": test_ids,
    "target_original": target,
}
with open("dados_processados.pkl", "wb") as f:
    pickle.dump(dados_processados, f)

print("Dados processados salvos em 'dados_processados.pkl'")
print("Chaves salvas: X_train, y_train, X_test, test_ids, target_original")

print("\nPre-processamento e engenharia de features concluidos. Iniciando modelagem baseline.")


# ---------------------------------------------------------------------------
# 24. Scorer RMSPE customizado
# ---------------------------------------------------------------------------
secao("24. SCORER RMSPE CUSTOMIZADO")


def rmspe(y_true_log, y_pred_log):
    """RMSPE no espaco original do preco, a partir de previsoes em log1p(preco).

    RMSPE = sqrt(mean(((y_true - y_pred) / y_true) ** 2)). Quanto MENOR, melhor.
    """
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)
    return np.sqrt(np.mean(((y_true - y_pred) / y_true) ** 2))


rmspe_scorer = make_scorer(rmspe, greater_is_better=False)

print("Funcao rmspe(y_true_log, y_pred_log) definida:")
print("  a) converte y_true_log e y_pred_log para o espaco original via np.expm1()")
print("  b) calcula RMSPE = sqrt(mean(((y_true - y_pred) / y_true) ** 2))")
print("  c) retorna o valor (quanto MENOR, melhor)")
print("Scorer sklearn criado com make_scorer(rmspe, greater_is_better=False).")


# ---------------------------------------------------------------------------
# 25. Baselines com 5-fold CV (dados de treino SEM outliers)
# ---------------------------------------------------------------------------
secao("25. BASELINES COM 5-FOLD CV (SEM OUTLIERS)")

with open("dados_processados.pkl", "rb") as f:
    dados_pkl = pickle.load(f)

X = dados_pkl["X_train"]
y = dados_pkl["y_train"]

# KFold simples: StratifiedKFold nao se aplica a regressao (nao ha classes para estratificar).
kf = KFold(n_splits=5, shuffle=True, random_state=42)

modelos = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=1.0),
    "RandomForestRegressor": RandomForestRegressor(n_estimators=200, random_state=42),
    "GradientBoostingRegressor": GradientBoostingRegressor(n_estimators=200, random_state=42),
    "LGBMRegressor": LGBMRegressor(n_estimators=400, verbose=-1, random_state=42),
    "XGBRegressor": XGBRegressor(n_estimators=400, verbosity=0, random_state=42),
    "CatBoostRegressor": CatBoostRegressor(iterations=400, verbose=0, random_seed=42),
}

resultados = []
for nome, modelo in modelos.items():
    scores = cross_val_score(modelo, X, y, cv=kf, scoring=rmspe_scorer, n_jobs=1)
    rmspe_scores = -scores  # make_scorer com greater_is_better=False inverte o sinal
    media = rmspe_scores.mean()
    desvio = rmspe_scores.std()
    resultados.append({"modelo": nome, "rmspe_medio": media, "rmspe_std": desvio})
    print(f"{nome:28s} RMSPE = {media:.4f} +/- {desvio:.4f}")

tab_resultados = pd.DataFrame(resultados).sort_values("rmspe_medio").reset_index(drop=True)
tab_resultados.index += 1

print("\nRanking dos modelos SEM outliers (do melhor/menor RMSPE ao pior):")
print(tab_resultados.to_string())


# ---------------------------------------------------------------------------
# 26. Baselines com 5-fold CV (dados de treino ORIGINAIS, com outliers) - comparacao
# ---------------------------------------------------------------------------
secao("26. BASELINES COM 5-FOLD CV (COM OUTLIERS, PARA COMPARACAO)")

resultados_outliers = []
for nome, modelo in modelos.items():
    scores = cross_val_score(
        modelo, X_train_com_outliers, y_train_com_outliers, cv=kf, scoring=rmspe_scorer, n_jobs=1
    )
    rmspe_scores = -scores
    media = rmspe_scores.mean()
    desvio = rmspe_scores.std()
    resultados_outliers.append({"modelo": nome, "rmspe_medio": media, "rmspe_std": desvio})
    print(f"{nome:28s} RMSPE = {media:.4f} +/- {desvio:.4f}")

tab_resultados_outliers = pd.DataFrame(resultados_outliers).sort_values("rmspe_medio").reset_index(drop=True)
tab_resultados_outliers.index += 1

print("\nRanking dos modelos COM outliers (do melhor/menor RMSPE ao pior):")
print(tab_resultados_outliers.to_string())


# ---------------------------------------------------------------------------
# 27. Comparacao: RMSPE com vs sem outliers
# ---------------------------------------------------------------------------
secao("27. COMPARACAO: RMSPE COM VS SEM OUTLIERS")

comparacao = (
    tab_resultados[["modelo", "rmspe_medio"]]
    .rename(columns={"rmspe_medio": "rmspe_sem_outliers"})
    .merge(
        tab_resultados_outliers[["modelo", "rmspe_medio"]].rename(columns={"rmspe_medio": "rmspe_com_outliers"}),
        on="modelo",
    )
)
comparacao["reducao_%"] = (1 - comparacao["rmspe_sem_outliers"] / comparacao["rmspe_com_outliers"]) * 100
comparacao = comparacao.sort_values("rmspe_sem_outliers").reset_index(drop=True)
comparacao.index += 1

print(comparacao.to_string())
print(
    f"\nMelhor modelo (dados sem outliers): {comparacao.iloc[0]['modelo']} "
    f"(RMSPE={comparacao.iloc[0]['rmspe_sem_outliers']:.4f})"
)

print("\nModelagem baseline concluida (com e sem tratamento de outliers).")


# ---------------------------------------------------------------------------
# 28. Optuna - setup (split 80/20 estratificado por faixas de log_preco)
# ---------------------------------------------------------------------------
secao("28. OPTUNA - SETUP")

optuna.logging.set_verbosity(optuna.logging.WARNING)

# Licao do Trabalho 1: split 80/20 simples + ~50 trials generalizou melhor no Kaggle
# do que buscas mais longas com 5-fold CV completo por trial.
bins_estratificacao = pd.qcut(y_train, q=5, labels=False)
X_opt_treino, X_opt_val, y_opt_treino, y_opt_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=bins_estratificacao
)

print(f"Split 80/20 estratificado por faixas de log_preco (5 bins via pd.qcut):")
print(f"  treino (80%)    : {X_opt_treino.shape}")
print(f"  validacao (20%) : {X_opt_val.shape}")


def avaliar_rmspe_val(modelo):
    """Treina no 80%, preve no 20% e calcula RMSPE no espaco original (expm1)."""
    modelo.fit(X_opt_treino, y_opt_treino)
    pred_log = modelo.predict(X_opt_val)
    return rmspe(y_opt_val, pred_log)


print("Funcao avaliar_rmspe_val() definida (treina 80%, avalia RMSPE no 20% restante).")


# ---------------------------------------------------------------------------
# 29. Optuna - LightGBM (50 trials)
# ---------------------------------------------------------------------------
secao("29. OPTUNA - LIGHTGBM (50 TRIALS)")


def objetivo_lgbm(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1500),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "num_leaves": trial.suggest_int("num_leaves", 8, 127),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample": trial.suggest_float("subsample", 0.4, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "random_state": 42,
        "verbose": -1,
    }
    return avaliar_rmspe_val(LGBMRegressor(**params))


study_lgbm = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
study_lgbm.optimize(objetivo_lgbm, n_trials=50, show_progress_bar=True)

print(f"\nMelhor RMSPE (validacao 20%) - LightGBM: {study_lgbm.best_value:.4f}")
print("Melhores parametros:")
for k, v in study_lgbm.best_params.items():
    print(f"  {k}: {v}")


# ---------------------------------------------------------------------------
# 30. Optuna - XGBoost (50 trials)
# ---------------------------------------------------------------------------
secao("30. OPTUNA - XGBOOST (50 TRIALS)")


def objetivo_xgb(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1500),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 30),
        "subsample": trial.suggest_float("subsample", 0.4, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "gamma": trial.suggest_float("gamma", 1e-8, 5.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "random_state": 42,
        "verbosity": 0,
    }
    return avaliar_rmspe_val(XGBRegressor(**params))


study_xgb = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
study_xgb.optimize(objetivo_xgb, n_trials=50, show_progress_bar=True)

print(f"\nMelhor RMSPE (validacao 20%) - XGBoost: {study_xgb.best_value:.4f}")
print("Melhores parametros:")
for k, v in study_xgb.best_params.items():
    print(f"  {k}: {v}")


# ---------------------------------------------------------------------------
# 31. Optuna - CatBoost (30 trials, mais lento)
# ---------------------------------------------------------------------------
secao("31. OPTUNA - CATBOOST (30 TRIALS)")


def objetivo_catboost(trial):
    params = {
        "iterations": trial.suggest_int("iterations", 200, 1000),
        "depth": trial.suggest_int("depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        "random_strength": trial.suggest_float("random_strength", 1e-8, 10.0, log=True),
        "random_seed": 42,
        "verbose": 0,
    }
    return avaliar_rmspe_val(CatBoostRegressor(**params))


study_catboost = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
study_catboost.optimize(objetivo_catboost, n_trials=30, show_progress_bar=True)

print(f"\nMelhor RMSPE (validacao 20%) - CatBoost: {study_catboost.best_value:.4f}")
print("Melhores parametros:")
for k, v in study_catboost.best_params.items():
    print(f"  {k}: {v}")


# ---------------------------------------------------------------------------
# 32. Resultados da otimizacao
# ---------------------------------------------------------------------------
secao("32. RESULTADOS DA OTIMIZACAO")

studies = {
    "LGBMRegressor": (study_lgbm, "best_lgbm.json"),
    "XGBRegressor": (study_xgb, "best_xgb.json"),
    "CatBoostRegressor": (study_catboost, "best_catboost.json"),
}

for nome, (study, arquivo) in studies.items():
    with open(arquivo, "w") as f:
        json.dump(study.best_params, f, indent=2)
    print(f"Parametros de {nome} salvos em '{arquivo}' (best_value={study.best_value:.4f})")

# Confirmar com 5-fold CV nos dados limpos (sem outliers), usando os melhores params encontrados
modelos_otimizados = {
    "LGBMRegressor": LGBMRegressor(**study_lgbm.best_params, random_state=42, verbose=-1),
    "XGBRegressor": XGBRegressor(**study_xgb.best_params, random_state=42, verbosity=0),
    "CatBoostRegressor": CatBoostRegressor(**study_catboost.best_params, random_seed=42, verbose=0),
}

print("\nConfirmando com 5-fold CV (dados sem outliers):")
resultados_otimizados = []
for nome, modelo in modelos_otimizados.items():
    scores = cross_val_score(modelo, X_train, y_train, cv=kf, scoring=rmspe_scorer, n_jobs=1)
    rmspe_scores = -scores
    media = rmspe_scores.mean()
    desvio = rmspe_scores.std()
    resultados_otimizados.append({"modelo": nome, "rmspe_medio": media, "rmspe_std": desvio})
    print(f"{nome:28s} RMSPE = {media:.4f} +/- {desvio:.4f}")

tab_otimizados = pd.DataFrame(resultados_otimizados)

# Tabela comparativa: baseline default (secao 25, sem outliers) vs otimizado (Optuna)
comparacao_final = (
    tab_resultados[tab_resultados["modelo"].isin(modelos_otimizados.keys())][["modelo", "rmspe_medio"]]
    .rename(columns={"rmspe_medio": "rmspe_baseline_default"})
    .merge(
        tab_otimizados[["modelo", "rmspe_medio"]].rename(columns={"rmspe_medio": "rmspe_otimizado"}),
        on="modelo",
    )
)
comparacao_final["melhora_%"] = (
    1 - comparacao_final["rmspe_otimizado"] / comparacao_final["rmspe_baseline_default"]
) * 100
comparacao_final = comparacao_final.sort_values("rmspe_otimizado").reset_index(drop=True)
comparacao_final.index += 1

print("\nComparacao: baseline default (5-fold CV) vs otimizado via Optuna (5-fold CV), dados sem outliers:")
print(comparacao_final.to_string())

print("\nOtimizacao de hiperparametros concluida.")


# ---------------------------------------------------------------------------
# 33. Ensemble - setup (carregar melhores parametros dos JSONs)
# ---------------------------------------------------------------------------
secao("33. ENSEMBLE - SETUP")

with open("best_lgbm.json") as f:
    params_lgbm = json.load(f)
with open("best_xgb.json") as f:
    params_xgb = json.load(f)
with open("best_catboost.json") as f:
    params_catboost = json.load(f)

print("Melhores parametros recarregados de best_lgbm.json, best_xgb.json e best_catboost.json.")


def criar_modelos_otimizados(seed):
    """Instancia os 3 modelos otimizados com uma seed especifica (para blend multi-seed)."""
    return {
        "LGBMRegressor": LGBMRegressor(**params_lgbm, random_state=seed, verbose=-1),
        "XGBRegressor": XGBRegressor(**params_xgb, random_state=seed, verbosity=0),
        "CatBoostRegressor": CatBoostRegressor(**params_catboost, random_seed=seed, verbose=0),
    }


# ---------------------------------------------------------------------------
# 34. Ensemble - Voting (media das 3 previsoes, 5-fold CV)
# ---------------------------------------------------------------------------
secao("34. ENSEMBLE - VOTING (MEDIA DOS 3 MODELOS, 5-FOLD CV)")

rmspe_voting_folds = []
for fold, (idx_tr, idx_val) in enumerate(kf.split(X_train), start=1):
    X_tr, X_val = X_train.iloc[idx_tr], X_train.iloc[idx_val]
    y_tr, y_val = y_train.iloc[idx_tr], y_train.iloc[idx_val]

    preds_val = []
    for nome, modelo in criar_modelos_otimizados(seed=42).items():
        modelo.fit(X_tr, y_tr)
        preds_val.append(modelo.predict(X_val))

    pred_media = np.mean(preds_val, axis=0)
    rmspe_fold = rmspe(y_val, pred_media)
    rmspe_voting_folds.append(rmspe_fold)
    print(f"Fold {fold}: RMSPE = {rmspe_fold:.4f}")

rmspe_voting_media = float(np.mean(rmspe_voting_folds))
rmspe_voting_std = float(np.std(rmspe_voting_folds))
print(f"\nVoting Ensemble (3 modelos): RMSPE = {rmspe_voting_media:.4f} +/- {rmspe_voting_std:.4f}")


# ---------------------------------------------------------------------------
# 35. Ensemble - Blend multi-seed (3 seeds x 3 modelos = 9 modelos, 5-fold CV)
# ---------------------------------------------------------------------------
secao("35. ENSEMBLE - BLEND MULTI-SEED (3 SEEDS X 3 MODELOS, 5-FOLD CV)")

seeds_blend_cv = [42, 123, 2026]

rmspe_blend_folds = []
for fold, (idx_tr, idx_val) in enumerate(kf.split(X_train), start=1):
    X_tr, X_val = X_train.iloc[idx_tr], X_train.iloc[idx_val]
    y_tr, y_val = y_train.iloc[idx_tr], y_train.iloc[idx_val]

    preds_val = []
    for seed in seeds_blend_cv:
        for nome, modelo in criar_modelos_otimizados(seed=seed).items():
            modelo.fit(X_tr, y_tr)
            preds_val.append(modelo.predict(X_val))

    pred_media = np.mean(preds_val, axis=0)
    rmspe_fold = rmspe(y_val, pred_media)
    rmspe_blend_folds.append(rmspe_fold)
    print(f"Fold {fold}: RMSPE = {rmspe_fold:.4f}  ({len(preds_val)} modelos no blend)")

rmspe_blend_media = float(np.mean(rmspe_blend_folds))
rmspe_blend_std = float(np.std(rmspe_blend_folds))
print(f"\nBlend multi-seed (9 modelos): RMSPE = {rmspe_blend_media:.4f} +/- {rmspe_blend_std:.4f}")


# ---------------------------------------------------------------------------
# 36. Comparacao: individuais vs Voting vs Blend
# ---------------------------------------------------------------------------
secao("36. COMPARACAO: INDIVIDUAIS VS VOTING VS BLEND")

comparacao_ensemble = pd.concat(
    [
        tab_otimizados[["modelo", "rmspe_medio", "rmspe_std"]],
        pd.DataFrame(
            [
                {"modelo": "Voting (3 modelos)", "rmspe_medio": rmspe_voting_media, "rmspe_std": rmspe_voting_std},
                {
                    "modelo": "Blend multi-seed (9 modelos)",
                    "rmspe_medio": rmspe_blend_media,
                    "rmspe_std": rmspe_blend_std,
                },
            ]
        ),
    ],
    ignore_index=True,
).sort_values("rmspe_medio").reset_index(drop=True)
comparacao_ensemble.index += 1

print("Todos avaliados com as MESMAS 5 dobras do KFold (mesma seed) -- comparacao justa:")
print(comparacao_ensemble.to_string())

melhor_abordagem = comparacao_ensemble.iloc[0]["modelo"]
print(f"\nMelhor abordagem pelo CV: {melhor_abordagem} (RMSPE={comparacao_ensemble.iloc[0]['rmspe_medio']:.4f})")


# ---------------------------------------------------------------------------
# 37. Geracao da submissao (blend multi-seed no dataset completo)
# ---------------------------------------------------------------------------
secao("37. GERACAO DA SUBMISSAO")

# 4. Escolha da abordagem para a submissao final. O blend multi-seed reduz a variancia
# associada a inicializacao aleatoria de cada modelo (menos dependente de uma unica seed
# "sortuda"), sendo a estrategia mais robusta para generalizar no leaderboard -- mesmo
# quando seu RMSPE medio de CV fica proximo ao do melhor modelo/estrategia individual.
print(f"Abordagem com menor RMSPE no CV: {melhor_abordagem}")
print("Estrategia adotada para a submissao final: Blend multi-seed (maior robustez a variancia entre seeds).")

# 5. Treinar cada modelo com 5 seeds diferentes no treino completo (sem outliers) e prever o teste
seeds_submissao = [42, 123, 2026, 7, 99]

previsoes_test = []
for seed in seeds_submissao:
    for nome, modelo in criar_modelos_otimizados(seed=seed).items():
        modelo.fit(X_train, y_train)
        pred_log = modelo.predict(X_test)
        previsoes_test.append(pred_log)
        print(f"  seed={seed:<5d} {nome:20s} treinado e previsao gerada.")

pred_log_media = np.mean(previsoes_test, axis=0)
print(f"\nTotal de modelos no blend final: {len(previsoes_test)} ({len(seeds_submissao)} seeds x 3 modelos)")

# Converter para preco (espaco original) e clipar valores nao positivos / muito baixos
PRECO_MINIMO = 10_000
preco_predito = np.expm1(pred_log_media)
n_clipados = int((preco_predito < PRECO_MINIMO).sum())
preco_predito = np.clip(preco_predito, PRECO_MINIMO, None)
print(f"Precos abaixo de R$ {PRECO_MINIMO:,.0f} clipados para o minimo: {n_clipados}")

submissao = pd.DataFrame({"Id": test_ids, "preco": preco_predito})

# Verificar formato contra o exemplo antes de salvar
exemplo = pd.read_csv("exemplo_arquivo_respostas.csv")
print(f"\nFormato da submissao: {submissao.shape}  |  formato do exemplo: {exemplo.shape}")
print(f"Colunas da submissao: {list(submissao.columns)}  |  colunas do exemplo: {list(exemplo.columns)}")
assert list(submissao.columns) == list(exemplo.columns), "Colunas da submissao nao batem com o exemplo!"
assert submissao.shape[0] == exemplo.shape[0], "Numero de linhas da submissao nao bate com o exemplo!"
assert submissao["Id"].isin(exemplo["Id"]).all(), "Ha Ids na submissao que nao existem no exemplo!"
assert submissao["Id"].is_unique, "Ha Ids duplicados na submissao!"
print("Formato validado com sucesso contra exemplo_arquivo_respostas.csv.")

submissao.to_csv("submissao_kaggle.csv", index=False)
print("\nSubmissao salva em 'submissao_kaggle.csv'.")


# ---------------------------------------------------------------------------
# 38. Distribuicao das previsoes
# ---------------------------------------------------------------------------
secao("38. DISTRIBUICAO DAS PREVISOES")

stats_pred = submissao["preco"].describe(percentiles=[0.25, 0.5, 0.75])
print("Estatisticas do preco predito (teste):")
print(stats_pred)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

preco_treino_limpo = np.expm1(y_train)
sns.histplot(
    preco_treino_limpo, bins=60, color="#4C72B0", stat="density",
    label="treino (sem outliers)", alpha=0.5, ax=axes[0],
)
sns.histplot(
    submissao["preco"], bins=60, color="#DD8452", stat="density",
    label="teste (previsto)", alpha=0.5, ax=axes[0],
)
axes[0].set_title("Preco: treino vs previsto (teste)")
axes[0].set_xlabel("preco (R$)")
axes[0].legend()

sns.histplot(
    y_train, bins=60, color="#4C72B0", stat="density",
    label="treino (sem outliers)", alpha=0.5, ax=axes[1],
)
sns.histplot(
    pred_log_media, bins=60, color="#DD8452", stat="density",
    label="teste (previsto)", alpha=0.5, ax=axes[1],
)
axes[1].set_title("log1p(preco): treino vs previsto (teste)")
axes[1].set_xlabel("log1p(preco)")
axes[1].legend()

fig.tight_layout()
fig.savefig("eda_submissao_distribuicao.png", dpi=120)
plt.close(fig)
print("\nFigura salva em eda_submissao_distribuicao.png")

print("\nEnsemble e geracao de submissao concluidos.")


# ---------------------------------------------------------------------------
# 39. Melhoria 1 - Target Encoding do bairro (variantes)
# ---------------------------------------------------------------------------
secao("39. MELHORIA 1 - TARGET ENCODING DO BAIRRO (VARIANTES)")

print("Contexto: score no Kaggle = 0.2535, CV do blend/voting (secao 36) = 0.2333.")
print("Gap de ~0.02 sugere leve overfitting -- mesma licao do Trabalho 1 com target encoding")
print("de alta cardinalidade. Testando variantes de encoding do bairro, cada uma avaliada")
print("com 5-fold CV usando o CatBoost otimizado (mesma metodologia em todas as variantes).\n")

# Recuperar a coluna 'bairro' bruta, alinhada com X_train (pos-remocao de outliers) e X_test
bairro_treino = treino.loc[idx_treino_limpo, "bairro"].reset_index(drop=True)
bairro_teste = teste["bairro"].reset_index(drop=True)

FATOR_SMOOTHING = 10
contagem_bairro = bairro_treino.value_counts()
media_bairro = (
    pd.DataFrame({"bairro": bairro_treino, "log_preco": y_train}).groupby("bairro")["log_preco"].mean()
)
media_global_log_preco = y_train.mean()
freq_bairro = bairro_treino.value_counts(normalize=True)
te_bairro_smoothed = (
    contagem_bairro * media_bairro + FATOR_SMOOTHING * media_global_log_preco
) / (contagem_bairro + FATOR_SMOOTHING)


def construir_variante_bairro(modo):
    """Retorna (X_train_variante, X_test_variante) trocando o encoding de bairro."""
    X_tr = X_train.drop(columns=["bairro_target_enc"]).copy()
    X_te = X_test.drop(columns=["bairro_target_enc"]).copy()

    if modo == "smoothed":
        X_tr["bairro_target_enc"] = bairro_treino.map(te_bairro_smoothed).values
        X_te["bairro_target_enc"] = bairro_teste.map(te_bairro_smoothed).fillna(media_global_log_preco).values
    elif modo == "freq":
        X_tr["bairro_freq_enc"] = bairro_treino.map(freq_bairro).values
        X_te["bairro_freq_enc"] = bairro_teste.map(freq_bairro).fillna(0.0).values
    elif modo == "smoothed_freq":
        X_tr["bairro_target_enc"] = bairro_treino.map(te_bairro_smoothed).values
        X_te["bairro_target_enc"] = bairro_teste.map(te_bairro_smoothed).fillna(media_global_log_preco).values
        X_tr["bairro_freq_enc"] = bairro_treino.map(freq_bairro).values
        X_te["bairro_freq_enc"] = bairro_teste.map(freq_bairro).fillna(0.0).values
    return X_tr, X_te


def cv_rmspe_catboost(X, y):
    """5-fold CV (mesmo 'kf' usado em todo o script) com o CatBoost otimizado."""
    scores = cross_val_score(
        CatBoostRegressor(**params_catboost, random_seed=42, verbose=0),
        X, y, cv=kf, scoring=rmspe_scorer, n_jobs=1,
    )
    return -scores


resultados_bairro = []

# a) TE atual (mediana de log_preco por bairro) -- ja computado na secao 32, reaproveitado
rmspe_atual = tab_otimizados.loc[tab_otimizados["modelo"] == "CatBoostRegressor", "rmspe_medio"].iloc[0]
std_atual = tab_otimizados.loc[tab_otimizados["modelo"] == "CatBoostRegressor", "rmspe_std"].iloc[0]
resultados_bairro.append({"variante": "a) TE atual (mediana)", "rmspe_medio": rmspe_atual, "rmspe_std": std_atual})
print(f"a) TE atual (mediana):   RMSPE = {rmspe_atual:.4f} +/- {std_atual:.4f}  (reaproveitado da secao 32)")

# b) TE com smoothing
X_tr_b, X_te_b = construir_variante_bairro("smoothed")
scores_b = cv_rmspe_catboost(X_tr_b, y_train)
resultados_bairro.append({"variante": "b) TE smoothed (m=10)", "rmspe_medio": scores_b.mean(), "rmspe_std": scores_b.std()})
print(f"b) TE smoothed (m=10):   RMSPE = {scores_b.mean():.4f} +/- {scores_b.std():.4f}")

# c) Frequency encoding
X_tr_c, X_te_c = construir_variante_bairro("freq")
scores_c = cv_rmspe_catboost(X_tr_c, y_train)
resultados_bairro.append({"variante": "c) Frequency encoding", "rmspe_medio": scores_c.mean(), "rmspe_std": scores_c.std()})
print(f"c) Frequency encoding:   RMSPE = {scores_c.mean():.4f} +/- {scores_c.std():.4f}")

# d) TE smoothed + frequency encoding combinados
X_tr_d, X_te_d = construir_variante_bairro("smoothed_freq")
scores_d = cv_rmspe_catboost(X_tr_d, y_train)
resultados_bairro.append({"variante": "d) TE smoothed + freq", "rmspe_medio": scores_d.mean(), "rmspe_std": scores_d.std()})
print(f"d) TE smoothed + freq:   RMSPE = {scores_d.mean():.4f} +/- {scores_d.std():.4f}")

tab_bairro = pd.DataFrame(resultados_bairro).sort_values("rmspe_medio").reset_index(drop=True)
tab_bairro.index += 1
print("\nRanking das variantes de encoding de bairro:")
print(tab_bairro.to_string())

variante_vencedora = tab_bairro.iloc[0]["variante"]
melhorou_melhoria1 = tab_bairro.iloc[0]["rmspe_medio"] < rmspe_atual
print(f"\nMelhor variante: {variante_vencedora}")
print(f"Melhoria 1 {'MELHOROU' if melhorou_melhoria1 else 'NAO melhorou'} o RMSPE em relacao ao TE atual.")

variantes_X_bairro = {
    "a) TE atual (mediana)": (X_train, X_test),
    "b) TE smoothed (m=10)": (X_tr_b, X_te_b),
    "c) Frequency encoding": (X_tr_c, X_te_c),
    "d) TE smoothed + freq": (X_tr_d, X_te_d),
}
X_train_melhoria1, X_test_melhoria1 = variantes_X_bairro[variante_vencedora]


# ---------------------------------------------------------------------------
# 40. Melhoria 2 - Features de interacao
# ---------------------------------------------------------------------------
secao("40. MELHORIA 2 - FEATURES DE INTERACAO")

col_bairro_base = "bairro_target_enc" if "bairro_target_enc" in X_train_melhoria1.columns else "bairro_freq_enc"
print(f"Coluna usada como base de 'preco_m2_estimado': {col_bairro_base}")


def adicionar_features_interacao(X):
    X = X.copy()
    X["preco_m2_estimado"] = X[col_bairro_base]
    X["quartos_x_area"] = X["quartos"] * X["log_area_util"]
    X["vagas_x_area"] = X["vagas"] * X["log_area_util"]
    X["suites_ratio_quartos"] = X["suites"] / (X["quartos"] + 1)
    X["luxo_x_area"] = X["total_amenidades"] * X["log_area_util"]
    for col_tipo in colunas_tipo:  # ['tipo_Apartamento', 'tipo_Casa', 'tipo_Loft', 'tipo_Quitinete']
        X[f"{col_tipo}_x_area"] = X[col_tipo] * X["log_area_util"]
    return X


rmspe_base_m2 = tab_bairro.iloc[0]["rmspe_medio"]
std_base_m2 = tab_bairro.iloc[0]["rmspe_std"]
print(f"Base (encoding vencedor da melhoria 1, reaproveitado): RMSPE = {rmspe_base_m2:.4f} +/- {std_base_m2:.4f}")

X_train_m2_extra = adicionar_features_interacao(X_train_melhoria1)
X_test_m2_extra = adicionar_features_interacao(X_test_melhoria1)

scores_m2_extra = cv_rmspe_catboost(X_train_m2_extra, y_train)
print(f"Com features de interacao:                             RMSPE = {scores_m2_extra.mean():.4f} +/- {scores_m2_extra.std():.4f}")

melhorou_melhoria2 = scores_m2_extra.mean() < rmspe_base_m2
print(f"\nMelhoria 2 {'MELHOROU' if melhorou_melhoria2 else 'NAO melhorou'} o RMSPE.")

if melhorou_melhoria2:
    X_train_melhoria2, X_test_melhoria2 = X_train_m2_extra, X_test_m2_extra
    rmspe_melhoria2, std_melhoria2 = float(scores_m2_extra.mean()), float(scores_m2_extra.std())
else:
    X_train_melhoria2, X_test_melhoria2 = X_train_melhoria1, X_test_melhoria1
    rmspe_melhoria2, std_melhoria2 = float(rmspe_base_m2), float(std_base_m2)


# ---------------------------------------------------------------------------
# 41. Melhoria 3 - Remocao mais agressiva de outliers (percentil 99)
# ---------------------------------------------------------------------------
secao("41. MELHORIA 3 - REMOCAO MAIS AGRESSIVA DE OUTLIERS (PERCENTIL 99)")

p99_log = y_train.quantile(0.99)
mask_p99 = y_train <= p99_log
n_removidos_p99 = int((~mask_p99).sum())

X_train_p99 = X_train_melhoria2.loc[mask_p99].reset_index(drop=True)
y_train_p99 = y_train.loc[mask_p99].reset_index(drop=True)

print(f"Percentil 99 de log_preco (dados ja sem outliers IQR): {p99_log:.4f}  (preco ~ R$ {np.expm1(p99_log):,.2f})")
print(f"Imoveis adicionais removidos (acima do p99): {n_removidos_p99}")
print(f"Treino: {len(y_train)} -> {len(y_train_p99)} imoveis")

scores_p99 = cv_rmspe_catboost(X_train_p99, y_train_p99)

print(f"\nRMSPE com outliers removidos via IQR (3x) apenas (melhoria 2, reaproveitado): {rmspe_melhoria2:.4f} +/- {std_melhoria2:.4f}")
print(f"RMSPE com corte adicional no percentil 99:                                    {scores_p99.mean():.4f} +/- {scores_p99.std():.4f}")

melhorou_melhoria3 = scores_p99.mean() < rmspe_melhoria2
print(f"\nMelhoria 3 {'MELHOROU' if melhorou_melhoria3 else 'NAO melhorou'} o RMSPE.")

if melhorou_melhoria3:
    X_train_melhoria3, y_train_melhoria3 = X_train_p99, y_train_p99
    rmspe_melhoria3, std_melhoria3 = float(scores_p99.mean()), float(scores_p99.std())
else:
    X_train_melhoria3, y_train_melhoria3 = X_train_melhoria2, y_train
    rmspe_melhoria3, std_melhoria3 = rmspe_melhoria2, std_melhoria2


# ---------------------------------------------------------------------------
# 42. Resumo das tentativas e geracao de novas submissoes
# ---------------------------------------------------------------------------
secao("42. RESUMO DAS TENTATIVAS E GERACAO DE NOVAS SUBMISSOES")


def gerar_submissao_blend(X_tr, y_tr, X_te, nome_arquivo):
    """Blend de 5 seeds x 3 modelos (LGBM, XGB, CatBoost) no dataset completo fornecido."""
    previsoes = []
    for seed in seeds_submissao:
        for nome, modelo in criar_modelos_otimizados(seed=seed).items():
            modelo.fit(X_tr, y_tr)
            previsoes.append(modelo.predict(X_te))
    pred_log_media = np.mean(previsoes, axis=0)
    preco_pred = np.clip(np.expm1(pred_log_media), PRECO_MINIMO, None)
    sub = pd.DataFrame({"Id": test_ids, "preco": preco_pred})
    sub.to_csv(nome_arquivo, index=False)
    print(f"Submissao salva em '{nome_arquivo}' ({len(sub)} linhas).")
    return sub


tentativas = [
    {"versao": "v1 (baseline, blend/voting secao 36)", "rmspe_cv": 0.2333, "melhorou": "-", "arquivo": "submissao_kaggle.csv"},
    {"versao": f"v2 (melhoria 1: {variante_vencedora})", "rmspe_cv": float(tab_bairro.iloc[0]["rmspe_medio"]), "melhorou": melhorou_melhoria1, "arquivo": None},
    {"versao": "v3 (melhoria 2: features de interacao)", "rmspe_cv": rmspe_melhoria2, "melhorou": melhorou_melhoria2, "arquivo": None},
    {"versao": "v4 (melhoria 3: corte p99)", "rmspe_cv": rmspe_melhoria3, "melhorou": melhorou_melhoria3, "arquivo": None},
]

candidatos_submissao = [
    (X_train_melhoria1, y_train, X_test_melhoria1),
    (X_train_melhoria2, y_train, X_test_melhoria2),
    (X_train_melhoria3, y_train_melhoria3, X_test_melhoria2),
]

contador_versao = 2
alguma_melhoria_gerou_submissao = False
for tentativa, (X_tr_cand, y_tr_cand, X_te_cand) in zip(tentativas[1:], candidatos_submissao):
    if tentativa["melhorou"]:
        nome_arquivo = f"submissao_v{contador_versao}.csv"
        gerar_submissao_blend(X_tr_cand, y_tr_cand, X_te_cand, nome_arquivo)
        tentativa["arquivo"] = nome_arquivo
        alguma_melhoria_gerou_submissao = True
    else:
        print(f"{tentativa['versao']}: nao melhorou o CV -- submissao NAO gerada.")
    contador_versao += 1

if not alguma_melhoria_gerou_submissao:
    print("\nNenhuma das 3 melhorias superou o CV baseline; 'submissao_kaggle.csv' (v1) permanece a melhor.")

tab_tentativas = pd.DataFrame(tentativas)
print("\nResumo de todas as tentativas:")
print(tab_tentativas.to_string(index=False))

print("\nIteracao de melhorias concluida.")


# ---------------------------------------------------------------------------
# 43. Melhoria 4 - Re-otimizar Optuna no dataset v4
# ---------------------------------------------------------------------------
secao("43. MELHORIA 4 - RE-OTIMIZAR OPTUNA NO DATASET V4")

print("Dataset v4: TE smoothed+freq (melhoria 1) + features de interacao (melhoria 2) + corte p99 (melhoria 3).")
print(f"Shape do treino v4: {X_train_melhoria3.shape}")

# 2. Novo split 80/20 estratificado (mesmo esquema da secao 28), agora sobre o dataset v4.
# Reatribuir as globais usadas por avaliar_rmspe_val() e pelas funcoes objetivo_* (secoes 29-31)
# reaproveita o MESMO codigo de otimizacao, so que apontando para os dados do dataset v4 --
# nada nessas funcoes ja foi usado de novo ate aqui, entao a reatribuicao e segura.
bins_estratificacao_v4 = pd.qcut(y_train_melhoria3, q=5, labels=False)
X_opt_treino, X_opt_val, y_opt_treino, y_opt_val = train_test_split(
    X_train_melhoria3, y_train_melhoria3, test_size=0.2, random_state=42, stratify=bins_estratificacao_v4
)
print(f"Split 80/20 estratificado (dataset v4): treino={X_opt_treino.shape}  validacao={X_opt_val.shape}")

# 3. Optuna: LightGBM (50), XGBoost (50), CatBoost (30) -- mesmas funcoes objetivo das secoes 29-31
print("\nRe-otimizando LightGBM (50 trials)...")
study_lgbm_v4 = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
study_lgbm_v4.optimize(objetivo_lgbm, n_trials=50, show_progress_bar=True)
print(f"Melhor RMSPE (validacao) - LightGBM v4: {study_lgbm_v4.best_value:.4f}")

print("\nRe-otimizando XGBoost (50 trials)...")
study_xgb_v4 = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
study_xgb_v4.optimize(objetivo_xgb, n_trials=50, show_progress_bar=True)
print(f"Melhor RMSPE (validacao) - XGBoost v4: {study_xgb_v4.best_value:.4f}")

print("\nRe-otimizando CatBoost (30 trials)...")
study_catboost_v4 = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
study_catboost_v4.optimize(objetivo_catboost, n_trials=30, show_progress_bar=True)
print(f"Melhor RMSPE (validacao) - CatBoost v4: {study_catboost_v4.best_value:.4f}")

# 4. Comparar RMSPE: params antigos (secao 32/33) vs params novos, 5-fold CV no dataset v4
print("\nComparando params antigos vs novos (5-fold CV, dataset v4), por modelo:")

resultados_v4_tuning = []
for nome, modelo_cls, params_antigos, params_novos, seed_kw in [
    ("LGBMRegressor", LGBMRegressor, params_lgbm, study_lgbm_v4.best_params, {"random_state": 42, "verbose": -1}),
    ("XGBRegressor", XGBRegressor, params_xgb, study_xgb_v4.best_params, {"random_state": 42, "verbosity": 0}),
    ("CatBoostRegressor", CatBoostRegressor, params_catboost, study_catboost_v4.best_params, {"random_seed": 42, "verbose": 0}),
]:
    scores_antigo = -cross_val_score(
        modelo_cls(**params_antigos, **seed_kw), X_train_melhoria3, y_train_melhoria3,
        cv=kf, scoring=rmspe_scorer, n_jobs=1,
    )
    scores_novo = -cross_val_score(
        modelo_cls(**params_novos, **seed_kw), X_train_melhoria3, y_train_melhoria3,
        cv=kf, scoring=rmspe_scorer, n_jobs=1,
    )
    melhora_modelo = bool(scores_novo.mean() < scores_antigo.mean())
    resultados_v4_tuning.append(
        {
            "modelo": nome,
            "rmspe_params_antigos": scores_antigo.mean(),
            "rmspe_params_novos": scores_novo.mean(),
            "melhorou": melhora_modelo,
        }
    )
    print(
        f"{nome:20s} antigos={scores_antigo.mean():.4f}  novos={scores_novo.mean():.4f}  "
        f"melhorou={melhora_modelo}"
    )

tab_v4_tuning = pd.DataFrame(resultados_v4_tuning)
print("\nComparacao params antigos vs novos (dataset v4):")
print(tab_v4_tuning.to_string(index=False))


def escolher_params(nome_modelo, params_antigos, params_novos):
    melhorou = tab_v4_tuning.loc[tab_v4_tuning["modelo"] == nome_modelo, "melhorou"].iloc[0]
    return params_novos if melhorou else params_antigos


# 5. Params finais por modelo (novos se melhoraram, senao mantem os antigos)
params_lgbm_final = escolher_params("LGBMRegressor", params_lgbm, study_lgbm_v4.best_params)
params_xgb_final = escolher_params("XGBRegressor", params_xgb, study_xgb_v4.best_params)
params_catboost_final = escolher_params("CatBoostRegressor", params_catboost, study_catboost_v4.best_params)

# Veredito agregado (voting ensemble, 3 modelos) no dataset v4: params antigos vs escolhidos
print("\nVeredito agregado -- voting ensemble (3 modelos) no dataset v4:")


def cv_rmspe_voting(criar_modelos_fn, X, y):
    """5-fold CV do voting ensemble; criar_modelos_fn() retorna um dict fresco {nome: modelo}."""
    scores_fold = []
    for idx_tr, idx_val in kf.split(X):
        X_tr, X_val = X.iloc[idx_tr], X.iloc[idx_val]
        y_tr, y_val = y.iloc[idx_tr], y.iloc[idx_val]
        preds = []
        for modelo in criar_modelos_fn().values():
            modelo.fit(X_tr, y_tr)
            preds.append(modelo.predict(X_val))
        pred_media = np.mean(preds, axis=0)
        scores_fold.append(rmspe(y_val, pred_media))
    return np.array(scores_fold)


def criar_modelos_antigos():
    return {
        "LGBMRegressor": LGBMRegressor(**params_lgbm, random_state=42, verbose=-1),
        "XGBRegressor": XGBRegressor(**params_xgb, random_state=42, verbosity=0),
        "CatBoostRegressor": CatBoostRegressor(**params_catboost, random_seed=42, verbose=0),
    }


def criar_modelos_novos():
    return {
        "LGBMRegressor": LGBMRegressor(**params_lgbm_final, random_state=42, verbose=-1),
        "XGBRegressor": XGBRegressor(**params_xgb_final, random_state=42, verbosity=0),
        "CatBoostRegressor": CatBoostRegressor(**params_catboost_final, random_seed=42, verbose=0),
    }


scores_voting_antigo_v4 = cv_rmspe_voting(criar_modelos_antigos, X_train_melhoria3, y_train_melhoria3)
scores_voting_novo_v4 = cv_rmspe_voting(criar_modelos_novos, X_train_melhoria3, y_train_melhoria3)

print(f"Params antigos: RMSPE = {scores_voting_antigo_v4.mean():.4f} +/- {scores_voting_antigo_v4.std():.4f}")
print(f"Params novos  : RMSPE = {scores_voting_novo_v4.mean():.4f} +/- {scores_voting_novo_v4.std():.4f}")

melhorou_melhoria4 = bool(scores_voting_novo_v4.mean() < scores_voting_antigo_v4.mean())
print(f"\nMelhoria 4 {'MELHOROU' if melhorou_melhoria4 else 'NAO melhorou'} o RMSPE (voting ensemble, dataset v4).")

if melhorou_melhoria4:
    rmspe_melhoria4, std_melhoria4 = float(scores_voting_novo_v4.mean()), float(scores_voting_novo_v4.std())
else:
    # Nao melhorou -- os params "finais" seguem sendo os antigos para as proximas secoes
    params_lgbm_final, params_xgb_final, params_catboost_final = params_lgbm, params_xgb, params_catboost
    rmspe_melhoria4, std_melhoria4 = float(scores_voting_antigo_v4.mean()), float(scores_voting_antigo_v4.std())


# ---------------------------------------------------------------------------
# 44. Melhoria 5 - Weighted blending
# ---------------------------------------------------------------------------
secao("44. MELHORIA 5 - WEIGHTED BLENDING")


def criar_modelos_finais():
    return {
        "LGBMRegressor": LGBMRegressor(**params_lgbm_final, random_state=42, verbose=-1),
        "XGBRegressor": XGBRegressor(**params_xgb_final, random_state=42, verbosity=0),
        "CatBoostRegressor": CatBoostRegressor(**params_catboost_final, random_seed=42, verbose=0),
    }


# 1. RMSPE individual de cada modelo (5-fold CV, dataset v4, params finais da melhoria 4)
rmspe_individuais = {}
for nome, modelo in criar_modelos_finais().items():
    scores = -cross_val_score(
        modelo, X_train_melhoria3, y_train_melhoria3, cv=kf, scoring=rmspe_scorer, n_jobs=1
    )
    rmspe_individuais[nome] = float(scores.mean())
    print(f"{nome:20s} RMSPE individual = {rmspe_individuais[nome]:.4f}")

# 2. Pesos inversamente proporcionais ao RMSPE
inv_rmspe = {nome: 1.0 / v for nome, v in rmspe_individuais.items()}
soma_inv = sum(inv_rmspe.values())
pesos = {nome: v / soma_inv for nome, v in inv_rmspe.items()}
print("\nPesos do blend (inversamente proporcionais ao RMSPE):")
for nome, peso in pesos.items():
    print(f"  {nome:20s} peso = {peso:.4f}")

# 3-4. Blend uniforme vs blend pesado (5-fold CV manual, dataset v4, pesos fixos definidos acima)
scores_uniforme_fold = []
scores_pesado_fold = []
for idx_tr, idx_val in kf.split(X_train_melhoria3):
    X_tr = X_train_melhoria3.iloc[idx_tr]
    X_val = X_train_melhoria3.iloc[idx_val]
    y_tr = y_train_melhoria3.iloc[idx_tr]
    y_val = y_train_melhoria3.iloc[idx_val]

    preds = {}
    for nome, modelo in criar_modelos_finais().items():
        modelo.fit(X_tr, y_tr)
        preds[nome] = modelo.predict(X_val)

    pred_uniforme = np.mean(list(preds.values()), axis=0)
    pred_pesado = sum(pesos[nome] * preds[nome] for nome in preds)

    scores_uniforme_fold.append(rmspe(y_val, pred_uniforme))
    scores_pesado_fold.append(rmspe(y_val, pred_pesado))

scores_uniforme_fold = np.array(scores_uniforme_fold)
scores_pesado_fold = np.array(scores_pesado_fold)

print(f"\nBlend uniforme: RMSPE = {scores_uniforme_fold.mean():.4f} +/- {scores_uniforme_fold.std():.4f}")
print(f"Blend pesado:   RMSPE = {scores_pesado_fold.mean():.4f} +/- {scores_pesado_fold.std():.4f}")

melhorou_melhoria5 = bool(scores_pesado_fold.mean() < scores_uniforme_fold.mean())
print(f"\nMelhoria 5 {'MELHOROU' if melhorou_melhoria5 else 'NAO melhorou'} o RMSPE (blend pesado vs uniforme).")

if melhorou_melhoria5:
    rmspe_melhoria5, std_melhoria5 = float(scores_pesado_fold.mean()), float(scores_pesado_fold.std())
else:
    rmspe_melhoria5, std_melhoria5 = float(scores_uniforme_fold.mean()), float(scores_uniforme_fold.std())


# ---------------------------------------------------------------------------
# 45. Melhoria 6 - Features extras do texto 'diferenciais'
# ---------------------------------------------------------------------------
secao("45. MELHORIA 6 - FEATURES EXTRAS DO TEXTO 'diferenciais'")

# Recalculadas direto de treino/teste originais (nunca modificados) -- evita depender de
# alinhamento por indice atraves de todo o pre-processamento/outliers ja aplicados.
features_luxo_treino = extrair_features_texto_luxo(treino.loc[idx_treino_limpo, "diferenciais"]).reset_index(drop=True)
features_luxo_teste = extrair_features_texto_luxo(teste["diferenciais"]).reset_index(drop=True)

# Mesmo corte de linhas do dataset v4 (melhoria 3 so remove linhas, nao muda a ordem)
if melhorou_melhoria3:
    features_luxo_treino_v4 = features_luxo_treino.loc[mask_p99.values].reset_index(drop=True)
else:
    features_luxo_treino_v4 = features_luxo_treino

X_train_m6_base = X_train_melhoria3
X_test_m6_base = X_test_melhoria2
y_train_m6 = y_train_melhoria3

X_train_m6_extra = pd.concat(
    [X_train_m6_base.reset_index(drop=True), features_luxo_treino_v4.reset_index(drop=True)], axis=1
)
X_test_m6_extra = pd.concat(
    [X_test_m6_base.reset_index(drop=True), features_luxo_teste.reset_index(drop=True)], axis=1
)


def cv_rmspe_catboost_final(X, y):
    scores = cross_val_score(
        CatBoostRegressor(**params_catboost_final, random_seed=42, verbose=0),
        X, y, cv=kf, scoring=rmspe_scorer, n_jobs=1,
    )
    return -scores


scores_m6_base = cv_rmspe_catboost_final(X_train_m6_base, y_train_m6)
scores_m6_extra = cv_rmspe_catboost_final(X_train_m6_extra, y_train_m6)

print(f"Base (dataset v4, sem features de texto):        RMSPE = {scores_m6_base.mean():.4f} +/- {scores_m6_base.std():.4f}")
print(f"Com features extras de texto (chars/keywords):   RMSPE = {scores_m6_extra.mean():.4f} +/- {scores_m6_extra.std():.4f}")

melhorou_melhoria6 = bool(scores_m6_extra.mean() < scores_m6_base.mean())
print(f"\nMelhoria 6 {'MELHOROU' if melhorou_melhoria6 else 'NAO melhorou'} o RMSPE.")

if melhorou_melhoria6:
    X_train_final, X_test_final = X_train_m6_extra, X_test_m6_extra
    rmspe_melhoria6, std_melhoria6 = float(scores_m6_extra.mean()), float(scores_m6_extra.std())
else:
    X_train_final, X_test_final = X_train_m6_base, X_test_m6_base
    rmspe_melhoria6, std_melhoria6 = float(scores_m6_base.mean()), float(scores_m6_base.std())


# ---------------------------------------------------------------------------
# 46. Gerar novas submissoes e resumo final
# ---------------------------------------------------------------------------
secao("46. GERAR NOVAS SUBMISSOES E RESUMO FINAL")


def gerar_submissao_blend_v2(X_tr, y_tr, X_te, nome_arquivo, params_lgbm_uso, params_xgb_uso, params_catboost_uso, pesos_uso=None):
    """Blend de 5 seeds x 3 modelos, com params e (opcionalmente) pesos customizados."""
    previsoes = {"LGBMRegressor": [], "XGBRegressor": [], "CatBoostRegressor": []}
    for seed in seeds_submissao:
        modelos_seed = {
            "LGBMRegressor": LGBMRegressor(**params_lgbm_uso, random_state=seed, verbose=-1),
            "XGBRegressor": XGBRegressor(**params_xgb_uso, random_state=seed, verbosity=0),
            "CatBoostRegressor": CatBoostRegressor(**params_catboost_uso, random_seed=seed, verbose=0),
        }
        for nome, modelo in modelos_seed.items():
            modelo.fit(X_tr, y_tr)
            previsoes[nome].append(modelo.predict(X_te))

    if pesos_uso is None:
        todas_previsoes = [p for lista in previsoes.values() for p in lista]
        pred_log_media = np.mean(todas_previsoes, axis=0)
    else:
        medias_por_modelo = {nome: np.mean(lista, axis=0) for nome, lista in previsoes.items()}
        pred_log_media = sum(pesos_uso[nome] * medias_por_modelo[nome] for nome in medias_por_modelo)

    preco_pred = np.clip(np.expm1(pred_log_media), PRECO_MINIMO, None)
    sub = pd.DataFrame({"Id": test_ids, "preco": preco_pred})
    sub.to_csv(nome_arquivo, index=False)
    print(f"Submissao salva em '{nome_arquivo}' ({len(sub)} linhas).")
    return sub


nova_rodada = []

# v5: efeito isolado do re-optuna (melhoria 4) -- dataset v4, blend uniforme
if melhorou_melhoria4:
    gerar_submissao_blend_v2(
        X_train_melhoria3, y_train_melhoria3, X_test_melhoria2,
        "submissao_v5.csv", params_lgbm_final, params_xgb_final, params_catboost_final,
    )
    arquivo_v5 = "submissao_v5.csv"
else:
    print("Melhoria 4 (re-optuna) nao melhorou -- 'submissao_v5.csv' NAO gerada.")
    arquivo_v5 = None
nova_rodada.append(
    {"versao": "v5 (melhoria 4: re-optuna no dataset v4)", "rmspe_cv": rmspe_melhoria4, "melhorou": melhorou_melhoria4, "arquivo": arquivo_v5}
)

# v6: melhor combinacao cumulativa (params da melhoria 4 + pesos da melhoria 5, se ajudou +
# features de texto da melhoria 6, se ajudou)
alguma_melhoria_5_ou_6 = melhorou_melhoria5 or melhorou_melhoria6
gerar_v6 = melhorou_melhoria4 or alguma_melhoria_5_ou_6
if gerar_v6:
    pesos_finais = pesos if melhorou_melhoria5 else None
    gerar_submissao_blend_v2(
        X_train_final, y_train_m6, X_test_final,
        "submissao_v6.csv", params_lgbm_final, params_xgb_final, params_catboost_final,
        pesos_uso=pesos_finais,
    )
    arquivo_v6 = "submissao_v6.csv"
else:
    print("Nenhuma melhoria adicional (4, 5 ou 6) sobre o dataset v4 -- 'submissao_v6.csv' NAO gerada.")
    arquivo_v6 = None
nova_rodada.append(
    {"versao": "v6 (melhor combo: melhorias 4+5+6)", "rmspe_cv": rmspe_melhoria6, "melhorou": gerar_v6, "arquivo": arquivo_v6}
)

tab_nova_rodada = pd.DataFrame(nova_rodada)
print("\nResumo desta rodada (melhorias 4-6):")
print(tab_nova_rodada.to_string(index=False))

print("\nNota sobre as metricas: v1-v4 e v5 (agregado) usam RMSPE do voting ensemble (3 modelos);")
print("as comparacoes internas das melhorias 1, 2, 3 e 6 usam CatBoost isolado (mais rapido de")
print("iterar); a melhoria 5 compara blend uniforme vs pesado. Bases nao sao 100% intercambiaveis,")
print("mas cada linha documenta corretamente o que foi comparado contra o que na sua propria etapa.")

tab_tentativas_completa = pd.concat([tab_tentativas, tab_nova_rodada], ignore_index=True)
print("\nResumo de TODAS as tentativas ate agora:")
print(tab_tentativas_completa.to_string(index=False))

print("\nSegunda rodada de melhorias concluida.")


# ---------------------------------------------------------------------------
# 47. Melhoria 7 - Stacking com meta-learner
# ---------------------------------------------------------------------------
secao("47. MELHORIA 7 - STACKING COM META-LEARNER")

NOMES_MODELOS_STACK = ["LGBMRegressor", "XGBRegressor", "CatBoostRegressor"]


def gerar_meta_features_oof(X, y):
    """OOF (out-of-fold) previsoes dos 3 modelos base, via o mesmo 'kf' usado em todo o script."""
    meta = np.zeros((len(X), 3))
    for idx_tr, idx_val in kf.split(X):
        X_tr, y_tr = X.iloc[idx_tr], y.iloc[idx_tr]
        X_val = X.iloc[idx_val]
        for col, modelo in enumerate(criar_modelos_finais().values()):
            modelo.fit(X_tr, y_tr)
            meta[idx_val, col] = modelo.predict(X_val)
    return pd.DataFrame(meta, columns=NOMES_MODELOS_STACK)


def gerar_previsao_stacking(X_tr, y_tr, X_te):
    """Stacking completo: OOF no treino -> treina Ridge (meta-learner) -> preve teste."""
    meta_treino = gerar_meta_features_oof(X_tr, y_tr)
    meta_learner_final = Ridge(alpha=1.0)
    meta_learner_final.fit(meta_treino, y_tr)

    meta_teste = np.zeros((len(X_te), 3))
    for col, modelo in enumerate(criar_modelos_finais().values()):
        modelo.fit(X_tr, y_tr)
        meta_teste[:, col] = modelo.predict(X_te)
    meta_teste_df = pd.DataFrame(meta_teste, columns=meta_treino.columns)

    return meta_learner_final.predict(meta_teste_df)


# 1-2. Meta-features OOF (5-fold CV) sobre o dataset atual (v4 + melhorias 1-3, sem melhoria 6)
meta_X_treino = gerar_meta_features_oof(X_train_final, y_train_m6)
print(f"Meta-features OOF geradas: {meta_X_treino.shape}")
print(meta_X_treino.describe())

# 3. Ridge como meta-learner sobre as meta-features -> log_preco
meta_learner = Ridge(alpha=1.0)
meta_learner.fit(meta_X_treino, y_train_m6)
print(f"\nCoeficientes do meta-learner: {dict(zip(meta_X_treino.columns, meta_learner.coef_))}")
print(f"Intercepto: {meta_learner.intercept_:.4f}")

# 5. Comparar via CV: stacking vs voting (melhoria 4) vs blend pesado (melhoria 5).
# O stacking reaproveita os MESMOS folds do 'kf' usados para gerar as OOF -- pratica padrao
# (nao e uma nested CV completa, mas o meta-learner e um Ridge de baixa capacidade sobre
# so 3 features, entao o otimismo residual e pequeno).
scores_stacking = -cross_val_score(
    Ridge(alpha=1.0), meta_X_treino, y_train_m6, cv=kf, scoring=rmspe_scorer, n_jobs=1
)
print(f"\nStacking (Ridge sobre OOF):                     RMSPE = {scores_stacking.mean():.4f} +/- {scores_stacking.std():.4f}")
print(f"Voting (melhoria 4, dataset v4, params novos):  RMSPE = {rmspe_melhoria4:.4f} +/- {std_melhoria4:.4f}")
print(f"Blend pesado (melhoria 5):                       RMSPE = {rmspe_melhoria5:.4f} +/- {std_melhoria5:.4f}")

rmspe_referencia_ensemble = min(rmspe_melhoria4, rmspe_melhoria5)
std_referencia_ensemble = std_melhoria4 if rmspe_melhoria4 <= rmspe_melhoria5 else std_melhoria5
melhorou_melhoria7 = bool(scores_stacking.mean() < rmspe_referencia_ensemble)
print(f"\nMelhoria 7 (stacking) {'MELHOROU' if melhorou_melhoria7 else 'NAO melhorou'} em relacao a voting/blend.")

if melhorou_melhoria7:
    rmspe_melhoria7, std_melhoria7 = float(scores_stacking.mean()), float(scores_stacking.std())
else:
    rmspe_melhoria7, std_melhoria7 = float(rmspe_referencia_ensemble), float(std_referencia_ensemble)


# ---------------------------------------------------------------------------
# 48. Melhoria 8 - Encoding de bairro com Leave-One-Out
# ---------------------------------------------------------------------------
secao("48. MELHORIA 8 - ENCODING DE BAIRRO COM LEAVE-ONE-OUT")

# Realinhar 'bairro' bruto ao dataset atual (mesmo corte de linhas da melhoria 3, se aplicado)
if melhorou_melhoria3:
    bairro_treino_v4 = bairro_treino.loc[mask_p99.values].reset_index(drop=True)
else:
    bairro_treino_v4 = bairro_treino

col_bairro_atual = "bairro_target_enc" if "bairro_target_enc" in X_train_final.columns else "bairro_freq_enc"
print(f"Coluna de TE substituida pelo LOO: {col_bairro_atual}")

soma_por_bairro = pd.DataFrame({"bairro": bairro_treino_v4, "log_preco": y_train_m6}).groupby("bairro")["log_preco"].sum()
contagem_por_bairro_v4 = bairro_treino_v4.value_counts()
media_por_bairro_v4 = pd.DataFrame({"bairro": bairro_treino_v4, "log_preco": y_train_m6}).groupby("bairro")["log_preco"].mean()
media_global_v4 = y_train_m6.mean()

soma_bairro_row = bairro_treino_v4.map(soma_por_bairro).values
contagem_bairro_row = bairro_treino_v4.map(contagem_por_bairro_v4).values
y_values = y_train_m6.values

# 1. LOO: media do bairro EXCLUINDO a propria amostra (fallback p/ bairros com 1 unica amostra)
divisor_loo_seguro = np.where(contagem_bairro_row > 1, contagem_bairro_row - 1, 1)  # evita divisao por zero
loo_te_treino = np.where(
    contagem_bairro_row > 1,
    (soma_bairro_row - y_values) / divisor_loo_seguro,
    media_global_v4,
)
# 2. Teste: media do bairro no treino completo (nao ha "propria amostra" a excluir)
loo_te_teste = bairro_teste.map(media_por_bairro_v4).fillna(media_global_v4).values

X_train_m8_base = X_train_final
X_test_m8_base = X_test_final

X_train_m8_loo = X_train_m8_base.copy()
X_train_m8_loo[col_bairro_atual] = loo_te_treino
X_test_m8_loo = X_test_m8_base.copy()
X_test_m8_loo[col_bairro_atual] = loo_te_teste


def avaliar_loo_te_cv(X_outras, bairro_serie, y):
    """5-fold CV honesto para o LOO target encoding.

    CUIDADO: aplicar o LOO globalmente (uma unica vez, fora da CV) e depois avaliar com
    cross_val_score vaza dados entre folds -- duas amostras do MESMO bairro em folds
    diferentes continuam se enxergando pela estatistica agregada compartilhada, mesmo cada
    uma excluindo so o proprio valor. Por isso o encoding e recalculado AQUI DENTRO de cada
    fold, usando somente o treino daquele fold (LOO no treino; media simples, sem exclusao,
    no fold de validacao -- que ja e honesto porque a validacao nao participa do agregado).
    """
    scores = []
    for idx_tr, idx_val in kf.split(X_outras):
        bairro_tr = bairro_serie.iloc[idx_tr].reset_index(drop=True)
        bairro_val = bairro_serie.iloc[idx_val].reset_index(drop=True)
        y_tr = y.iloc[idx_tr].reset_index(drop=True)
        y_val = y.iloc[idx_val]

        soma_fold = pd.DataFrame({"bairro": bairro_tr, "y": y_tr}).groupby("bairro")["y"].sum()
        contagem_fold = bairro_tr.value_counts()
        media_fold = pd.DataFrame({"bairro": bairro_tr, "y": y_tr}).groupby("bairro")["y"].mean()
        media_global_fold = y_tr.mean()

        soma_row = bairro_tr.map(soma_fold).values
        contagem_row = bairro_tr.map(contagem_fold).values
        divisor_seguro = np.where(contagem_row > 1, contagem_row - 1, 1)
        loo_tr = np.where(contagem_row > 1, (soma_row - y_tr.values) / divisor_seguro, media_global_fold)
        te_val = bairro_val.map(media_fold).fillna(media_global_fold).values

        X_tr_fold = X_outras.iloc[idx_tr].copy()
        X_tr_fold[col_bairro_atual] = loo_tr
        X_val_fold = X_outras.iloc[idx_val].copy()
        X_val_fold[col_bairro_atual] = te_val

        modelo = CatBoostRegressor(**params_catboost_final, random_seed=42, verbose=0)
        modelo.fit(X_tr_fold, y_tr)
        pred = modelo.predict(X_val_fold)
        scores.append(rmspe(y_val, pred))
    return np.array(scores)


# 3. Comparar via CV com o smoothed TE atual (vencedor da melhoria 1) -- LOO reavaliado
# fold-a-fold (ver docstring de avaliar_loo_te_cv) para nao inflar o RMSPE por vazamento.
scores_te_atual = cv_rmspe_catboost_final(X_train_m8_base, y_train_m6)
scores_loo = avaliar_loo_te_cv(X_train_m8_base, bairro_treino_v4, y_train_m6)

print(f"TE smoothed atual (melhoria 1):  RMSPE = {scores_te_atual.mean():.4f} +/- {scores_te_atual.std():.4f}")
print(f"LOO target encoding:             RMSPE = {scores_loo.mean():.4f} +/- {scores_loo.std():.4f}")

melhorou_melhoria8 = bool(scores_loo.mean() < scores_te_atual.mean())
print(f"\nMelhoria 8 {'MELHOROU' if melhorou_melhoria8 else 'NAO melhorou'} o RMSPE.")

if melhorou_melhoria8:
    X_train_m8_win, X_test_m8_win = X_train_m8_loo, X_test_m8_loo
    rmspe_melhoria8, std_melhoria8 = float(scores_loo.mean()), float(scores_loo.std())
else:
    X_train_m8_win, X_test_m8_win = X_train_m8_base, X_test_m8_base
    rmspe_melhoria8, std_melhoria8 = float(scores_te_atual.mean()), float(scores_te_atual.std())


# ---------------------------------------------------------------------------
# 49. Melhoria 9 - Features de preco por m2
# ---------------------------------------------------------------------------
secao("49. MELHORIA 9 - FEATURES DE PRECO POR M2")

area_util_treino_m9 = X_train_m8_win["area_util"]
preco_v4 = np.expm1(y_train_m6)
preco_por_m2 = preco_v4 / area_util_treino_m9.replace(0, np.nan)

FATOR_SMOOTHING_M2 = 10
mediana_preco_m2_global = float(preco_por_m2.median())

tmp_m2 = pd.DataFrame({"bairro": bairro_treino_v4, "preco_m2": preco_por_m2})
contagem_bairro_m2 = tmp_m2.groupby("bairro")["preco_m2"].count()
mediana_bairro_m2 = tmp_m2.groupby("bairro")["preco_m2"].median()

# 1. bairro_preco_m2 = mediana(preco / area_util) por bairro, com smoothing (mesma logica da
# melhoria 1, agora aplicada a mediana do preco/m2 em vez da mediana/media de log_preco)
bairro_preco_m2_smoothed = (
    contagem_bairro_m2 * mediana_bairro_m2 + FATOR_SMOOTHING_M2 * mediana_preco_m2_global
) / (contagem_bairro_m2 + FATOR_SMOOTHING_M2)

# CUIDADO com leakage: bairro_preco_m2 calculado SOMENTE com dados de treino (acima).
bairro_preco_m2_treino = bairro_treino_v4.map(bairro_preco_m2_smoothed).fillna(mediana_preco_m2_global).values
bairro_preco_m2_teste = bairro_teste.map(bairro_preco_m2_smoothed).fillna(mediana_preco_m2_global).values

# 2-3. preco_m2_estimado = bairro_preco_m2 * area_util ; log_preco_m2_estimado = log1p(...)
preco_m2_estimado_treino = bairro_preco_m2_treino * X_train_m8_win["area_util"].values
preco_m2_estimado_teste = bairro_preco_m2_teste * X_test_m8_win["area_util"].values

if "preco_m2_estimado" in X_train_m8_win.columns:
    print("Nota: 'preco_m2_estimado' ja existia (era um alias de bairro_target_enc, criado na")
    print("melhoria 2) -- sobrescrevendo com o calculo real de preco por m2 desta melhoria.")

X_train_m9_extra = X_train_m8_win.copy()
X_train_m9_extra["bairro_preco_m2"] = bairro_preco_m2_treino
X_train_m9_extra["preco_m2_estimado"] = preco_m2_estimado_treino
X_train_m9_extra["log_preco_m2_estimado"] = np.log1p(np.clip(preco_m2_estimado_treino, 0, None))

X_test_m9_extra = X_test_m8_win.copy()
X_test_m9_extra["bairro_preco_m2"] = bairro_preco_m2_teste
X_test_m9_extra["preco_m2_estimado"] = preco_m2_estimado_teste
X_test_m9_extra["log_preco_m2_estimado"] = np.log1p(np.clip(preco_m2_estimado_teste, 0, None))

# Base = vencedor da melhoria 8, reaproveitado (evita recomputar 5 fits do CatBoost de novo)
rmspe_base_m9, std_base_m9 = rmspe_melhoria8, std_melhoria8
scores_m9_extra = cv_rmspe_catboost_final(X_train_m9_extra, y_train_m6)

print(f"Base (vencedor da melhoria 8, reaproveitado): RMSPE = {rmspe_base_m9:.4f} +/- {std_base_m9:.4f}")
print(f"Com features de preco por m2:                  RMSPE = {scores_m9_extra.mean():.4f} +/- {scores_m9_extra.std():.4f}")

melhorou_melhoria9 = bool(scores_m9_extra.mean() < rmspe_base_m9)
print(f"\nMelhoria 9 {'MELHOROU' if melhorou_melhoria9 else 'NAO melhorou'} o RMSPE.")

if melhorou_melhoria9:
    X_train_m9_win, X_test_m9_win = X_train_m9_extra, X_test_m9_extra
    rmspe_melhoria9, std_melhoria9 = float(scores_m9_extra.mean()), float(scores_m9_extra.std())
else:
    X_train_m9_win, X_test_m9_win = X_train_m8_win, X_test_m8_win
    rmspe_melhoria9, std_melhoria9 = rmspe_base_m9, std_base_m9


# ---------------------------------------------------------------------------
# 50. Melhoria 10 - Binning de variaveis numericas
# ---------------------------------------------------------------------------
secao("50. MELHORIA 10 - BINNING DE VARIAVEIS NUMERICAS")

# 1. area_util_bin: qcut definido no TREINO, bordas reaplicadas ao teste (evita leakage/
# inconsistencia de rotulos entre treino e teste)
area_util_bin_treino, bin_edges = pd.qcut(
    X_train_m9_win["area_util"], q=10, labels=False, duplicates="drop", retbins=True
)
bin_edges_ajustados = bin_edges.copy()
bin_edges_ajustados[0] = -np.inf
bin_edges_ajustados[-1] = np.inf
area_util_bin_teste = pd.cut(X_test_m9_win["area_util"], bins=bin_edges_ajustados, labels=False, include_lowest=True)

# 2. quartos_vagas: interacao categorica simples (quartos*10 + vagas)
quartos_vagas_treino = X_train_m9_win["quartos"] * 10 + X_train_m9_win["vagas"]
quartos_vagas_teste = X_test_m9_win["quartos"] * 10 + X_test_m9_win["vagas"]

X_train_m10_extra = X_train_m9_win.copy()
X_train_m10_extra["area_util_bin"] = area_util_bin_treino.values
X_train_m10_extra["quartos_vagas"] = quartos_vagas_treino.values

X_test_m10_extra = X_test_m9_win.copy()
X_test_m10_extra["area_util_bin"] = area_util_bin_teste.values
X_test_m10_extra["quartos_vagas"] = quartos_vagas_teste.values

print(f"area_util_bin: {int(area_util_bin_treino.nunique())} bins (qcut, q=10, duplicates='drop')")

rmspe_base_m10, std_base_m10 = rmspe_melhoria9, std_melhoria9
scores_m10_extra = cv_rmspe_catboost_final(X_train_m10_extra, y_train_m6)

print(f"\nBase (vencedor da melhoria 9, reaproveitado): RMSPE = {rmspe_base_m10:.4f} +/- {std_base_m10:.4f}")
print(f"Com binning (area_util_bin + quartos_vagas):   RMSPE = {scores_m10_extra.mean():.4f} +/- {scores_m10_extra.std():.4f}")

melhorou_melhoria10 = bool(scores_m10_extra.mean() < rmspe_base_m10)
print(f"\nMelhoria 10 {'MELHOROU' if melhorou_melhoria10 else 'NAO melhorou'} o RMSPE.")

if melhorou_melhoria10:
    X_train_m10_win, X_test_m10_win = X_train_m10_extra, X_test_m10_extra
    rmspe_melhoria10, std_melhoria10 = float(scores_m10_extra.mean()), float(scores_m10_extra.std())
else:
    X_train_m10_win, X_test_m10_win = X_train_m9_win, X_test_m9_win
    rmspe_melhoria10, std_melhoria10 = rmspe_base_m10, std_base_m10


# ---------------------------------------------------------------------------
# 51. Gerar novas submissoes (v7 stacking, v8 melhor combo) e resumo final
# ---------------------------------------------------------------------------
secao("51. GERAR NOVAS SUBMISSOES (V7, V8) E RESUMO FINAL")

# v7: efeito isolado do stacking (melhoria 7), sobre o dataset v4 (mesmo da comparacao original)
if melhorou_melhoria7:
    pred_log_v7 = gerar_previsao_stacking(X_train_final, y_train_m6, X_test_final)
    preco_v7 = np.clip(np.expm1(pred_log_v7), PRECO_MINIMO, None)
    sub_v7 = pd.DataFrame({"Id": test_ids, "preco": preco_v7})
    sub_v7.to_csv("submissao_v7.csv", index=False)
    print(f"Submissao salva em 'submissao_v7.csv' ({len(sub_v7)} linhas).")
    arquivo_v7 = "submissao_v7.csv"
else:
    print("Melhoria 7 (stacking) nao superou voting/blend -- 'submissao_v7.csv' NAO gerada.")
    arquivo_v7 = None

# v8: melhor combo -- dataset final (melhorias 8+9+10 encadeadas) + stacking (se melhoria 7
# venceu) ou blend uniforme/pesado (se nao), sempre gerada como o checkpoint mais atual.
print(f"\nDataset final (melhorias 8+9+10): {X_train_m10_win.shape}")
if melhorou_melhoria7:
    pred_log_v8 = gerar_previsao_stacking(X_train_m10_win, y_train_m6, X_test_m10_win)
    preco_v8 = np.clip(np.expm1(pred_log_v8), PRECO_MINIMO, None)
    sub_v8 = pd.DataFrame({"Id": test_ids, "preco": preco_v8})
    sub_v8.to_csv("submissao_v8.csv", index=False)
    print(f"Submissao salva em 'submissao_v8.csv' ({len(sub_v8)} linhas) -- combo com stacking.")
else:
    pesos_finais_v8 = pesos if melhorou_melhoria5 else None
    gerar_submissao_blend_v2(
        X_train_m10_win, y_train_m6, X_test_m10_win,
        "submissao_v8.csv", params_lgbm_final, params_xgb_final, params_catboost_final,
        pesos_uso=pesos_finais_v8,
    )
arquivo_v8 = "submissao_v8.csv"

terceira_rodada = pd.DataFrame(
    [
        {"versao": "v7 (melhoria 7: stacking)", "rmspe_cv": rmspe_melhoria7, "melhorou": melhorou_melhoria7, "arquivo": arquivo_v7},
        {"versao": "v8 (melhor combo: melhorias 7+8+9+10)", "rmspe_cv": rmspe_melhoria10, "melhorou": (melhorou_melhoria7 or melhorou_melhoria8 or melhorou_melhoria9 or melhorou_melhoria10), "arquivo": arquivo_v8},
    ]
)
print("\nResumo desta rodada (melhorias 7-10):")
print(terceira_rodada.to_string(index=False))

print("\nNota sobre as metricas (continuacao): v7 e a comparacao da melhoria 7 usam a mesma base")
print("de voting/blend das secoes 43-44; as melhorias 8, 9 e 10 usam CatBoost isolado (mesmo padrao")
print("das melhorias 1-3 e 6). O rmspe_cv de v8 reflete o CatBoost isolado no dataset final, NAO o")
print("stacking/blend que a submissao realmente usa -- mesma ressalva ja documentada para v6.")

tab_tentativas_final = pd.concat([tab_tentativas_completa, terceira_rodada], ignore_index=True)
print("\nResumo de TODAS as tentativas (v1-v8):")
print(tab_tentativas_final.to_string(index=False))

print("\nTerceira rodada de melhorias concluida.")
