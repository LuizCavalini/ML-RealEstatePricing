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
ensemble (voting e blend multi-seed) e geracao da submissao (secoes 33-38).
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
