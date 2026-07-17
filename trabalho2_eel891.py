"""
Trabalho 2 - EEL891 - Analise Exploratoria de Dados
Problema de regressao: estimar preco de venda de imoveis.
Metrica de avaliacao: RMSPE (Root Mean Square Percentage Error).

Apenas EDA. Nenhum modelo e treinado aqui.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

print("\nEDA concluida. Nenhum modelo foi treinado neste script.")
