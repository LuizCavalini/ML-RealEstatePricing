"""
================================================================================
EEL891 - Introdução ao Aprendizado de Máquina (2025-2)
Trabalho 2 - Regressão de Preços de Imóveis (Kaggle)
Aluno: Luiz Felipe Píccoli Cavalini
Resultado: RMSPE 0.2393 no leaderboard público do Kaggle
================================================================================

PIPELINE:
  1. Carregamento e EDA
  2. Tratamento de outliers (IQR 3x + percentil 99)
  3. Pré-processamento e imputação
  4. Engenharia de features
  5. Encoding (one-hot + smoothed target encoding + frequency encoding)
  6. Modelagem: 8 modelos diversos
  7. Ensemble com pesos otimizados (SLSQP sobre previsões OOF)
  8. Geração da submissão

NOTA: o script trabalho2_eel891.py contém a exploração completa, incluindo
todas as abordagens testadas e descartadas descritas no relatório
(relatorio_eel891.tex) -- LOO encoding, stacking, blending de submissões,
feature selection, multi-seed, binning, etc. Este script contém SOMENTE o
caminho que produziu o melhor resultado, para leitura e reprodução rápidas.
Os hiperparâmetros abaixo foram obtidos via Optuna (150/150/80 trials para
LGBM/XGB/CatBoost, 25 trials para RF/ExtraTrees); o script de busca está
em otimizacao_local.py.

INSTALAR:
  pip install pandas numpy matplotlib seaborn scikit-learn lightgbm xgboost \
              catboost scipy

RODAR:
  python trabalho2_final.py
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from scipy.optimize import minimize
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 160)
sns.set_theme(style="whitegrid")

TREINO_PATH = "conjunto_de_treinamento.csv"
TESTE_PATH = "conjunto_de_teste.csv"
EXEMPLO_PATH = "exemplo_arquivo_respostas.csv"

COLS_BINARIAS = [
    "churrasqueira", "estacionamento", "piscina", "playground", "quadra",
    "s_festas", "s_jogos", "s_ginastica", "sauna", "vista_mar",
]
COLS_NUMERICAS = ["quartos", "suites", "vagas", "area_util", "area_extra"]

FATOR_SMOOTHING = 10       # target encoding do bairro (log_preco)
FATOR_SMOOTHING_M2 = 10    # target encoding do preco/m2 por bairro
PRECO_MINIMO = 10_000      # clip de seguranca na submissao
SEED = 42


def secao(titulo):
    print("\n" + "=" * 80)
    print(titulo)
    print("=" * 80)


# =============================================================================
# 1. CARREGAMENTO E EDA
# =============================================================================
secao("1. CARREGAMENTO E EDA")

treino = pd.read_csv(TREINO_PATH)
teste = pd.read_csv(TESTE_PATH)
print(f"Treino: {treino.shape}   Teste: {teste.shape}")
print(f"Nulos no treino: {int(treino.isnull().sum().sum())}   Nulos no teste: {int(teste.isnull().sum().sum())}")

# O preco bruto e fortemente assimetrico (cauda longa de imoveis caros); log1p(preco) se
# aproxima de uma normal -- por isso todo o treinamento usa log1p(preco) como target (a
# metrica RMSPE penaliza erro PERCENTUAL, que no espaco log vira aproximadamente erro
# absoluto, alinhando a loss padrao dos modelos com a metrica real de avaliacao).
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
sns.histplot(treino["preco"], bins=60, ax=axes[0], color="#4C72B0")
axes[0].set_title("Distribuicao do preco")
axes[0].set_xlabel("preco (R$)")
sns.histplot(np.log1p(treino["preco"]), bins=60, ax=axes[1], color="#DD8452")
axes[1].set_title("Distribuicao do log1p(preco)")
axes[1].set_xlabel("log1p(preco)")
fig.tight_layout()
fig.savefig("eda_preco.png", dpi=120)
plt.close(fig)

# Correlacoes lineares das numericas com o preco bruto sao todas fracas (|r| < 0.05) --
# reforca a necessidade de features de interacao nao-lineares (secao 4).
corr = treino[COLS_NUMERICAS + ["preco"]].corr()
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
ax.set_title("Correlacao entre numericas e preco")
fig.tight_layout()
fig.savefig("eda_correlacoes.png", dpi=120)
plt.close(fig)
print("Figuras salvas: eda_preco.png, eda_correlacoes.png")


# =============================================================================
# 2. TRATAMENTO DE OUTLIERS
# =============================================================================
secao("2. TRATAMENTO DE OUTLIERS")

target = treino["preco"].copy()
test_ids = teste["Id"].copy()

treino_prep = treino.copy()
treino_prep["log_preco"] = np.log1p(treino_prep["preco"])
teste_prep = teste.copy()
treino_prep["is_train"] = 1
teste_prep["is_train"] = 0
dados = pd.concat([treino_prep, teste_prep], axis=0, ignore_index=True, sort=False)
dados = dados.drop(columns=["Id", "preco"])

# Baselines sem tratamento de outliers chegavam a RMSPE de 200%-4800% -- sinal de que
# poucos imoveis com precos absurdos dominavam a metrica percentual. IQR x3 (mais
# permissivo que o 1.5 classico, para nao descartar imoveis de alto padrao legitimos)
# sobre log_preco captura exatamente esses casos (ex.: R$630M, R$340M, um provavel erro
# de digitacao de R$750).
log_preco_treino = dados.loc[dados["is_train"] == 1, "log_preco"]
q1, q3 = log_preco_treino.quantile(0.25), log_preco_treino.quantile(0.75)
iqr = q3 - q1
limite_inferior, limite_superior = q1 - 3 * iqr, q3 + 3 * iqr
mask_outlier_iqr = (dados["is_train"] == 1) & (
    (dados["log_preco"] > limite_superior) | (dados["log_preco"] < limite_inferior)
)
n_outliers_iqr = int(mask_outlier_iqr.sum())
dados = dados.drop(index=dados.index[mask_outlier_iqr])
print(f"Outliers removidos via IQR (3x) sobre log_preco: {n_outliers_iqr}")

y_train = dados.loc[dados["is_train"] == 1, "log_preco"].reset_index(drop=True)
X_train = dados.loc[dados["is_train"] == 1].drop(columns=["is_train", "log_preco"]).reset_index(drop=True)
X_test = dados.loc[dados["is_train"] == 0].drop(columns=["is_train", "log_preco"]).reset_index(drop=True)

# Corte adicional, mais agressivo, no percentil 99 -- reduziu ainda mais o RMSPE em
# validacao cruzada (0.2306 -> 0.2240). So aplicado ao treino; o teste nunca e alterado.
p99_log = y_train.quantile(0.99)
mask_p99 = y_train <= p99_log
n_removidos_p99 = int((~mask_p99).sum())

bairro_treino = X_train["bairro"].copy()  # preservado p/ o encoding de bairro (secao 5)
X_train = X_train.loc[mask_p99].reset_index(drop=True)
y_train = y_train.loc[mask_p99].reset_index(drop=True)
bairro_treino = bairro_treino.loc[mask_p99.values].reset_index(drop=True)

print(f"Imoveis adicionais removidos (corte percentil 99): {n_removidos_p99}")
print(f"Treino: {len(treino)} -> {len(X_train)} imoveis (teste inalterado: {len(X_test)})")


# =============================================================================
# 3. PRE-PROCESSAMENTO
# =============================================================================
secao("3. PRE-PROCESSAMENTO")

# 'diferenciais' e texto livre com vocabulario fixo (combinacoes das 10 amenidades
# binarias) -- extrai-se contagem de palavras e flag de nulo antes de descartar a coluna.
X_train["n_palavras_diferenciais"] = X_train["diferenciais"].fillna("").str.split().apply(len)
X_train["tem_diferenciais"] = X_train["diferenciais"].notnull().astype(int)
X_test["n_palavras_diferenciais"] = X_test["diferenciais"].fillna("").str.split().apply(len)
X_test["tem_diferenciais"] = X_test["diferenciais"].notnull().astype(int)
X_train = X_train.drop(columns=["diferenciais"])
X_test = X_test.drop(columns=["diferenciais"])

# Imputacao defensiva (este dataset nao tem nulos, mas o pipeline fica robusto a novos
# dados): area_extra ausente = sem area extra; demais numericas -> mediana do treino.
X_train["area_extra"] = X_train["area_extra"].fillna(0)
X_test["area_extra"] = X_test["area_extra"].fillna(0)
for col in [c for c in COLS_NUMERICAS if c != "area_extra"]:
    mediana = X_train[col].median()
    X_train[col] = X_train[col].fillna(mediana)
    X_test[col] = X_test[col].fillna(mediana)

print(f"NaN em X_train: {int(X_train.isnull().sum().sum())}   NaN em X_test: {int(X_test.isnull().sum().sum())}")


# =============================================================================
# 4. ENGENHARIA DE FEATURES
# =============================================================================
secao("4. ENGENHARIA DE FEATURES")

for X in (X_train, X_test):
    X["area_total"] = X["area_util"] + X["area_extra"]
    X["log_area_util"] = np.log1p(X["area_util"])
    X["log_area_total"] = np.log1p(X["area_total"])
    X["area_por_quarto"] = X["area_util"] / (X["quartos"] + 1)
    X["tem_suite"] = (X["suites"] > 0).astype(int)
    X["proporcao_suites"] = X["suites"] / (X["quartos"] + 1)
    X["total_amenidades"] = X[COLS_BINARIAS].sum(axis=1)
    X["categoria_luxo"] = ((X["piscina"] + X["sauna"] + X["vista_mar"]) >= 2).astype(int)
    X["tem_area_extra"] = (X["area_extra"] > 0).astype(int)

print("Features de tamanho/amenidades criadas: area_total, log_area_util, log_area_total,")
print("area_por_quarto, tem_suite, proporcao_suites, total_amenidades, categoria_luxo, tem_area_extra")


# =============================================================================
# 5. ENCODING
# =============================================================================
secao("5. ENCODING")

# One-hot para 'tipo' (4 categorias, baixa cardinalidade)
colunas_antes = set(X_train.columns)
X_train = pd.get_dummies(X_train, columns=["tipo"], prefix="tipo", dtype=int)
X_test = pd.get_dummies(X_test, columns=["tipo"], prefix="tipo", dtype=int)
colunas_tipo = sorted(set(X_train.columns) - colunas_antes)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)  # garante mesmas colunas/ordem

# Label (binaria) para 'tipo_vendedor'
X_train["tipo_vendedor_imobiliaria"] = (X_train["tipo_vendedor"] == "Imobiliaria").astype(int)
X_test["tipo_vendedor_imobiliaria"] = (X_test["tipo_vendedor"] == "Imobiliaria").astype(int)
X_train = X_train.drop(columns=["tipo_vendedor"])
X_test = X_test.drop(columns=["tipo_vendedor"])

# Bairro (66 categorias): target encoding com smoothing (m=10, regulariza bairros raros
# puxando-os para a media global) + frequency encoding. Testado contra TE sem smoothing
# (0.2355), frequency sozinho (0.2405) e Leave-One-Out (0.3879, muito ruidoso em bairros
# esparsos) -- a combinacao smoothed TE + freq foi a que venceu (0.2320 em CV).
bairro_teste = X_test["bairro"].copy()
contagem_bairro = bairro_treino.value_counts()
media_bairro = pd.DataFrame({"bairro": bairro_treino, "log_preco": y_train}).groupby("bairro")["log_preco"].mean()
media_global_log_preco = float(y_train.mean())
freq_bairro = bairro_treino.value_counts(normalize=True)
te_bairro_smoothed = (
    contagem_bairro * media_bairro + FATOR_SMOOTHING * media_global_log_preco
) / (contagem_bairro + FATOR_SMOOTHING)

X_train["bairro_target_enc"] = bairro_treino.map(te_bairro_smoothed).values
X_test["bairro_target_enc"] = bairro_teste.map(te_bairro_smoothed).fillna(media_global_log_preco).values
X_train["bairro_freq_enc"] = bairro_treino.map(freq_bairro).values
X_test["bairro_freq_enc"] = bairro_teste.map(freq_bairro).fillna(0.0).values
X_train = X_train.drop(columns=["bairro"])
X_test = X_test.drop(columns=["bairro"])

# Features de interacao: capturam efeitos nao-lineares que a correlacao linear (Figura
# eda_correlacoes.png) nao encontra -- ex.: o efeito de uma vaga extra difere conforme o
# tamanho do imovel.
for X in (X_train, X_test):
    X["quartos_x_area"] = X["quartos"] * X["log_area_util"]
    X["vagas_x_area"] = X["vagas"] * X["log_area_util"]
    X["suites_ratio_quartos"] = X["suites"] / (X["quartos"] + 1)
    X["luxo_x_area"] = X["total_amenidades"] * X["log_area_util"]
    for col_tipo in colunas_tipo:
        X[f"{col_tipo}_x_area"] = X[col_tipo] * X["log_area_util"]

# Preco por m2 estimado por bairro (mediana com smoothing, calculada SOMENTE no treino
# para evitar leakage) x area do imovel -- da ao modelo uma estimativa de preco ja
# informada por localizacao E tamanho, em vez de deixa-lo aprender essa interacao do zero.
area_util_treino = X_train["area_util"]
preco_treino_real = np.expm1(y_train)
preco_por_m2 = preco_treino_real / area_util_treino.replace(0, np.nan)
mediana_preco_m2_global = float(preco_por_m2.median())
tmp_m2 = pd.DataFrame({"bairro": bairro_treino, "preco_m2": preco_por_m2})
contagem_bairro_m2 = tmp_m2.groupby("bairro")["preco_m2"].count()
mediana_bairro_m2 = tmp_m2.groupby("bairro")["preco_m2"].median()
bairro_preco_m2_smoothed = (
    contagem_bairro_m2 * mediana_bairro_m2 + FATOR_SMOOTHING_M2 * mediana_preco_m2_global
) / (contagem_bairro_m2 + FATOR_SMOOTHING_M2)

bairro_preco_m2_treino = bairro_treino.map(bairro_preco_m2_smoothed).fillna(mediana_preco_m2_global).values
bairro_preco_m2_teste = bairro_teste.map(bairro_preco_m2_smoothed).fillna(mediana_preco_m2_global).values
X_train["bairro_preco_m2"] = bairro_preco_m2_treino
X_test["bairro_preco_m2"] = bairro_preco_m2_teste
X_train["preco_m2_estimado"] = bairro_preco_m2_treino * X_train["area_util"].values
X_test["preco_m2_estimado"] = bairro_preco_m2_teste * X_test["area_util"].values
X_train["log_preco_m2_estimado"] = np.log1p(np.clip(X_train["preco_m2_estimado"], 0, None))
X_test["log_preco_m2_estimado"] = np.log1p(np.clip(X_test["preco_m2_estimado"], 0, None))

assert list(X_train.columns) == list(X_test.columns), "X_train e X_test com colunas diferentes!"
print(f"Dataset final: X_train={X_train.shape}  X_test={X_test.shape}")
assert X_train.isnull().sum().sum() == 0 and X_test.isnull().sum().sum() == 0, "Ha NaN no dataset final!"


# =============================================================================
# 6. RMSPE E SCORER
# =============================================================================
secao("6. METRICA RMSPE")


def rmspe(y_true_log, y_pred_log):
    """RMSPE no espaco original do preco, a partir de previsoes em log1p(preco)."""
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)
    return np.sqrt(np.mean(((y_true - y_pred) / y_true) ** 2))


rmspe_scorer = make_scorer(rmspe, greater_is_better=False)
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
print("RMSPE = sqrt(mean(((preco_real - preco_previsto) / preco_real) ** 2))")


# =============================================================================
# 7. MODELOS (8 modelos diversos, hiperparametros ja otimizados)
# =============================================================================
secao("7. MODELOS")

# Hiperparametros obtidos via Optuna (ver otimizacao_local.py: 150/150/80 trials p/
# LGBM/XGB/CatBoost, split 80/20 estratificado; RF/ExtraTrees com 25 trials).
PARAMS_LGBM = {
    "n_estimators": 834, "max_depth": 8, "num_leaves": 51,
    "learning_rate": 0.0059391787857261055, "min_child_samples": 13,
    "subsample": 0.43873497200283185, "colsample_bytree": 0.4627319750592007,
    "reg_alpha": 1.4724745895076162e-07, "reg_lambda": 0.5206040152624476,
}
PARAMS_XGB = {
    "n_estimators": 1036, "max_depth": 10, "learning_rate": 0.005404689095118521,
    "min_child_weight": 19, "subsample": 0.6596134978099941,
    "colsample_bytree": 0.41794101340216117, "gamma": 9.788842085765603e-05,
    "reg_alpha": 0.10858753203006233, "reg_lambda": 0.6379662185058624,
}
PARAMS_CATBOOST = {
    "iterations": 653, "depth": 10, "learning_rate": 0.01648849563100044,
    "l2_leaf_reg": 1.1705531967011489, "bagging_temperature": 0.48556430144049684,
    "random_strength": 0.00642006701797595,
}
PARAMS_RF = {
    "n_estimators": 333, "max_depth": 14, "min_samples_split": 7,
    "min_samples_leaf": 1, "max_features": 0.30390979886712716,
}
PARAMS_ET = {
    "n_estimators": 351, "max_depth": 14, "min_samples_split": 20,
    "min_samples_leaf": 3, "max_features": 0.9443310021868961,
}
RIDGE_ALPHA = 31.6228
ELASTICNET_ALPHA = 0.0139
ELASTICNET_L1_RATIO = 0.10
KNN_N_NEIGHBORS = 10


def criar_modelos(seed=SEED):
    """8 modelos: 3 gradient boosters + 2 bagging + 2 lineares + 1 baseado em vizinhanca.

    Diagnostico central do trabalho: os 5 modelos baseados em arvore (boosters + bagging)
    tem correlacao >= 0.99 entre suas previsoes (todos aprendem sobre as mesmas features
    da mesma forma), enquanto Ridge/ElasticNet/KNN sao estruturalmente diferentes e
    correlacionam bem menos (~0.96) -- essa diversidade e o que o ensemble ponderado
    (secao 9) explora. Lineares e KNN sao sensiveis a escala: vao dentro de um Pipeline
    com StandardScaler, ajustado somente no treino de cada fold/fit.
    """
    return {
        "LGBMRegressor": LGBMRegressor(**PARAMS_LGBM, random_state=seed, verbose=-1),
        "XGBRegressor": XGBRegressor(**PARAMS_XGB, random_state=seed, verbosity=0),
        "CatBoostRegressor": CatBoostRegressor(**PARAMS_CATBOOST, random_seed=seed, verbose=0),
        "RandomForestRegressor": RandomForestRegressor(**PARAMS_RF, random_state=seed, n_jobs=1),
        "ExtraTreesRegressor": ExtraTreesRegressor(**PARAMS_ET, random_state=seed, n_jobs=1),
        "Ridge": Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=RIDGE_ALPHA, random_state=seed))]),
        "ElasticNet": Pipeline([
            ("scaler", StandardScaler()),
            ("elasticnet", ElasticNet(alpha=ELASTICNET_ALPHA, l1_ratio=ELASTICNET_L1_RATIO, max_iter=5000, random_state=seed)),
        ]),
        "KNeighborsRegressor": Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsRegressor(n_neighbors=KNN_N_NEIGHBORS))]),
    }


print(f"8 modelos definidos: {list(criar_modelos().keys())}")


# =============================================================================
# 8. AVALIACAO 5-FOLD CV + MATRIZ DE CORRELACAO OOF
# =============================================================================
secao("8. AVALIACAO 5-FOLD CV E CORRELACAO OOF")


def gerar_oof_com_scores(modelos_dict, X, y):
    """Previsoes out-of-fold (OOF) + RMSPE por fold, numa unica passada de treino --
    serve tanto para a tabela de RMSPE individual quanto para a matriz de correlacao."""
    oof = {nome: np.zeros(len(X)) for nome in modelos_dict}
    scores_por_modelo = {nome: [] for nome in modelos_dict}
    for idx_tr, idx_val in kf.split(X):
        X_tr, y_tr = X.iloc[idx_tr], y.iloc[idx_tr]
        X_val, y_val = X.iloc[idx_val], y.iloc[idx_val]
        for nome, modelo in modelos_dict.items():
            modelo.fit(X_tr, y_tr)
            pred = modelo.predict(X_val)
            oof[nome][idx_val] = pred
            scores_por_modelo[nome].append(rmspe(y_val, pred))
    oof_df = pd.DataFrame(oof)
    resultados = pd.DataFrame(
        [{"modelo": nome, "rmspe_medio": np.mean(s), "rmspe_std": np.std(s)} for nome, s in scores_por_modelo.items()]
    ).sort_values("rmspe_medio").reset_index(drop=True)
    return oof_df, resultados


oof_treino, tab_resultados = gerar_oof_com_scores(criar_modelos(), X_train, y_train)
print("RMSPE individual (5-fold CV):")
print(tab_resultados.to_string(index=False))

matriz_correlacao = oof_treino.corr()
print("\nMatriz de correlacao das previsoes OOF:")
print(matriz_correlacao.round(3).to_string())


# =============================================================================
# 9. OTIMIZACAO DE PESOS (SLSQP sobre o OOF)
# =============================================================================
secao("9. OTIMIZACAO DE PESOS DO ENSEMBLE")

# Blend uniforme dos 3 boosters nao superava RMSPE 0.222 em CV. Como os modelos baseados
# em arvore sao quase perfeitamente correlacionados (secao 8), um blend uniforme com os
# modelos diversos (mais fracos individualmente) pioraria o resultado. A solucao e
# otimizar os PESOS do blend via scipy.optimize (SLSQP), com restricoes w>=0 e soma=1,
# minimizando diretamente o RMSPE das previsoes OOF ponderadas -- isso permite que um
# modelo fraco mas descorrelacionado (KNN) contribua sem dominar o ensemble.


def objetivo_pesos(pesos, oof_matrix, y_true):
    pred = oof_matrix.values @ pesos
    return rmspe(y_true, pred)


n_modelos = oof_treino.shape[1]
pesos_iniciais = np.ones(n_modelos) / n_modelos
restricao_soma_1 = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
limites = [(0.0, 1.0)] * n_modelos

resultado_opt = minimize(
    objetivo_pesos, pesos_iniciais, args=(oof_treino, y_train),
    method="SLSQP", bounds=limites, constraints=[restricao_soma_1],
    options={"maxiter": 1000, "ftol": 1e-10},
)
pesos_dict = dict(zip(oof_treino.columns, resultado_opt.x))

print("Pesos otimizados:")
for nome, peso in sorted(pesos_dict.items(), key=lambda x: -x[1]):
    print(f"  {nome:24s} peso = {peso:.4f}")

pred_ensemble_oof = oof_treino.values @ resultado_opt.x
rmspe_ensemble = rmspe(y_train, pred_ensemble_oof)
print(f"\nRMSPE do ensemble ponderado (OOF completo): {rmspe_ensemble:.4f}")


# =============================================================================
# 10. GERACAO DA SUBMISSAO
# =============================================================================
secao("10. GERACAO DA SUBMISSAO")

modelos_finais = criar_modelos(seed=SEED)
previsoes_teste = {}
for nome, modelo in modelos_finais.items():
    modelo.fit(X_train, y_train)
    previsoes_teste[nome] = modelo.predict(X_test)
    print(f"  {nome:24s} treinado no dataset completo.")

previsoes_teste_df = pd.DataFrame(previsoes_teste)[list(pesos_dict.keys())]
pred_log_final = (previsoes_teste_df * pd.Series(pesos_dict)).sum(axis=1).values

preco_final = np.expm1(pred_log_final)
n_clipados = int((preco_final < PRECO_MINIMO).sum())
preco_final = np.clip(preco_final, PRECO_MINIMO, None)
print(f"\nPrecos abaixo de R$ {PRECO_MINIMO:,.0f} clipados: {n_clipados}")

submissao = pd.DataFrame({"Id": test_ids, "preco": preco_final})

exemplo = pd.read_csv(EXEMPLO_PATH)
assert list(submissao.columns) == list(exemplo.columns), "Colunas nao batem com o exemplo!"
assert submissao.shape[0] == exemplo.shape[0], "Numero de linhas nao bate com o exemplo!"
assert submissao["Id"].isin(exemplo["Id"]).all(), "Ha Ids que nao existem no exemplo!"
assert submissao["Id"].is_unique, "Ha Ids duplicados!"

submissao.to_csv("submissao_final.csv", index=False)
print(f"Submissao salva em 'submissao_final.csv' ({len(submissao)} linhas, formato validado).")


# =============================================================================
# 11. FEATURE IMPORTANCE
# =============================================================================
secao("11. FEATURE IMPORTANCE (CATBOOST)")

modelo_catboost_final = modelos_finais["CatBoostRegressor"]
importancias = pd.Series(modelo_catboost_final.feature_importances_, index=X_train.columns)
importancias_pct = (importancias / importancias.sum() * 100).sort_values(ascending=False)
top20 = importancias_pct.head(20).sort_values()

fig, ax = plt.subplots(figsize=(9, 8))
ax.barh(top20.index, top20.values, color="#1F4E79")
ax.set_xlabel("Importancia (%)")
ax.set_title("Top 20 features por importancia (CatBoost)")
fig.tight_layout()
fig.savefig("feature_importance.png", dpi=120)
plt.close(fig)

print("Top 10 features:")
print(importancias_pct.head(10).to_string())
print("\nFigura salva em feature_importance.png")


# =============================================================================
# 12. RESUMO FINAL
# =============================================================================
secao("12. RESUMO FINAL")

print(f"RMSPE do ensemble (OOF, 5-fold CV): {rmspe_ensemble:.4f}")
print("Pesos finais:")
for nome, peso in sorted(pesos_dict.items(), key=lambda x: -x[1]):
    if peso > 1e-4:
        print(f"  {nome:24s} {peso:.4f}")
print("Submissao final: submissao_final.csv")
print("Score de referencia no Kaggle (pipeline completo, trabalho2_eel891.py): RMSPE 0.2393")
print("\nPipeline concluido.")
