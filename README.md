# ML-RealEstatePricing

Trabalho 2 da disciplina **EEL891 — Introdução ao Aprendizado de Máquina** (UFRJ, 2025-2).

Regressão multivariável para estimativa de preços de imóveis residenciais, utilizando ensemble ponderado de modelos diversos com otimização bayesiana.

**Melhor resultado:** RMSPE de **0.2393** no leaderboard público do Kaggle.

## Estrutura
```
├── trabalho2_eel891.py       # Pipeline completo (EDA → submissão)
├── otimizacao_local.py       # Otimização estendida com mais trials
├── relatorio_eel891.tex/.pdf # Relatório
├── conjunto_de_*.csv         # Dados do Kaggle
└── eda_*.png                 # Visualizações
```

## Pipeline

1. **EDA** — distribuição log-normal do preço, 20 features, correlações fracas no espaço linear
2. **Pré-processamento** — remoção de outliers (IQR 3x + percentil 99), imputação de nulos
3. **Feature Engineering** — área total, preço/m² estimado por bairro, interações, contagem de amenidades
4. **Encoding** — one-hot (tipo), smoothed target encoding + frequency encoding (bairro)
5. **Modelagem** — 8 modelos: LightGBM, XGBoost, CatBoost, RandomForest, ExtraTrees, Ridge, ElasticNet, KNN
6. **Otimização** — Optuna bayesiano (TPE), 150/150/80 trials
7. **Ensemble** — pesos otimizados via scipy (SLSQP) sobre previsões out-of-fold

## Resultados

| Versão | RMSPE Kaggle |
|---|---|
| Baseline | 0.2535 |
| + encoding, features, outliers | 0.2479 |
| + re-otimização Optuna | 0.2469 |
| + Optuna estendido (mais trials) | 0.2437 |
| **+ ensemble ponderado diverso** | **0.2393** |

**Achado principal:** ensembles de gradient boosting apresentaram correlação >0.998 entre si, limitando o ganho de blending uniforme. A adição de modelos estruturalmente diferentes (KNN) com pesos otimizados quebrou esse teto.

## Como Executar

```bash
pip install pandas numpy matplotlib seaborn scikit-learn lightgbm xgboost catboost optuna scipy
python trabalho2_eel891.py
```

## Autor

**Luiz Felipe Píccoli Cavalini** — Engenharia de Computação e Informação, UFRJ
