# ML-RealEstatePricing

Assignment 2 for **EEL891 — Introduction to Machine Learning** (UFRJ, 2025-2).

Multivariable regression to estimate residential property prices, using a
weighted ensemble of diverse models with Bayesian optimization.

**Best result:** RMSPE of **0.2393** on the public Kaggle leaderboard.

## Structure

├── trabalho2_eel891.py # Full pipeline (EDA → submission)
├── otimizacao_local.py # Extended optimization with more trials
├── relatorio_eel891.tex/.pdf # Report (Portuguese)
├── conjunto_de_.csv # Kaggle data
└── eda_.png # Visualizations


## Pipeline

1. **EDA** — log-normal price distribution, 20 features, weak correlations in linear space
2. **Preprocessing** — outlier removal (3× IQR + 99th percentile), null imputation
3. **Feature engineering** — total area, estimated price per m² by neighbourhood, interactions, amenity count
4. **Encoding** — one-hot (property type), smoothed target encoding + frequency encoding (neighbourhood)
5. **Modelling** — 8 models: LightGBM, XGBoost, CatBoost, RandomForest, ExtraTrees, Ridge, ElasticNet, KNN
6. **Optimization** — Bayesian Optuna (TPE), 150/150/80 trials
7. **Ensemble** — weights optimized with scipy (SLSQP) over out-of-fold predictions

## Results

| Version | Kaggle RMSPE |
|---|---|
| Baseline | 0.2535 |
| + encoding, features, outlier removal | 0.2479 |
| + Optuna re-tuning | 0.2469 |
| + extended Optuna (more trials) | 0.2437 |
| **+ weighted diverse ensemble** | **0.2393** |

**Main finding:** the gradient boosting models correlated above 0.998 with
each other, capping the gain from uniform blending. Adding structurally
different models (KNN) with optimized weights broke that ceiling.

## Running

```bash
pip install pandas numpy matplotlib seaborn scikit-learn lightgbm xgboost catboost optuna scipy
python trabalho2_eel891.py
```

## Author

**Luiz Felipe Píccoli Cavalini** — Computer and Information Engineering, UFRJ
