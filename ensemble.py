"""Ensemble com pesos otimizados sobre previsoes out-of-fold."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize


class WeightedEnsemble:
    """Combina varios ModelWrapper com pesos otimizados.

    O ensemble nao sabe — e nao precisa saber — quais modelos esta carregando:
    chama `fit` e `predict` em cada um e confia no contrato. Trocar LightGBM
    por outro modelo nao exige mudar uma linha aqui.
    """

    def __init__(self, models, metric, cv):
        self._models = list(models)
        self._metric = metric            # funcao (y_true, y_pred) -> erro
        self._cv = cv                    # objeto de split, ex. KFold
        self._weights = None
        self._oof = None

    # ---- propriedades somente leitura -------------------------------------

    @property
    def model_names(self):
        return [m.name for m in self._models]

    @property
    def weights(self):
        if self._weights is None:
            raise RuntimeError("Pesos ainda nao otimizados — chame fit() antes.")
        return dict(zip(self.model_names, self._weights))

    @property
    def oof_predictions(self):
        if self._oof is None:
            raise RuntimeError("OOF ainda nao gerado — chame fit() antes.")
        return self._oof

    # ---- etapas -----------------------------------------------------------

    def generate_oof(self, X, y):
        """Previsoes out-of-fold e RMSPE por modelo, numa unica passada."""
        oof = {m.name: np.zeros(len(X)) for m in self._models}
        scores = {m.name: [] for m in self._models}

        for idx_tr, idx_val in self._cv.split(X):
            X_tr, y_tr = X.iloc[idx_tr], y.iloc[idx_tr]
            X_val, y_val = X.iloc[idx_val], y.iloc[idx_val]

            for model in self._models:          # polimorfismo: todos respondem igual
                pred = model.fit(X_tr, y_tr).predict(X_val)
                oof[model.name][idx_val] = pred
                scores[model.name].append(self._metric(y_val, pred))

        self._oof = pd.DataFrame(oof)
        resumo = (
            pd.DataFrame(
                [{"modelo": n, "rmspe_medio": np.mean(s), "rmspe_std": np.std(s)}
                 for n, s in scores.items()]
            )
            .sort_values("rmspe_medio")
            .reset_index(drop=True)
        )
        return self._oof, resumo

    def optimize_weights(self, y):
        """SLSQP minimizando a metrica sobre o OOF, com w >= 0 e soma = 1."""
        n = self._oof.shape[1]
        objetivo = lambda w: self._metric(y, self._oof.values @ w)

        resultado = minimize(
            objetivo,
            np.ones(n) / n,
            method="SLSQP",
            bounds=[(0.0, 1.0)] * n,
            constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}],
            options={"maxiter": 1000, "ftol": 1e-10},
        )
        self._weights = resultado.x
        return self._weights

    def fit(self, X, y):
        """Gera OOF, otimiza pesos e retreina cada modelo no dataset completo."""
        _, resumo = self.generate_oof(X, y)
        self.optimize_weights(y)
        for model in self._models:
            model.fit(X, y)
        return resumo

    # A combinacao final e um produto escalar (BLAS). A versao pre-refactor
    # acumulava os termos em outra ordem, entao as previsoes diferem em ate
    # ~4e-15 em termos relativos -- ruido de ponto flutuante, nao mudanca de
    # modelo. Equivalencia confirmada com np.allclose(rtol=1e-9).
    def predict(self, X):
        preds = np.column_stack([m.predict(X) for m in self._models])
        return preds @ self._weights

    def get(self, name):
        for model in self._models:
            if model.name == name:
                return model
        raise KeyError(
            f"Modelo '{name}' nao esta no ensemble. Disponiveis: {self.model_names}"
        )

    def oof_score(self, y):
        return self._metric(y, self._oof.values @ self._weights)

    def correlation_matrix(self):
        return self.oof_predictions.corr()
