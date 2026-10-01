"""Hierarquia de modelos do ensemble.

Cada biblioteca (LightGBM, XGBoost, CatBoost, scikit-learn) tem uma API
ligeiramente diferente: o nome do parametro de semente muda, o de verbosidade
muda, e alguns modelos precisam de normalizacao antes do treino. Essas
diferencas ficam encapsuladas aqui, de modo que quem usa as classes so precisa
conhecer `fit` e `predict`.
"""

from abc import ABC, abstractmethod

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


class ModelWrapper(ABC):
    """Contrato comum a todos os modelos do ensemble.

    Classe abstrata: nao pode ser instanciada. Cada subclasse so precisa
    informar como construir seu estimador; o ciclo de treino e previsao e
    herdado daqui e escrito uma unica vez.
    """

    #: subclasses que dependem de escala sobrescrevem para True
    needs_scaling = False

    def __init__(self, params=None, seed=42):
        self._params = params or {}      # estado interno: encapsulado
        self._seed = seed
        self._estimator = None           # so existe depois do fit

    # ---- contrato que as subclasses implementam ---------------------------

    @abstractmethod
    def _create_estimator(self):
        """Instancia o estimador da biblioteca correspondente."""

    # ---- comportamento herdado por todas ----------------------------------

    @property
    def name(self):
        return type(self).__name__.replace("Model", "")

    @property
    def is_fitted(self):
        return self._estimator is not None

    def _build(self):
        """Envolve o estimador num Pipeline com scaler quando necessario."""
        estimator = self._create_estimator()
        if not self.needs_scaling:
            return estimator
        return Pipeline([("scaler", StandardScaler()), ("estimator", estimator)])

    def fit(self, X, y):
        self._estimator = self._build()
        self._estimator.fit(X, y)
        return self                      # permite encadear: model.fit(X, y).predict(Z)

    def predict(self, X):
        if not self.is_fitted:
            raise RuntimeError(f"{self.name} ainda nao foi treinado — chame fit() antes.")
        return self._estimator.predict(X)

    @property
    def estimator(self):
        """Estimador da biblioteca, por baixo do Pipeline quando houver."""
        if not self.is_fitted:
            raise RuntimeError(f"{self.name} ainda nao foi treinado.")
        if self.needs_scaling:
            return self._estimator.named_steps["estimator"]
        return self._estimator

    def __repr__(self):
        estado = "treinado" if self.is_fitted else "nao treinado"
        return f"<{self.name} ({estado})>"


# ---------------------------------------------------------------------------
# Gradient boosting
# ---------------------------------------------------------------------------

class LGBMModel(ModelWrapper):
    def _create_estimator(self):
        return LGBMRegressor(**self._params, random_state=self._seed, verbose=-1)


class XGBModel(ModelWrapper):
    def _create_estimator(self):
        return XGBRegressor(**self._params, random_state=self._seed, verbosity=0)


class CatBoostModel(ModelWrapper):
    def _create_estimator(self):
        # CatBoost chama a semente de random_seed, nao random_state
        return CatBoostRegressor(**self._params, random_seed=self._seed, verbose=0)


# ---------------------------------------------------------------------------
# Bagging
# ---------------------------------------------------------------------------

class RandomForestModel(ModelWrapper):
    def _create_estimator(self):
        return RandomForestRegressor(**self._params, random_state=self._seed, n_jobs=1)


class ExtraTreesModel(ModelWrapper):
    def _create_estimator(self):
        return ExtraTreesRegressor(**self._params, random_state=self._seed, n_jobs=1)


# ---------------------------------------------------------------------------
# Sensiveis a escala — needs_scaling = True injeta o StandardScaler
# ---------------------------------------------------------------------------

class RidgeModel(ModelWrapper):
    needs_scaling = True

    def _create_estimator(self):
        return Ridge(**self._params, random_state=self._seed)


class ElasticNetModel(ModelWrapper):
    needs_scaling = True

    def _create_estimator(self):
        return ElasticNet(**self._params, random_state=self._seed)


class KNNModel(ModelWrapper):
    needs_scaling = True

    def _create_estimator(self):
        # KNN nao usa semente — a ausencia de random_state e parte do contrato
        return KNeighborsRegressor(**self._params)
