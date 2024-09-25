from abc import ABC, abstractmethod
import os
import sys
import numpy as np
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
density_ratio_estimation_dir = os.path.abspath(os.path.join(parent_dir, "density_ratio_estimation/scripts/"))
sys.path.append(density_ratio_estimation_dir)
import density_ratio_estimators as dre

class Model(ABC):
    """Abstract base class for models."""
    @abstractmethod
    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """Fit the model to the data."""
        pass

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict using the model."""
        pass

class KNNModel(Model):
    def __init__(self, k_numer: int, k_denom: int, p: float) -> None:
        self.density_ratio_estimator = dre.KNNDensityRatioEstimator(k_numer, k_denom)
        self.p = p

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        x_numer = x[y]
        x_denom = x[~y]
        self.density_ratio_estimator.fit(x_numer, x_denom)
        ratio_hat = self.density_ratio_estimator.predict(x)
        self.pred_threshold = np.quantile(ratio_hat, self.p, method="inverted_cdf")

    def predict(self, x: np.ndarray) -> np.ndarray:
        ratio_hat = self.density_ratio_estimator.predict(x)
        return ratio_hat >= self.pred_threshold

class KNN2Model(Model):
    def __init__(self, k: int, p: float) -> None:
        self.density_ratio_estimator = dre.KNN2DensityRatioEstimator(k)
        self.p = p

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        x_numer = x[y]
        x_denom = x[~y]
        self.density_ratio_estimator.fit(x_numer, x_denom)
        preds = self.density_ratio_estimator.predict(x)
        self.pred_threshold = np.quantile(preds, self.p, method="inverted_cdf")

    def predict(self, x: np.ndarray) -> np.ndarray:
        preds = self.density_ratio_estimator.predict(x)
        return preds >= self.pred_threshold
