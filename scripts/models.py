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
    def __init__(self, y_threshold: float, k: int) -> None:
        self.y_threshold = y_threshold
        self.density_ratio_estimator = dre.KNNDensityRatioEstimator(k=k)

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        x_numer = x[y >= self.y_threshold]
        x_denom = x[y < self.y_threshold]
        self.density_ratio_estimator.fit(x_numer, x_denom)
        y_pred = self.density_ratio_estimator.predict(x)
        p = np.mean(y <= self.y_threshold)
        self.y_pred_threshold = np.quantile(y_pred, p)

    def predict(self, x: np.ndarray) -> np.ndarray:
        y_pred = self.density_ratio_estimator.predict(x)
        return y_pred >= self.y_pred_threshold
