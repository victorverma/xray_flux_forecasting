import os
import sys
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
density_ratio_estimation_dir = os.path.abspath(os.path.join(parent_dir, "density_ratio_estimation/scripts/"))
tscv_dir = os.path.abspath(os.path.join(parent_dir, "tscv/src/tscv/"))
sys.path.append(density_ratio_estimation_dir)
sys.path.append(tscv_dir)
import numpy as np
from density_ratio_estimators import *
from model import Model
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from typing import Optional

class KNNModel(Model):
    def __init__(self, k_numer: int, k_denom: int, p: float, label: str) -> None:
        self.density_ratio_estimator = KNNDensityRatioEstimator(k_numer, k_denom)
        self.p = p
        self._label = label

    @property
    def label(self) -> str:
        return self._label

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
    def __init__(self, k: int, p: float, label: str) -> None:
        self.density_ratio_estimator = KNN2DensityRatioEstimator(k)
        self.p = p
        self._label = label

    @property
    def label(self) -> str:
        return self._label

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        x_numer = x[y]
        x_denom = x[~y]
        self.density_ratio_estimator.fit(x_numer, x_denom)
        preds = self.density_ratio_estimator.predict(x)
        self.pred_threshold = np.quantile(preds, self.p, method="inverted_cdf")

    def predict(self, x: np.ndarray) -> np.ndarray:
        preds = self.density_ratio_estimator.predict(x)
        return preds >= self.pred_threshold

class KNN3Model(Model):
    def __init__(self, k: int, label: str) -> None:
        self.classifier = KNeighborsClassifier(k)
        self._label = label

    @property
    def label(self) -> str:
        return self._label

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        self.classifier.fit(x, y)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.classifier.predict(x)

class RuLSIFModel(Model):
    def __init__(self, p: float, label: str, sigma_range="auto", lambda_range="auto", kernel_num=100, verbose=True) -> None:
        self.density_ratio_estimator = RuLSIFDensityRatioEstimator(sigma_range, lambda_range, kernel_num, verbose)
        self.p = p
        self._label = label

    @property
    def label(self) -> str:
        return self._label

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        x_numer = x[y]
        x_denom = x[~y]
        self.density_ratio_estimator.fit(x_numer, x_denom)
        ratio_hat = self.density_ratio_estimator.predict(x)
        self.pred_threshold = np.quantile(ratio_hat, self.p, method="inverted_cdf")

    def predict(self, x: np.ndarray) -> np.ndarray:
        ratio_hat = self.density_ratio_estimator.predict(x)
        return ratio_hat >= self.pred_threshold

class LogRegModel(Model):
    def __init__(self, p: float, label: str, penalty: Optional[str] = None, max_iter: int = 100, verbose: int = 0) -> None:
        self.density_ratio_estimator = LogRegDensityRatioEstimator(penalty, max_iter, verbose)
        self.p = p
        self._label = label

    @property
    def label(self) -> str:
        return self._label

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        x_numer = x[y]
        x_denom = x[~y]
        self.density_ratio_estimator.fit(x_numer, x_denom)
        ratio_hat = self.density_ratio_estimator.predict(x)
        self.pred_threshold = np.quantile(ratio_hat, self.p, method="inverted_cdf")

    def predict(self, x: np.ndarray) -> np.ndarray:
        ratio_hat = self.density_ratio_estimator.predict(x)
        return ratio_hat >= self.pred_threshold

class LogReg2Model(Model):
    def __init__(self, label: str, penalty: Optional[str] = None, max_iter: int = 100, verbose: int = 0) -> None:
        self.model_fitter = LogisticRegression(penalty=penalty, max_iter=max_iter, verbose=verbose)
        self._label = label

    @property
    def label(self) -> str:
        return self._label

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        self.model_fitter.fit(x, y)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.model_fitter.predict(x)
