import os
import sys
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
tscv_dir = os.path.abspath(os.path.join(parent_dir, "tscv/src/tscv/"))
sys.path.append(tscv_dir)
import numpy as np
from preprocessor import Preprocessor
from typing import Optional, Tuple

class IdentityPreprocessor(Preprocessor):
    def __init__(self, r: int = 1, y_threshold: Optional[float] = None, p: Optional[float] = None) -> None:
        self.r = r
        self.y_threshold = y_threshold
        self.p = p

    def _binarize(self, y: np.ndarray, y_threshold: float) -> np.ndarray:
        """
        Flag values in an array that are at or above a threshold.

        :param y: NumPy array of shape (n,); is assumed to have been validated using `_validate_arrays`.
        :param y_threshold: Float giving the threshold; is assumed to have been validated using `_validate_init_params`.
        :return: NumPy array of shape (n,) containing the flags.
        """
        return y >= y_threshold

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        if self.y_threshold is None:
            self.y_threshold = np.quantile(y, self.p, method="inverted_cdf")

    def transform(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return Preprocessor._embed(x, self.r), self._binarize(y[(self.r - 1):], self.y_threshold)

class StandardizePreprocessor(Preprocessor):
    def __init__(self, r: int = 1, y_threshold: Optional[float] = None, p: Optional[float] = None) -> None:
        self.r = r
        self.y_threshold = y_threshold
        self.p = p

    def _binarize(self, y: np.ndarray, y_threshold: float) -> np.ndarray:
        """
        Flag values in an array that are at or above a threshold.

        :param y: NumPy array of shape (n,); is assumed to have been validated using `_validate_arrays`.
        :param y_threshold: Float giving the threshold; is assumed to have been validated using `_validate_init_params`.
        :return: NumPy array of shape (n,) containing the flags.
        """
        return y >= y_threshold

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        if self.y_threshold is None:
            self.y_threshold = np.quantile(y, self.p, method="inverted_cdf")
        self.means = np.mean(x, axis=0)
        self.sds = np.std(x, axis=0)

    def transform(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x = (x - self.means) / self.sds
        return Preprocessor._embed(x, self.r), self._binarize(y[(self.r - 1):], self.y_threshold)
