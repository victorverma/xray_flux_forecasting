from abc import ABC, abstractmethod
from typing import final, Tuple
import numpy as np

class Preprocessor(ABC):
    """Abstract base class for preprocessors."""

    @abstractmethod
    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the preprocessor to training data.

        :param x: NumPy array containing covariate training data.
        :param y: NumPy array containing response training data.
        """
        pass

    @abstractmethod
    def transform(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform the data with the preprocessor.

        :param x: NumPy array containing covariate data.
        :param y: NumPy array containing response data.
        :return: Tuple whose two entries are the values of x and y after preprocessing.
        """
        pass

    def fit_transform(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit the preprocessor to training data and then transform the data with the fitted preprocessor.

        :param x: NumPy array containing covariate training data.
        :param y: NumPy array containing response training data.
        :return: Tuple whose two entries are the values of x and y after preprocessing.
        """
        self.fit(x, y)
        return self.transform(x, y)

    @final
    def _embed(x: np.ndarray, d: int) -> np.ndarray:
        """
        Concatenate consecutive rows of an array. This mimics the functionality of stats::embed() in R.

        :param x: NumPy array of shape (n, p).
        :param d: Integer equal to the number of consecutive rows to concatenate.
        :return: NumPy array of shape (n - d + 1, d * p) containing the concatenated rows.
        """
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a NumPy array")
        if x.ndim != 2:
            raise TypeError("x must be 2D")
        if not isinstance(d, int):
            raise TypeError("d must be an integer")
        n, p = x.shape
        if d < 1 or d > n:
            raise ValueError("d must be between one and the number of rows in x")
        x_ = np.zeros((n - d + 1, d * p))
        for i in range(n - d + 1):
            x_[i] = x[i:(i + d)][::-1].flatten()
        return x_

    @final
    def _binarize(y: np.ndarray, threshold: float) -> np.ndarray:
        """
        Flag values in an array that are at or above a threshold.

        :param y: NumPy array of shape (n,).
        :param threshold: Float.
        :return: NumPy array of shape (n,) containing the flags.
        """
        if not isinstance(y, np.ndarray):
            raise TypeError("y must be a NumPy array")
        if y.ndim != 1:
            raise TypeError("y must be 1D")
        if not isinstance(threshold, float):
            raise TypeError("y_threshold must be a float")
        return y >= threshold

class IdentityPreprocessor(Preprocessor):
    def __init__(self, y_threshold: float) -> None:
        self.y_threshold = y_threshold

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        pass

    def transform(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return x, Preprocessor._binarize(y, self.y_threshold)

class StandardizePreprocessor(Preprocessor):
    def __init__(self, y_threshold: float) -> None:
        self.y_threshold = y_threshold

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        self.means = np.mean(x, axis=0)
        self.sds = np.std(x, axis=0)

    def transform(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return (x - self.means) / self.sds, Preprocessor._binarize(y, self.y_threshold)