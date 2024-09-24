from abc import ABC, abstractmethod
from typing import final, Tuple
import numpy as np
import warnings

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
    def _embed(x: np.ndarray, r: int) -> np.ndarray:
        """
        Concatenate consecutive rows of an array. This mimics the functionality of stats::embed() in R.

        :param x: NumPy array of shape (n, d).
        :param r: Integer equal to the number of consecutive rows to concatenate.
        :return: NumPy array of shape (n - r + 1, r * d) containing the concatenated rows.
        """
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a NumPy array")
        if x.ndim != 2:
            raise TypeError("x must be 2D")
        if not isinstance(r, int):
            raise TypeError("r must be an integer")
        n, d = x.shape
        if r < 1 or r > n:
            raise ValueError("r must be between one and the number of rows in x")
        x_ = np.zeros((n - r + 1, r * d))
        for i in range(n - r + 1):
            x_[i] = x[i:(i + r)][::-1].flatten()
        return x_

    @final
    def _validate(threshold: float = None, p: float = None) -> None:
        """
        Check whether the threshold defining extremeness or its quantile level are valid.

        Specify either `threshold` or `p`, but not both. If both are specified,
        `threshold` will be used.

        :param threshold: Float giving the threshold explicitly.
        :param p: Float giving the quantile level of the threshold; must be in (0, 1).
        """
        if threshold is not None and p is not None:
            warnings.warn("both threshold and p were specified, using threshold")
        if threshold is None and p is None:
            raise ValueError("must specify either threshold or p")

        if threshold is not None:
            if not isinstance(threshold, float):
                raise TypeError("threshold must be a float")
        else:
            if not isinstance(p, float):
                raise TypeError("p must be a float")
            if not (0 < p < 1):
                raise ValueError("p must be in (0, 1)")

    @final
    def _binarize(y: np.ndarray, threshold: float = None, p: float = None) -> np.ndarray:
        """
        Flag values in an array that are at or above a threshold.

        Specify either `threshold` or `p`, but not both. If both are specified,
        `threshold` will be used. `threshold` and `p` are assumed to have been validated
        using `_validate`.

        :param y: NumPy array of shape (n,).
        :param threshold: Float giving the threshold explicitly.
        :param p: Float giving the quantile level of the threshold; must be in (0, 1).
        :return: NumPy array of shape (n,) containing the flags.
        """
        if not isinstance(y, np.ndarray):
            raise TypeError("y must be a NumPy array")
        if y.ndim != 1:
            raise TypeError("y must be 1D")

        if threshold is not None:
            return y >= threshold
        else:
            return y >= np.quantile(y, p, method="inverted_cdf")

class IdentityPreprocessor(Preprocessor):
    def __init__(self, y_threshold: float = None, p: float = None) -> None:
        Preprocessor._validate(y_threshold, p)
        self.y_threshold = y_threshold
        self.p = p

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        pass

    def transform(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return x, Preprocessor._binarize(y, self.y_threshold, self.p)

class StandardizePreprocessor(Preprocessor):
    def __init__(self, y_threshold: float = None, p: float = None) -> None:
        Preprocessor._validate(y_threshold, p)
        self.y_threshold = y_threshold
        self.p = p

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        self.means = np.mean(x, axis=0)
        self.sds = np.std(x, axis=0)

    def transform(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return (x - self.means) / self.sds, Preprocessor._binarize(y, self.y_threshold, self.p)