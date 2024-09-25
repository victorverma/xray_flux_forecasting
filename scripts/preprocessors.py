from abc import ABC, abstractmethod
from typing import final, Optional, Tuple
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
    def _validate(r: int, y_threshold: Optional[float] = None, p: Optional[float] = None) -> None:
        """
        Validate parameters of other internal methods. See the docstrings of those
        methods for detailed descriptions of the parameters.

        :param r: Integer; parameter of `_embed`.
        :param y_threshold: Float; parameter of `_binarize`.
        :param p: Float; parameter of `_binarize`.
        :raises TypeError: If any parameter is of the wrong type.
        :raises ValueError: If `r` is not positive or if `p` is not in (0, 1).
        """
        if not isinstance(r, int):
            raise TypeError("r must be an integer")
        if r <= 0:
            raise ValueError("r must be positive")

        if y_threshold is not None and p is not None:
            warnings.warn("both y_threshold and p were specified, using y_threshold")
        if y_threshold is None and p is None:
            raise ValueError("must specify either y_threshold or p")

        if y_threshold is not None:
            if not isinstance(y_threshold, float):
                raise TypeError("y_threshold must be a float")
        else:
            if not isinstance(p, float):
                raise TypeError("p must be a float")
            if not (0 < p < 1):
                raise ValueError("p must be in (0, 1)")

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
        n, d = x.shape
        if n < r:
            raise ValueError("the number of rows in x must be at least r")

        if r == 1:
            return x
        x_ = np.zeros((n - r + 1, r * d))
        for i in range(n - r + 1):
            x_[i] = x[i:(i + r)][::-1].flatten()
        return x_

    @final
    def _binarize(y: np.ndarray, y_threshold: Optional[float] = None, p: Optional[float] = None) -> np.ndarray:
        """
        Flag values in an array that are at or above a threshold.

        Specify either `y_threshold` or `p`, but not both. If both are specified,
        `y_threshold` will be used. `y_threshold` and `p` are assumed to have been validated
        using `_validate`.

        :param y: NumPy array of shape (n,).
        :param y_threshold: Float giving the threshold explicitly.
        :param p: Float giving the quantile level of the threshold; must be in (0, 1).
        :return: NumPy array of shape (n,) containing the flags.
        """
        if not isinstance(y, np.ndarray):
            raise TypeError("y must be a NumPy array")
        if y.ndim != 1:
            raise TypeError("y must be 1D")

        if y_threshold is not None:
            return y >= y_threshold
        else:
            return y >= np.quantile(y, p, method="inverted_cdf")

class IdentityPreprocessor(Preprocessor):
    def __init__(self, r: int, y_threshold: Optional[float] = None, p: Optional[float] = None) -> None:
        Preprocessor._validate(r, y_threshold, p)
        self.r = r
        self.y_threshold = y_threshold
        self.p = p

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        pass

    def transform(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return Preprocessor._embed(x, self.r), Preprocessor._binarize(y[self.r:], self.y_threshold, self.p)

class StandardizePreprocessor(Preprocessor):
    def __init__(self, r: int, y_threshold: Optional[float] = None, p: Optional[float] = None) -> None:
        Preprocessor._validate(r, y_threshold, p)
        self.y_threshold = y_threshold
        self.p = p

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        self.means = np.mean(x, axis=0)
        self.sds = np.std(x, axis=0)

    def transform(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x = (x - self.means) / self.sds
        return Preprocessor._embed(x, self.r), Preprocessor._binarize(y[self.r:], self.y_threshold, self.p)
