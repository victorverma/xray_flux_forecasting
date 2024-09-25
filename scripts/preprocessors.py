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
    def _validate_init_params(r: int, y_threshold: Optional[float] = None, p: Optional[float] = None) -> None:
        """
        Validate parameters that all subclasses should use for initialization.

        :param r: Integer giving the number of consecutive covariate vectors to concatenate.
        :param y_threshold: Float giving an explicit threshold that defines extremeness.
        :param p: Float giving the quantile level of the threshold that defines extremeness.
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
    def _validate_arrays(x: np.ndarray, y: np.ndarray) -> None:
        """
        Validate arrays passed into `fit` and `transform`.

        :param x: NumPy array of shape (n, d) whose rows are covariate vectors.
        :param y: NumPy array of shape (n,) whose entries are response values.
        """
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a NumPy array")
        if x.ndim != 2:
            raise TypeError("x must be 2D")
        if not isinstance(y, np.ndarray):
            raise TypeError("y must be a NumPy array")
        if y.ndim != 1:
            raise TypeError("y must be 1D")
        if x.shape[0] != y.size:
            raise ValueError("number of rows in x must equal length of y")

    @final
    def _embed(x: np.ndarray, r: int) -> np.ndarray:
        """
        Concatenate consecutive rows of an array. This mimics the functionality of stats::embed() in R.

        :param x: NumPy array of shape (n, d); is assumed to have been validated using `_validate_arrays`.
        :param r: Integer equal to the number of consecutive rows to concatenate; is assumed to have been validated using `_validate_init_params`.
        :return: NumPy array of shape (n - r + 1, r * d) containing the concatenated rows.
        """
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
    def _binarize(y: np.ndarray, y_threshold: float) -> np.ndarray:
        """
        Flag values in an array that are at or above a threshold.

        :param y: NumPy array of shape (n,); is assumed to have been validated using `_validate_arrays`.
        :param y_threshold: Float giving the threshold; is assumed to have been validated using `_validate_init_params`.
        :return: NumPy array of shape (n,) containing the flags.
        """
        return y >= y_threshold

class IdentityPreprocessor(Preprocessor):
    def __init__(self, r: int = 1, y_threshold: Optional[float] = None, p: Optional[float] = None) -> None:
        Preprocessor._validate_init_params(r, y_threshold, p)
        self.r = r
        self.y_threshold = y_threshold
        self.p = p

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        if self.y_threshold is None:
            self.y_threshold = np.quantile(y, self.p, method="inverted_cdf")

    def transform(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        Preprocessor._validate_arrays(x, y)
        return Preprocessor._embed(x, self.r), Preprocessor._binarize(y[self.r:], self.y_threshold)

class StandardizePreprocessor(Preprocessor):
    def __init__(self, r: int = 1, y_threshold: Optional[float] = None, p: Optional[float] = None) -> None:
        Preprocessor._validate_init_params(r, y_threshold, p)
        self.r = r
        self.y_threshold = y_threshold
        self.p = p

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        Preprocessor._validate_arrays(x, y)
        if self.y_threshold is None:
            self.y_threshold = np.quantile(y, self.p, method="inverted_cdf")
        self.means = np.mean(x, axis=0)
        self.sds = np.std(x, axis=0)

    def transform(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        Preprocessor._validate_arrays(x, y)
        x = (x - self.means) / self.sds
        return Preprocessor._embed(x, self.r), Preprocessor._binarize(y[self.r:], self.y_threshold)
