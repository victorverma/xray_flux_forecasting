from abc import ABC, abstractmethod
from typing import Tuple
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

class IdentityPreprocessor(Preprocessor):
    def __init__(self, y_threshold: float) -> None:
        self.y_threshold = y_threshold

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        pass

    def transform(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return x, y >= self.y_threshold
