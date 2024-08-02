from abc import ABC, abstractmethod
from sklearn.metrics import confusion_matrix
import numpy as np

class Metric(ABC):
     """Abstract base class for metrics."""
     @abstractmethod
     def evaluate(self, y_test: np.ndarray, y_pred: np.ndarray) -> float:
         """
         Compute the value of the metric using the test response values and their predictions.

         :param y_test: NumPy array containing the test response values.
         :param y_pred: NumPy array containing the predictions of the test response values.
         :return: Float that equals the computed value of the metric.
         """
         pass

class tpr(Metric):
     def evaluate(self, y_test: np.ndarray, y_pred: np.ndarray) -> float:
        _, _, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        return tp / (tp + fn)

class fpr(Metric):
     def evaluate(self, y_test: np.ndarray, y_pred: np.ndarray) -> float:
          tn, fp, _, _ = confusion_matrix(y_test, y_pred).ravel()
          return fp / (fp + tn)

class tss(Metric):
     def evaluate(self, y_test: np.ndarray, y_pred: np.ndarray) -> float:
          tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
          return tp / (tp + fn) - fp / (fp + tn)

class precision(Metric):
     def evaluate(self, y_test: np.ndarray, y_pred: np.ndarray) -> float:
          _, fp, _, tp = confusion_matrix(y_test, y_pred).ravel()
          return tp / (tp + fp)