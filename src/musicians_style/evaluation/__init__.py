"""Ewaluacja obiektywna i subiektywna (Wymagania 6.x, 7.x).

Pakiet udostępnia:

* :func:`euclidean` / :func:`mahalanobis` - funkcje odległości *Wektorów_Cech*
  (zadanie 11.1, Wymagania 6.1, 6.2).
* :class:`ObjectiveEvaluator` - ewaluacja obiektywna z testami statystycznymi
  (zadanie 11.2, Wymagania 6.1-6.5) wraz z modelami danych raportu
  (:class:`EvaluationReport`, :class:`StatTestResult`, :class:`DescriptiveStats`).
"""

from .distance import euclidean, mahalanobis
from .objective import (
    DescriptiveStats,
    EvaluationReport,
    Metric,
    ObjectiveEvaluator,
    SeriesStats,
    StatTestResult,
)

__all__ = [
    "euclidean",
    "mahalanobis",
    "ObjectiveEvaluator",
    "EvaluationReport",
    "StatTestResult",
    "DescriptiveStats",
    "SeriesStats",
    "Metric",
]
