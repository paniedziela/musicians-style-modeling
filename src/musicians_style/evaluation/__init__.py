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
from .plots import (
    generate_comparison_plots,
    plot_interval_histogram,
    plot_pitch_class_histogram,
    plot_scalar_boxplots,
)
from .subjective import (
    AudioRenderer,
    FormQuestion,
    FormSpec,
    FormType,
    ListeningPair,
    ListeningSet,
    MetricStats,
    Response,
    SignificanceResult,
    SubjectiveEvaluator,
    SubjectiveReport,
    default_fluidsynth_renderer,
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
    "generate_comparison_plots",
    "plot_pitch_class_histogram",
    "plot_interval_histogram",
    "plot_scalar_boxplots",
    "SubjectiveEvaluator",
    "ListeningSet",
    "ListeningPair",
    "AudioRenderer",
    "default_fluidsynth_renderer",
    "FormSpec",
    "FormQuestion",
    "FormType",
    "Response",
    "SubjectiveReport",
    "MetricStats",
    "SignificanceResult",
]
