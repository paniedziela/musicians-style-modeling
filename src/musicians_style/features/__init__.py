"""Ekstraktor_Cech i model danych Wektora_Cech (Wymagania 2.x).

Pakiet udostępnia:

* :class:`FeatureVector` / :class:`AggregatedFeatures` - model danych
  *Wektora_Cech* i jego agregatu statystycznego (zadanie 3.1).
* :data:`NEUTRAL_FEATURE_VECTOR` - *Wektor_Cech* neutralny dla plików pustych
  (Wymagania 2.7, 11.8).
* :class:`FeatureExtractor` - obliczanie *Wektora_Cech* z *Reprezentacji_Wewnętrznej*
  (zadanie 3.2, Wymagania 2.1, 2.2, 2.5, 2.7, 2.8).
"""

from .constants import NEUTRAL_FEATURE_VECTOR
from .extractor import FeatureExtractor
from .types import (
    FEATURE_VECTOR_LENGTH,
    INTERVAL_HISTOGRAM_BINS,
    PITCH_CLASS_BINS,
    AggregatedFeatures,
    FeatureVector,
)

__all__ = [
    "FeatureExtractor",
    "FeatureVector",
    "AggregatedFeatures",
    "NEUTRAL_FEATURE_VECTOR",
    "FEATURE_VECTOR_LENGTH",
    "PITCH_CLASS_BINS",
    "INTERVAL_HISTOGRAM_BINS",
]
