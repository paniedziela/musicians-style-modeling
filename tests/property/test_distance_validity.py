# Feature: musicians-style-modeling, Property 10
"""Property 10: Walidność funkcji odległości w ewaluacji (zadanie 11.6).

**Validates: Requirements 6.1, 6.2**

Sekcja *Correctness Properties* (``design.md``):

    *For any* *Wektorów_Cech* ``f1``, ``f2`` i agregowanego *Wektora_Cech* ``D``
    zwróconego przez ``AggregatedFeatures``, funkcja odległości ``distance(f1, D)``
    (zarówno euklidesowa jak i Mahalanobisa) jest **skończoną**, **nieujemną**
    liczbą rzeczywistą i **deterministyczną** względem wejść.

Wymagania ``requirements.md`` (Requirement 6 - Ewaluacja obiektywna):

* 6.1 - System oblicza dla każdego *Utworu_Wyjściowego* odległość *Wektora_Cech*
  od agregowanego wektora *Zbioru_Stylu* artysty docelowego.
* 6.2 - System oblicza dla każdego *Utworu_Wyjściowego* odległość *Wektora_Cech*
  od *Wektora_Cech* *Utworu_Wejściowego* (kontrola zachowania struktury źródłowej).

Aby pomiary odległości z 6.1/6.2 były użyteczne w teście statystycznym (6.3-6.5),
muszą być wartościami liczbowymi o dobrze określonych własnościach: skończonością,
nieujemnością i powtarzalnością (determinizm). Ten test weryfikuje te niezmienniki
dla obu funkcji odległości z :mod:`musicians_style.evaluation.distance`.

Strategie ``feature_vector_strategy`` (skończone *Wektory_Cech*) oraz
``psd_covariance`` (dodatnio półokreślona macierz kowariancji) pochodzą ze
wspólnego, przetestowanego modułu ``tests/property/strategies.py`` - testy nie
redefiniują własnych generatorów.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from .profiles import property_settings
from hypothesis import given

from musicians_style.evaluation.distance import euclidean, mahalanobis
from musicians_style.features.types import FEATURE_VECTOR_LENGTH, FeatureVector

from .strategies import feature_vector_strategy, psd_covariance


@pytest.mark.property
@property_settings(max_examples=200, deadline=None)
@given(f1=feature_vector_strategy(), f2=feature_vector_strategy())
def test_euclidean_distance_is_valid(f1: FeatureVector, f2: FeatureVector) -> None:
    """Odległość euklidesowa jest skończona, nieujemna i deterministyczna (Property 10).

    Weryfikuje niezmienniki Property 10 (Wymagania 6.1, 6.2) dla
    :func:`musicians_style.evaluation.distance.euclidean`:

    * **skończoność** - wynik jest skończoną liczbą rzeczywistą (``math.isfinite``),
    * **nieujemność** - wynik ``>= 0`` (norma L2),
    * **determinizm** - dwa wywołania na tych samych wejściach dają identyczną wartość,
    * **symetria** - ``euclidean(f1, f2) == euclidean(f2, f1)``,
    * **identyczność (tożsamość)** - ``euclidean(f1, f1) == 0.0``.
    """
    a = f1.as_array()
    b = f2.as_array()

    distance = euclidean(a, b)

    # Skończoność i nieujemność (Property 10).
    assert math.isfinite(distance), "Odległość euklidesowa musi być skończona."
    assert distance >= 0.0, "Odległość euklidesowa musi być nieujemna."

    # Determinizm: powtórne wywołanie daje identyczną wartość.
    assert euclidean(a, b) == distance, "Odległość euklidesowa musi być deterministyczna."

    # Symetria metryki: d(f1, f2) == d(f2, f1).
    assert euclidean(b, a) == distance, "Odległość euklidesowa musi być symetryczna."

    # Identyczność: odległość wektora od samego siebie wynosi dokładnie 0.0.
    assert euclidean(a, a) == 0.0, "euclidean(f, f) musi wynosić 0.0."


@pytest.mark.property
@property_settings(max_examples=200, deadline=None)
@given(
    f1=feature_vector_strategy(),
    f2=feature_vector_strategy(),
    covariance=psd_covariance(),
)
def test_mahalanobis_distance_is_valid(
    f1: FeatureVector, f2: FeatureVector, covariance: np.ndarray
) -> None:
    """Odległość Mahalanobisa jest skończona, nieujemna i deterministyczna (Property 10).

    Weryfikuje niezmienniki Property 10 (Wymagania 6.1, 6.2) dla
    :func:`musicians_style.evaluation.distance.mahalanobis` z losową dodatnio
    półokreśloną macierzą kowariancji (``psd_covariance``, rozmiar
    :data:`FEATURE_VECTOR_LENGTH`, konstrukcja niskorzędowa ``A · Aᵀ``):

    * **skończoność** - wynik jest skończoną liczbą rzeczywistą (``math.isfinite``),
    * **nieujemność** - wynik ``>= 0`` (forma kwadratowa z pseudoodwrotnością PSD),
    * **determinizm** - dwa wywołania na tych samych wejściach dają identyczną wartość.

    Symetria ``d(f1, f2) == d(f2, f1)`` zachodzi dla symetrycznej kowariancji
    (a ``psd_covariance`` taką generuje), więc jest dodatkowo asertowana.
    """
    a = f1.as_array()
    b = f2.as_array()

    # Macierz kowariancji ma bok zgodny z długością Wektora_Cech.
    assert covariance.shape == (FEATURE_VECTOR_LENGTH, FEATURE_VECTOR_LENGTH)

    distance = mahalanobis(a, b, covariance)

    # Skończoność i nieujemność (Property 10).
    assert math.isfinite(distance), "Odległość Mahalanobisa musi być skończona."
    assert distance >= 0.0, "Odległość Mahalanobisa musi być nieujemna."

    # Determinizm: powtórne wywołanie daje identyczną wartość.
    assert mahalanobis(a, b, covariance) == distance, (
        "Odległość Mahalanobisa musi być deterministyczna."
    )

    # Symetria dla symetrycznej macierzy kowariancji (opcjonalna w Property 10).
    assert mahalanobis(b, a, covariance) == distance, (
        "Odległość Mahalanobisa musi być symetryczna dla symetrycznej kowariancji."
    )
