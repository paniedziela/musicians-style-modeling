"""Testy jednostkowe funkcji odległości (``musicians_style.evaluation.distance``).

Zakres (zadanie 11.1, Wymagania 6.1, 6.2):
* :func:`euclidean` - poprawność wartości, niezmienniki (skończoność,
  nieujemność, symetria, identyczność), walidacja wejść,
* :func:`mahalanobis` - poprawność dla macierzy jednostkowej i pełnej,
  obsługa macierzy osobliwej (pseudoodwrotność), walidacja wejść.

Te funkcje są czyste i deterministyczne - wykorzystywane przez *Funkcję_Dopasowania*
(zadanie 6.2) oraz *ObjectiveEvaluator* (zadanie 11.2).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from musicians_style.evaluation.distance import (
    euclidean,
    mahalanobis,
    mahalanobis_from_inverse,
    prepare_mahalanobis,
)
from musicians_style.features.types import (
    FEATURE_VECTOR_LENGTH,
    INTERVAL_HISTOGRAM_BINS,
    PITCH_CLASS_BINS,
    FeatureVector,
)


# --------------------------------------------------------------------------- #
# Pomocnicze dane
# --------------------------------------------------------------------------- #
def _sample_vector(**overrides: object) -> FeatureVector:
    pch = np.full(PITCH_CLASS_BINS, 1.0 / PITCH_CLASS_BINS, dtype=np.float64)
    ivl = np.zeros(INTERVAL_HISTOGRAM_BINS, dtype=np.float64)
    ivl[12] = 1.0
    params: dict[str, object] = dict(
        tempo_bpm=120.0,
        key="C major",
        pitch_class_histogram=pch,
        interval_histogram=ivl,
        note_density_per_s=2.5,
        mean_note_duration_s=0.4,
        std_note_duration_s=0.1,
        rest_ratio=0.2,
    )
    params.update(overrides)
    return FeatureVector(**params)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# euclidean - poprawność wartości
# --------------------------------------------------------------------------- #
def test_euclidean_known_value() -> None:
    a = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    b = np.array([3.0, 4.0, 0.0], dtype=np.float64)
    assert euclidean(a, b) == pytest.approx(5.0)


def test_euclidean_identical_vectors_is_zero() -> None:
    a = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    assert euclidean(a, a) == 0.0


def test_euclidean_is_symmetric() -> None:
    a = np.array([1.0, -2.0, 3.5], dtype=np.float64)
    b = np.array([-1.0, 0.0, 2.0], dtype=np.float64)
    assert euclidean(a, b) == pytest.approx(euclidean(b, a))


def test_euclidean_non_negative_and_finite() -> None:
    a = np.array([10.0, -5.0, 100.0], dtype=np.float64)
    b = np.array([-3.0, 7.0, 0.0], dtype=np.float64)
    d = euclidean(a, b)
    assert d >= 0.0
    assert math.isfinite(d)


def test_euclidean_returns_builtin_float() -> None:
    a = np.array([1.0, 2.0], dtype=np.float64)
    b = np.array([0.0, 0.0], dtype=np.float64)
    assert type(euclidean(a, b)) is float


def test_euclidean_deterministic() -> None:
    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    b = np.array([4.0, 3.0, 2.0, 1.0], dtype=np.float64)
    assert euclidean(a, b) == euclidean(a, b)


# --------------------------------------------------------------------------- #
# euclidean - akceptacja FeatureVector i jego as_array()
# --------------------------------------------------------------------------- #
def test_euclidean_accepts_feature_vector_objects() -> None:
    fv1 = _sample_vector(tempo_bpm=120.0)
    fv2 = _sample_vector(tempo_bpm=130.0)
    # Akceptacja obiektów z as_array() oraz jawnych tablic daje ten sam wynik.
    via_objects = euclidean(fv1, fv2)
    via_arrays = euclidean(fv1.as_array(), fv2.as_array())
    assert via_objects == pytest.approx(via_arrays)
    # Różnica tylko w tempo (130 - 120 = 10) → odległość 10.
    assert via_objects == pytest.approx(10.0)


def test_euclidean_identical_feature_vectors_is_zero() -> None:
    fv = _sample_vector()
    assert euclidean(fv, fv) == 0.0


# --------------------------------------------------------------------------- #
# euclidean - walidacja wejść
# --------------------------------------------------------------------------- #
def test_euclidean_dimension_mismatch_raises() -> None:
    a = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    b = np.array([1.0, 2.0], dtype=np.float64)
    with pytest.raises(ValueError, match="identyczne wymiary"):
        euclidean(a, b)


def test_euclidean_non_finite_input_raises() -> None:
    a = np.array([1.0, np.nan, 3.0], dtype=np.float64)
    b = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    with pytest.raises(ValueError, match="nieskończone"):
        euclidean(a, b)


def test_euclidean_inf_input_raises() -> None:
    a = np.array([1.0, np.inf, 3.0], dtype=np.float64)
    b = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    with pytest.raises(ValueError, match="nieskończone"):
        euclidean(a, b)


def test_euclidean_two_dimensional_input_raises() -> None:
    a = np.zeros((2, 3), dtype=np.float64)
    b = np.zeros((2, 3), dtype=np.float64)
    with pytest.raises(ValueError, match="jednowymiarowym"):
        euclidean(a, b)


def test_euclidean_empty_input_raises() -> None:
    a = np.array([], dtype=np.float64)
    b = np.array([], dtype=np.float64)
    with pytest.raises(ValueError, match="pustym"):
        euclidean(a, b)


# --------------------------------------------------------------------------- #
# mahalanobis - poprawność wartości
# --------------------------------------------------------------------------- #
def test_mahalanobis_identity_covariance_equals_euclidean() -> None:
    a = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    b = np.array([4.0, 6.0, 3.0], dtype=np.float64)
    cov = np.eye(3, dtype=np.float64)
    assert mahalanobis(a, b, cov) == pytest.approx(euclidean(a, b))


def test_mahalanobis_identical_vectors_is_zero() -> None:
    a = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    cov = np.eye(3, dtype=np.float64)
    assert mahalanobis(a, a, cov) == 0.0


def test_mahalanobis_diagonal_covariance_scales() -> None:
    # Cov = diag(4) → odległość = sqrt(Σ δ²/4) = euclidean/2.
    a = np.array([0.0, 0.0], dtype=np.float64)
    b = np.array([4.0, 0.0], dtype=np.float64)
    cov = np.diag([4.0, 4.0]).astype(np.float64)
    assert mahalanobis(a, b, cov) == pytest.approx(2.0)


def test_mahalanobis_known_full_covariance() -> None:
    # δ = [1, 1], Σ = [[2, 0], [0, 8]] → δᵀ Σ⁻¹ δ = 1/2 + 1/8 = 0.625.
    a = np.array([1.0, 1.0], dtype=np.float64)
    b = np.array([0.0, 0.0], dtype=np.float64)
    cov = np.array([[2.0, 0.0], [0.0, 8.0]], dtype=np.float64)
    assert mahalanobis(a, b, cov) == pytest.approx(math.sqrt(0.625))


def test_prepared_mahalanobis_matches_historical_distance() -> None:
    a = np.array([1.0, 2.0, -1.0])
    b = np.array([0.0, 1.0, 3.0])
    cov = np.array([[2.0, 0.2, 0.0], [0.2, 1.0, 0.1], [0.0, 0.1, 3.0]])
    expected = mahalanobis(a, b, cov)
    inverse = prepare_mahalanobis(cov, dimension=3)
    assert mahalanobis_from_inverse(a, b, inverse) == pytest.approx(expected)


def test_mahalanobis_non_negative_and_finite() -> None:
    a = np.array([5.0, -3.0, 2.0], dtype=np.float64)
    b = np.array([-1.0, 4.0, 0.0], dtype=np.float64)
    cov = np.array(
        [[2.0, 0.5, 0.0], [0.5, 3.0, 0.1], [0.0, 0.1, 1.0]], dtype=np.float64
    )
    d = mahalanobis(a, b, cov)
    assert d >= 0.0
    assert math.isfinite(d)


def test_mahalanobis_singular_covariance_uses_pseudoinverse() -> None:
    # Macierz osobliwa (zerowa) - pinv = 0 → forma kwadratowa = 0.
    a = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    b = np.array([4.0, 5.0, 6.0], dtype=np.float64)
    cov = np.zeros((3, 3), dtype=np.float64)
    d = mahalanobis(a, b, cov)
    assert math.isfinite(d)
    assert d == pytest.approx(0.0)


def test_mahalanobis_rank_deficient_covariance_finite() -> None:
    # Kowariancja o niepełnym rzędzie (powielona kolumna).
    a = np.array([1.0, 0.0, 2.0], dtype=np.float64)
    b = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    base = np.array([[1.0, 1.0, 0.0]], dtype=np.float64)
    cov = base.T @ base  # rząd 1, symetryczna, dodatnio półokreślona
    d = mahalanobis(a, b, cov)
    assert math.isfinite(d)
    assert d >= 0.0


def test_mahalanobis_deterministic() -> None:
    a = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    b = np.array([3.0, 2.0, 1.0], dtype=np.float64)
    cov = np.array(
        [[2.0, 0.3, 0.1], [0.3, 1.5, 0.2], [0.1, 0.2, 1.0]], dtype=np.float64
    )
    assert mahalanobis(a, b, cov) == mahalanobis(a, b, cov)


def test_mahalanobis_accepts_feature_vector_objects() -> None:
    fv1 = _sample_vector(tempo_bpm=120.0)
    fv2 = _sample_vector(tempo_bpm=130.0)
    cov = np.eye(FEATURE_VECTOR_LENGTH, dtype=np.float64)
    via_objects = mahalanobis(fv1, fv2, cov)
    via_arrays = mahalanobis(fv1.as_array(), fv2.as_array(), cov)
    assert via_objects == pytest.approx(via_arrays)


# --------------------------------------------------------------------------- #
# mahalanobis - walidacja wejść
# --------------------------------------------------------------------------- #
def test_mahalanobis_vector_dimension_mismatch_raises() -> None:
    a = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    b = np.array([1.0, 2.0], dtype=np.float64)
    cov = np.eye(3, dtype=np.float64)
    with pytest.raises(ValueError, match="identyczne wymiary"):
        mahalanobis(a, b, cov)


def test_mahalanobis_non_square_covariance_raises() -> None:
    a = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    b = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    cov = np.zeros((3, 2), dtype=np.float64)
    with pytest.raises(ValueError, match="kwadratowa"):
        mahalanobis(a, b, cov)


def test_mahalanobis_covariance_side_mismatch_raises() -> None:
    a = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    b = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    cov = np.eye(2, dtype=np.float64)
    with pytest.raises(ValueError, match="Bok macierzy kowariancji"):
        mahalanobis(a, b, cov)


def test_mahalanobis_non_finite_covariance_raises() -> None:
    a = np.array([1.0, 2.0], dtype=np.float64)
    b = np.array([0.0, 0.0], dtype=np.float64)
    cov = np.array([[1.0, np.nan], [0.0, 1.0]], dtype=np.float64)
    with pytest.raises(ValueError, match="nieskończone"):
        mahalanobis(a, b, cov)


def test_mahalanobis_non_finite_vector_raises() -> None:
    a = np.array([1.0, np.inf], dtype=np.float64)
    b = np.array([0.0, 0.0], dtype=np.float64)
    cov = np.eye(2, dtype=np.float64)
    with pytest.raises(ValueError, match="nieskończone"):
        mahalanobis(a, b, cov)
