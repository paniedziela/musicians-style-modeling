"""Testy jednostkowe modelu danych *Wektora_Cech* (``musicians_style.features``).

Zakres (zadanie 3.1, Wymagania 2.1, 2.3, 2.7):
* :class:`FeatureVector` - walidacja kształtów, niemutowalność, ``as_array``
  (stała długość, układ indeksów), równość i haszowanie (idempotencja),
* :class:`AggregatedFeatures` - walidacja kształtu macierzy kowariancji,
  równość/haszowanie,
* :data:`NEUTRAL_FEATURE_VECTOR` - wartości neutralne dla plików pustych.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from musicians_style.features.constants import (
    NEUTRAL_KEY,
    NEUTRAL_TEMPO_BPM,
    NEUTRAL_FEATURE_VECTOR,
)
from musicians_style.features.types import (
    FEATURE_VECTOR_LENGTH,
    INTERVAL_HISTOGRAM_BINS,
    PITCH_CLASS_BINS,
    AggregatedFeatures,
    FeatureVector,
)


# --------------------------------------------------------------------------- #
# Pomocnicze dane
# --------------------------------------------------------------------------- #
def _sample_vector(**overrides: object) -> FeatureVector:
    pch = np.full(PITCH_CLASS_BINS, 1.0 / PITCH_CLASS_BINS, dtype=np.float64)
    ivl = np.zeros(INTERVAL_HISTOGRAM_BINS, dtype=np.float64)
    ivl[12] = 1.0  # interwał 0 (najczęstszy w neutralnym śladzie)
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


def _identity_covariance() -> np.ndarray:
    return np.eye(FEATURE_VECTOR_LENGTH, dtype=np.float64)


# --------------------------------------------------------------------------- #
# FeatureVector - konstrukcja i walidacja kształtów
# --------------------------------------------------------------------------- #
def test_constructs_with_valid_fields() -> None:
    fv = _sample_vector()
    assert fv.tempo_bpm == 120.0
    assert fv.key == "C major"
    assert fv.pitch_class_histogram.shape == (PITCH_CLASS_BINS,)
    assert fv.interval_histogram.shape == (INTERVAL_HISTOGRAM_BINS,)


def test_scalar_fields_normalized_to_builtin_types() -> None:
    fv = _sample_vector(tempo_bpm=np.float64(130.0), rest_ratio=np.float64(0.5))
    assert type(fv.tempo_bpm) is float
    assert type(fv.rest_ratio) is float


def test_arrays_coerced_to_float64() -> None:
    fv = _sample_vector(
        pitch_class_histogram=np.ones(PITCH_CLASS_BINS, dtype=np.int32)
    )
    assert fv.pitch_class_histogram.dtype == np.float64


def test_invalid_pitch_class_shape_raises() -> None:
    with pytest.raises(ValueError, match="pitch_class_histogram"):
        _sample_vector(pitch_class_histogram=np.zeros(11, dtype=np.float64))


def test_invalid_interval_shape_raises() -> None:
    with pytest.raises(ValueError, match="interval_histogram"):
        _sample_vector(interval_histogram=np.zeros(24, dtype=np.float64))


# --------------------------------------------------------------------------- #
# FeatureVector - niemutowalność
# --------------------------------------------------------------------------- #
def test_is_frozen_scalar_assignment_raises() -> None:
    fv = _sample_vector()
    with pytest.raises(dataclasses.FrozenInstanceError):
        fv.tempo_bpm = 200.0  # type: ignore[misc]


def test_stored_arrays_are_read_only() -> None:
    fv = _sample_vector()
    assert fv.pitch_class_histogram.flags.writeable is False
    with pytest.raises(ValueError):
        fv.pitch_class_histogram[0] = 0.5


def test_constructor_copies_source_array() -> None:
    src = np.full(PITCH_CLASS_BINS, 1.0 / PITCH_CLASS_BINS, dtype=np.float64)
    fv = _sample_vector(pitch_class_histogram=src)
    # mutacja oryginalnej tablicy nie wpływa na zapisaną kopię
    src[0] = 99.0
    assert fv.pitch_class_histogram[0] == pytest.approx(1.0 / PITCH_CLASS_BINS)


# --------------------------------------------------------------------------- #
# FeatureVector.as_array - stała długość i układ
# --------------------------------------------------------------------------- #
def test_as_array_has_constant_length() -> None:
    fv = _sample_vector()
    assert fv.as_array().shape == (FEATURE_VECTOR_LENGTH,)
    assert fv.as_array().dtype == np.float64


def test_as_array_length_independent_of_input() -> None:
    a = _sample_vector(note_density_per_s=0.0)
    b = _sample_vector(note_density_per_s=1000.0)
    assert a.as_array().shape == b.as_array().shape == (FEATURE_VECTOR_LENGTH,)


def test_as_array_layout() -> None:
    ivl = np.arange(INTERVAL_HISTOGRAM_BINS, dtype=np.float64)
    pch = np.linspace(0.0, 1.0, PITCH_CLASS_BINS, dtype=np.float64)
    fv = _sample_vector(
        tempo_bpm=99.0,
        pitch_class_histogram=pch,
        interval_histogram=ivl,
        note_density_per_s=3.0,
        mean_note_duration_s=0.5,
        std_note_duration_s=0.25,
        rest_ratio=0.75,
    )
    arr = fv.as_array()
    assert arr[0] == 99.0
    np.testing.assert_array_equal(arr[1 : 1 + PITCH_CLASS_BINS], pch)
    start = 1 + PITCH_CLASS_BINS
    np.testing.assert_array_equal(
        arr[start : start + INTERVAL_HISTOGRAM_BINS], ivl
    )
    tail = arr[start + INTERVAL_HISTOGRAM_BINS :]
    np.testing.assert_array_equal(tail, np.array([3.0, 0.5, 0.25, 0.75]))


def test_as_array_returns_writeable_copy() -> None:
    fv = _sample_vector()
    arr = fv.as_array()
    assert arr.flags.writeable is True
    arr[0] = -1.0  # nie wpływa na wektor źródłowy
    assert fv.tempo_bpm == 120.0


# --------------------------------------------------------------------------- #
# FeatureVector - równość i haszowanie (idempotencja)
# --------------------------------------------------------------------------- #
def test_equality_value_based() -> None:
    assert _sample_vector() == _sample_vector()


def test_equality_hash_consistent() -> None:
    assert hash(_sample_vector()) == hash(_sample_vector())


def test_usable_in_set() -> None:
    assert len({_sample_vector(), _sample_vector()}) == 1


def test_inequality_on_scalar_field() -> None:
    assert _sample_vector() != _sample_vector(tempo_bpm=121.0)


def test_inequality_on_array_field() -> None:
    other = np.zeros(PITCH_CLASS_BINS, dtype=np.float64)
    other[0] = 1.0
    assert _sample_vector() != _sample_vector(pitch_class_histogram=other)


def test_equality_with_other_type_returns_not_implemented() -> None:
    assert _sample_vector().__eq__(object()) is NotImplemented


def test_equality_handles_nan_arrays() -> None:
    # Wektory częściowe (Wymaganie 2.8) mogą zawierać NaN; NaN == NaN dla równości.
    nan_hist = np.full(PITCH_CLASS_BINS, np.nan, dtype=np.float64)
    a = _sample_vector(pitch_class_histogram=nan_hist)
    b = _sample_vector(pitch_class_histogram=nan_hist)
    assert a == b


# --------------------------------------------------------------------------- #
# AggregatedFeatures
# --------------------------------------------------------------------------- #
def test_aggregated_constructs_with_valid_covariance() -> None:
    agg = AggregatedFeatures(
        mean=_sample_vector(),
        median=_sample_vector(),
        std=_sample_vector(),
        covariance=_identity_covariance(),
    )
    assert agg.covariance.shape == (FEATURE_VECTOR_LENGTH, FEATURE_VECTOR_LENGTH)
    assert agg.covariance.dtype == np.float64
    assert agg.covariance.flags.writeable is False


def test_aggregated_invalid_covariance_shape_raises() -> None:
    with pytest.raises(ValueError, match="covariance"):
        AggregatedFeatures(
            mean=_sample_vector(),
            median=_sample_vector(),
            std=_sample_vector(),
            covariance=np.eye(FEATURE_VECTOR_LENGTH - 1, dtype=np.float64),
        )


def test_aggregated_equality_and_hash() -> None:
    def build() -> AggregatedFeatures:
        return AggregatedFeatures(
            mean=_sample_vector(),
            median=_sample_vector(),
            std=_sample_vector(),
            covariance=_identity_covariance(),
        )

    assert build() == build()
    assert hash(build()) == hash(build())


def test_aggregated_inequality_on_covariance() -> None:
    base = AggregatedFeatures(
        mean=_sample_vector(),
        median=_sample_vector(),
        std=_sample_vector(),
        covariance=_identity_covariance(),
    )
    other_cov = _identity_covariance()
    other_cov[0, 1] = 0.5
    other = AggregatedFeatures(
        mean=_sample_vector(),
        median=_sample_vector(),
        std=_sample_vector(),
        covariance=other_cov,
    )
    assert base != other


def test_aggregated_is_frozen() -> None:
    agg = AggregatedFeatures(
        mean=_sample_vector(),
        median=_sample_vector(),
        std=_sample_vector(),
        covariance=_identity_covariance(),
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        agg.covariance = _identity_covariance()  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# NEUTRAL_FEATURE_VECTOR (Wymagania 2.7, 11.8)
# --------------------------------------------------------------------------- #
def test_neutral_vector_values() -> None:
    n = NEUTRAL_FEATURE_VECTOR
    assert n.tempo_bpm == NEUTRAL_TEMPO_BPM == 120.0
    assert n.key == NEUTRAL_KEY == "C major"
    assert n.note_density_per_s == 0.0
    assert n.mean_note_duration_s == 0.0
    assert n.std_note_duration_s == 0.0
    assert n.rest_ratio == 1.0


def test_neutral_vector_histograms() -> None:
    n = NEUTRAL_FEATURE_VECTOR
    # pitch class histogram: rozkład jednostajny sumujący się do 1
    assert n.pitch_class_histogram.shape == (PITCH_CLASS_BINS,)
    assert n.pitch_class_histogram.sum() == pytest.approx(1.0, abs=1e-9)
    np.testing.assert_allclose(
        n.pitch_class_histogram, 1.0 / PITCH_CLASS_BINS
    )
    # interval histogram: zera
    assert n.interval_histogram.shape == (INTERVAL_HISTOGRAM_BINS,)
    np.testing.assert_array_equal(
        n.interval_histogram, np.zeros(INTERVAL_HISTOGRAM_BINS)
    )


def test_neutral_vector_rest_ratio_in_unit_interval() -> None:
    assert 0.0 <= NEUTRAL_FEATURE_VECTOR.rest_ratio <= 1.0


def test_neutral_vector_as_array_constant_length() -> None:
    assert NEUTRAL_FEATURE_VECTOR.as_array().shape == (FEATURE_VECTOR_LENGTH,)


def test_neutral_vector_equality_idempotent() -> None:
    assert NEUTRAL_FEATURE_VECTOR == NEUTRAL_FEATURE_VECTOR
    assert hash(NEUTRAL_FEATURE_VECTOR) == hash(NEUTRAL_FEATURE_VECTOR)
