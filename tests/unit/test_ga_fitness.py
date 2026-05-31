"""Testy jednostkowe *Funkcji_Dopasowania* *Algorytmu_Genetycznego* (zadanie 6.2).

Weryfikują kontrakt :func:`musicians_style.ga.fitness.fitness` opisany w sekcji
*Funkcja_Dopasowania* (``design.md``) i Wymaganiu 4.3:

* dopasowanie = ``-distance(extract(apply_transformation(x, g)), target.mean)``,
* wsparcie dla metryki euklidesowej i Mahalanobisa (``target.covariance``),
* maksymalizacja dopasowania == minimalizacja odległości (wartości ``<= 0``),
* determinizm względem wejść (brak losowości),
* czystość (brak modyfikacji wejścia), walidacja nieznanej metryki,
* spójność z bezpośrednim złożeniem komponentów (extract + distance).
"""

from __future__ import annotations

import numpy as np
import pytest

from musicians_style.evaluation.distance import euclidean, mahalanobis
from musicians_style.features.extractor import FeatureExtractor
from musicians_style.features.types import (
    FEATURE_VECTOR_LENGTH,
    INTERVAL_HISTOGRAM_BINS,
    PITCH_CLASS_BINS,
    AggregatedFeatures,
    FeatureVector,
)
from musicians_style.ga import IDENTITY_GENOME, Genome, apply_transformation, fitness
from musicians_style.midi.types import InternalRepr, MetaEvent, NoteEvent


# --------------------------------------------------------------------------- #
# Pomocnicze dane
# --------------------------------------------------------------------------- #
def _sample_repr() -> InternalRepr:
    notes = (
        NoteEvent(tick=0, channel=0, pitch=60, velocity=100, duration_ticks=480),
        NoteEvent(tick=480, channel=0, pitch=64, velocity=80, duration_ticks=240),
        NoteEvent(tick=960, channel=0, pitch=67, velocity=40, duration_ticks=120),
        NoteEvent(tick=1200, channel=0, pitch=72, velocity=90, duration_ticks=480),
    )
    meta = (
        MetaEvent(tick=0, kind="tempo", payload={"tempo": 500_000}),
        MetaEvent(tick=0, kind="time_signature", payload={"numerator": 4, "denominator": 4}),
    )
    return InternalRepr(ticks_per_beat=480, notes=notes, meta=meta, smf_format=1)


def _feature_vector(**overrides: object) -> FeatureVector:
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


def _aggregated(
    mean: FeatureVector, covariance: np.ndarray | None = None
) -> AggregatedFeatures:
    if covariance is None:
        covariance = np.eye(FEATURE_VECTOR_LENGTH, dtype=np.float64)
    return AggregatedFeatures(
        mean=mean, median=mean, std=mean, covariance=covariance
    )


# --------------------------------------------------------------------------- #
# Definicja dopasowania (Wymaganie 4.3)
# --------------------------------------------------------------------------- #
def test_fitness_equals_negative_euclidean_distance() -> None:
    repr_ = _sample_repr()
    target = _aggregated(_feature_vector())
    genome = Genome(3.0, 1.2, 0.8, 10.0)

    extractor = FeatureExtractor()
    expected_distance = euclidean(
        extractor.extract(apply_transformation(repr_, genome)).as_array(),
        target.mean.as_array(),
    )

    assert fitness(genome, repr_, target, "euclidean") == pytest.approx(
        -expected_distance
    )


def test_fitness_equals_negative_mahalanobis_distance() -> None:
    repr_ = _sample_repr()
    # Niejednostkowa, dodatnio półokreślona macierz kowariancji.
    rng = np.random.default_rng(0)
    a = rng.standard_normal((FEATURE_VECTOR_LENGTH, FEATURE_VECTOR_LENGTH))
    cov = a @ a.T + np.eye(FEATURE_VECTOR_LENGTH)
    target = _aggregated(_feature_vector(tempo_bpm=130.0), covariance=cov)
    genome = Genome(-2.0, 0.9, 1.1, -5.0)

    extractor = FeatureExtractor()
    expected_distance = mahalanobis(
        extractor.extract(apply_transformation(repr_, genome)).as_array(),
        target.mean.as_array(),
        cov,
    )

    assert fitness(genome, repr_, target, "mahalanobis") == pytest.approx(
        -expected_distance
    )


def test_fitness_default_metric_is_euclidean() -> None:
    repr_ = _sample_repr()
    target = _aggregated(_feature_vector())
    genome = Genome(1.0, 1.0, 1.0, 0.0)
    assert fitness(genome, repr_, target) == pytest.approx(
        fitness(genome, repr_, target, "euclidean")
    )


# --------------------------------------------------------------------------- #
# Maksymalizacja == minimalizacja odległości
# --------------------------------------------------------------------------- #
def test_fitness_is_non_positive() -> None:
    repr_ = _sample_repr()
    target = _aggregated(_feature_vector())
    for genome in (
        IDENTITY_GENOME,
        Genome(5.0, 1.5, 0.7, 20.0),
        Genome(-7.0, 0.6, 1.4, -15.0),
    ):
        assert fitness(genome, repr_, target) <= 0.0


def test_fitness_is_zero_when_features_match_target_mean() -> None:
    # Gdy średnia Zbioru_Stylu jest dokładnie cechami x po transformacji,
    # odległość == 0, więc dopasowanie == 0 (maksimum).
    repr_ = _sample_repr()
    extractor = FeatureExtractor()
    exact_mean = extractor.extract(apply_transformation(repr_, IDENTITY_GENOME))
    target = _aggregated(exact_mean)
    assert fitness(IDENTITY_GENOME, repr_, target) == pytest.approx(0.0)


def test_closer_genome_has_higher_fitness() -> None:
    # Cel: cechy x transponowanego o +5 półtonów. Genom +5 powinien mieć
    # dopasowanie nie gorsze (wyższe) niż genom oddalony (np. +0).
    repr_ = _sample_repr()
    extractor = FeatureExtractor()
    target_mean = extractor.extract(apply_transformation(repr_, Genome(5.0, 1.0, 1.0, 0.0)))
    target = _aggregated(target_mean)

    fit_close = fitness(Genome(5.0, 1.0, 1.0, 0.0), repr_, target)
    fit_far = fitness(Genome(0.0, 1.0, 1.0, 0.0), repr_, target)
    assert fit_close >= fit_far
    assert fit_close == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# Determinizm i czystość
# --------------------------------------------------------------------------- #
def test_fitness_is_deterministic() -> None:
    repr_ = _sample_repr()
    target = _aggregated(_feature_vector())
    genome = Genome(2.0, 1.3, 0.7, 15.0)
    assert fitness(genome, repr_, target) == fitness(genome, repr_, target)


def test_fitness_does_not_mutate_input_repr() -> None:
    repr_ = _sample_repr()
    snapshot = _sample_repr()
    target = _aggregated(_feature_vector())
    fitness(Genome(4.0, 1.5, 0.5, 12.0), repr_, target, "mahalanobis")
    assert repr_ == snapshot


def test_fitness_returns_builtin_float() -> None:
    repr_ = _sample_repr()
    target = _aggregated(_feature_vector())
    assert type(fitness(IDENTITY_GENOME, repr_, target)) is float


def test_fitness_accepts_injected_extractor() -> None:
    repr_ = _sample_repr()
    target = _aggregated(_feature_vector())
    genome = Genome(3.0, 1.0, 1.0, 0.0)
    shared = FeatureExtractor()
    assert fitness(genome, repr_, target, extractor=shared) == pytest.approx(
        fitness(genome, repr_, target)
    )


# --------------------------------------------------------------------------- #
# Walidacja metryki
# --------------------------------------------------------------------------- #
def test_fitness_unknown_metric_raises() -> None:
    repr_ = _sample_repr()
    target = _aggregated(_feature_vector())
    with pytest.raises(ValueError, match="Nieobsługiwana metryka"):
        fitness(IDENTITY_GENOME, repr_, target, "manhattan")  # type: ignore[arg-type]


def test_fitness_handles_empty_repr() -> None:
    # Pusty Utwór_Wejściowy -> Wektor_Cech neutralny; funkcja nie rzuca wyjątku.
    empty = InternalRepr(ticks_per_beat=480)
    target = _aggregated(_feature_vector())
    value = fitness(IDENTITY_GENOME, empty, target)
    assert value <= 0.0
    assert np.isfinite(value)
