# Feature: musicians-style-modeling, Property 8
"""Property 8: Monotoniczność Algorytmu_Genetycznego w trybie elitaryzmu (zadanie 6.7).

**Validates: Requirements 4.7, 11.5**

Sekcja *Correctness Properties* (``design.md``):

    *For any* uruchomienia ``GA.run(...)`` z ``elitism_k >= 1``, dla każdej pary
    kolejnych pokoleń ``n`` i ``n+1``, najlepsza wartość *Funkcji_Dopasowania* w
    pokoleniu ``n+1`` jest nie gorsza niż najlepsza wartość w pokoleniu ``n``.

Wymagania (``requirements.md``):

* 4.7 - WHILE algorytm jest uruchomiony w trybie elitaryzmu, THE
  Algorytm_Genetyczny SHALL zachowywać monotoniczność najlepszej wartości
  funkcji dopasowania.
* 11.5 - FOR ALL pokoleń ``n`` w trybie elitaryzmu, THE Algorytm_Genetyczny
  SHALL spełniać własność monotoniczności: najlepsza wartość *Funkcji_Dopasowania*
  w pokoleniu ``n+1`` SHALL być nie gorsza niż w pokoleniu ``n``.

Przy ``elitism_k >= 1`` najlepszy osobnik pokolenia ``n`` jest przenoszony
**bez zmian** (wraz z deterministycznie odtwarzalnym dopasowaniem) do pokolenia
``n+1``, więc ciąg ``history.best_fitness`` jest niemalejący. Porównanie używa
małej tolerancji ``1e-9`` na zaokrąglenia zmiennoprzecinkowe.

Strategie ``genome_strategy`` (do zbudowania osiągalnego celu) pochodzą ze
wspólnego modułu ``tests/property/strategies.py`` (nie są tu redefiniowane).
Konfiguracje GA są **małe** dla szybkości testu.
"""

from __future__ import annotations

import numpy as np
import pytest
from .profiles import property_settings
from hypothesis import given
from hypothesis import strategies as st

from musicians_style.config import GAConfig
from musicians_style.features.extractor import FeatureExtractor
from musicians_style.features.types import FEATURE_VECTOR_LENGTH, AggregatedFeatures
from musicians_style.ga import GeneticAlgorithm, Genome, apply_transformation
from musicians_style.midi.types import InternalRepr, MetaEvent, NoteEvent

from .strategies import genome_strategy

#: Tolerancja porównań zmiennoprzecinkowych dla warunku monotoniczności.
_TOL = 1e-9


def _sample_input() -> InternalRepr:
    """Mały, ustalony *Utwór_Wejściowy* (cztery nuty + meta-zdarzenia)."""
    notes = (
        NoteEvent(tick=0, channel=0, pitch=60, velocity=100, duration_ticks=480),
        NoteEvent(tick=480, channel=0, pitch=64, velocity=80, duration_ticks=240),
        NoteEvent(tick=960, channel=0, pitch=67, velocity=40, duration_ticks=120),
        NoteEvent(tick=1200, channel=0, pitch=72, velocity=90, duration_ticks=480),
    )
    meta = (
        MetaEvent(tick=0, kind="tempo", payload={"tempo": 500_000}),
        MetaEvent(
            tick=0, kind="time_signature", payload={"numerator": 4, "denominator": 4}
        ),
    )
    return InternalRepr(ticks_per_beat=480, notes=notes, meta=meta, smf_format=1)


def _achievable_target(
    x_input: InternalRepr, genome: Genome
) -> AggregatedFeatures:
    """Buduje osiągalny agregat *Zbioru_Stylu* z cech ``x`` po transformacji ``genome``."""
    mean = FeatureExtractor().extract(apply_transformation(x_input, genome))
    covariance = np.eye(FEATURE_VECTOR_LENGTH, dtype=np.float64)
    return AggregatedFeatures(
        mean=mean, median=mean, std=mean, covariance=covariance
    )


@pytest.mark.property
@property_settings(max_examples=50, deadline=None)
@given(
    seed=st.integers(min_value=0, max_value=2**32 - 1),
    target_genome=genome_strategy(),
    elitism_k=st.integers(min_value=1, max_value=3),
    crossover=st.sampled_from(("uniform", "single_point")),
)
def test_best_fitness_is_non_decreasing_with_elitism(
    seed: int, target_genome: Genome, elitism_k: int, crossover: str
) -> None:
    """``best_fitness`` jest niemalejące przy ``elitism_k >= 1`` (Property 8).

    Dla każdej pary kolejnych pokoleń ``(n, n+1)`` weryfikuje
    ``best_fitness[n+1] >= best_fitness[n] - 1e-9`` (Wymagania 4.7, 11.5).
    """
    x_input = _sample_input()
    target = _achievable_target(x_input, target_genome)
    config = GAConfig(
        population_size=10,
        generations=8,
        tournament_size=3,
        crossover=crossover,
        mutation_sigma={
            "transpose_semitones": 1.5,
            "rhythm_density_factor": 0.1,
            "note_duration_factor": 0.1,
            "velocity_offset": 5.0,
        },
        elitism_k=elitism_k,
        fitness_metric="euclidean",
        stagnation_generations=1000,  # wysoki próg → wiele pokoleń do sprawdzenia
    )

    _, history = GeneticAlgorithm().run(x_input, target, config, seed=seed)

    best = history.best_fitness
    assert len(best) >= 1
    for n in range(len(best) - 1):
        assert best[n + 1] >= best[n] - _TOL, (
            f"Naruszono monotoniczność elitaryzmu w pokoleniu {n}: "
            f"{best[n]} -> {best[n + 1]} (Property 8 / Wymaganie 4.7)."
        )
