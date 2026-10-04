# Feature: musicians-style-modeling, Property 7
"""Property 7: Determinizm Algorytmu_Genetycznego (zadanie 6.6).

**Validates: Requirements 4.4, 11.4**

Sekcja *Correctness Properties* (``design.md``):

    *For any* ustalonego *Seed* ``s``, *Utworu_Wejściowego* ``x``,
    *Zbioru_Stylu* ``D`` i konfiguracji GA ``c``, dwa niezależne uruchomienia
    ``GA.run(x, D, c, seed=s)`` produkują bit-identyczne populacje końcowe i
    bit-identyczne historie dopasowania.

Wymagania (``requirements.md``):

* 4.4 - WHEN użytkownik podaje *Seed*, THE Algorytm_Genetyczny SHALL produkować
  deterministyczny przebieg ewolucji (identyczna populacja końcowa dla tych
  samych danych wejściowych i tej samej liczby pokoleń).
* 11.4 - FOR ALL ustalonych *Seed* i identycznych konfiguracji wejściowych, THE
  Algorytm_Genetyczny SHALL spełniać własność determinizmu: dwa niezależne
  uruchomienia SHALL produkować identyczne populacje końcowe.

Jedynym źródłem losowości przebiegu jest :class:`numpy.random.Generator`
zainicjalizowany *Seedem*, a *Funkcja_Dopasowania* i ``apply_transformation`` są
czyste. Dlatego dwa wywołania ``run`` z tym samym *Seedem* muszą zwrócić
bit-identyczny najlepszy genotyp oraz :class:`History` (krotki ``best_fitness``,
``mean_fitness``, ``worst_fitness`` i ``best_genome``).

Strategia genotypów (``genome_strategy``) wykorzystywana do zbudowania
*osiągalnego* celu optymalizacji pochodzi ze wspólnego modułu
``tests/property/strategies.py`` (nie jest tu redefiniowana). *Utwór_Wejściowy*
i konfiguracje GA są celowo **małe**, aby test pozostał szybki.
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
from musicians_style.ga import (
    GeneticAlgorithm,
    Genome,
    History,
    apply_transformation,
)
from musicians_style.midi.types import InternalRepr, MetaEvent, NoteEvent

from .strategies import genome_strategy


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
    """Buduje osiągalny agregat *Zbioru_Stylu* z cech ``x`` po transformacji ``genome``.

    Cel optymalizacji jest osiągalny (istnieje genotyp dający odległość 0), co
    czyni przebieg nietrywialnym, lecz przewidywalnym. Determinizm jest jednak
    własnością przebiegu **niezależną** od konkretnego celu.
    """
    mean = FeatureExtractor().extract(apply_transformation(x_input, genome))
    covariance = np.eye(FEATURE_VECTOR_LENGTH, dtype=np.float64)
    return AggregatedFeatures(
        mean=mean, median=mean, std=mean, covariance=covariance
    )


def _tiny_config(crossover: str) -> GAConfig:
    """Mała konfiguracja GA dla szybkich przebiegów property (population ~8, gen ~5)."""
    return GAConfig(
        population_size=8,
        generations=5,
        tournament_size=3,
        crossover=crossover,
        mutation_sigma={
            "transpose_semitones": 1.5,
            "rhythm_density_factor": 0.1,
            "note_duration_factor": 0.1,
            "velocity_offset": 5.0,
        },
        elitism_k=2,
        fitness_metric="euclidean",
        stagnation_generations=1000,  # wysoki próg → pełne max_generations
    )


@pytest.mark.property
@property_settings(max_examples=50, deadline=None)
@given(
    seed=st.integers(min_value=0, max_value=2**32 - 1),
    target_genome=genome_strategy(),
    crossover=st.sampled_from(("uniform", "single_point")),
)
def test_ga_run_is_bit_identical_for_same_seed(
    seed: int, target_genome: Genome, crossover: str
) -> None:
    """Dwa przebiegi ``run`` z tym samym *Seedem* są bit-identyczne (Property 7).

    Weryfikuje równość najlepszego genotypu oraz pełnej :class:`History`
    (``best_fitness``/``mean_fitness``/``worst_fitness`` jako krotki ``float``
    oraz ``best_genome`` jako krotka :class:`Genome`). Drobna wariacja
    konfiguracji (operator krzyżowania) potwierdza determinizm dla obu wariantów
    (Wymagania 4.4, 11.4).
    """
    x_input = _sample_input()
    target = _achievable_target(x_input, target_genome)
    config = _tiny_config(crossover)

    ga = GeneticAlgorithm()
    best_a, hist_a = ga.run(x_input, target, config, seed=seed)
    best_b, hist_b = ga.run(x_input, target, config, seed=seed)

    # Bit-identyczny najlepszy genotyp.
    assert best_a == best_b

    # Bit-identyczna historia: statystyki dopasowania i najlepsze genotypy.
    assert isinstance(hist_a, History)
    assert hist_a.best_fitness == hist_b.best_fitness
    assert hist_a.mean_fitness == hist_b.mean_fitness
    assert hist_a.worst_fitness == hist_b.worst_fitness
    assert hist_a.best_genome == hist_b.best_genome
    assert hist_a.stop_reason == hist_b.stop_reason
    # Pełna równość obiektu History (frozen dataclass z porównaniem po wartości).
    assert hist_a == hist_b
