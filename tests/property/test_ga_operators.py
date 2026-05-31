# Feature: musicians-style-modeling, Property 9
"""Property 9: Operatory genetyczne zachowują liczność populacji (zadanie 6.8).

**Validates: Requirements 4.1, 4.2**

Sekcja *Correctness Properties* (``design.md``):

    *For any* populacji ``P`` o rozmiarze ``N`` i każdej konfiguracji operatorów
    (selekcja turniejowa, krzyżowanie jednopunktowe lub jednorodne, mutacja
    gaussowska), populacja po pełnym cyklu ``next_generation(P, config)`` ma
    dokładnie ``N`` osobników, a każdy gen w każdym osobniku mieści się w
    zadeklarowanym zakresie typu.

Wymagania (``requirements.md``):

* 4.1 - THE Algorytm_Genetyczny SHALL operować na populacji osobników
  reprezentujących transformacje zakodowane jako wektory parametrów rzeczywistych
  (parametry mogą być ujemne).
* 4.2 - THE Algorytm_Genetyczny SHALL stosować operatory genetyczne: selekcję
  turniejową, krzyżowanie jednopunktowe lub jednorodne, mutację gaussowską.

Test weryfikuje kontrakty na dwóch poziomach:

1. **Poziom operatora** - oba operatory krzyżowania przyjmują 2 rodziców i
   zwracają 2 potomków (zachowanie liczności), selekcja turniejowa zwraca
   istniejący :class:`Genome` z populacji, a mutacja gaussowska zwraca
   :class:`Genome`. Wszystkie produkowane geny są skończone (Wymaganie 4.1 -
   parametry rzeczywiste, dopuszczalne ujemne, lecz zawsze skończone).
2. **Poziom pełnego cyklu** (białoskrzynkowo, jak ``tests/unit/test_ga_algorithm.py``) -
   ``GeneticAlgorithm._next_generation`` zachowuje liczność ``N`` populacji, a
   wszystkie geny w nowej populacji są skończone i mieszczą się w zakresach
   roboczych (saturacja w pętli *Algorytmu_Genetycznego*).

Strategia ``genome_strategy`` pochodzi ze wspólnego, przetestowanego modułu
``tests/property/strategies.py`` (nie jest tu redefiniowana). Jedynym źródłem
losowości operatorów jest zalążkowany :class:`numpy.random.Generator`.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from musicians_style.config import GAConfig
from musicians_style.features.extractor import FeatureExtractor
from musicians_style.features.types import FEATURE_VECTOR_LENGTH, AggregatedFeatures
from musicians_style.ga import (
    GENE_NAMES,
    WORKING_RANGES,
    GeneticAlgorithm,
    Genome,
    apply_transformation,
    gaussian_mutate,
    single_point_crossover,
    tournament_select,
    uniform_crossover,
)
from musicians_style.midi.types import InternalRepr, MetaEvent, NoteEvent

from .strategies import genome_strategy


def _genes(g: Genome) -> tuple[float, float, float, float]:
    """Geny ``g`` w kanonicznej kolejności :data:`GENE_NAMES`."""
    return (
        g.transpose_semitones,
        g.rhythm_density_factor,
        g.note_duration_factor,
        g.velocity_offset,
    )


def _assert_finite_genome(g: Genome) -> None:
    """Asercja: ``g`` jest :class:`Genome` o skończonych genach (Wymaganie 4.1)."""
    assert isinstance(g, Genome)
    for name, value in zip(GENE_NAMES, _genes(g)):
        assert isinstance(value, float)
        assert math.isfinite(value), f"Gen {name!r} nie jest skończony: {value!r}"


def _sample_input() -> InternalRepr:
    """Mały, ustalony *Utwór_Wejściowy* na potrzeby pełnego cyklu pokolenia."""
    notes = (
        NoteEvent(tick=0, channel=0, pitch=60, velocity=100, duration_ticks=480),
        NoteEvent(tick=480, channel=0, pitch=64, velocity=80, duration_ticks=240),
        NoteEvent(tick=960, channel=0, pitch=67, velocity=40, duration_ticks=120),
    )
    meta = (MetaEvent(tick=0, kind="tempo", payload={"tempo": 500_000}),)
    return InternalRepr(ticks_per_beat=480, notes=notes, meta=meta, smf_format=1)


def _achievable_target(x_input: InternalRepr) -> AggregatedFeatures:
    """Osiągalny agregat *Zbioru_Stylu* dla ewaluacji w pełnym cyklu pokolenia."""
    mean = FeatureExtractor().extract(x_input)
    covariance = np.eye(FEATURE_VECTOR_LENGTH, dtype=np.float64)
    return AggregatedFeatures(
        mean=mean, median=mean, std=mean, covariance=covariance
    )


# --------------------------------------------------------------------------- #
# Poziom operatora: krzyżowanie zachowuje liczność (2 rodziców → 2 potomków)
# --------------------------------------------------------------------------- #
@pytest.mark.property
@settings(max_examples=200, deadline=None)
@given(
    p1=genome_strategy(),
    p2=genome_strategy(),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
def test_crossover_operators_return_two_finite_children(
    p1: Genome, p2: Genome, seed: int
) -> None:
    """Oba operatory krzyżowania zwracają dokładnie 2 skończone potomki (Property 9).

    Krzyżowanie jest operacją zachowującą liczność na poziomie pary rodzic-
    potomek: 2 rodziców → 2 potomków (Wymaganie 4.2). Geny potomków są
    rzeczywiste i skończone (Wymaganie 4.1).
    """
    for crossover_op in (single_point_crossover, uniform_crossover):
        rng = np.random.default_rng(seed)
        children = crossover_op(p1, p2, rng)
        assert isinstance(children, tuple)
        assert len(children) == 2
        for child in children:
            _assert_finite_genome(child)


@pytest.mark.property
@settings(max_examples=200, deadline=None)
@given(genome=genome_strategy(), seed=st.integers(min_value=0, max_value=2**32 - 1))
def test_gaussian_mutate_returns_finite_genome(genome: Genome, seed: int) -> None:
    """Mutacja gaussowska zwraca pojedynczy, skończony :class:`Genome` (Property 9)."""
    rng = np.random.default_rng(seed)
    sigma = {
        "transpose_semitones": 1.5,
        "rhythm_density_factor": 0.1,
        "note_duration_factor": 0.1,
        "velocity_offset": 5.0,
    }
    mutated = gaussian_mutate(genome, sigma, rng)
    _assert_finite_genome(mutated)


# --------------------------------------------------------------------------- #
# Poziom operatora: selekcja turniejowa zwraca osobnika z populacji
# --------------------------------------------------------------------------- #
@pytest.mark.property
@settings(max_examples=200, deadline=None)
@given(
    population=st.lists(genome_strategy(), min_size=2, max_size=20),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
    k=st.integers(min_value=1, max_value=7),
)
def test_tournament_select_returns_member_of_population(
    population: list[Genome], seed: int, k: int
) -> None:
    """Selekcja turniejowa zwraca istniejący :class:`Genome` z populacji (Property 9).

    Zwycięzca turnieju MUSI być jednym z osobników wejściowej populacji
    (operator selekcji nie tworzy nowych genotypów). Dopasowania są
    deterministyczne względem indeksu, więc test nie zależy od cech muzycznych.
    """
    rng = np.random.default_rng(seed)
    fitnesses = [float(i) for i in range(len(population))]

    winner = tournament_select(population, fitnesses, k, rng)

    _assert_finite_genome(winner)
    # Zwycięzca jest tożsamy (identycznością) z którymś osobnikiem populacji.
    assert any(winner is member for member in population)


# --------------------------------------------------------------------------- #
# Pełny cykl pokolenia (białoskrzynkowo): liczność N zachowana, geny w zakresie
# --------------------------------------------------------------------------- #
@pytest.mark.property
@settings(max_examples=200, deadline=None)
@given(
    population_size=st.integers(min_value=2, max_value=20),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
    elitism_k=st.integers(min_value=0, max_value=3),
    crossover=st.sampled_from(("uniform", "single_point")),
)
def test_next_generation_preserves_population_size_and_ranges(
    population_size: int, seed: int, elitism_k: int, crossover: str
) -> None:
    """Pełny cykl ``_next_generation`` zachowuje liczność ``N`` i zakresy (Property 9).

    Po pełnym cyklu *selekcja → krzyżowanie → mutacja → elitaryzm* liczność
    populacji pozostaje ``N`` (Wymaganie 4.1/4.2), a każdy gen jest skończony i
    mieści się w :data:`~musicians_style.ga.WORKING_RANGES` (saturacja na
    poziomie pętli *Algorytmu_Genetycznego*).
    """
    x_input = _sample_input()
    target = _achievable_target(x_input)
    config = GAConfig(
        population_size=population_size,
        generations=3,
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
        stagnation_generations=1000,
    )

    ga = GeneticAlgorithm()
    extractor = FeatureExtractor()
    rng = np.random.default_rng(seed)

    population = ga._init_population(population_size, rng)  # noqa: SLF001 - białoskrzynkowo
    assert len(population) == population_size
    fitnesses = ga._evaluate_all(  # noqa: SLF001
        population, x_input, target, config.fitness_metric, extractor
    )

    # Jeden pełny cykl tworzenia kolejnego pokolenia (najprostsza wierna kontrola).
    new_population, new_fitnesses = ga._next_generation(  # noqa: SLF001
        population, fitnesses, x_input, target, config.fitness_metric,
        extractor, config, rng,
    )

    assert len(new_population) == population_size
    assert len(new_fitnesses) == population_size
    for genome in new_population:
        _assert_finite_genome(genome)
        for name, value in zip(GENE_NAMES, _genes(genome)):
            low, high = WORKING_RANGES[name]
            assert low - 1e-9 <= value <= high + 1e-9, (
                f"Gen {name!r}={value} poza zakresem roboczym "
                f"[{low}, {high}] (Property 9)."
            )


@pytest.mark.property
@settings(max_examples=200, deadline=None)
@given(
    population_size=st.integers(min_value=2, max_value=12),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
def test_full_run_history_lengths_are_consistent(
    population_size: int, seed: int
) -> None:
    """``ga.run`` kończy się ze spójnymi długościami :class:`History` (Property 9).

    Uzupełniająca kontrola czarnoskrzynkowa: pełny przebieg kończy się
    poprawnie, a wszystkie krotki historii mają tę samą długość, co potwierdza
    spójną liczność rejestrowanych pokoleń.
    """
    x_input = _sample_input()
    target = _achievable_target(x_input)
    config = GAConfig(
        population_size=population_size,
        generations=5,
        tournament_size=3,
        crossover="uniform",
        mutation_sigma={
            "transpose_semitones": 1.5,
            "rhythm_density_factor": 0.1,
            "note_duration_factor": 0.1,
            "velocity_offset": 5.0,
        },
        elitism_k=2,
        fitness_metric="euclidean",
        stagnation_generations=1000,
    )

    best_genome, history = GeneticAlgorithm().run(x_input, target, config, seed=seed)

    _assert_finite_genome(best_genome)
    n = history.num_generations
    assert n == config.generations + 1
    assert len(history.best_fitness) == n
    assert len(history.mean_fitness) == n
    assert len(history.worst_fitness) == n
    assert len(history.best_genome) == n
