"""Testy jednostkowe głównej pętli *Algorytmu_Genetycznego* (zadanie 6.4).

Weryfikują kontrakt :class:`musicians_style.ga.algorithm.GeneticAlgorithm`
opisany w sekcji *Algorytm_Genetyczny* (``design.md``) i Wymaganiach 4.4-4.7:

* **determinizm** (Wymaganie 4.4, Property 7): dwa przebiegi z tym samym
  *Seedem*, identycznym wejściem i konfiguracją zwracają bit-identyczny najlepszy
  genotyp oraz historię dopasowania,
* **monotoniczność elitaryzmu** (Wymaganie 4.7, Property 8): przy
  ``elitism_k >= 1`` najlepsze dopasowanie jest niemalejące między pokoleniami,
* **warunki stopu** (Wymaganie 4.5): zakończenie po ``max_generations`` oraz po
  stagnacji najlepszego dopasowania,
* **historia per pokolenie** (Wymaganie 4.6): rejestrowane są
  ``(best_fitness, mean_fitness, worst_fitness, best_genome)`` dla każdego
  pokolenia, a opcjonalny plik ``ga.jsonl`` zawiera te statystyki,
* **zakresy robocze**: populacja inicjalizowana i utrzymywana w
  :data:`~musicians_style.ga.algorithm.WORKING_RANGES`.

Używane są małe populacje i krótkie przebiegi dla szybkości testów.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from musicians_style.config import GAConfig
from musicians_style.features.extractor import FeatureExtractor
from musicians_style.ga import (
    WORKING_RANGES,
    GeneticAlgorithm,
    Genome,
    History,
    apply_transformation,
)
from musicians_style.ga.algorithm import GENE_NAMES
from musicians_style.midi.types import InternalRepr, MetaEvent, NoteEvent


# --------------------------------------------------------------------------- #
# Pomocnicze dane
# --------------------------------------------------------------------------- #
def _sample_repr() -> InternalRepr:
    """Niewielki *Utwór_Wejściowy* z czterema nutami i meta-zdarzeniami."""
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


def _target_for(genome: Genome) -> "object":
    """Buduje agregat *Zbioru_Stylu* równy cechom ``x`` po transformacji ``genome``.

    Dzięki temu cel optymalizacji jest osiągalny (istnieje genotyp dający
    odległość 0), co czyni testy konwergencji i stagnacji przewidywalnymi.
    """
    from musicians_style.features.types import (
        FEATURE_VECTOR_LENGTH,
        AggregatedFeatures,
    )

    extractor = FeatureExtractor()
    mean = extractor.extract(apply_transformation(_sample_repr(), genome))
    covariance = np.eye(FEATURE_VECTOR_LENGTH, dtype=np.float64)
    return AggregatedFeatures(
        mean=mean, median=mean, std=mean, covariance=covariance
    )


def _small_config(**overrides: object) -> GAConfig:
    """Mała konfiguracja *Algorytmu_Genetycznego* na potrzeby szybkich testów."""
    params: dict[str, object] = dict(
        population_size=12,
        generations=8,
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
        stagnation_generations=30,  # wysoki próg → stop po max_generations
    )
    params.update(overrides)
    return GAConfig(**params)  # type: ignore[arg-type]


def _genes(g: Genome) -> tuple[float, float, float, float]:
    return (
        g.transpose_semitones,
        g.rhythm_density_factor,
        g.note_duration_factor,
        g.velocity_offset,
    )


# --------------------------------------------------------------------------- #
# Determinizm (Wymaganie 4.4, Property 7)
# --------------------------------------------------------------------------- #
def test_run_is_deterministic_for_same_seed() -> None:
    x = _sample_repr()
    target = _target_for(Genome(5.0, 1.2, 0.9, 10.0))
    config = _small_config()

    ga = GeneticAlgorithm()
    best_a, hist_a = ga.run(x, target, config, seed=2024)
    best_b, hist_b = ga.run(x, target, config, seed=2024)

    assert best_a == best_b
    assert hist_a == hist_b


def test_run_different_seed_changes_trajectory() -> None:
    x = _sample_repr()
    target = _target_for(Genome(-4.0, 0.8, 1.3, -12.0))
    config = _small_config()

    ga = GeneticAlgorithm()
    _, hist_a = ga.run(x, target, config, seed=1)
    _, hist_b = ga.run(x, target, config, seed=2)

    # Inny Seed → inna trajektoria (z bardzo wysokim prawdopodobieństwem).
    assert hist_a.best_genome != hist_b.best_genome


def test_run_deterministic_with_injected_shared_extractor() -> None:
    x = _sample_repr()
    target = _target_for(Genome(3.0, 1.0, 1.0, 0.0))
    config = _small_config()

    best_a, hist_a = GeneticAlgorithm(FeatureExtractor()).run(
        x, target, config, seed=7
    )
    best_b, hist_b = GeneticAlgorithm(FeatureExtractor()).run(
        x, target, config, seed=7
    )
    assert best_a == best_b
    assert hist_a == hist_b


# --------------------------------------------------------------------------- #
# Monotoniczność elitaryzmu (Wymaganie 4.7, Property 8)
# --------------------------------------------------------------------------- #
def test_elitism_makes_best_fitness_non_decreasing() -> None:
    x = _sample_repr()
    target = _target_for(Genome(6.0, 1.4, 0.7, 20.0))
    config = _small_config(elitism_k=2, generations=15)

    _, history = GeneticAlgorithm().run(x, target, config, seed=42)

    best = history.best_fitness
    for n in range(len(best) - 1):
        assert best[n + 1] >= best[n] - 1e-12, (
            f"naruszono monotoniczność w pokoleniu {n}: "
            f"{best[n]} -> {best[n + 1]}"
        )


def test_returned_best_genome_matches_last_generation_best_with_elitism() -> None:
    x = _sample_repr()
    target = _target_for(Genome(2.0, 1.1, 0.95, 5.0))
    config = _small_config(elitism_k=3, generations=10)

    best_genome, history = GeneticAlgorithm().run(x, target, config, seed=11)
    # Przy elitaryzmie najlepszy genom całego przebiegu == najlepszy w ostatnim
    # zarejestrowanym pokoleniu (monotoniczność).
    assert best_genome == history.best_genome[-1]


def test_without_elitism_best_fitness_may_not_be_monotone_but_run_succeeds() -> None:
    # Bez elitaryzmu (elitism_k=0) monotoniczność nie jest gwarantowana, lecz
    # przebieg ma się zakończyć poprawnie, a zwrócony genom być najlepszym
    # napotkanym (jego dopasowanie >= najlepsze z ostatniego pokolenia... niekoniecznie,
    # ale >= max po wszystkich pokoleniach).
    x = _sample_repr()
    target = _target_for(Genome(1.0, 1.0, 1.0, 0.0))
    config = _small_config(elitism_k=0, generations=8)

    best_genome, history = GeneticAlgorithm().run(x, target, config, seed=3)
    assert isinstance(best_genome, Genome)
    # Najlepszy zwrócony >= maksimum best_fitness z historii.
    extractor = FeatureExtractor()
    from musicians_style.ga import fitness

    returned_fit = fitness(best_genome, x, target, "euclidean", extractor=extractor)
    assert returned_fit >= max(history.best_fitness) - 1e-9


# --------------------------------------------------------------------------- #
# Warunki stopu (Wymaganie 4.5)
# --------------------------------------------------------------------------- #
def test_stop_after_max_generations() -> None:
    x = _sample_repr()
    target = _target_for(Genome(5.0, 1.0, 1.0, 0.0))
    config = _small_config(generations=6, stagnation_generations=1000)

    _, history = GeneticAlgorithm().run(x, target, config, seed=5)

    # Pokolenie 0 + 6 pokoleń ewolucji = 7 zarejestrowanych pokoleń.
    assert history.num_generations == config.generations + 1
    assert history.stop_reason == "max_generations"


def test_stop_on_stagnation() -> None:
    x = _sample_repr()
    target = _target_for(Genome(5.0, 1.0, 1.0, 0.0))
    # Wysoki elitaryzm + niskie sigma sprawiają, że po znalezieniu optimum
    # najlepsze dopasowanie nie poprawia się → stagnacja kończy przebieg
    # przed osiągnięciem dużej liczby pokoleń.
    config = _small_config(
        population_size=10,
        generations=500,
        elitism_k=4,
        stagnation_generations=3,
    )

    _, history = GeneticAlgorithm().run(x, target, config, seed=99)

    assert history.stop_reason == "stagnation"
    # Zakończono wcześniej niż po max_generations.
    assert history.num_generations < config.generations + 1
    # Ostatnie pokolenia mają niemalejące, ostatecznie stałe best_fitness.
    best = history.best_fitness
    assert best[-1] >= best[0] - 1e-12


def test_zero_generations_records_only_initial_population() -> None:
    x = _sample_repr()
    target = _target_for(Genome(0.0, 1.0, 1.0, 0.0))
    config = _small_config(generations=0)

    best_genome, history = GeneticAlgorithm().run(x, target, config, seed=0)
    assert history.num_generations == 1  # tylko pokolenie 0
    assert isinstance(best_genome, Genome)
    assert history.stop_reason == "max_generations"


# --------------------------------------------------------------------------- #
# Historia per pokolenie (Wymaganie 4.6)
# --------------------------------------------------------------------------- #
def test_history_lengths_are_consistent() -> None:
    x = _sample_repr()
    target = _target_for(Genome(3.0, 1.0, 1.0, 0.0))
    config = _small_config(generations=5, stagnation_generations=1000)

    _, history = GeneticAlgorithm().run(x, target, config, seed=8)

    n = history.num_generations
    assert n == 6
    assert len(history.best_fitness) == n
    assert len(history.mean_fitness) == n
    assert len(history.worst_fitness) == n
    assert len(history.best_genome) == n


def test_history_best_ge_mean_ge_worst_each_generation() -> None:
    x = _sample_repr()
    target = _target_for(Genome(4.0, 1.2, 0.8, 8.0))
    config = _small_config(generations=8, stagnation_generations=1000)

    _, history = GeneticAlgorithm().run(x, target, config, seed=13)

    for best, mean, worst in zip(
        history.best_fitness, history.mean_fitness, history.worst_fitness
    ):
        assert best >= mean - 1e-12
        assert mean >= worst - 1e-12


def test_history_best_fitness_matches_best_genome() -> None:
    # best_genome[g] musi mieć dopasowanie równe best_fitness[g].
    x = _sample_repr()
    target = _target_for(Genome(2.0, 1.1, 0.9, 4.0))
    config = _small_config(generations=6, stagnation_generations=1000)

    from musicians_style.ga import fitness

    _, history = GeneticAlgorithm().run(x, target, config, seed=21)
    extractor = FeatureExtractor()
    for genome, best in zip(history.best_genome, history.best_fitness):
        recomputed = fitness(genome, x, target, "euclidean", extractor=extractor)
        assert recomputed == best  # deterministyczne, bit-identyczne


def test_ga_jsonl_log_is_written(tmp_path: Path) -> None:
    x = _sample_repr()
    target = _target_for(Genome(5.0, 1.0, 1.0, 0.0))
    config = _small_config(generations=4, stagnation_generations=1000)

    log_path = tmp_path / "logs" / "ga.jsonl"
    _, history = GeneticAlgorithm().run(
        x, target, config, seed=1, log_path=log_path
    )

    assert log_path.exists()
    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    # Jedna linia na każde zarejestrowane pokolenie.
    assert len(lines) == history.num_generations

    for idx, line in enumerate(lines):
        record = json.loads(line)
        # Pola wymagane formatem logu (design.md).
        for field in ("ts", "level", "component", "msg"):
            assert field in record
        assert record["component"] == "ga"
        assert record["generation"] == idx
        assert record["best_fitness"] == history.best_fitness[idx]
        assert set(record["best_genome"]) == set(GENE_NAMES)


def test_no_log_file_created_when_log_path_none(tmp_path: Path) -> None:
    x = _sample_repr()
    target = _target_for(Genome(0.0, 1.0, 1.0, 0.0))
    config = _small_config(generations=3)

    GeneticAlgorithm().run(x, target, config, seed=0, log_path=None)
    # Brak ścieżki → brak plików w katalogu tymczasowym.
    assert list(tmp_path.iterdir()) == []


# --------------------------------------------------------------------------- #
# Zakresy robocze populacji
# --------------------------------------------------------------------------- #
def test_initial_population_within_working_ranges() -> None:
    ga = GeneticAlgorithm()
    rng = np.random.default_rng(0)
    population = ga._init_population(200, rng)  # noqa: SLF001 - test białoskrzynkowy
    for genome in population:
        for name, value in zip(GENE_NAMES, _genes(genome)):
            low, high = WORKING_RANGES[name]
            assert low <= value <= high


def test_best_genome_stays_within_working_ranges() -> None:
    x = _sample_repr()
    target = _target_for(Genome(12.0, 2.0, 0.5, 32.0))
    config = _small_config(generations=12, stagnation_generations=1000)

    best_genome, history = GeneticAlgorithm().run(x, target, config, seed=77)

    for genome in (*history.best_genome, best_genome):
        for name, value in zip(GENE_NAMES, _genes(genome)):
            low, high = WORKING_RANGES[name]
            assert low - 1e-9 <= value <= high + 1e-9


# --------------------------------------------------------------------------- #
# Walidacja konfiguracji i liczność populacji
# --------------------------------------------------------------------------- #
def test_population_size_is_preserved_each_generation() -> None:
    x = _sample_repr()
    target = _target_for(Genome(1.0, 1.0, 1.0, 0.0))
    config = _small_config(population_size=9, generations=5, elitism_k=2)

    # Białoskrzynkowo sprawdzamy zachowanie liczności w _next_generation.
    ga = GeneticAlgorithm()
    extractor = FeatureExtractor()
    rng = np.random.default_rng(0)
    population = ga._init_population(config.population_size, rng)  # noqa: SLF001
    fitnesses = ga._evaluate_all(  # noqa: SLF001
        population, x, target, config.fitness_metric, extractor
    )
    for _ in range(config.generations):
        population, fitnesses = ga._next_generation(  # noqa: SLF001
            population, fitnesses, x, target, config.fitness_metric,
            extractor, config, rng,
        )
        assert len(population) == config.population_size
        assert len(fitnesses) == config.population_size


def test_invalid_config_raises() -> None:
    x = _sample_repr()
    target = _target_for(Genome(0.0, 1.0, 1.0, 0.0))
    ga = GeneticAlgorithm()

    import pytest

    with pytest.raises(ValueError):
        ga.run(x, target, _small_config(population_size=0), seed=0)
    with pytest.raises(ValueError):
        ga.run(x, target, _small_config(tournament_size=0), seed=0)
    with pytest.raises(ValueError):
        ga.run(x, target, _small_config(elitism_k=-1), seed=0)
    with pytest.raises(ValueError):
        ga.run(x, target, _small_config(crossover="manhattan"), seed=0)
    with pytest.raises(ValueError):
        ga.run(x, target, _small_config(generations=-1), seed=0)


def test_single_point_crossover_run_is_deterministic() -> None:
    x = _sample_repr()
    target = _target_for(Genome(3.0, 1.0, 1.0, 0.0))
    config = _small_config(crossover="single_point")

    best_a, hist_a = GeneticAlgorithm().run(x, target, config, seed=55)
    best_b, hist_b = GeneticAlgorithm().run(x, target, config, seed=55)
    assert best_a == best_b
    assert hist_a == hist_b


def test_elitism_k_exceeding_population_is_clamped() -> None:
    x = _sample_repr()
    target = _target_for(Genome(2.0, 1.0, 1.0, 0.0))
    # elitism_k > population_size: cała populacja staje się elitą; przebieg
    # kończy się poprawnie (brak potomków, populacja stała).
    config = _small_config(population_size=5, elitism_k=10, generations=4)

    best_genome, history = GeneticAlgorithm().run(x, target, config, seed=0)
    assert isinstance(best_genome, Genome)
    assert history.num_generations == config.generations + 1
    # Cała populacja przeniesiona bez zmian → best_fitness stałe.
    assert all(
        abs(b - history.best_fitness[0]) < 1e-12 for b in history.best_fitness
    )


def test_mahalanobis_metric_run_succeeds_and_is_deterministic() -> None:
    x = _sample_repr()
    target = _target_for(Genome(4.0, 1.0, 1.0, 0.0))
    config = _small_config(fitness_metric="mahalanobis")

    best_a, hist_a = GeneticAlgorithm().run(x, target, config, seed=31)
    best_b, hist_b = GeneticAlgorithm().run(x, target, config, seed=31)
    assert best_a == best_b
    assert hist_a == hist_b
    assert isinstance(hist_a, History)


def test_history_is_immutable() -> None:
    x = _sample_repr()
    target = _target_for(Genome(0.0, 1.0, 1.0, 0.0))
    _, history = GeneticAlgorithm().run(x, target, _small_config(generations=2), seed=0)

    import pytest
    from dataclasses import FrozenInstanceError

    with pytest.raises(FrozenInstanceError):
        history.best_fitness = ()  # type: ignore[misc]

    # replace tworzy nową instancję bez modyfikacji oryginału.
    clone = replace(history)
    assert clone == history
