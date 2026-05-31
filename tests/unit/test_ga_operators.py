"""Testy jednostkowe operatorów genetycznych *Algorytmu_Genetycznego* (zadanie 6.3).

Weryfikują kontrakt operatorów z
:mod:`musicians_style.ga.operators` (sekcja *Algorytm_Genetyczny* w ``design.md``,
Wymaganie 4.2):

* :func:`tournament_select` - selekcja turniejowa zwracająca istniejący,
  najlepiej dopasowany genotyp,
* :func:`single_point_crossover` - krzyżowanie jednopunktowe,
* :func:`uniform_crossover` - krzyżowanie jednorodne BLX-α (α = 0.5),
* :func:`gaussian_mutate` - mutacja gaussowska parametrów rzeczywistych.

Sprawdzane niezmienniki:

* operatory **nie modyfikują** wejścia (czystość, :class:`Genome` jest
  niemutowalny) i zwracają **nowe** instancje :class:`Genome`,
* wszystkie operatory są **deterministyczne** względem przekazanego
  :class:`numpy.random.Generator` (ten sam *Seed* → identyczny wynik),
* korzystają **wyłącznie** z przekazanego ``rng`` (brak globalnego RNG),
* krzyżowania zachowują liczność populacji (2 rodziców → 2 potomków),
* potomki BLX-α mieszczą się w przedziale mieszania
  ``[c_min - α·d, c_max + α·d]``.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from musicians_style.ga import (
    BLX_ALPHA,
    GENE_NAMES,
    Genome,
    gaussian_mutate,
    single_point_crossover,
    tournament_select,
    uniform_crossover,
)


def _genes(g: Genome) -> tuple[float, float, float, float]:
    return (
        g.transpose_semitones,
        g.rhythm_density_factor,
        g.note_duration_factor,
        g.velocity_offset,
    )


def _sample_population() -> list[Genome]:
    return [
        Genome(0.0, 1.0, 1.0, 0.0),
        Genome(2.0, 1.5, 0.8, 10.0),
        Genome(-3.0, 0.5, 1.2, -5.0),
        Genome(5.0, 0.9, 1.1, 20.0),
    ]


# -- tournament_select -------------------------------------------------------


def test_tournament_select_returns_member_of_population() -> None:
    pop = _sample_population()
    fitnesses = [-1.0, -0.5, -2.0, -0.1]
    rng = np.random.default_rng(0)
    winner = tournament_select(pop, fitnesses, k=3, rng=rng)
    assert winner in pop


def test_tournament_select_picks_best_with_large_tournament() -> None:
    # Dla dwuelementowej populacji i bardzo dużego turnieju (losowanie ze
    # zwracaniem) prawdopodobieństwo pominięcia najlepszego osobnika wynosi
    # (1/2)^k - dla k=40 jest pomijalne, więc wynik jest praktycznie pewny.
    pop = [Genome(0.0, 1.0, 1.0, 0.0), Genome(2.0, 1.5, 0.8, 10.0)]
    fitnesses = [-2.0, -0.1]  # najlepszy = indeks 1
    best = pop[1]
    rng = np.random.default_rng(123)
    results = [
        tournament_select(pop, fitnesses, k=40, rng=rng) for _ in range(20)
    ]
    assert all(r == best for r in results)


def test_tournament_select_k1_returns_drawn_individual() -> None:
    # Turniej o rozmiarze 1 zwraca po prostu wylosowanego osobnika - wynik jest
    # zdeterminowany stanem rng (niezależnie od dopasowań).
    pop = _sample_population()
    fitnesses = [-1.0, -0.5, -2.0, -0.1]
    rng = np.random.default_rng(2024)
    idx = int(np.random.default_rng(2024).integers(0, len(pop), size=1)[0])
    assert tournament_select(pop, fitnesses, k=1, rng=rng) == pop[idx]


def test_tournament_select_is_deterministic_for_same_seed() -> None:
    pop = _sample_population()
    fitnesses = [-1.0, -0.5, -2.0, -0.1]
    a = tournament_select(pop, fitnesses, k=2, rng=np.random.default_rng(7))
    b = tournament_select(pop, fitnesses, k=2, rng=np.random.default_rng(7))
    assert a == b


def test_tournament_select_validates_arguments() -> None:
    pop = _sample_population()
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        tournament_select([], [], k=2, rng=rng)
    with pytest.raises(ValueError):
        tournament_select(pop, [0.0, 1.0], k=2, rng=rng)  # niezgodne długości
    with pytest.raises(ValueError):
        tournament_select(pop, [0.0] * 4, k=0, rng=rng)  # k < 1


# -- single_point_crossover --------------------------------------------------


def test_single_point_crossover_returns_two_new_genomes() -> None:
    p1 = Genome(0.0, 1.0, 1.0, 0.0)
    p2 = Genome(2.0, 2.0, 2.0, 2.0)
    rng = np.random.default_rng(0)
    c1, c2 = single_point_crossover(p1, p2, rng)
    assert isinstance(c1, Genome) and isinstance(c2, Genome)
    assert c1 is not p1 and c1 is not p2
    assert c2 is not p1 and c2 is not p2


def test_single_point_crossover_preserves_genes_as_partition() -> None:
    # Każdy gen potomka pochodzi od jednego z rodziców (brak nowych wartości).
    p1 = Genome(0.0, 0.0, 0.0, 0.0)
    p2 = Genome(1.0, 1.0, 1.0, 1.0)
    rng = np.random.default_rng(3)
    c1, c2 = single_point_crossover(p1, p2, rng)
    for value in _genes(c1) + _genes(c2):
        assert value in (0.0, 1.0)
    # Suma genów obu potomków zachowuje sumę genów rodziców (wymiana ogonów).
    assert sum(_genes(c1)) + sum(_genes(c2)) == sum(_genes(p1)) + sum(_genes(p2))


def test_single_point_crossover_is_deterministic_for_same_seed() -> None:
    p1 = Genome(0.0, 1.0, 1.0, 0.0)
    p2 = Genome(2.0, 2.0, 2.0, 2.0)
    r1 = single_point_crossover(p1, p2, np.random.default_rng(42))
    r2 = single_point_crossover(p1, p2, np.random.default_rng(42))
    assert r1 == r2


def test_single_point_crossover_does_not_mutate_parents() -> None:
    p1 = Genome(0.0, 1.0, 1.0, 0.0)
    p2 = Genome(2.0, 2.0, 2.0, 2.0)
    snap1, snap2 = _genes(p1), _genes(p2)
    single_point_crossover(p1, p2, np.random.default_rng(1))
    assert _genes(p1) == snap1 and _genes(p2) == snap2


# -- uniform_crossover (BLX-α) -----------------------------------------------


def test_uniform_crossover_returns_two_new_genomes() -> None:
    p1 = Genome(0.0, 1.0, 1.0, 0.0)
    p2 = Genome(4.0, 2.0, 0.5, 10.0)
    c1, c2 = uniform_crossover(p1, p2, np.random.default_rng(0))
    assert isinstance(c1, Genome) and isinstance(c2, Genome)


def test_uniform_crossover_children_within_blx_interval() -> None:
    p1 = Genome(0.0, 1.0, 1.0, 0.0)
    p2 = Genome(4.0, 2.0, 0.5, 10.0)
    rng = np.random.default_rng(5)
    for _ in range(50):
        c1, c2 = uniform_crossover(p1, p2, rng)
        for child in (c1, c2):
            for g1, g2, gene in zip(_genes(p1), _genes(p2), _genes(child)):
                c_min, c_max = min(g1, g2), max(g1, g2)
                spread = (c_max - c_min) * BLX_ALPHA
                assert c_min - spread - 1e-9 <= gene <= c_max + spread + 1e-9


def test_uniform_crossover_equal_parents_yield_same_gene() -> None:
    # Gdy geny rodziców są równe (d = 0), przedział degeneruje się do punktu.
    p = Genome(3.0, 1.0, 1.0, -2.0)
    c1, c2 = uniform_crossover(p, p, np.random.default_rng(9))
    assert _genes(c1) == _genes(p)
    assert _genes(c2) == _genes(p)


def test_uniform_crossover_is_deterministic_for_same_seed() -> None:
    p1 = Genome(0.0, 1.0, 1.0, 0.0)
    p2 = Genome(4.0, 2.0, 0.5, 10.0)
    r1 = uniform_crossover(p1, p2, np.random.default_rng(11))
    r2 = uniform_crossover(p1, p2, np.random.default_rng(11))
    assert r1 == r2


# -- gaussian_mutate ---------------------------------------------------------


def test_gaussian_mutate_returns_new_genome() -> None:
    g = Genome(0.0, 1.0, 1.0, 0.0)
    sigma = {name: 1.0 for name in GENE_NAMES}
    mutated = gaussian_mutate(g, sigma, np.random.default_rng(0))
    assert isinstance(mutated, Genome)
    assert mutated is not g


def test_gaussian_mutate_zero_sigma_is_identity() -> None:
    g = Genome(2.0, 1.5, 0.8, 10.0)
    sigma = {name: 0.0 for name in GENE_NAMES}
    mutated = gaussian_mutate(g, sigma, np.random.default_rng(0))
    assert mutated == g


def test_gaussian_mutate_missing_key_is_unchanged_gene() -> None:
    g = Genome(2.0, 1.5, 0.8, 10.0)
    # Tylko transpozycja mutowana; pozostałe geny bez klucza → bez zmian.
    sigma = {"transpose_semitones": 1.0}
    mutated = gaussian_mutate(g, sigma, np.random.default_rng(0))
    assert mutated.rhythm_density_factor == g.rhythm_density_factor
    assert mutated.note_duration_factor == g.note_duration_factor
    assert mutated.velocity_offset == g.velocity_offset
    assert mutated.transpose_semitones != g.transpose_semitones


def test_gaussian_mutate_is_deterministic_for_same_seed() -> None:
    g = Genome(0.0, 1.0, 1.0, 0.0)
    sigma = {name: 1.0 for name in GENE_NAMES}
    a = gaussian_mutate(g, sigma, np.random.default_rng(99))
    b = gaussian_mutate(g, sigma, np.random.default_rng(99))
    assert a == b


def test_gaussian_mutate_rejects_negative_sigma() -> None:
    g = Genome(0.0, 1.0, 1.0, 0.0)
    sigma = {"transpose_semitones": -1.0}
    with pytest.raises(ValueError):
        gaussian_mutate(g, sigma, np.random.default_rng(0))


def test_gaussian_mutate_does_not_mutate_input() -> None:
    g = Genome(0.0, 1.0, 1.0, 0.0)
    snapshot = _genes(g)
    gaussian_mutate(g, {name: 2.0 for name in GENE_NAMES}, np.random.default_rng(0))
    assert _genes(g) == snapshot


def test_genome_remains_immutable() -> None:
    g = Genome(0.0, 1.0, 1.0, 0.0)
    with pytest.raises(FrozenInstanceError):
        g.transpose_semitones = 1.0  # type: ignore[misc]
