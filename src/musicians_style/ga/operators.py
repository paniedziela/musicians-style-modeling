"""Operatory genetyczne *Algorytmu_Genetycznego* (Wymaganie 4.2).

Moduł implementuje operatory ewolucyjne działające na genotypie
:class:`~musicians_style.ga.types.Genome` (sekcja *Algorytm_Genetyczny* w
``design.md``):

* :func:`tournament_select` - **selekcja turniejowa** o rozmiarze ``k``,
* :func:`single_point_crossover` - **krzyżowanie jednopunktowe**,
* :func:`uniform_crossover` - **krzyżowanie jednorodne** typu BLX-α
  (mieszanie parametrów rzeczywistych z α = 0.5),
* :func:`gaussian_mutate` - **mutacja gaussowska** parametrów rzeczywistych.

Determinizm i brak globalnego RNG (Wymagania 4.4, 11.4)
=======================================================

Wszystkie operatory przyjmują :class:`numpy.random.Generator` jako jawny
parametr ``rng`` i korzystają **wyłącznie** z niego jako źródła losowości.
Żaden operator nie sięga po globalny stan ``numpy.random`` ani moduł ``random``.
Dzięki temu *Algorytm_Genetyczny* (zadanie 6.4) jest w pełni powtarzalny dla
ustalonego *Seed* - dwa przebiegi z generatorem zainicjalizowanym tym samym
ziarnem dają identyczne wyniki (Property 7, Wymaganie 11.4, zadanie 6.6).

Czystość operatorów
===================

Operatory **nie modyfikują** swoich argumentów. :class:`Genome` jest
niemutowalny (``frozen=True``), a operatory krzyżowania i mutacji zwracają
**nowe** instancje :class:`Genome`. :func:`tournament_select` zwraca istniejącą
(niemutowalną) instancję zwycięzcy turnieju, co jest bezpieczne ze względu na
niemutowalność genotypu.

Konwencja zwracanych wartości
=============================

Operatory krzyżowania (:func:`single_point_crossover`, :func:`uniform_crossover`)
zwracają **krotkę dwóch potomków** ``(child1, child2)`` - spójnie pomiędzy
oboma operatorami, co upraszcza budowę kolejnego pokolenia w pętli ewolucji.

Parametry rzeczywiste pozostają **nieograniczone** (mogą być ujemne, Wymaganie
4.1). Operatory celowo nie nakładają saturacji na zakresy robocze - ewentualne
przycięcie do dopuszczalnych przedziałów (np. ``[-12, +12]`` dla transpozycji)
realizowane jest na poziomie pętli *Algorytmu_Genetycznego* (zadanie 6.4),
natomiast saturacja wartości MIDI odbywa się dopiero przy aplikacji genotypu w
:func:`~musicians_style.ga.transformation.apply_transformation`.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from .types import Genome

__all__ = [
    "GENE_NAMES",
    "BLX_ALPHA",
    "tournament_select",
    "single_point_crossover",
    "uniform_crossover",
    "gaussian_mutate",
]

#: Kanoniczna kolejność parametrów (genów) genotypu :class:`Genome`. Spójna z
#: kolejnością pól dataclass oraz kluczami ``ga.mutation_sigma`` w konfiguracji
#: (:class:`~musicians_style.config.GAConfig`).
GENE_NAMES: tuple[str, ...] = (
    "transpose_semitones",
    "rhythm_density_factor",
    "note_duration_factor",
    "velocity_offset",
)

#: Współczynnik α krzyżowania jednorodnego BLX-α (Wymaganie 4.2, ``design.md``).
BLX_ALPHA: float = 0.5


def _genes(genome: Genome) -> tuple[float, float, float, float]:
    """Zwraca parametry ``genome`` w kanonicznej kolejności :data:`GENE_NAMES`."""
    return (
        genome.transpose_semitones,
        genome.rhythm_density_factor,
        genome.note_duration_factor,
        genome.velocity_offset,
    )


def _from_genes(values: Sequence[float]) -> Genome:
    """Buduje :class:`Genome` z sekwencji czterech genów (rzutowanych na float)."""
    return Genome(
        transpose_semitones=float(values[0]),
        rhythm_density_factor=float(values[1]),
        note_duration_factor=float(values[2]),
        velocity_offset=float(values[3]),
    )


def tournament_select(
    population: Sequence[Genome],
    fitnesses: Sequence[float],
    k: int,
    rng: np.random.Generator,
) -> Genome:
    """Selekcja turniejowa o rozmiarze ``k`` (Wymaganie 4.2).

    Losuje ``k`` osobników z populacji (indeksy pobierane z ``rng``, **ze
    zwracaniem**) i zwraca tego o najwyższym dopasowaniu. Losowanie ze
    zwracaniem jest klasycznym wariantem selekcji turniejowej i działa poprawnie
    również, gdy ``k`` przekracza liczność populacji.

    Przy remisie dopasowań zwracany jest osobnik o najniższym indeksie wśród
    wylosowanych (``argmax`` wskazuje pierwsze maksimum) - zachowuje to
    determinizm wyniku dla ustalonego stanu ``rng``.

    Args:
        population: populacja osobników (niepusta sekwencja :class:`Genome`).
        fitnesses: wartości *Funkcji_Dopasowania* wyrównane indeksami z
            ``population`` (wyższa wartość = lepsze dopasowanie).
        k: rozmiar turnieju (liczba losowanych osobników), ``>= 1``.
        rng: generator liczb pseudolosowych - jedyne źródło losowości.

    Returns:
        Zwycięski :class:`Genome` (istniejąca, niemutowalna instancja z
        ``population``).

    Raises:
        ValueError: gdy ``population`` jest pusta, długości ``population`` i
            ``fitnesses`` są różne, lub ``k < 1``.
    """
    n = len(population)
    if n == 0:
        raise ValueError("Selekcja turniejowa wymaga niepustej populacji.")
    if len(fitnesses) != n:
        raise ValueError(
            "Długość 'fitnesses' musi być równa liczności populacji "
            f"(otrzymano {len(fitnesses)} dla populacji o rozmiarze {n})."
        )
    if k < 1:
        raise ValueError(f"Rozmiar turnieju 'k' musi być >= 1, otrzymano {k}.")

    # Losowanie ze zwracaniem - indeksy uczestników turnieju.
    indices = rng.integers(0, n, size=k)
    contender_fitnesses = np.asarray(
        [fitnesses[int(i)] for i in indices], dtype=np.float64
    )
    winner_index = int(indices[int(np.argmax(contender_fitnesses))])
    return population[winner_index]


def single_point_crossover(
    p1: Genome,
    p2: Genome,
    rng: np.random.Generator,
) -> tuple[Genome, Genome]:
    """Krzyżowanie jednopunktowe dwóch genotypów (Wymaganie 4.2).

    Dla czteroparametrowego genotypu losowany jest punkt cięcia ``point`` ze
    zbioru ``{1, 2, 3}`` (pomiędzy genami) i wymieniane są „ogony" rodziców:

    * ``child1 = p1[:point] + p2[point:]``,
    * ``child2 = p2[:point] + p1[point:]``.

    Operator nie modyfikuje rodziców i zwraca dwie nowe instancje
    :class:`Genome`.

    Args:
        p1: pierwszy rodzic.
        p2: drugi rodzic.
        rng: generator liczb pseudolosowych - jedyne źródło losowości.

    Returns:
        Krotka ``(child1, child2)`` dwóch nowych potomków.
    """
    g1 = _genes(p1)
    g2 = _genes(p2)

    # Punkt cięcia w {1, 2, 3} - co najmniej jeden gen z każdego rodzica.
    point = int(rng.integers(1, len(GENE_NAMES)))

    child1 = g1[:point] + g2[point:]
    child2 = g2[:point] + g1[point:]
    return _from_genes(child1), _from_genes(child2)


def uniform_crossover(
    p1: Genome,
    p2: Genome,
    rng: np.random.Generator,
) -> tuple[Genome, Genome]:
    """Krzyżowanie jednorodne BLX-α dla parametrów rzeczywistych (Wymaganie 4.2).

    Realizuje mieszanie BLX-α (*blend crossover*) z α = :data:`BLX_ALPHA`
    (0.5). Dla każdego genu osobno wyznaczany jest przedział mieszania:

    * ``c_min = min(g1, g2)``, ``c_max = max(g1, g2)``, ``d = c_max - c_min``,
    * gen potomka losowany jednostajnie z ``[c_min - α·d, c_max + α·d]``.

    Każdy z dwóch potomków otrzymuje **niezależnie** wylosowane geny (mieszanie
    realizowane per-gen), co jest typowym wariantem BLX-α. Gdy geny rodziców są
    równe (``d = 0``), przedział degeneruje się do punktu i gen potomka jest
    równy genowi rodziców.

    Args:
        p1: pierwszy rodzic.
        p2: drugi rodzic.
        rng: generator liczb pseudolosowych - jedyne źródło losowości.

    Returns:
        Krotka ``(child1, child2)`` dwóch nowych potomków o genach
        rzeczywistych (potencjalnie ujemnych, Wymaganie 4.1).
    """
    g1 = _genes(p1)
    g2 = _genes(p2)

    child1: list[float] = []
    child2: list[float] = []
    for gene1, gene2 in zip(g1, g2):
        c_min = min(gene1, gene2)
        c_max = max(gene1, gene2)
        spread = (c_max - c_min) * BLX_ALPHA
        low = c_min - spread
        high = c_max + spread
        # Dwa niezależne losowania - po jednym genie dla każdego potomka.
        child1.append(float(rng.uniform(low, high)))
        child2.append(float(rng.uniform(low, high)))

    return _from_genes(child1), _from_genes(child2)


def gaussian_mutate(
    genome: Genome,
    sigma_per_param: Mapping[str, float],
    rng: np.random.Generator,
) -> Genome:
    """Mutacja gaussowska parametrów rzeczywistych (Wymaganie 4.2).

    Do każdego genu dodawany jest niezależny szum gaussowski:
    ``gene' = gene + N(0, σ_gene)``, gdzie odchylenie ``σ_gene`` pochodzi z
    ``sigma_per_param`` (klucz = nazwa parametru). Próbki losowane są wyłącznie
    z przekazanego ``rng``.

    Mapowanie ``sigma_per_param`` odpowiada strukturze
    :attr:`~musicians_style.config.GAConfig.mutation_sigma` i jest kluczowane
    nazwami parametrów z :data:`GENE_NAMES` (``transpose_semitones``,
    ``rhythm_density_factor``, ``note_duration_factor``, ``velocity_offset``).
    Brak klucza dla danego parametru jest interpretowany jako ``σ = 0`` (gen nie
    podlega mutacji). Wartość ``σ = 0`` daje zerowy przyrost (gen niezmieniony).

    Operator nie modyfikuje wejścia i zwraca nową instancję :class:`Genome`.

    Args:
        genome: mutowany genotyp.
        sigma_per_param: odchylenia standardowe szumu per parametr (nieujemne).
        rng: generator liczb pseudolosowych - jedyne źródło losowości.

    Returns:
        Nowy, zmutowany :class:`Genome`.

    Raises:
        ValueError: gdy którekolwiek odchylenie ``σ`` jest ujemne.
    """
    mutated: list[float] = []
    for name, value in zip(GENE_NAMES, _genes(genome)):
        sigma = float(sigma_per_param.get(name, 0.0))
        if sigma < 0.0:
            raise ValueError(
                f"Odchylenie standardowe mutacji dla parametru {name!r} musi być "
                f"nieujemne, otrzymano {sigma}."
            )
        if sigma == 0.0:
            mutated.append(value)
        else:
            mutated.append(value + float(rng.normal(0.0, sigma)))

    return _from_genes(mutated)
