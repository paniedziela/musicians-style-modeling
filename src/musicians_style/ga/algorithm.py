"""Główna pętla *Algorytmu_Genetycznego* - :class:`GeneticAlgorithm` (Wymaganie 4.4-4.7).

Moduł implementuje pełny przebieg ewolucyjny *Algorytmu_Genetycznego* (sekcja
*Algorytm_Genetyczny* w ``design.md``), spinający wcześniej zdefiniowane
komponenty pakietu :mod:`musicians_style.ga`:

* genotyp :class:`~musicians_style.ga.types.Genome` (Wymaganie 4.1),
* operatory genetyczne (Wymaganie 4.2): selekcja turniejowa
  (:func:`~musicians_style.ga.operators.tournament_select`), krzyżowanie
  jednopunktowe / jednorodne BLX-α
  (:func:`~musicians_style.ga.operators.single_point_crossover`,
  :func:`~musicians_style.ga.operators.uniform_crossover`) oraz mutacja
  gaussowska (:func:`~musicians_style.ga.operators.gaussian_mutate`),
* *Funkcja_Dopasowania* :func:`~musicians_style.ga.fitness.fitness`
  (Wymaganie 4.3).

Przebieg ewolucji
=================

1. **Inicjalizacja** populacji o liczności ``config.population_size`` przez
   losowanie genotypów w **zakresach roboczych** (patrz
   :data:`WORKING_RANGES`) z jedynego źródła losowości
   :class:`numpy.random.Generator` zainicjalizowanego *Seedem*.
2. **Ewaluacja** każdego osobnika *Funkcją_Dopasowania*.
3. **Pętla pokoleń** (Wymaganie 4.4): w każdym pokoleniu kolejno
   *selekcja → krzyżowanie → mutacja → ewaluacja → elitaryzm*. ``elitism_k``
   najlepszych osobników przechodzi **bezpośrednio** do następnego pokolenia
   (Wymaganie 4.7), pozostałe ``population_size - elitism_k`` miejsc wypełniają
   potomkowie powstali z selekcji turniejowej, krzyżowania i mutacji rodziców.

Warunki stopu (Wymaganie 4.5)
=============================

* osiągnięcie ``config.generations`` pokoleń (``max_generations``),
* **stagnacja**: brak poprawy najlepszego dopasowania przez
  ``config.stagnation_generations`` kolejnych pokoleń.

Determinizm (Wymaganie 4.4, Property 7)
=======================================

Jedynym źródłem losowości jest :class:`numpy.random.Generator` utworzony z
*Seeda* (``numpy.random.default_rng(seed)``) i przekazywany do **wszystkich**
operatorów. Nie używa się globalnego stanu ``numpy.random`` ani modułu
``random``. *Funkcja_Dopasowania* i :func:`~musicians_style.ga.transformation.
apply_transformation` są czyste i deterministyczne. Dlatego dwa przebiegi z tym
samym *Seedem*, identycznym wejściem i konfiguracją dają **bit-identyczną**
populację końcową oraz historię dopasowania (zadanie 6.6).

Monotoniczność elitaryzmu (Wymaganie 4.7, Property 8)
=====================================================

Przy ``elitism_k >= 1`` najlepszy osobnik pokolenia ``n`` jest zachowywany
**bez zmian** w pokoleniu ``n+1`` (wraz ze swoją - deterministycznie
odtwarzalną - wartością dopasowania), więc najlepsze dopasowanie jest
**niemalejące** w kolejnych pokoleniach (zadanie 6.7).

Logowanie (Wymaganie 4.6)
=========================

W każdym pokoleniu zapisywana jest krotka ``(best_fitness, mean_fitness,
worst_fitness, best_genome)``. Statystyki te trafiają zarówno do zwracanej
:class:`History`, jak i - opcjonalnie - do pliku ``ga.jsonl`` (format JSON-lines
zgodny z sekcją *Format wpisu logu* w ``design.md``). Ścieżka pliku jest
**wstrzykiwana** parametrem ``log_path`` metody :meth:`GeneticAlgorithm.run`;
gdy jest ``None``, żaden plik nie jest tworzony (brak efektów ubocznych na
systemie plików - ułatwia testowanie).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import numpy as np

from ..config import GAConfig
from ..evaluation.distance import prepare_mahalanobis
from ..features.extractor import FeatureExtractor
from ..features.types import AggregatedFeatures
from ..logging import get_logger
from ..midi.types import InternalRepr
from .fitness import fitness
from .operators import (
    GENE_NAMES,
    gaussian_mutate,
    single_point_crossover,
    tournament_select,
    uniform_crossover,
)
from .types import Genome

__all__ = ["WORKING_RANGES", "History", "GeneticAlgorithm"]

#: Zakresy robocze parametrów genotypu (sekcja *Algorytm_Genetyczny* w
#: ``design.md``). Inicjalizacja populacji losuje każdy gen **jednostajnie** w
#: tych granicach, a potomkowie powstali z krzyżowania/mutacji są do nich
#: **przycinani** (saturacja), dzięki czemu przeszukiwanie pozostaje w
#: muzycznie sensownym, interpretowalnym obszarze. Operatory genetyczne celowo
#: nie nakładają tych ograniczeń (zob. docstring :mod:`musicians_style.ga.
#: operators`) - egzekwowane są dopiero tutaj, w pętli *Algorytmu_Genetycznego*.
#:
#: * ``transpose_semitones`` - ``[-12, +12]`` (oktawa w górę/dół),
#: * ``rhythm_density_factor`` - ``[0.5, 2.0]`` (1.0 = bez zmian),
#: * ``note_duration_factor`` - ``[0.5, 2.0]`` (1.0 = bez zmian),
#: * ``velocity_offset`` - ``[-32, +32]``.
WORKING_RANGES: dict[str, tuple[float, float]] = {
    "transpose_semitones": (-12.0, 12.0),
    "rhythm_density_factor": (0.5, 2.0),
    "note_duration_factor": (0.5, 2.0),
    "velocity_offset": (-32.0, 32.0),
}

#: Tolerancja uznania poprawy najlepszego dopasowania (warunek stagnacji,
#: Wymaganie 4.5). Poprawa musi przekroczyć ten próg, aby wyzerować licznik
#: stagnacji. Przy elitaryzmie brak poprawy oznacza dokładną równość wartości
#: (najlepszy osobnik jest przenoszony bez zmian), więc dowolny mały dodatni
#: próg działa poprawnie.
_IMPROVEMENT_TOL: float = 1e-12


@dataclass(frozen=True)
class History:
    """Historia przebiegu *Algorytmu_Genetycznego* (Wymaganie 4.6).

    Niemutowalny zapis statystyk *Funkcji_Dopasowania* dla **każdego** pokolenia
    (włącznie z pokoleniem początkowym o indeksie 0). Wszystkie krotki mają tę
    samą długość, równą liczbie zarejestrowanych pokoleń
    (:attr:`num_generations`). Niemutowalność i porównywalność po wartości
    (krotki :class:`Genome` i ``float``) pozwalają asertować bit-identyczność
    dwóch przebiegów w teście determinizmu (Property 7, zadanie 6.6).

    Atrybuty:
        best_fitness: najlepsza (maksymalna) wartość dopasowania w każdym
            pokoleniu. Przy ``elitism_k >= 1`` ciąg jest **niemalejący**
            (Property 8, Wymaganie 4.7).
        mean_fitness: średnia wartość dopasowania w każdym pokoleniu.
        worst_fitness: najgorsza (minimalna) wartość dopasowania w pokoleniu.
        best_genome: najlepszy genotyp w każdym pokoleniu (osobnik o
            ``best_fitness``).
        stop_reason: powód zakończenia - ``"max_generations"`` (osiągnięto
            ``config.generations``) lub ``"stagnation"`` (brak poprawy przez
            ``config.stagnation_generations`` pokoleń).
    """

    best_fitness: tuple[float, ...]
    mean_fitness: tuple[float, ...]
    worst_fitness: tuple[float, ...]
    best_genome: tuple[Genome, ...]
    stop_reason: str

    @property
    def num_generations(self) -> int:
        """Liczba zarejestrowanych pokoleń (z pokoleniem początkowym włącznie)."""
        return len(self.best_fitness)


class GeneticAlgorithm:
    """*Algorytm_Genetyczny* transferu stylu - pełna pętla ewolucji (Wymaganie 4.4-4.7).

    Klasa spina operatory genetyczne i *Funkcję_Dopasowania* w deterministyczny,
    powtarzalny proces optymalizacji genotypu :class:`Genome` minimalizującego
    odległość *Wektora_Cech* przekształconego *Utworu_Wejściowego* od
    agregowanego *Wektora_Cech* *Zbioru_Stylu* (Wymaganie 4.3).

    Instancja jest bezstanowa względem pojedynczego przebiegu - cały stan
    (populacja, generator losowy, historia) jest lokalny dla wywołania
    :meth:`run`, dzięki czemu tę samą instancję można bezpiecznie wykorzystać
    wielokrotnie.
    """

    def __init__(self, extractor: FeatureExtractor | None = None) -> None:
        """Inicjalizuje *Algorytm_Genetyczny*.

        Args:
            extractor: opcjonalny, współdzielony *Ekstraktor_Cech* przekazywany
                do *Funkcji_Dopasowania* przy każdej ewaluacji (Wymaganie 4.3,
                wydajność). Gdy ``None``, w :meth:`run` tworzona jest jedna
                instancja na przebieg. *Ekstraktor_Cech* jest bezstanowy
                względem danych, więc reużycie nie narusza determinizmu.
        """
        self._extractor = extractor
        self._log = get_logger("ga")

    # ------------------------------------------------------------------ #
    # API publiczne
    # ------------------------------------------------------------------ #
    def run(
        self,
        x_input: InternalRepr,
        style_aggregated: AggregatedFeatures,
        config: GAConfig,
        seed: int,
        *,
        log_path: Path | str | None = None,
        prepared_inverse_covariance: object | None = None,
    ) -> tuple[Genome, History]:
        """Uruchamia pełny przebieg ewolucji i zwraca najlepszy genotyp z historią.

        Args:
            x_input: *Reprezentacja_Wewnętrzna* *Utworu_Wejściowego* poddawanego
                transformacji.
            style_aggregated: agregat statystyczny *Zbioru_Stylu* artysty
                docelowego (cel optymalizacji); wykorzystywane są ``mean`` oraz -
                dla metryki Mahalanobisa - ``covariance`` (Wymaganie 4.3).
            config: parametry *Algorytmu_Genetycznego*
                (:class:`~musicians_style.config.GAConfig`).
            seed: *Seed* inicjalizujący jedyny generator losowy przebiegu
                (Wymaganie 4.4).
            log_path: opcjonalna ścieżka pliku ``ga.jsonl``; gdy podana,
                statystyki każdego pokolenia zapisywane są jako linie JSON
                (Wymaganie 4.6). Gdy ``None``, nie powstają żadne pliki.
            prepared_inverse_covariance: opcjonalna, wcześniej obliczona
                pseudoodwrotność kowariancji profilu celu. Pozwala współdzielić
                cache między wieloma przebiegami bez zmiany wyników; gdy
                ``None`` i metryką jest Mahalanobis, macierz jest liczona raz
                na przebieg.

        Returns:
            Krotka ``(best_genome, history)``: najlepszy znaleziony genotyp w
            całym przebiegu oraz :class:`History` ze statystykami każdego
            pokolenia. Przy ``elitism_k >= 1`` ``best_genome`` jest tożsamy z
            najlepszym osobnikiem ostatniego pokolenia (monotoniczność,
            Wymaganie 4.7); dla ``elitism_k == 0`` zwracany jest najlepszy
            osobnik napotkany kiedykolwiek w przebiegu.

        Raises:
            ValueError: gdy konfiguracja jest niepoprawna (np. niedodatnia
                liczność populacji, nieujemna liczba pokoleń, nieznany operator
                krzyżowania) lub gdy *Funkcja_Dopasowania* odrzuci metrykę
                (zob. :func:`~musicians_style.ga.fitness.fitness`).
        """
        self._validate_config(config)

        # Jedyne źródło losowości całego przebiegu (Wymaganie 4.4, Property 7).
        rng = np.random.default_rng(seed)
        extractor = self._extractor if self._extractor is not None else FeatureExtractor()
        metric = config.fitness_metric
        # The target profile is fixed throughout one run.  Preparing the
        # pseudoinverse once preserves the historical fitness exactly while
        # avoiding an expensive SVD for every genome evaluation.
        if metric == "mahalanobis" and prepared_inverse_covariance is None:
            prepared_inverse_covariance = prepare_mahalanobis(
                style_aggregated.covariance,
                dimension=style_aggregated.mean.as_array().size,
            )

        self._log.info(
            "start Algorytmu_Genetycznego",
            seed=seed,
            population_size=config.population_size,
            generations=config.generations,
            elitism_k=config.elitism_k,
            crossover=config.crossover,
            fitness_metric=metric,
            stagnation_generations=config.stagnation_generations,
        )

        # --- pokolenie 0: inicjalizacja i ewaluacja ---
        population = self._init_population(config.population_size, rng)
        fitnesses = self._evaluate_all(
            population,
            x_input,
            style_aggregated,
            metric,
            extractor,
            prepared_inverse_covariance,
        )

        best_list: list[float] = []
        mean_list: list[float] = []
        worst_list: list[float] = []
        genome_list: list[Genome] = []

        handle = self._open_log(log_path)
        try:
            best, mean, worst, best_genome = self._stats(population, fitnesses)
            self._record(
                best_list, mean_list, worst_list, genome_list,
                best, mean, worst, best_genome,
            )
            self._write_log(handle, 0, best, mean, worst, best_genome)

            overall_best_genome = best_genome
            overall_best_fitness = best
            previous_best = best
            stagnation_counter = 0
            stop_reason = "max_generations"

            # --- pętla pokoleń (Wymaganie 4.4) ---
            for generation in range(1, config.generations + 1):
                population, fitnesses = self._next_generation(
                    population, fitnesses, x_input, style_aggregated,
                    metric, extractor, config, rng, prepared_inverse_covariance,
                )
                best, mean, worst, best_genome = self._stats(population, fitnesses)
                self._record(
                    best_list, mean_list, worst_list, genome_list,
                    best, mean, worst, best_genome,
                )
                self._write_log(handle, generation, best, mean, worst, best_genome)

                # Najlepszy osobnik w całym przebiegu (odporne na elitism_k == 0).
                if best > overall_best_fitness:
                    overall_best_fitness = best
                    overall_best_genome = best_genome

                # Warunek stagnacji (Wymaganie 4.5).
                if best > previous_best + _IMPROVEMENT_TOL:
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1
                previous_best = best

                if (
                    config.stagnation_generations >= 1
                    and stagnation_counter >= config.stagnation_generations
                ):
                    stop_reason = "stagnation"
                    self._log.info(
                        "stop: stagnacja najlepszego dopasowania",
                        generation=generation,
                        stagnation_generations=config.stagnation_generations,
                        best_fitness=best,
                    )
                    break
            else:
                self._log.info(
                    "stop: osiągnięto maksymalną liczbę pokoleń",
                    generations=config.generations,
                    best_fitness=overall_best_fitness,
                )
        finally:
            if handle is not None:
                handle.close()

        history = History(
            best_fitness=tuple(best_list),
            mean_fitness=tuple(mean_list),
            worst_fitness=tuple(worst_list),
            best_genome=tuple(genome_list),
            stop_reason=stop_reason,
        )
        return overall_best_genome, history

    # ------------------------------------------------------------------ #
    # Inicjalizacja i ewaluacja
    # ------------------------------------------------------------------ #
    def _init_population(
        self, size: int, rng: np.random.Generator
    ) -> list[Genome]:
        """Losuje populację początkową w zakresach roboczych (:data:`WORKING_RANGES`).

        Każdy gen losowany jest **jednostajnie** w swoim przedziale roboczym, w
        kanonicznej kolejności :data:`~musicians_style.ga.operators.GENE_NAMES`
        (transpozycja, gęstość, długość nut, dynamika) - ustalona kolejność
        losowań jest istotna dla determinizmu (Property 7).
        """
        t_lo, t_hi = WORKING_RANGES["transpose_semitones"]
        r_lo, r_hi = WORKING_RANGES["rhythm_density_factor"]
        d_lo, d_hi = WORKING_RANGES["note_duration_factor"]
        v_lo, v_hi = WORKING_RANGES["velocity_offset"]

        population: list[Genome] = []
        for _ in range(size):
            population.append(
                Genome(
                    transpose_semitones=float(rng.uniform(t_lo, t_hi)),
                    rhythm_density_factor=float(rng.uniform(r_lo, r_hi)),
                    note_duration_factor=float(rng.uniform(d_lo, d_hi)),
                    velocity_offset=float(rng.uniform(v_lo, v_hi)),
                )
            )
        return population

    def _evaluate_all(
        self,
        population: list[Genome],
        x_input: InternalRepr,
        target: AggregatedFeatures,
        metric: str,
        extractor: FeatureExtractor,
        prepared_inverse_covariance: object | None = None,
    ) -> list[float]:
        """Oblicza wartość *Funkcji_Dopasowania* dla każdego osobnika populacji."""
        return [
            fitness(
                genome,
                x_input,
                target,
                metric,  # type: ignore[arg-type]
                extractor=extractor,
                prepared_inverse_covariance=prepared_inverse_covariance,
            )
            for genome in population
        ]

    # ------------------------------------------------------------------ #
    # Pojedyncze pokolenie: selekcja → krzyżowanie → mutacja → elitaryzm
    # ------------------------------------------------------------------ #
    def _next_generation(
        self,
        population: list[Genome],
        fitnesses: list[float],
        x_input: InternalRepr,
        target: AggregatedFeatures,
        metric: str,
        extractor: FeatureExtractor,
        config: GAConfig,
        rng: np.random.Generator,
        prepared_inverse_covariance: object | None = None,
    ) -> tuple[list[Genome], list[float]]:
        """Tworzy kolejne pokolenie (Wymaganie 4.4, 4.7).

        Kolejność operacji: wyłonienie elity (``elitism_k`` najlepszych
        osobników przenoszonych bezpośrednio wraz z ich dopasowaniem), a
        następnie wypełnienie pozostałych miejsc potomkami powstałymi przez
        selekcję turniejową, krzyżowanie i mutację (z przycięciem do zakresów
        roboczych). Liczność populacji jest zachowana.

        Zachowanie elity bez zmian gwarantuje monotoniczność najlepszego
        dopasowania (Property 8): nowa populacja zawiera najlepszego osobnika
        poprzedniej wraz z jego (deterministycznie odtwarzalną) wartością
        dopasowania.
        """
        n = len(population)
        k_elite = self._effective_elitism_k(config.elitism_k, n)

        # Elitaryzm: indeksy posortowane malejąco wg dopasowania. ``sorted`` jest
        # stabilne, więc remisy rozstrzyga rosnący indeks - wynik deterministyczny.
        order = sorted(range(n), key=lambda i: fitnesses[i], reverse=True)
        elite_indices = order[:k_elite]
        elites = [population[i] for i in elite_indices]
        elite_fitnesses = [fitnesses[i] for i in elite_indices]

        n_offspring = n - k_elite
        crossover_op = self._crossover_op(config.crossover)
        sigma: Mapping[str, float] = config.mutation_sigma

        offspring: list[Genome] = []
        while len(offspring) < n_offspring:
            parent1 = tournament_select(
                population, fitnesses, config.tournament_size, rng
            )
            parent2 = tournament_select(
                population, fitnesses, config.tournament_size, rng
            )
            child1, child2 = crossover_op(parent1, parent2, rng)
            child1 = self._clamp_genome(gaussian_mutate(child1, sigma, rng))
            offspring.append(child1)
            if len(offspring) < n_offspring:
                child2 = self._clamp_genome(gaussian_mutate(child2, sigma, rng))
                offspring.append(child2)

        offspring_fitnesses = self._evaluate_all(
            offspring,
            x_input,
            target,
            metric,
            extractor,
            prepared_inverse_covariance,
        )

        new_population = elites + offspring
        new_fitnesses = elite_fitnesses + offspring_fitnesses
        return new_population, new_fitnesses

    # ------------------------------------------------------------------ #
    # Statystyki, rejestrowanie i logowanie pokolenia (Wymaganie 4.6)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _stats(
        population: list[Genome], fitnesses: list[float]
    ) -> tuple[float, float, float, Genome]:
        """Zwraca ``(best, mean, worst, best_genome)`` dla danego pokolenia.

        ``best`` to maksimum dopasowania (najmniejsza odległość), ``worst`` -
        minimum, ``mean`` - średnia. ``best_genome`` to osobnik o najwyższym
        dopasowaniu (``argmax`` wskazuje pierwsze maksimum - deterministycznie).
        """
        arr = np.asarray(fitnesses, dtype=np.float64)
        best_idx = int(np.argmax(arr))
        best = float(arr[best_idx])
        mean = float(arr.mean())
        worst = float(arr.min())
        return best, mean, worst, population[best_idx]

    @staticmethod
    def _record(
        best_list: list[float],
        mean_list: list[float],
        worst_list: list[float],
        genome_list: list[Genome],
        best: float,
        mean: float,
        worst: float,
        best_genome: Genome,
    ) -> None:
        """Dopisuje statystyki pokolenia do akumulatorów historii."""
        best_list.append(best)
        mean_list.append(mean)
        worst_list.append(worst)
        genome_list.append(best_genome)

    def _write_log(
        self,
        handle,  # type: ignore[no-untyped-def]
        generation: int,
        best: float,
        mean: float,
        worst: float,
        best_genome: Genome,
    ) -> None:
        """Zapisuje statystyki pokolenia jako linię JSON do ``ga.jsonl`` (Wymaganie 4.6).

        Wpis zawiera pola wymagane formatem logu (``ts``, ``level``,
        ``component``, ``msg``) oraz statystyki dopasowania i najlepszy genotyp.
        Gdy ``handle`` jest ``None`` (brak ``log_path``), funkcja nic nie robi.
        """
        if handle is None:
            return
        record = {
            "ts": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "level": "INFO",
            "component": "ga",
            "generation": generation,
            "best_fitness": best,
            "mean_fitness": mean,
            "worst_fitness": worst,
            "best_genome": {
                "transpose_semitones": best_genome.transpose_semitones,
                "rhythm_density_factor": best_genome.rhythm_density_factor,
                "note_duration_factor": best_genome.note_duration_factor,
                "velocity_offset": best_genome.velocity_offset,
            },
            "msg": "statystyki pokolenia",
        }
        handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")

    @staticmethod
    def _open_log(log_path: Path | str | None):  # type: ignore[no-untyped-def]
        """Otwiera plik ``ga.jsonl`` do zapisu (tryb nadpisania) lub zwraca ``None``.

        Katalog nadrzędny jest tworzony w razie potrzeby. Gdy ``log_path`` jest
        ``None``, nie powstają żadne efekty uboczne na systemie plików.
        """
        if log_path is None:
            return None
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path.open("w", encoding="utf-8")

    # ------------------------------------------------------------------ #
    # Pomocnicze: walidacja, dobór operatorów, przycinanie zakresów
    # ------------------------------------------------------------------ #
    @staticmethod
    def _validate_config(config: GAConfig) -> None:
        """Waliduje parametry przebiegu przed rozpoczęciem ewolucji.

        Raises:
            ValueError: dla niepoprawnych wartości (liczność populacji < 1,
                ujemna liczba pokoleń, rozmiar turnieju < 1, ujemny
                ``elitism_k`` lub nieznany operator krzyżowania).
        """
        if config.population_size < 1:
            raise ValueError(
                "Liczność populacji 'population_size' musi być >= 1, "
                f"otrzymano {config.population_size}."
            )
        if config.generations < 0:
            raise ValueError(
                "Liczba pokoleń 'generations' musi być >= 0, "
                f"otrzymano {config.generations}."
            )
        if config.tournament_size < 1:
            raise ValueError(
                "Rozmiar turnieju 'tournament_size' musi być >= 1, "
                f"otrzymano {config.tournament_size}."
            )
        if config.elitism_k < 0:
            raise ValueError(
                "Parametr elitaryzmu 'elitism_k' musi być >= 0, "
                f"otrzymano {config.elitism_k}."
            )
        if config.crossover not in ("single_point", "uniform"):
            raise ValueError(
                "Nieobsługiwany operator krzyżowania "
                f"{config.crossover!r}; dozwolone wartości to 'single_point' "
                "lub 'uniform'."
            )

    def _effective_elitism_k(self, elitism_k: int, n: int) -> int:
        """Przycina ``elitism_k`` do liczności populacji (``min(elitism_k, n)``).

        Gdy ``elitism_k`` przekracza liczność populacji, cała populacja staje się
        elitą (brak potomków). Fakt przycięcia jest odnotowywany w logu.
        """
        if elitism_k > n:
            self._log.warning(
                "elitism_k przekracza liczność populacji - przycięto",
                elitism_k=elitism_k,
                population_size=n,
            )
            return n
        return elitism_k

    @staticmethod
    def _crossover_op(name: str):  # type: ignore[no-untyped-def]
        """Zwraca funkcję krzyżowania odpowiadającą ``config.crossover``."""
        if name == "single_point":
            return single_point_crossover
        if name == "uniform":
            return uniform_crossover
        # Walidacja w _validate_config czyni tę gałąź nieosiągalną.
        raise ValueError(f"Nieobsługiwany operator krzyżowania: {name!r}.")

    @staticmethod
    def _clamp_genome(genome: Genome) -> Genome:
        """Przycina geny do zakresów roboczych :data:`WORKING_RANGES` (saturacja).

        Krzyżowanie BLX-α i mutacja gaussowska mogą wyprowadzić geny poza
        przedziały robocze; przycięcie utrzymuje populację w muzycznie sensownym
        obszarze przeszukiwania (zob. docstring :data:`WORKING_RANGES`).
        """
        values = {
            "transpose_semitones": genome.transpose_semitones,
            "rhythm_density_factor": genome.rhythm_density_factor,
            "note_duration_factor": genome.note_duration_factor,
            "velocity_offset": genome.velocity_offset,
        }
        clamped = {}
        for name in GENE_NAMES:
            low, high = WORKING_RANGES[name]
            clamped[name] = float(min(high, max(low, values[name])))
        return Genome(
            transpose_semitones=clamped["transpose_semitones"],
            rhythm_density_factor=clamped["rhythm_density_factor"],
            note_duration_factor=clamped["note_duration_factor"],
            velocity_offset=clamped["velocity_offset"],
        )
