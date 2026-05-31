"""Algorytm_Genetyczny: genom, operatory, funkcja dopasowania (Wymagania 4.x).

Pakiet udostępnia komponenty *Algorytmu_Genetycznego* realizującego ewolucyjny
transfer stylu (sekcja *Algorytm_Genetyczny* w ``design.md``):

* :class:`Genome` - niskowymiarowy genotyp (4 parametry rzeczywiste, Wymaganie 4.1),
* :data:`IDENTITY_GENOME` - transformacja tożsamościowa (Property 5, Wymaganie 11.7),
* :func:`apply_transformation` - czysta funkcja aplikująca genotyp do
  *Reprezentacji_Wewnętrznej* MIDI,
* :func:`fitness` - *Funkcja_Dopasowania* oceniająca osobnika jako ujemną
  odległość *Wektora_Cech* od stylu docelowego (Wymaganie 4.3),
* operatory genetyczne (Wymaganie 4.2): :func:`tournament_select`,
  :func:`single_point_crossover`, :func:`uniform_crossover`,
  :func:`gaussian_mutate` - wszystkie deterministyczne względem przekazanego
  :class:`numpy.random.Generator` i niemodyfikujące wejścia,
* :class:`GeneticAlgorithm` - główna pętla ewolucji z elitaryzmem i warunkami
  stopu (Wymagania 4.4-4.7) oraz :class:`History` ze statystykami pokoleń;
  :data:`WORKING_RANGES` definiuje zakresy robocze genotypu.
"""

from .algorithm import WORKING_RANGES, GeneticAlgorithm, History
from .fitness import FitnessMetric, fitness
from .operators import (
    BLX_ALPHA,
    GENE_NAMES,
    gaussian_mutate,
    single_point_crossover,
    tournament_select,
    uniform_crossover,
)
from .transformation import apply_transformation
from .types import IDENTITY_GENOME, Genome

__all__ = [
    "Genome",
    "IDENTITY_GENOME",
    "apply_transformation",
    "fitness",
    "FitnessMetric",
    "tournament_select",
    "single_point_crossover",
    "uniform_crossover",
    "gaussian_mutate",
    "GENE_NAMES",
    "BLX_ALPHA",
    "GeneticAlgorithm",
    "History",
    "WORKING_RANGES",
]
