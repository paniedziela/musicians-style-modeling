"""Funkcja_Dopasowania *Algorytmu_Genetycznego* - :func:`fitness` (Wymaganie 4.3).

Moduł implementuje *czystą*, deterministyczną *Funkcję_Dopasowania* oceniającą
osobnika (:class:`~musicians_style.ga.types.Genome`) *Algorytmu_Genetycznego*
(sekcja *Algorytm_Genetyczny* / *Funkcja_Dopasowania* w ``design.md``).

Dopasowanie osobnika definiowane jest jako **ujemna odległość** *Wektora_Cech*
*Utworu_Wejściowego* po zastosowaniu transformacji genotypu od agregowanego
(średniego) *Wektora_Cech* *Zbioru_Stylu*:

``fitness(g) = -distance(extract(apply_transformation(x, g)), target.mean)``

Maksymalizacja dopasowania jest więc równoważna **minimalizacji odległości** od
stylu docelowego - im bliżej średniego *Wektora_Cech* *Zbioru_Stylu*, tym wyższa
(mniej ujemna, w granicy zero) wartość *Funkcji_Dopasowania* (Wymaganie 4.3).

Metryki odległości (Wymaganie 4.3)
==================================

* ``"euclidean"`` - odległość euklidesowa (L2) między spłaszczonymi
  *Wektorami_Cech* (:meth:`~musicians_style.features.types.FeatureVector.as_array`).
* ``"mahalanobis"`` - odległość Mahalanobisa wykorzystująca macierz kowariancji
  *Zbioru_Stylu* (``target_aggregated.covariance``); patrz
  :func:`musicians_style.evaluation.distance.mahalanobis`.

Determinizm (Wymaganie 4.3)
===========================

Funkcja jest **deterministyczna względem wejść** i pozbawiona losowości: jej
wynik zależy wyłącznie od ``genome``, ``x_input``, ``target_aggregated`` oraz
``metric``. Wszystkie wykorzystywane komponenty są czyste -
:func:`~musicians_style.ga.transformation.apply_transformation` jest czystą
transformacją MIDI, :meth:`~musicians_style.features.extractor.FeatureExtractor.extract`
jest deterministyczny (Wymaganie 2.4), a funkcje odległości są deterministyczne
(Wymagania 6.1, 6.2). Dla powtarzalności *Algorytmu_Genetycznego* (Wymaganie 4.4)
*Funkcja_Dopasowania* nie wprowadza żadnego dodatkowego stanu losowego.

Wstrzykiwanie *Ekstraktora_Cech*
================================

*Ekstraktor_Cech* jest bezstanowy względem danych wejściowych, więc *Funkcję_
Dopasowania* można wywoływać z domyślnie utworzoną instancją. Dla wydajności
(uniknięcie tworzenia loggera i instancji w każdej ewaluacji populacji) oraz
testowalności funkcja przyjmuje opcjonalny parametr ``extractor`` - pętla
*Algorytmu_Genetycznego* (zadanie 6.4) powinna przekazać jedną współdzieloną
instancję.
"""

from __future__ import annotations

from typing import Literal

from ..evaluation.distance import euclidean, mahalanobis_from_inverse, prepare_mahalanobis
from ..features.extractor import FeatureExtractor
from ..features.types import AggregatedFeatures
from ..midi.types import InternalRepr
from .transformation import apply_transformation
from .types import Genome

__all__ = ["FitnessMetric", "fitness"]

#: Dozwolone metryki odległości *Funkcji_Dopasowania* (Wymaganie 4.3).
FitnessMetric = Literal["euclidean", "mahalanobis"]

#: Domyślny, współdzielony *Ekstraktor_Cech* używany, gdy wywołujący nie poda
#: własnej instancji. Jest bezstanowy względem danych wejściowych, więc może być
#: bezpiecznie reużywany pomiędzy wywołaniami (Wymaganie 2.4).
_DEFAULT_EXTRACTOR = FeatureExtractor()


def fitness(
    genome: Genome,
    x_input: InternalRepr,
    target_aggregated: AggregatedFeatures,
    metric: FitnessMetric = "euclidean",
    *,
    extractor: FeatureExtractor | None = None,
    prepared_inverse_covariance: object | None = None,
) -> float:
    """Oblicza dopasowanie osobnika *Algorytmu_Genetycznego* (Wymaganie 4.3).

    Aplikuje genotyp ``genome`` do *Utworu_Wejściowego* ``x_input``, wyznacza
    *Wektor_Cech* wyniku i zwraca **ujemną odległość** tego wektora od średniego
    *Wektora_Cech* *Zbioru_Stylu* (``target_aggregated.mean``). Maksymalizacja
    zwracanej wartości odpowiada minimalizacji odległości od stylu docelowego.

    Funkcja jest czysta i deterministyczna względem wejść (brak losowości):
    ``fitness = -distance(extract(apply_transformation(x_input, genome)),
    target_aggregated.mean)``.

    Args:
        genome: genotyp transformacji ocenianego osobnika.
        x_input: *Reprezentacja_Wewnętrzna* *Utworu_Wejściowego*.
        target_aggregated: agregat statystyczny *Zbioru_Stylu*; wykorzystywane
            są ``mean`` (punkt odniesienia) oraz - dla metryki Mahalanobisa -
            ``covariance``.
        metric: metryka odległości, ``"euclidean"`` (domyślnie) lub
            ``"mahalanobis"`` (Wymaganie 4.3).
        extractor: opcjonalny *Ekstraktor_Cech*. Gdy ``None``, używana jest
            współdzielona instancja domyślna. Przekazanie jednej instancji w
            pętli ewolucji ogranicza narzut tworzenia obiektów.
        prepared_inverse_covariance: opcjonalna pseudoodwrotność kowariancji
            przygotowana raz dla stałego profilu celu; dotyczy metryki
            Mahalanobisa i zachowuje zgodność z wywołaniem bez cache'u.

    Returns:
        Wartość *Funkcji_Dopasowania* jako ``float`` - ujemna odległość
        (``<= 0``; równa ``0`` przy idealnym dopasowaniu). Im wyższa, tym lepsze
        dopasowanie do stylu docelowego.

    Raises:
        ValueError: gdy ``metric`` ma nieobsługiwaną wartość lub gdy funkcje
            odległości odrzucą wejścia (np. niezgodne wymiary, wartości
            nieskończone) - zgodnie z kontraktem
            :mod:`musicians_style.evaluation.distance`.
    """
    used_extractor = extractor if extractor is not None else _DEFAULT_EXTRACTOR

    transformed = apply_transformation(x_input, genome)
    feature_vec = used_extractor.extract(transformed)

    candidate = feature_vec.as_array()
    target = target_aggregated.mean.as_array()

    if metric == "euclidean":
        distance = euclidean(candidate, target)
    elif metric == "mahalanobis":
        inverse = (
            prepare_mahalanobis(target_aggregated.covariance, dimension=candidate.size)
            if prepared_inverse_covariance is None
            else prepared_inverse_covariance
        )
        distance = mahalanobis_from_inverse(candidate, target, inverse)
    else:
        raise ValueError(
            "Nieobsługiwana metryka odległości "
            f"{metric!r}; dozwolone wartości to 'euclidean' lub 'mahalanobis'."
        )

    # Maksymalizujemy dopasowanie == minimalizujemy odległość (Wymaganie 4.3).
    return -distance
