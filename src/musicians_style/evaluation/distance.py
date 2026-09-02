"""Funkcje odległości *Wektorów_Cech* w ewaluacji obiektywnej (Wymagania 6.1, 6.2).

Moduł dostarcza dwie czyste, deterministyczne funkcje mierzące odległość pomiędzy
*Wektorami_Cech* (zob. :meth:`musicians_style.features.types.FeatureVector.as_array`):

- :func:`euclidean` - odległość euklidesowa (L2),
- :func:`mahalanobis` - odległość Mahalanobisa wykorzystująca macierz kowariancji
  *Zbioru_Stylu* (:class:`musicians_style.features.types.AggregatedFeatures`).

Obie funkcje są wykorzystywane przez:

- *Funkcję_Dopasowania* *Algorytmu_Genetycznego* (zadanie 6.2, Wymaganie 4.3),
- *ObjectiveEvaluator* w ewaluacji obiektywnej (zadanie 11.2, Wymagania 6.1, 6.2).

Kontrakt poprawnościowy (*Property 10*, Wymagania 6.1, 6.2)
==========================================================

Dla dowolnych poprawnych wektorów wejściowych obie funkcje zwracają wartość, która
jest jednocześnie:

- **skończona** (nie ``NaN`` ani ``±inf``),
- **nieujemna** (``>= 0``),
- **deterministyczna** względem wejść (brak wewnętrznego stanu i losowości).

Wejścia są walidowane *przed* obliczeniem:

- oba *Wektory_Cech* muszą być jednowymiarowe (1-D) i mieć **identyczną długość**,
- wszystkie wartości wejściowe muszą być **skończone** (brak ``NaN``/``inf``),
- macierz kowariancji (dla :func:`mahalanobis`) musi być kwadratowa o boku równym
  długości wektorów oraz skończona.

Naruszenie któregokolwiek warunku skutkuje zgłoszeniem :class:`ValueError`.

Decyzja: odwracanie macierzy kowariancji
========================================

Macierz kowariancji *Zbioru_Stylu* bywa **osobliwa** (np. cechy stałe w zbiorze,
liczba próbek mniejsza niż wymiar cech). Dlatego do odległości Mahalanobisa używamy
**pseudoodwrotności Moore'a-Penrose'a** (:func:`numpy.linalg.pinv`) zamiast klasycznej
odwrotności - jest ona zdefiniowana również dla macierzy osobliwych i numerycznie
stabilna. Macierz jest dodatkowo symetryzowana (``(C + Cᵀ) / 2``) przed odwróceniem,
co dla symetrycznej, dodatnio półokreślonej kowariancji gwarantuje, że forma
kwadratowa ``δᵀ · pinv(C) · δ`` jest nieujemna z dokładnością do błędu numerycznego
(drobne wartości ujemne są przycinane do zera).
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "euclidean",
    "mahalanobis",
    "prepare_mahalanobis",
    "mahalanobis_from_inverse",
]


def _as_feature_array(value: object, name: str) -> np.ndarray:
    """Konwertuje wejście na skończony, jednowymiarowy wektor ``float64``.

    Akceptuje obiekty udostępniające metodę ``as_array`` (np.
    :class:`~musicians_style.features.types.FeatureVector`) oraz dowolne dane
    konwertowalne do tablicy NumPy (``np.ndarray``, lista, krotka).

    Args:
        value: wejściowy *Wektor_Cech* lub obiekt z metodą ``as_array``.
        name: nazwa argumentu używana w komunikatach błędów.

    Returns:
        Jednowymiarowa tablica ``float64`` (niezależna kopia danych).

    Raises:
        ValueError: gdy wejście nie jest jednowymiarowe, jest puste lub zawiera
            wartości nieskończone (``NaN``/``inf``).
    """
    if hasattr(value, "as_array"):
        value = value.as_array()  # type: ignore[union-attr]

    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(
            f"Argument '{name}' musi być wektorem jednowymiarowym (1-D), "
            f"otrzymano tablicę o kształcie {arr.shape}."
        )
    if arr.size == 0:
        raise ValueError(f"Argument '{name}' nie może być pustym wektorem.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(
            f"Argument '{name}' zawiera wartości nieskończone (NaN/inf); "
            "wymagane są wyłącznie wartości skończone."
        )
    return arr


def _validate_same_length(f1: np.ndarray, f2: np.ndarray) -> None:
    """Sprawdza, że dwa *Wektory_Cech* mają identyczną długość.

    Raises:
        ValueError: gdy długości wektorów się różnią.
    """
    if f1.shape != f2.shape:
        raise ValueError(
            "Wektory cech muszą mieć identyczne wymiary, otrzymano "
            f"{f1.shape} oraz {f2.shape}."
        )


def euclidean(f1: object, f2_or_aggregated: object) -> float:
    """Oblicza odległość euklidesową (L2) pomiędzy dwoma *Wektorami_Cech*.

    Czysta funkcja deterministyczna (Wymagania 6.1, 6.2): ``sqrt(Σ (f1 - f2)²)``.
    Zwracana wartość jest zawsze skończona i nieujemna (*Property 10*).

    Args:
        f1: pierwszy *Wektor_Cech* (np. ``feature_vec.as_array()``); akceptowany
            jest również obiekt z metodą ``as_array``.
        f2_or_aggregated: drugi *Wektor_Cech* odniesienia, np.
            ``feature_vec.as_array()`` lub ``aggregated.mean.as_array()``.

    Returns:
        Odległość euklidesowa jako ``float`` (``>= 0``, skończona).

    Raises:
        ValueError: przy niezgodności wymiarów wejść lub wartościach
            nieskończonych (``NaN``/``inf``) na wejściu.
    """
    a = _as_feature_array(f1, "f1")
    b = _as_feature_array(f2_or_aggregated, "f2_or_aggregated")
    _validate_same_length(a, b)

    distance = float(np.linalg.norm(a - b))

    # Niezmiennik wyniku (Property 10): skończoność i nieujemność.
    if not np.isfinite(distance):
        raise ValueError(
            "Odległość euklidesowa nie jest skończona; sprawdź wartości wejściowe."
        )
    # Norma L2 jest z definicji nieujemna; zabezpieczenie przed ujemnym zerem.
    return max(distance, 0.0)


def mahalanobis(
    f1: object, f2_or_aggregated: object, covariance: object
) -> float:
    """Oblicza odległość Mahalanobisa pomiędzy dwoma *Wektorami_Cech*.

    Czysta funkcja deterministyczna (Wymagania 6.1, 6.2):
    ``sqrt((f1 - f2)ᵀ · Σ⁻¹ · (f1 - f2))``, gdzie ``Σ⁻¹`` jest pseudoodwrotnością
    Moore'a-Penrose'a symetryzowanej macierzy kowariancji (patrz docstring modułu).
    Zwracana wartość jest zawsze skończona i nieujemna (*Property 10*).

    Args:
        f1: pierwszy *Wektor_Cech* (np. ``feature_vec.as_array()``).
        f2_or_aggregated: drugi *Wektor_Cech* odniesienia, np.
            ``aggregated.mean.as_array()``.
        covariance: macierz kowariancji cech *Zbioru_Stylu* o kształcie
            ``(n, n)``, gdzie ``n`` to długość *Wektorów_Cech* (np.
            ``aggregated.covariance``).

    Returns:
        Odległość Mahalanobisa jako ``float`` (``>= 0``, skończona).

    Raises:
        ValueError: przy niezgodności wymiarów wejść, niekwadratowej macierzy
            kowariancji o niewłaściwym boku, lub wartościach nieskończonych
            (``NaN``/``inf``) na wejściu.
    """
    a = _as_feature_array(f1, "f1")
    b = _as_feature_array(f2_or_aggregated, "f2_or_aggregated")
    _validate_same_length(a, b)

    inv_cov = prepare_mahalanobis(covariance, dimension=a.shape[0])
    return mahalanobis_from_inverse(a, b, inv_cov)


def prepare_mahalanobis(
    covariance: object, *, dimension: int | None = None
) -> np.ndarray:
    """Prepare the pseudoinverse used by :func:`mahalanobis`.

    Callers evaluating many candidates against one style profile can compute
    this once and pass the result to :func:`mahalanobis_from_inverse`.  The
    validation and symmetrisation match the historical implementation.
    """
    cov = np.asarray(covariance, dtype=np.float64)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError(
            "Macierz kowariancji musi być kwadratowa (2-D), otrzymano kształt "
            f"{cov.shape}."
        )
    if dimension is not None and cov.shape[0] != dimension:
        raise ValueError(
            "Bok macierzy kowariancji musi być równy długości wektorów cech "
            f"({dimension}), otrzymano {cov.shape[0]}."
        )
    if not np.all(np.isfinite(cov)):
        raise ValueError(
            "Macierz kowariancji zawiera wartości nieskończone (NaN/inf); "
            "wymagane są wyłącznie wartości skończone."
        )
    inverse = np.asarray(np.linalg.pinv(0.5 * (cov + cov.T)), dtype=np.float64)
    if not np.all(np.isfinite(inverse)):
        raise ValueError("Pseudoodwrotność macierzy kowariancji nie jest skończona.")
    inverse.flags.writeable = False
    return inverse


def mahalanobis_from_inverse(
    f1: object, f2_or_aggregated: object, inverse_covariance: object
) -> float:
    """Compute Mahalanobis distance from a precomputed pseudoinverse."""
    a = _as_feature_array(f1, "f1")
    b = _as_feature_array(f2_or_aggregated, "f2_or_aggregated")
    _validate_same_length(a, b)
    inverse = np.asarray(inverse_covariance, dtype=np.float64)
    if inverse.ndim != 2 or inverse.shape != (a.shape[0], a.shape[0]):
        raise ValueError(
            "Macierz pseudoodwrotna musi mieć kształt zgodny z wektorami cech; "
            f"otrzymano {inverse.shape} dla wymiaru {a.shape[0]}."
        )
    if not np.all(np.isfinite(inverse)):
        raise ValueError(
            "Macierz pseudoodwrotna zawiera wartości nieskończone (NaN/inf); "
            "wymagane są wyłącznie wartości skończone."
        )
    delta = a - b
    quadratic = float(delta @ inverse @ delta)
    if not np.isfinite(quadratic):
        raise ValueError(
            "Forma kwadratowa odległości Mahalanobisa nie jest skończona; "
            "sprawdź wartości wejściowe i macierz pseudoodwrotną."
        )
    return float(np.sqrt(max(quadratic, 0.0)))
