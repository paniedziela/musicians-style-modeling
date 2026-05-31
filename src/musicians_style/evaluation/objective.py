"""ObjectiveEvaluator - ewaluacja obiektywna transferu stylu (Wymaganie 6).

Moduł implementuje :class:`ObjectiveEvaluator` realizujący ilościową ocenę
skuteczności transferu stylu (sekcja *Ewaluator* w ``design.md``). Dla zbioru par
``(Utwór_Wejściowy, Utwór_Wyjściowy)`` oraz agregatu statystycznego
*Zbioru_Stylu* artysty docelowego (:class:`~musicians_style.features.types.AggregatedFeatures`)
ewaluator oblicza trzy serie odległości *Wektorów_Cech* i - przy wystarczającej
liczności próby - przeprowadza test statystyczny porównujący odległość od stylu
docelowego przed i po transferze.

Obliczane odległości (Wymagania 6.1, 6.2)
=========================================

Dla każdej poprawnej pary ``(input, output)``:

* ``dist_to_style_before`` - odległość *Wektora_Cech* *Utworu_Wejściowego* od
  średniego *Wektora_Cech* *Zbioru_Stylu* (Wymaganie 6.1, kontrola "przed"),
* ``dist_to_style_after``  - odległość *Wektora_Cech* *Utworu_Wyjściowego* od
  średniego *Wektora_Cech* *Zbioru_Stylu* (Wymaganie 6.1, kontrola "po"),
* ``dist_input_output``    - odległość *Wektora_Cech* *Utworu_Wyjściowego* od
  *Wektora_Cech* *Utworu_Wejściowego* (Wymaganie 6.2, kontrola zachowania
  struktury źródłowej).

Metryka odległości jest konfigurowalna: domyślnie euklidesowa
(:func:`~musicians_style.evaluation.distance.euclidean`), opcjonalnie Mahalanobisa
(:func:`~musicians_style.evaluation.distance.mahalanobis`) z wykorzystaniem
macierzy kowariancji *Zbioru_Stylu* (``style_aggregated.covariance``).

Reguła doboru testu statystycznego (Wymagania 6.3, 6.4)
=======================================================

Niech ``n`` oznacza liczbę **poprawnych** par (par, dla których udało się
obliczyć odległości). Pary z plikami niepoprawnymi są pomijane i nie wliczają się
do ``n`` (analogicznie do Wymagania 1.3 dla akwizycji).

* ``n >= min_pairs_for_test`` (domyślnie 10): przeprowadzany jest test
  porównujący odległości od stylu **przed** i **po** transferze:

  - jeżeli różnice ``before - after`` mają rozkład bliski normalnemu wg testu
    **Shapiro-Wilka** (``p_shapiro > alpha``) → **test t-Studenta dla prób
    zależnych** (:func:`scipy.stats.ttest_rel`),
  - w przeciwnym razie → **test rang Wilcoxona**
    (:func:`scipy.stats.wilcoxon`),

  na poziomie istotności ``alpha`` (domyślnie 0,05). Zgodnie z Wymaganiem 6.3
  statystyki opisowe są w tej sytuacji **pomijane** (``descriptive_stats = None``).

* ``n < min_pairs_for_test``: test jest **pomijany** (``statistical_test = None``),
  w logu zapisywane jest ostrzeżenie o niewystarczającej liczności próby
  (Wymaganie 6.4), a raport zawiera wyłącznie **statystyki opisowe** serii
  odległości.

Raportowanie (Wymaganie 6.5)
============================

Metoda :meth:`ObjectiveEvaluator.write_report` zapisuje :class:`EvaluationReport`
do pliku ``objective_report.json`` (sekcja *Format manifestu eksperymentu* w
``design.md``), raportując m.in. **wartości p** uzyskane w teście statystycznym.
Wartości nieskończone/``NaN`` są zapisywane jako ``null`` (poprawny JSON).

Determinizm
===========

Obliczenia są deterministyczne względem wejść: ekstrakcja cech jest czysta
(Wymaganie 2.4), a testy statystyczne ``scipy`` są deterministyczne dla
ustalonych danych wejściowych.
"""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal

import numpy as np
from scipy.stats import shapiro, ttest_rel, wilcoxon

from musicians_style.evaluation.distance import euclidean, mahalanobis
from musicians_style.features.extractor import FeatureExtractor
from musicians_style.features.types import AggregatedFeatures, FeatureVector
from musicians_style.logging import get_logger
from musicians_style.midi.parser import MidiParser

__all__ = [
    "Metric",
    "StatTestResult",
    "SeriesStats",
    "DescriptiveStats",
    "EvaluationReport",
    "ObjectiveEvaluator",
]

#: Dozwolone metryki odległości w ewaluacji obiektywnej.
Metric = Literal["euclidean", "mahalanobis"]

#: Domyślny poziom istotności testu statystycznego (Wymaganie 6.3).
_DEFAULT_ALPHA = 0.05

#: Domyślna minimalna liczność próby pozwalająca na test statystyczny
#: (Wymagania 6.3, 6.4) - 10 par.
_DEFAULT_MIN_PAIRS = 10


def _json_safe_float(value: float) -> float | None:
    """Zwraca ``value`` jeśli skończone, w przeciwnym razie ``None``.

    Wartości ``NaN``/``±inf`` nie są poprawnym JSON-em w trybie ścisłym, dlatego
    przy serializacji raportu (Wymaganie 6.5) zastępujemy je ``null``.
    """
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


@dataclass(frozen=True)
class StatTestResult:
    """Wynik testu statystycznego porównującego odległości przed/po transferze.

    Atrybuty:
        test: nazwa testu - ``"t-test"`` (test t-Studenta dla prób zależnych)
            lub ``"wilcoxon"`` (test rang Wilcoxona).
        statistic: wartość statystyki testowej.
        p_value: uzyskana wartość p (Wymaganie 6.5).
        alpha: przyjęty poziom istotności (Wymaganie 6.3).
        significant: ``True`` gdy ``p_value < alpha`` (różnica istotna
            statystycznie).
        normality_p_value: wartość p testu Shapiro-Wilka, która zdecydowała o
            doborze testu (``None``, gdy test normalności był nieokreślony, np.
            stałe różnice).
    """

    test: Literal["t-test", "wilcoxon"]
    statistic: float
    p_value: float
    alpha: float
    significant: bool
    normality_p_value: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serializuje wynik do słownika bezpiecznego dla JSON (Wymaganie 6.5)."""
        return {
            "test": self.test,
            "statistic": _json_safe_float(self.statistic),
            "p_value": _json_safe_float(self.p_value),
            "alpha": self.alpha,
            "significant": self.significant,
            "normality_p_value": (
                None
                if self.normality_p_value is None
                else _json_safe_float(self.normality_p_value)
            ),
        }


@dataclass(frozen=True)
class SeriesStats:
    """Statystyki opisowe pojedynczej serii odległości.

    Atrybuty:
        count: liczba pomiarów.
        mean: średnia arytmetyczna.
        std: odchylenie standardowe (populacyjne, ``ddof=0``).
        median: mediana.
        minimum: wartość najmniejsza.
        maximum: wartość największa.
    """

    count: int
    mean: float
    std: float
    median: float
    minimum: float
    maximum: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "mean": _json_safe_float(self.mean),
            "std": _json_safe_float(self.std),
            "median": _json_safe_float(self.median),
            "min": _json_safe_float(self.minimum),
            "max": _json_safe_float(self.maximum),
        }


@dataclass(frozen=True)
class DescriptiveStats:
    """Statystyki opisowe trzech serii odległości (raport dla ``n < 10``).

    Raportowane wyłącznie, gdy liczność próby jest niewystarczająca do
    przeprowadzenia testu statystycznego (Wymaganie 6.4).
    """

    to_style_before: SeriesStats
    to_style_after: SeriesStats
    input_output: SeriesStats

    def to_dict(self) -> dict[str, Any]:
        return {
            "to_style_before": self.to_style_before.to_dict(),
            "to_style_after": self.to_style_after.to_dict(),
            "input_output": self.input_output.to_dict(),
        }


@dataclass
class EvaluationReport:
    """Raport ewaluacji obiektywnej (sekcja *Ewaluator* w ``design.md``).

    Atrybuty:
        distances_to_style_before: odległości *Utworów_Wejściowych* od stylu
            docelowego (Wymaganie 6.1).
        distances_to_style_after: odległości *Utworów_Wyjściowych* od stylu
            docelowego (Wymaganie 6.1).
        distances_input_output: odległości *Utworów_Wyjściowych* od
            *Utworów_Wejściowych* (Wymaganie 6.2).
        statistical_test: wynik testu statystycznego lub ``None`` gdy
            ``n < min_pairs_for_test`` (Wymagania 6.3, 6.4).
        descriptive_stats: statystyki opisowe serii odległości; wypełniane tylko
            gdy ``n < min_pairs_for_test`` (Wymaganie 6.4), w przeciwnym razie
            ``None`` (Wymaganie 6.3 - pominięcie statystyk opisowych).
        plots: ścieżki wygenerowanych wykresów (PDF/EPS); puste na tym etapie -
            wykresy realizuje zadanie 11.3 (Wymaganie 6.6).
    """

    distances_to_style_before: list[float]
    distances_to_style_after: list[float]
    distances_input_output: list[float]
    statistical_test: StatTestResult | None = None
    descriptive_stats: DescriptiveStats | None = None
    plots: list[Path] = field(default_factory=list)

    @property
    def n_pairs(self) -> int:
        """Liczba poprawnych par (par, dla których obliczono odległości)."""
        return len(self.distances_to_style_before)

    def to_dict(self) -> dict[str, Any]:
        """Serializuje raport do słownika gotowego do zapisu jako JSON.

        Wartości p testu statystycznego są raportowane zgodnie z Wymaganiem 6.5.
        """
        return {
            "n_pairs": self.n_pairs,
            "metric": None,  # uzupełniane przez ObjectiveEvaluator.write_report
            "distances": {
                "to_style_before": [
                    _json_safe_float(float(v)) for v in self.distances_to_style_before
                ],
                "to_style_after": [
                    _json_safe_float(float(v)) for v in self.distances_to_style_after
                ],
                "input_output": [
                    _json_safe_float(float(v)) for v in self.distances_input_output
                ],
            },
            "statistical_test": (
                None if self.statistical_test is None else self.statistical_test.to_dict()
            ),
            "descriptive_stats": (
                None if self.descriptive_stats is None else self.descriptive_stats.to_dict()
            ),
            "plots": [str(p) for p in self.plots],
        }


class ObjectiveEvaluator:
    """Ewaluacja obiektywna transferu stylu (Wymaganie 6).

    Ewaluator jest bezstanowy względem danych wejściowych - jedną instancję można
    bezpiecznie współdzielić pomiędzy ewaluacjami. Stan obiektu stanowią wyłącznie
    konfiguracja (metryka, ``alpha``, próg liczności) oraz współdzielone,
    bezstanowe komponenty (:class:`MidiParser`, :class:`FeatureExtractor`, logger).
    """

    def __init__(
        self,
        *,
        metric: Metric = "euclidean",
        alpha: float = _DEFAULT_ALPHA,
        min_pairs_for_test: int = _DEFAULT_MIN_PAIRS,
        parser: MidiParser | None = None,
        extractor: FeatureExtractor | None = None,
    ) -> None:
        """Inicjalizuje ewaluator obiektywny.

        Args:
            metric: metryka odległości *Wektorów_Cech* (``"euclidean"`` lub
                ``"mahalanobis"``); metryka Mahalanobisa wykorzystuje macierz
                kowariancji *Zbioru_Stylu* (Wymaganie 4.3 / 6.x).
            alpha: poziom istotności testu statystycznego (Wymaganie 6.3).
            min_pairs_for_test: minimalna liczba poprawnych par pozwalająca na
                przeprowadzenie testu (Wymagania 6.3, 6.4).
            parser: opcjonalny *Parser_MIDI* (domyślnie tworzony wewnętrznie).
            extractor: opcjonalny *Ekstraktor_Cech* (domyślnie tworzony
                wewnętrznie).

        Raises:
            ValueError: gdy ``metric`` jest spoza dozwolonego zbioru.
        """
        if metric not in ("euclidean", "mahalanobis"):
            raise ValueError(
                "Nieobsługiwana metryka odległości: "
                f"{metric!r} (dozwolone 'euclidean', 'mahalanobis')."
            )
        self.metric: Metric = metric
        self.alpha = float(alpha)
        self.min_pairs_for_test = int(min_pairs_for_test)
        self._parser = parser if parser is not None else MidiParser()
        self._extractor = extractor if extractor is not None else FeatureExtractor()
        self._log = get_logger("objective_evaluator")

    # -- API publiczne -------------------------------------------------------

    def evaluate(
        self,
        pairs: Iterable[tuple[Path | str, Path | str]],
        style_aggregated: AggregatedFeatures,
    ) -> EvaluationReport:
        """Ocenia zbiór par ``(Utwór_Wejściowy, Utwór_Wyjściowy)``.

        Dla każdej pary obliczane są trzy odległości *Wektorów_Cech* (Wymagania
        6.1, 6.2). Pary, dla których któregokolwiek z plików nie udało się
        sparsować (niepoprawny MIDI), są pomijane wraz z wpisem ostrzegawczym w
        logu i nie wliczają się do liczności próby.

        Args:
            pairs: iterowalna kolekcja par ścieżek ``(input_midi, output_midi)``.
            style_aggregated: agregat statystyczny *Zbioru_Stylu* artysty
                docelowego (dostarcza ``mean`` oraz ``covariance``).

        Returns:
            :class:`EvaluationReport` z seriami odległości i - zależnie od
            liczności - wynikiem testu statystycznego albo statystykami opisowymi.
        """
        style_mean = style_aggregated.mean.as_array()
        covariance = np.asarray(style_aggregated.covariance, dtype=np.float64)

        before: list[float] = []
        after: list[float] = []
        input_output: list[float] = []

        for input_path, output_path in pairs:
            measurement = self._measure_pair(
                input_path, output_path, style_mean, covariance
            )
            if measurement is None:
                continue
            d_before, d_after, d_io = measurement
            before.append(d_before)
            after.append(d_after)
            input_output.append(d_io)

        return self.build_report(before, after, input_output)

    def build_report(
        self,
        distances_to_style_before: list[float],
        distances_to_style_after: list[float],
        distances_input_output: list[float],
    ) -> EvaluationReport:
        """Buduje :class:`EvaluationReport` z gotowych serii odległości.

        Wydzielona, czysta logika raportowania (niezależna od wczytywania MIDI)
        realizująca regułę doboru testu (Wymagania 6.3, 6.4). Umożliwia także
        bezpośrednie testowanie scenariuszy ``n < 10`` oraz ``n >= 10``.

        Args:
            distances_to_style_before: odległości "przed" (Wymaganie 6.1).
            distances_to_style_after: odległości "po" (Wymaganie 6.1).
            distances_input_output: odległości wejście-wyjście (Wymaganie 6.2).

        Returns:
            :class:`EvaluationReport`.

        Raises:
            ValueError: gdy trzy serie odległości mają różną długość.
        """
        n = len(distances_to_style_before)
        if not (len(distances_to_style_after) == n == len(distances_input_output)):
            raise ValueError(
                "Serie odległości muszą mieć identyczną długość, otrzymano "
                f"before={len(distances_to_style_before)}, "
                f"after={len(distances_to_style_after)}, "
                f"input_output={len(distances_input_output)}."
            )

        statistical_test: StatTestResult | None
        descriptive_stats: DescriptiveStats | None

        if n >= self.min_pairs_for_test:
            # Wymaganie 6.3: test statystyczny; statystyki opisowe pomijane.
            statistical_test = self._select_and_run_test(
                np.asarray(distances_to_style_before, dtype=np.float64),
                np.asarray(distances_to_style_after, dtype=np.float64),
            )
            descriptive_stats = None
            self._log.info(
                "ewaluacja obiektywna: przeprowadzono test statystyczny",
                n_pairs=n,
                test=statistical_test.test,
                p_value=_json_safe_float(statistical_test.p_value),
                significant=statistical_test.significant,
            )
        else:
            # Wymaganie 6.4: pominięcie testu, log ostrzeżenia, statystyki opisowe.
            statistical_test = None
            descriptive_stats = self._descriptive_stats(
                distances_to_style_before,
                distances_to_style_after,
                distances_input_output,
            )
            self._log.warning(
                "niewystarczająca liczność próby do testu statystycznego - "
                "raport zawiera wyłącznie statystyki opisowe",
                n_pairs=n,
                min_pairs_for_test=self.min_pairs_for_test,
            )

        return EvaluationReport(
            distances_to_style_before=[float(v) for v in distances_to_style_before],
            distances_to_style_after=[float(v) for v in distances_to_style_after],
            distances_input_output=[float(v) for v in distances_input_output],
            statistical_test=statistical_test,
            descriptive_stats=descriptive_stats,
            plots=[],
        )

    def write_report(
        self, report: EvaluationReport, path: Path | str
    ) -> Path:
        """Zapisuje raport do pliku ``objective_report.json`` (Wymaganie 6.5).

        Args:
            report: raport ewaluacji do zapisania.
            path: docelowa ścieżka pliku JSON (katalogi nadrzędne są tworzone).

        Returns:
            Ścieżka zapisanego pliku.
        """
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)

        payload = report.to_dict()
        payload["metric"] = self.metric  # uzupełnienie informacji o metryce

        with target.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)

        self._log.info(
            "zapisano raport ewaluacji obiektywnej",
            path=str(target),
            n_pairs=report.n_pairs,
        )
        return target

    # -- obliczanie odległości ----------------------------------------------

    def _measure_pair(
        self,
        input_path: Path | str,
        output_path: Path | str,
        style_mean: np.ndarray,
        covariance: np.ndarray,
    ) -> tuple[float, float, float] | None:
        """Oblicza trzy odległości dla pojedynczej pary lub zwraca ``None``.

        ``None`` jest zwracane, gdy któregokolwiek z plików nie udało się
        sparsować (niepoprawny MIDI) - para jest wówczas pomijana, a fakt
        odnotowywany w logu.
        """
        try:
            fv_input = self._extract(input_path)
            fv_output = self._extract(output_path)
        except Exception as exc:  # noqa: BLE001 - niepoprawny plik nie może przerwać ewaluacji
            self._log.warning(
                "pominięto parę z powodu błędu wczytania/parsowania pliku MIDI",
                input=str(input_path),
                output=str(output_path),
                problem=repr(exc),
            )
            return None

        input_array = fv_input.as_array()
        output_array = fv_output.as_array()

        d_before = self._distance(input_array, style_mean, covariance)
        d_after = self._distance(output_array, style_mean, covariance)
        d_io = self._distance(output_array, input_array, covariance)
        return d_before, d_after, d_io

    def _extract(self, path: Path | str) -> FeatureVector:
        """Parsuje plik MIDI i zwraca jego *Wektor_Cech*."""
        internal = self._parser.parse(path)
        return self._extractor.extract(internal, source=path)

    def _distance(
        self, a: np.ndarray, b: np.ndarray, covariance: np.ndarray
    ) -> float:
        """Oblicza odległość zgodnie z wybraną metryką (euclidean/mahalanobis)."""
        if self.metric == "mahalanobis":
            return mahalanobis(a, b, covariance)
        return euclidean(a, b)

    # -- dobór i przeprowadzenie testu statystycznego -----------------------

    def _select_and_run_test(
        self, before: np.ndarray, after: np.ndarray
    ) -> StatTestResult:
        """Dobiera i przeprowadza test statystyczny (Wymaganie 6.3).

        Test normalności Shapiro-Wilka na różnicach ``before - after`` decyduje o
        wyborze testu t-Studenta dla prób zależnych (różnice normalne) albo testu
        rang Wilcoxona (różnice nienormalne).
        """
        diffs = before - after
        is_normal, normality_p = self._is_normal(diffs)

        if is_normal:
            statistic, p_value = self._run_ttest(before, after)
            test_name: Literal["t-test", "wilcoxon"] = "t-test"
        else:
            statistic, p_value = self._run_wilcoxon(before, after)
            test_name = "wilcoxon"

        significant = bool(math.isfinite(p_value) and p_value < self.alpha)
        return StatTestResult(
            test=test_name,
            statistic=float(statistic),
            p_value=float(p_value),
            alpha=float(self.alpha),
            significant=significant,
            normality_p_value=normality_p,
        )

    def _is_normal(self, diffs: np.ndarray) -> tuple[bool, float | None]:
        """Ocenia normalność różnic testem Shapiro-Wilka.

        Zwraca parę ``(czy_normalne, p_value_shapiro)``. Dla stałych różnic (brak
        wariancji) lub mniej niż 3 obserwacji test jest nieokreślony - zwracane
        jest ``(False, None)``, co kieruje przepływ do testu Wilcoxona.
        """
        if diffs.size < 3 or float(np.ptp(diffs)) == 0.0:
            return False, None
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = shapiro(diffs)
        except (ValueError, FloatingPointError):
            return False, None
        p_value = float(result.pvalue)
        if not math.isfinite(p_value):
            return False, None
        return (p_value > self.alpha), p_value

    def _run_ttest(
        self, before: np.ndarray, after: np.ndarray
    ) -> tuple[float, float]:
        """Test t-Studenta dla prób zależnych z obsługą przypadków degenerowanych."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = ttest_rel(before, after)
        statistic = float(result.statistic)
        p_value = float(result.pvalue)

        if not math.isfinite(p_value):
            diffs = before - after
            if np.allclose(diffs, 0.0):
                # Brak różnicy - test nieistotny statystycznie.
                return 0.0, 1.0
            # Stała, niezerowa różnica (zerowa wariancja) - różnica idealnie
            # spójna; traktujemy jako istotną (p -> 0).
            return statistic, 0.0
        return statistic, p_value

    def _run_wilcoxon(
        self, before: np.ndarray, after: np.ndarray
    ) -> tuple[float, float]:
        """Test rang Wilcoxona z obsługą przypadku zerowych różnic."""
        diffs = before - after
        if np.allclose(diffs, 0.0):
            # Wilcoxon nie jest zdefiniowany, gdy wszystkie różnice są zerowe.
            return 0.0, 1.0
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = wilcoxon(before, after)
        except ValueError:
            # Np. wszystkie różnice zerowe po odrzuceniu - brak istotności.
            return 0.0, 1.0
        statistic = float(result.statistic)
        p_value = float(result.pvalue)
        if not math.isfinite(p_value):
            return statistic, 1.0
        return statistic, p_value

    # -- statystyki opisowe (Wymaganie 6.4) ---------------------------------

    def _descriptive_stats(
        self,
        before: list[float],
        after: list[float],
        input_output: list[float],
    ) -> DescriptiveStats:
        """Buduje statystyki opisowe trzech serii odległości."""
        return DescriptiveStats(
            to_style_before=self._series_stats(before),
            to_style_after=self._series_stats(after),
            input_output=self._series_stats(input_output),
        )

    @staticmethod
    def _series_stats(values: list[float]) -> SeriesStats:
        """Oblicza statystyki opisowe pojedynczej serii (puste serie → zera)."""
        arr = np.asarray(values, dtype=np.float64)
        if arr.size == 0:
            return SeriesStats(
                count=0, mean=0.0, std=0.0, median=0.0, minimum=0.0, maximum=0.0
            )
        return SeriesStats(
            count=int(arr.size),
            mean=float(arr.mean()),
            std=float(arr.std()),
            median=float(np.median(arr)),
            minimum=float(arr.min()),
            maximum=float(arr.max()),
        )
