"""Ekstraktor_Cech - obliczanie *Wektora_Cech* z *Reprezentacji_Wewnętrznej*.

Moduł implementuje :class:`FeatureExtractor` realizujący Wymaganie 2: dla danej
:class:`~musicians_style.midi.types.InternalRepr` oblicza *Wektor_Cech* o stałej
długości (sekcja *Ekstraktor_Cech* w ``design.md``).

Obliczane cechy (Wymaganie 2.1)
===============================

* ``tempo_bpm`` - tempo średnie w BPM. Gdy plik nie zawiera meta-zdarzenia
  tempa, przyjmowane jest 120 BPM zgodnie ze specyfikacją MIDI i fakt ten jest
  odnotowywany w logu (Wymaganie 2.2). Przy wielu zdarzeniach tempa raportowana
  jest średnia arytmetyczna wartości BPM.
* ``key`` - tonacja dominująca wyznaczana metodą korelacji profili
  Krumhansla-Schmucklera (osobne profile dur i moll, 24 kandydatów).
* ``pitch_class_histogram`` - histogram 12 klas wysokości (zliczanie nut modulo
  12), znormalizowany do sumy 1.
* ``interval_histogram`` - histogram interwałów melodycznych monofonicznego
  śladu (różnice kolejnych nut, gdzie w każdym kroku czasowym wybierana jest
  najwyższa nuta); interwały przycinane do zakresu [-12, +12] półtonów (25
  kubełków), znormalizowany do sumy 1 gdy występuje co najmniej jeden interwał.
* ``note_density_per_s`` - liczba nut na sekundę.
* ``mean_note_duration_s`` / ``std_note_duration_s`` - średnia i odchylenie
  standardowe długości nut w sekundach.
* ``rest_ratio`` - proporcja czasu bez brzmiącej nuty w przedziale [0, 1].

Konwersja czasu (ticki → sekundy)
---------------------------------
Czas trwania nut przeliczany jest na sekundy przy użyciu pojedynczego,
reprezentatywnego tempa (średnia BPM zdarzeń tempa lub 120 BPM). Jest to celowe
uproszczenie - utwory ze zmiennym tempem są aproksymowane stałym tempem
średnim, co jest wystarczające dla statystycznej charakterystyki stylu i
zachowuje determinizm (Wymaganie 2.4).

Przypadki brzegowe
------------------
* **Plik pusty / wyłącznie pauzy** (zero nut): zwracany jest
  :data:`~musicians_style.features.constants.NEUTRAL_FEATURE_VECTOR` i fakt ten
  jest odnotowywany w logu (Wymagania 2.7, 11.8).
* **Częściowy błąd numeryczny** (degenerowane wartości liczbowe, niespójna
  struktura): zgodnie z Wymaganiem 2.8 w logu zapisywana jest nazwa pliku, opis
  problemu oraz częściowe wartości; degenerowane pola są zastępowane wartościami
  neutralnymi, a *Ekstraktor_Cech* zwraca częściowy (lecz poprawny) *Wektor_Cech*
  zamiast zgłaszać nieobsłużony wyjątek.

Determinizm (Wymaganie 2.4)
---------------------------
Wszystkie operacje są czysto numeryczne i pozbawione zewnętrznego stanu, dlatego
wielokrotne wywołanie :meth:`FeatureExtractor.extract` dla tej samej
``InternalRepr`` zwraca bit-identyczny *Wektor_Cech* (*Property 2*, zadanie 3.3).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from musicians_style.features.constants import NEUTRAL_FEATURE_VECTOR
from musicians_style.features.types import (
    INTERVAL_HISTOGRAM_BINS,
    PITCH_CLASS_BINS,
    FeatureVector,
)
from musicians_style.logging import get_logger
from musicians_style.midi.types import InternalRepr

__all__ = ["FeatureExtractor"]

#: Domyślne tempo (BPM) przyjmowane dla plików bez meta-zdarzenia tempa
#: (Wymaganie 2.2) oraz dla degenerowanych zdarzeń tempa.
_DEFAULT_TEMPO_BPM = 120.0

#: Liczba mikrosekund w minucie - do konwersji ``tempo`` (μs/ćwierćnutę) → BPM.
_MICROSECONDS_PER_MINUTE = 60_000_000.0

#: Maksymalny interwał melodyczny (w półtonach) reprezentowany w histogramie.
#: Interwały spoza [-_MAX_INTERVAL, +_MAX_INTERVAL] są przycinane do granicy.
_MAX_INTERVAL = 12

#: Indeks kubełka odpowiadającego interwałowi 0 (unison) w histogramie interwałów.
_INTERVAL_ZERO_BIN = _MAX_INTERVAL  # == 12 dla 25 kubełków [-12..+12]

#: Profil Krumhansla-Kesslera dla tonacji durowej (12 klas wysokości).
_KS_MAJOR_PROFILE = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
    dtype=np.float64,
)

#: Profil Krumhansla-Kesslera dla tonacji molowej (12 klas wysokości).
_KS_MINOR_PROFILE = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17],
    dtype=np.float64,
)

#: Nazwy klas wysokości (notacja krzyżykowa) dla etykiety tonacji.
_PITCH_CLASS_NAMES = (
    "C",
    "C#",
    "D",
    "D#",
    "E",
    "F",
    "F#",
    "G",
    "G#",
    "A",
    "A#",
    "B",
)


class FeatureExtractor:
    """Oblicza *Wektor_Cech* z *Reprezentacji_Wewnętrznej* (Wymaganie 2).

    Klasa jest bezstanowa względem danych wejściowych (jedyny stan to logger),
    więc jedną instancję można bezpiecznie współdzielić. Metoda
    :meth:`extract` jest czysta i deterministyczna (Wymaganie 2.4).
    """

    def __init__(self) -> None:
        self._log = get_logger("feature_extractor")

    # -- API publiczne -------------------------------------------------------

    def extract(
        self, repr_: InternalRepr, *, source: Path | str | None = None
    ) -> FeatureVector:
        """Oblicza *Wektor_Cech* dla pojedynczej *Reprezentacji_Wewnętrznej*.

        Args:
            repr_: *Reprezentacja_Wewnętrzna* pliku MIDI.
            source: opcjonalna nazwa/ścieżka pliku źródłowego, wykorzystywana
                wyłącznie do wzbogacenia wpisów logu (Wymagania 2.2, 2.7, 2.8).

        Returns:
            :class:`FeatureVector` o stałej długości (Wymaganie 2.3). Dla pliku
            pustego lub wyłącznie z pauzami zwracany jest
            :data:`NEUTRAL_FEATURE_VECTOR` (Wymaganie 2.7). W przypadku
            częściowego błędu numerycznego zwracany jest częściowy *Wektor_Cech*
            z degenerowanymi polami zastąpionymi wartościami neutralnymi
            (Wymaganie 2.8).
        """
        file_name = self._source_name(source)

        # Plik pusty / wyłącznie pauzy → wektor neutralny (Wymagania 2.7, 11.8).
        if len(repr_.notes) == 0:
            self._log.info(
                "plik pusty lub wyłącznie pauzy - zwracam Wektor_Cech neutralny",
                file=file_name,
            )
            return NEUTRAL_FEATURE_VECTOR

        try:
            return self._extract_nonempty(repr_, file_name)
        except Exception as exc:  # noqa: BLE001 - awaria liczbowa nie może przerwać Systemu
            # Niespodziewany błąd techniczny mimo obecności nut (Wymaganie 2.8):
            # logujemy i zwracamy wektor neutralny jako bezpieczny fallback.
            self._log.warning(
                "ekstrakcja nie powiodła się z powodu błędu technicznego - "
                "zwracam Wektor_Cech neutralny",
                file=file_name,
                problem=repr(exc),
            )
            return NEUTRAL_FEATURE_VECTOR

    # -- rdzeń ekstrakcji ----------------------------------------------------

    def _extract_nonempty(
        self, repr_: InternalRepr, file_name: str | None
    ) -> FeatureVector:
        """Oblicza *Wektor_Cech* dla repr. zawierającej co najmniej jedną nutę.

        Każda cecha jest walidowana pod kątem skończoności; pola degenerowane
        (``NaN``/``inf``) są zastępowane wartościami neutralnymi, a fakt
        odnotowywany jako częściowy błąd (Wymaganie 2.8).
        """
        problems: list[str] = []

        tempo_bpm = self._estimate_tempo(repr_, file_name)
        seconds_per_tick = self._seconds_per_tick(repr_, tempo_bpm)

        pitch_class_histogram = self._pitch_class_histogram(repr_)
        interval_histogram = self._interval_histogram(repr_)
        key = self._estimate_key(pitch_class_histogram)

        durations_ticks = np.array(
            [max(0, note.duration_ticks) for note in repr_.notes],
            dtype=np.float64,
        )
        durations_s = durations_ticks * seconds_per_tick
        mean_note_duration_s = float(durations_s.mean())
        std_note_duration_s = float(durations_s.std())

        total_ticks = max(
            note.tick + max(0, note.duration_ticks) for note in repr_.notes
        )
        total_duration_s = total_ticks * seconds_per_tick

        if total_duration_s > 0:
            note_density_per_s = len(repr_.notes) / total_duration_s
        else:
            # Zerowy zakres czasowy (np. wszystkie nuty o zerowej długości w
            # tym samym ticku) - gęstość na sekundę jest nieokreślona.
            note_density_per_s = math.inf

        rest_ratio = self._rest_ratio(repr_, total_ticks)

        # --- walidacja skończoności (Wymaganie 2.8) ---
        tempo_bpm = self._validated_scalar(
            tempo_bpm, "tempo_bpm", NEUTRAL_FEATURE_VECTOR.tempo_bpm, problems
        )
        note_density_per_s = self._validated_scalar(
            note_density_per_s,
            "note_density_per_s",
            NEUTRAL_FEATURE_VECTOR.note_density_per_s,
            problems,
        )
        mean_note_duration_s = self._validated_scalar(
            mean_note_duration_s,
            "mean_note_duration_s",
            NEUTRAL_FEATURE_VECTOR.mean_note_duration_s,
            problems,
        )
        std_note_duration_s = self._validated_scalar(
            std_note_duration_s,
            "std_note_duration_s",
            NEUTRAL_FEATURE_VECTOR.std_note_duration_s,
            problems,
        )
        rest_ratio = self._validated_scalar(
            rest_ratio, "rest_ratio", NEUTRAL_FEATURE_VECTOR.rest_ratio, problems
        )
        pitch_class_histogram = self._validated_array(
            pitch_class_histogram,
            "pitch_class_histogram",
            np.asarray(NEUTRAL_FEATURE_VECTOR.pitch_class_histogram),
            problems,
        )
        interval_histogram = self._validated_array(
            interval_histogram,
            "interval_histogram",
            np.asarray(NEUTRAL_FEATURE_VECTOR.interval_histogram),
            problems,
        )

        feature_vector = FeatureVector(
            tempo_bpm=tempo_bpm,
            key=key,
            pitch_class_histogram=pitch_class_histogram,
            interval_histogram=interval_histogram,
            note_density_per_s=note_density_per_s,
            mean_note_duration_s=mean_note_duration_s,
            std_note_duration_s=std_note_duration_s,
            rest_ratio=rest_ratio,
        )

        if problems:
            # Wymaganie 2.8: nazwa pliku, opis problemu i częściowe wartości.
            self._log.warning(
                "częściowy błąd numeryczny podczas ekstrakcji - zwracam "
                "częściowy Wektor_Cech",
                file=file_name,
                problem="; ".join(problems),
                partial={
                    "tempo_bpm": tempo_bpm,
                    "key": key,
                    "note_density_per_s": note_density_per_s,
                    "mean_note_duration_s": mean_note_duration_s,
                    "std_note_duration_s": std_note_duration_s,
                    "rest_ratio": rest_ratio,
                },
            )

        return feature_vector

    # -- tempo i konwersja czasu --------------------------------------------

    def _estimate_tempo(
        self, repr_: InternalRepr, file_name: str | None
    ) -> float:
        """Wyznacza tempo średnie w BPM (Wymagania 2.1, 2.2).

        Zbiera dodatnie zdarzenia tempa (payload ``{"tempo": <μs/ćwierćnutę>}``)
        i zwraca średnią arytmetyczną odpowiadających im wartości BPM. Gdy plik
        nie zawiera prawidłowego zdarzenia tempa, zwraca 120 BPM i odnotowuje
        ten fakt w logu (Wymaganie 2.2).
        """
        bpms: list[float] = []
        for event in repr_.meta:
            if event.kind != "tempo":
                continue
            tempo_us = event.payload.get("tempo")
            if isinstance(tempo_us, (int, float)) and tempo_us > 0:
                bpms.append(_MICROSECONDS_PER_MINUTE / float(tempo_us))

        if not bpms:
            self._log.info(
                "brak meta-zdarzenia tempa - przyjęto domyślne 120 BPM",
                file=file_name,
            )
            return _DEFAULT_TEMPO_BPM

        return float(np.mean(bpms))

    @staticmethod
    def _seconds_per_tick(repr_: InternalRepr, tempo_bpm: float) -> float:
        """Liczba sekund przypadająca na pojedynczy tick SMF.

        ``sekundy/tick = (60 / BPM) / ticks_per_beat``. Zwraca ``0.0`` gdy
        ``ticks_per_beat`` lub ``tempo_bpm`` są niepoprawne - degeneracja
        zostanie wykryta przez walidację skończoności cech zależnych.
        """
        ticks_per_beat = repr_.ticks_per_beat
        if ticks_per_beat <= 0 or tempo_bpm <= 0:
            return 0.0
        seconds_per_beat = 60.0 / tempo_bpm
        return seconds_per_beat / ticks_per_beat

    # -- histogramy ----------------------------------------------------------

    @staticmethod
    def _pitch_class_histogram(repr_: InternalRepr) -> np.ndarray:
        """Histogram 12 klas wysokości (modulo 12), znormalizowany do sumy 1.

        Każda nuta zliczana jest dokładnie raz w kubełku ``pitch % 12``. Suma
        histogramu wynosi 1 (niezmiennik *Wektora_Cech*, Wymaganie 2.3 /
        *Property 3*), gdyż liczba nut jest dodatnia.
        """
        histogram = np.zeros(PITCH_CLASS_BINS, dtype=np.float64)
        for note in repr_.notes:
            histogram[note.pitch % PITCH_CLASS_BINS] += 1.0
        total = histogram.sum()
        if total > 0:
            histogram /= total
        return histogram

    @staticmethod
    def _monophonic_pitches(repr_: InternalRepr) -> list[int]:
        """Buduje monofoniczny ślad melodyczny (najwyższa nuta na onset).

        Nuty grupowane są po ticku rozpoczęcia (``tick``); w każdej grupie
        wybierana jest najwyższa wysokość. Zwracana jest lista wysokości
        uporządkowana rosnąco wg ticku - podstawa obliczenia interwałów
        melodycznych.
        """
        highest_by_onset: dict[int, int] = {}
        for note in repr_.notes:
            current = highest_by_onset.get(note.tick)
            if current is None or note.pitch > current:
                highest_by_onset[note.tick] = note.pitch
        return [highest_by_onset[tick] for tick in sorted(highest_by_onset)]

    def _interval_histogram(self, repr_: InternalRepr) -> np.ndarray:
        """Histogram interwałów melodycznych monofonicznego śladu.

        Interwały (różnice kolejnych wysokości) są przycinane do zakresu
        [-12, +12] półtonów i zliczane w 25 kubełkach (indeks ``interwał + 12``).
        Histogram jest normalizowany do sumy 1, gdy występuje co najmniej jeden
        interwał; w przeciwnym razie (mniej niż dwie nuty melodyczne) zwracana
        jest tablica zer - zgodnie z konwencją *Wektora_Cech* neutralnego.
        """
        histogram = np.zeros(INTERVAL_HISTOGRAM_BINS, dtype=np.float64)
        pitches = self._monophonic_pitches(repr_)
        for previous, current in zip(pitches, pitches[1:]):
            interval = current - previous
            clamped = max(-_MAX_INTERVAL, min(_MAX_INTERVAL, interval))
            histogram[clamped + _INTERVAL_ZERO_BIN] += 1.0
        total = histogram.sum()
        if total > 0:
            histogram /= total
        return histogram

    # -- tonacja (Krumhansl-Schmuckler) -------------------------------------

    def _estimate_key(self, pitch_class_histogram: np.ndarray) -> str:
        """Wyznacza tonację dominującą metodą profili Krumhansla-Schmucklera.

        Dla każdej z 24 tonacji (12 durowych, 12 molowych) profil odniesienia
        jest cyklicznie obracany do toniki i korelowany (korelacja Pearsona) z
        histogramem klas wysokości. Wybierana jest tonacja o najwyższej
        korelacji; przy remisach decyduje stała kolejność (najpierw dur C..B,
        następnie moll C..B). Gdy histogram jest jednostajny (zerowa wariancja),
        zwracana jest neutralna tonacja ``"C major"``.

        Args:
            pitch_class_histogram: znormalizowany histogram klas wysokości.

        Returns:
            Etykieta tonacji, np. ``"C major"`` lub ``"A minor"``.
        """
        x = np.asarray(pitch_class_histogram, dtype=np.float64)
        if not np.all(np.isfinite(x)) or float(x.std()) == 0.0:
            # Brak wariancji (rozkład jednostajny) - korelacja nieokreślona.
            return "C major"

        best_key = "C major"
        best_corr = -math.inf
        for tonic in range(PITCH_CLASS_BINS):
            major_profile = np.roll(_KS_MAJOR_PROFILE, tonic)
            corr = self._pearson(x, major_profile)
            if corr > best_corr:
                best_corr = corr
                best_key = f"{_PITCH_CLASS_NAMES[tonic]} major"
        for tonic in range(PITCH_CLASS_BINS):
            minor_profile = np.roll(_KS_MINOR_PROFILE, tonic)
            corr = self._pearson(x, minor_profile)
            if corr > best_corr:
                best_corr = corr
                best_key = f"{_PITCH_CLASS_NAMES[tonic]} minor"
        return best_key

    @staticmethod
    def _pearson(x: np.ndarray, y: np.ndarray) -> float:
        """Współczynnik korelacji Pearsona dwóch wektorów (deterministyczny).

        Zwraca ``-inf`` gdy którykolwiek z wektorów ma zerową wariancję - taka
        wartość nigdy nie zostaje wybrana jako najlepsze dopasowanie tonacji.
        """
        xc = x - x.mean()
        yc = y - y.mean()
        denom = math.sqrt(float(xc @ xc) * float(yc @ yc))
        if denom == 0.0:
            return -math.inf
        return float(xc @ yc) / denom

    # -- proporcja pauz ------------------------------------------------------

    @staticmethod
    def _rest_ratio(repr_: InternalRepr, total_ticks: int) -> float:
        """Proporcja czasu bez brzmiącej nuty w przedziale [0, ``total_ticks``].

        Przedziały ``[tick, tick + duration_ticks)`` poszczególnych nut są
        scalane (z uwzględnieniem nakładania), a ich łączna długość stanowi czas
        aktywny. ``rest_ratio = 1 - czas_aktywny / total_ticks``. Cisza na
        początku utworu (nuty rozpoczynające się po ticku 0) jest wliczana do
        pauz. Wynik jest przycinany do [0, 1] (Wymaganie 2.1, *Property 3*).
        """
        if total_ticks <= 0:
            # Zerowy zakres czasowy - brak mierzalnego udziału pauz.
            return 0.0

        intervals = sorted(
            (note.tick, note.tick + note.duration_ticks)
            for note in repr_.notes
            if note.duration_ticks > 0
        )

        active_ticks = 0
        current_start: int | None = None
        current_end = 0
        for start, end in intervals:
            if current_start is None:
                current_start, current_end = start, end
            elif start <= current_end:
                current_end = max(current_end, end)
            else:
                active_ticks += current_end - current_start
                current_start, current_end = start, end
        if current_start is not None:
            active_ticks += current_end - current_start

        rest_ratio = 1.0 - active_ticks / total_ticks
        return float(min(1.0, max(0.0, rest_ratio)))

    # -- walidacja skończoności (Wymaganie 2.8) -----------------------------

    def _validated_scalar(
        self,
        value: float,
        name: str,
        fallback: float,
        problems: list[str],
    ) -> float:
        """Zwraca ``value`` jeśli skończone, w przeciwnym razie ``fallback``.

        Niefinitna wartość (``NaN``/``inf``) jest odnotowywana w ``problems``
        jako częściowy błąd numeryczny (Wymaganie 2.8).
        """
        if math.isfinite(value):
            return value
        problems.append(f"{name}={value!r} (zastąpiono wartością neutralną {fallback})")
        return fallback

    def _validated_array(
        self,
        array: np.ndarray,
        name: str,
        fallback: np.ndarray,
        problems: list[str],
    ) -> np.ndarray:
        """Zwraca ``array`` jeśli w pełni skończona, inaczej ``fallback``.

        Tablica zawierająca ``NaN``/``inf`` jest odnotowywana w ``problems``
        i zastępowana neutralną tablicą zastępczą (Wymaganie 2.8).
        """
        if np.all(np.isfinite(array)):
            return array
        problems.append(f"{name} zawiera niefinitne wartości (zastąpiono neutralnymi)")
        return fallback

    # -- pomocnicze ----------------------------------------------------------

    @staticmethod
    def _source_name(source: Path | str | None) -> str | None:
        """Normalizuje źródło do nazwy pliku (na potrzeby logowania)."""
        if source is None:
            return None
        if isinstance(source, Path):
            return source.name
        return str(source)
