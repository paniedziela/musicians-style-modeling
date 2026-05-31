"""Akwizytor_Danych - pozyskanie *Zbioru_Stylu* z lokalnego katalogu (Wymaganie 1).

Moduł implementuje :class:`DatasetAcquirer` - komponent odpowiedzialny za
zebranie plików MIDI artysty docelowego z lokalnego katalogu, ich walidację oraz
utworzenie *Manifestu_Zbioru* (:class:`~musicians_style.data.manifest.Manifest`).

Metoda :meth:`DatasetAcquirer.acquire_local` realizuje kryteria akceptacji
Wymagania 1:

* **1.1 / 1.2** - akceptacja katalogu z co najmniej 30 poprawnymi plikami MIDI;
  gdy liczba poprawnych plików jest mniejsza niż :data:`MIN_RECOMMENDED_FILES`,
  zapisywane jest ostrzeżenie wskazujące minimalną zalecaną liczność, a
  przetwarzanie jest kontynuowane.
* **1.3** - plik niezgodny ze specyfikacją *Standard MIDI File* (SMF) jest
  pomijany; do logu trafia wpis z nazwą pliku i przyczyną odrzucenia,
  a przetwarzanie pozostałych plików jest kontynuowane. Zgodność ze SMF jest
  sprawdzana przy użyciu :meth:`~musicians_style.midi.parser.MidiParser.validate`.
* **1.4** - akceptowane są pliki o czasie trwania w przedziale
  ``[MIN_DURATION_S, MAX_DURATION_S]`` (czyli ``[5 s, 30 min]``); pliki krótsze
  lub dłuższe są odrzucane z wpisem w logu zawierającym nazwę pliku i przyczynę.
* **1.5** - jeśli po walidacji wszystkich plików liczba poprawnych plików wynosi
  zero (niezależnie od przyczyny), zgłaszany jest wyjątek
  :class:`~musicians_style.errors.EmptyDatasetError` z opisem przyczyny.
  Sam akwizytor wyłącznie zgłasza wyjątek - zakończenie procesu niezerowym kodem
  wyjścia (``exit_code == 2``) jest zadaniem warstwy CLI.

Kolejność walidacji jest spójna z Wymaganiem 5.5: kontrola struktury SMF
(``MidiParser.validate``) odbywa się **przed** jakimkolwiek przetwarzaniem treści,
w tym przed obliczeniem długości czasowej pliku.
"""

from __future__ import annotations

from pathlib import Path

from mido import MidiFile, merge_tracks, tick2second

from ..errors import EmptyDatasetError, MidiValidationError
from ..logging import get_logger
from ..midi.parser import MidiParser
from .manifest import (
    FileEntry,
    Manifest,
    compute_sha256,
    current_git_commit,
)

__all__ = [
    "DatasetAcquirer",
    "MIN_RECOMMENDED_FILES",
    "MIN_DURATION_S",
    "MAX_DURATION_S",
    "LOCAL_SOURCE",
]

#: Minimalna zalecana liczność *Zbioru_Stylu* dla jednego artysty (Wymaganie 1.1).
#: Mniejsza liczba poprawnych plików skutkuje ostrzeżeniem (Wymaganie 1.2).
MIN_RECOMMENDED_FILES = 30

#: Minimalny dopuszczalny czas trwania pliku MIDI w sekundach (Wymaganie 1.4).
MIN_DURATION_S = 5.0

#: Maksymalny dopuszczalny czas trwania pliku MIDI w sekundach (30 minut, Wymaganie 1.4).
MAX_DURATION_S = 30.0 * 60.0

#: Wartość pola ``source`` wpisu :class:`FileEntry` dla plików lokalnych.
LOCAL_SOURCE = "local"

#: Rozszerzenia plików rozpoznawanych jako pliki MIDI (małymi literami).
_MIDI_SUFFIXES = (".mid", ".midi")

#: Domyślne tempo MIDI (mikrosekundy na ćwierćnutę) = 120 BPM (specyfikacja MIDI 1.0).
_DEFAULT_TEMPO_US = 500000

#: Domyślna rozdzielczość czasowa używana, gdy ``ticks_per_beat`` jest niepoprawne.
_DEFAULT_TICKS_PER_BEAT = 480


class DatasetAcquirer:
    """Pozyskuje *Zbiór_Stylu* z lokalnego katalogu i tworzy *Manifest_Zbioru*.

    Klasa deleguje kontrolę zgodności ze SMF do
    :class:`~musicians_style.midi.parser.MidiParser`. Parser można wstrzyknąć
    przez konstruktor (ułatwia to testy izolowane - Wymaganie 1.3/1.4), a w razie
    jego braku tworzona jest domyślna instancja.

    Instancja jest bezstanowa względem pojedynczego pozyskania - tę samą instancję
    można wielokrotnie użyć dla różnych katalogów.
    """

    def __init__(self, parser: MidiParser | None = None) -> None:
        """Inicjalizuje akwizytor.

        Args:
            parser: instancja :class:`MidiParser` używana do walidacji struktury
                SMF. Gdy ``None``, tworzona jest domyślna instancja.
        """
        self._parser = parser if parser is not None else MidiParser()
        self._log = get_logger("acquirer")

    # -- API publiczne -------------------------------------------------------

    def acquire_local(
        self,
        directory: Path | str,
        artist_id: str | None = None,
    ) -> Manifest:
        """Pozyskuje i waliduje pliki MIDI z lokalnego katalogu (Wymaganie 1).

        Przebieg:

        1. Wyszukanie kandydujących plików MIDI (rozszerzenia ``.mid`` / ``.midi``)
           w katalogu, w deterministycznej (alfabetycznej) kolejności.
        2. Dla każdego pliku: kontrola struktury SMF (Wymaganie 1.3) **przed**
           obliczeniem długości (Wymaganie 5.5), a następnie kontrola czasu
           trwania w przedziale ``[MIN_DURATION_S, MAX_DURATION_S]`` (Wymaganie 1.4).
        3. Ostrzeżenie, gdy liczba poprawnych plików jest mniejsza niż
           :data:`MIN_RECOMMENDED_FILES` (Wymaganie 1.2).
        4. Zgłoszenie :class:`EmptyDatasetError`, gdy liczba poprawnych plików
           wynosi zero (Wymaganie 1.5).

        Args:
            directory: ścieżka katalogu zawierającego pliki MIDI artysty.
            artist_id: identyfikator artysty; gdy ``None``, używana jest nazwa
                katalogu.

        Returns:
            :class:`Manifest` zawierający wpisy wyłącznie dla poprawnych plików.

        Raises:
            EmptyDatasetError: gdy po walidacji liczba poprawnych plików == 0
                (Wymaganie 1.5).
        """
        directory = Path(directory)
        resolved_artist = artist_id if artist_id else directory.name or "unknown"

        candidates = self._discover_midi_files(directory)

        entries: list[FileEntry] = []
        n_skipped_invalid = 0
        n_rejected_duration = 0

        for path in candidates:
            try:
                duration_s, tracks = self._probe_local_file(path)
            except MidiValidationError as exc:
                # Wymaganie 1.3: plik niezgodny ze SMF - pominięcie + log, kontynuacja.
                n_skipped_invalid += 1
                self._log.warning(
                    "file rejected",
                    file=path.name,
                    reason=exc.message,
                )
                continue

            if not (MIN_DURATION_S <= duration_s <= MAX_DURATION_S):
                # Wymaganie 1.4: czas trwania spoza [5 s, 30 min] - odrzucenie + log.
                n_rejected_duration += 1
                self._log.warning(
                    "file rejected",
                    file=path.name,
                    reason=(
                        f"czas trwania {duration_s:.3f} s poza dopuszczalnym "
                        f"przedziałem [{MIN_DURATION_S:.0f} s, {MAX_DURATION_S:.0f} s]"
                    ),
                )
                continue

            entries.append(
                FileEntry(
                    path=self._relative_path(path, directory),
                    sha256=compute_sha256(path),
                    duration_s=duration_s,
                    tracks=tracks,
                    source=LOCAL_SOURCE,
                )
            )

        valid_count = len(entries)

        # Wymaganie 1.5: brak poprawnych plików - wyjątek z opisem przyczyny.
        if valid_count == 0:
            reason = self._empty_reason(
                directory=directory,
                n_candidates=len(candidates),
                n_skipped_invalid=n_skipped_invalid,
                n_rejected_duration=n_rejected_duration,
            )
            self._log.error(
                "empty dataset",
                directory=str(directory),
                reason=reason,
            )
            raise EmptyDatasetError(reason)

        # Wymaganie 1.2: liczba poprawnych plików poniżej zalecanej liczności.
        if valid_count < MIN_RECOMMENDED_FILES:
            self._log.warning(
                "dataset below recommended size",
                artist_id=resolved_artist,
                valid_files=valid_count,
                min_required=MIN_RECOMMENDED_FILES,
            )

        manifest = Manifest(
            artist_id=resolved_artist,
            files=entries,
            git_commit=current_git_commit(),
        )
        self._log.info(
            "dataset acquired",
            artist_id=resolved_artist,
            valid_files=valid_count,
            skipped_invalid=n_skipped_invalid,
            rejected_duration=n_rejected_duration,
        )
        return manifest

    # -- walidacja i metadane pojedynczego pliku -----------------------------

    def _probe_local_file(self, path: Path) -> tuple[float, int]:
        """Waliduje strukturę SMF i zwraca ``(duration_s, tracks)``.

        Kontrola struktury (``MidiParser.validate``) jest wykonywana przed
        obliczeniem długości czasowej (Wymaganie 5.5). Błędy dekodowania treści,
        które nie zostały wykryte na poziomie struktury chunków, są opakowywane w
        :class:`MidiValidationError`, tak aby warstwa wywołująca mogła je
        jednolicie potraktować jako niezgodność ze SMF (Wymaganie 1.3).

        Args:
            path: ścieżka pliku MIDI.

        Returns:
            Krotka ``(duration_s, tracks)`` - długość w sekundach i liczba ścieżek.

        Raises:
            MidiValidationError: gdy struktura lub treść pliku jest niezgodna ze SMF.
        """
        # Wymaganie 1.3 / 5.5: walidacja struktury przed przetwarzaniem treści.
        self._parser.validate(path)

        try:
            midi_file = MidiFile(path)
            tracks = len(midi_file.tracks)
            duration_s = _duration_seconds(midi_file)
        except MidiValidationError:
            raise
        except Exception as exc:  # noqa: BLE001 - mido sygnalizuje błędy różnymi typami
            raise MidiValidationError(
                f"błąd podczas dekodowania treści MIDI: {exc}",
                file=path,
            ) from exc

        return duration_s, tracks

    # -- pomocnicze ----------------------------------------------------------

    @staticmethod
    def _discover_midi_files(directory: Path) -> list[Path]:
        """Zwraca posortowaną listę kandydujących plików MIDI w katalogu.

        Wyszukiwanie jest płaskie (bez rekurencji) i obejmuje pliki o
        rozszerzeniu ``.mid`` lub ``.midi`` (niezależnie od wielkości liter).
        Brak katalogu lub nie-katalog skutkuje pustą listą - przypadek ten jest
        następnie obsłużony jako pusty zbiór (Wymaganie 1.5).
        """
        if not directory.is_dir():
            return []
        files = [
            entry
            for entry in directory.iterdir()
            if entry.is_file() and entry.suffix.lower() in _MIDI_SUFFIXES
        ]
        return sorted(files, key=lambda p: p.name)

    @staticmethod
    def _relative_path(path: Path, directory: Path) -> str:
        """Zwraca ścieżkę pliku względem katalogu zbioru (lub nazwę pliku)."""
        try:
            return str(path.relative_to(directory))
        except ValueError:
            return path.name

    @staticmethod
    def _empty_reason(
        directory: Path,
        n_candidates: int,
        n_skipped_invalid: int,
        n_rejected_duration: int,
    ) -> str:
        """Buduje opis przyczyny pustego *Zbioru_Stylu* (Wymaganie 1.5)."""
        if n_candidates == 0:
            return (
                f"Katalog '{directory}' nie zawiera plików MIDI "
                f"(rozszerzenia {', '.join(_MIDI_SUFFIXES)})."
            )
        return (
            f"Brak poprawnych plików MIDI w katalogu '{directory}': "
            f"znaleziono {n_candidates} kandydatów, "
            f"{n_skipped_invalid} pominięto jako niezgodne ze SMF, "
            f"{n_rejected_duration} odrzucono z powodu czasu trwania poza "
            f"przedziałem [{MIN_DURATION_S:.0f} s, {MAX_DURATION_S:.0f} s]."
        )


def _duration_seconds(midi_file: MidiFile) -> float:
    """Oblicza długość pliku MIDI w sekundach z uwzględnieniem zmian tempa.

    Funkcja scala ścieżki (``merge_tracks``) i sumuje czas wszystkich komunikatów,
    aktualizując bieżące tempo na podstawie meta-zdarzeń ``set_tempo``. W
    przeciwieństwie do ``MidiFile.length`` działa dla wszystkich formatów SMF
    (w tym formatu 2) bez zgłaszania wyjątku. Brak meta-zdarzenia tempa oznacza
    domyślne 120 BPM zgodnie ze specyfikacją MIDI 1.0.

    Args:
        midi_file: wczytany obiekt ``mido.MidiFile``.

    Returns:
        Długość pliku w sekundach (>= 0.0).
    """
    ticks_per_beat = midi_file.ticks_per_beat or _DEFAULT_TICKS_PER_BEAT
    if ticks_per_beat <= 0:
        ticks_per_beat = _DEFAULT_TICKS_PER_BEAT

    tempo = _DEFAULT_TEMPO_US
    seconds = 0.0
    for message in merge_tracks(midi_file.tracks):
        seconds += tick2second(int(message.time), ticks_per_beat, tempo)
        if message.type == "set_tempo":
            tempo = message.tempo
    return seconds
