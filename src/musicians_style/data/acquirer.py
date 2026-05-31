"""Akwizytor_Danych - pozyskanie *Zbioru_Stylu* z lokalnego katalogu lub z YouTube (Wymaganie 1).

Moduł implementuje :class:`DatasetAcquirer` - komponent odpowiedzialny za
zebranie plików MIDI artysty docelowego (z lokalnego katalogu lub fragmentów
audio z serwisu YouTube), ich walidację oraz utworzenie *Manifestu_Zbioru*
(:class:`~musicians_style.data.manifest.Manifest`).

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

Metoda :meth:`DatasetAcquirer.acquire_youtube` realizuje Wymaganie 1.6: dla listy
identyfikatorów utworów YouTube pobierane są **fragmenty audio o długości nie
większej niż 15 sekund** (zgodnie z ograniczeniami praw autorskich opisanymi w
korespondencji z promotorem). Pobieranie jest realizowane przez wstrzykiwalny
komponent :class:`YoutubeAudioDownloader` (domyślnie :class:`YtDlpAudioDownloader`
oparty o bibliotekę ``yt-dlp``); wyodrębnienie go za interfejsem umożliwia
testowanie akwizytora bez rzeczywistych połączeń sieciowych. Konwersja audio na
MIDI pozostaje **poza zakresem** pracy (brak wiarygodnego komponentu) - przy braku
wstrzykniętego konwertera konwersja jest pomijana z wpisem ostrzeżenia w logu
(log + skip), a pobrany fragment audio jest zapisywany w manifeście jako wpis o
źródle ``"youtube"``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

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
    "MAX_YOUTUBE_CLIP_S",
    "LOCAL_SOURCE",
    "YOUTUBE_SOURCE",
    "YoutubeClip",
    "YoutubeAudioDownloader",
    "AudioToMidiConverter",
    "YtDlpAudioDownloader",
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

#: Wartość pola ``source`` wpisu :class:`FileEntry` dla fragmentów z YouTube.
YOUTUBE_SOURCE = "youtube"

#: Twardy limit długości fragmentu audio pobieranego z YouTube w sekundach
#: (Wymaganie 1.6 oraz 7.3 - ograniczenia praw autorskich). Wartość ``max_duration_s``
#: przekazana do :meth:`DatasetAcquirer.acquire_youtube` jest przycinana do tej
#: granicy, niezależnie od żądanej przez wywołującego długości.
MAX_YOUTUBE_CLIP_S = 15

#: Rozszerzenia plików rozpoznawanych jako pliki MIDI (małymi literami).
_MIDI_SUFFIXES = (".mid", ".midi")

#: Domyślne tempo MIDI (mikrosekundy na ćwierćnutę) = 120 BPM (specyfikacja MIDI 1.0).
_DEFAULT_TEMPO_US = 500000

#: Domyślna rozdzielczość czasowa używana, gdy ``ticks_per_beat`` jest niepoprawne.
_DEFAULT_TICKS_PER_BEAT = 480


@dataclass(frozen=True)
class YoutubeClip:
    """Wynik pobrania pojedynczego fragmentu audio z YouTube (Wymaganie 1.6).

    Atrybuty:
        video_id: identyfikator utworu YouTube, którego dotyczy fragment.
        path: ścieżka zapisanego na dysku pliku audio fragmentu.
        duration_s: rzeczywista długość pobranego fragmentu w sekundach
            (powinna być ``<=`` żądanego limitu ``max_duration_s``).
    """

    video_id: str
    path: Path
    duration_s: float


@runtime_checkable
class YoutubeAudioDownloader(Protocol):
    """Interfejs komponentu pobierającego fragmenty audio z YouTube.

    Wyodrębnienie pobierania za tym protokołem pozwala wstrzyknąć atrapę
    (mock) w testach i tym samym weryfikować logikę :class:`DatasetAcquirer`
    **bez rzeczywistych połączeń sieciowych**. Domyślną, produkcyjną
    implementacją jest :class:`YtDlpAudioDownloader` oparty o bibliotekę
    ``yt-dlp``.
    """

    def download_clip(
        self,
        video_id: str,
        output_dir: Path,
        max_duration_s: int,
    ) -> YoutubeClip:
        """Pobiera fragment audio utworu ``video_id`` o długości ``<= max_duration_s``.

        Implementacja SHALL ograniczać długość pobieranego fragmentu do nie
        więcej niż ``max_duration_s`` sekund (Wymaganie 1.6).

        Args:
            video_id: identyfikator utworu YouTube.
            output_dir: katalog, w którym zapisywany jest pobrany fragment.
            max_duration_s: maksymalna długość fragmentu w sekundach.

        Returns:
            :class:`YoutubeClip` opisujący pobrany fragment.

        Raises:
            Exception: dowolny błąd pobierania (sieć, niedostępny utwór,
                błąd ``yt-dlp``); akwizytor przechwytuje go i pomija dany
                identyfikator z wpisem w logu.
        """
        ...


@runtime_checkable
class AudioToMidiConverter(Protocol):
    """Interfejs (opcjonalnego) konwertera audio → MIDI.

    Konwersja audio na MIDI pozostaje **poza zakresem** niniejszej pracy
    inżynierskiej (brak wiarygodnego komponentu - decyzja projektowa). Protokół
    zdefiniowano, aby umożliwić wstrzyknięcie konwertera w przyszłości lub w
    testach. Gdy konwerter nie jest dostępny, :meth:`DatasetAcquirer.acquire_youtube`
    pomija konwersję z wpisem ostrzeżenia w logu (log + skip).
    """

    def convert(self, audio_path: Path, output_dir: Path) -> Path:
        """Konwertuje plik audio na plik MIDI i zwraca ścieżkę wyniku."""
        ...


class YtDlpAudioDownloader:
    """Domyślna implementacja :class:`YoutubeAudioDownloader` oparta o ``yt-dlp``.

    Klasa pobiera wyłącznie wstępny fragment audio o długości nie większej niż
    ``max_duration_s`` sekund, korzystając z opcji ``download_ranges`` biblioteki
    ``yt-dlp``. Import ``yt-dlp`` jest **leniwy** (wykonywany dopiero w
    :meth:`download_clip`), dzięki czemu sam moduł akwizytora ładuje się bez tej
    zależności, a testy mogą używać atrapy bez instalowania ani uruchamiania
    ``yt-dlp``.

    Uwaga prawna (Wymaganie 1.6): pobieranie fragmentów dłuższych niż
    :data:`MAX_YOUTUBE_CLIP_S` sekund jest niedozwolone - długość żądana jest
    dodatkowo przycinana w :meth:`DatasetAcquirer.acquire_youtube`.
    """

    def download_clip(
        self,
        video_id: str,
        output_dir: Path,
        max_duration_s: int,
    ) -> YoutubeClip:
        """Pobiera fragment ``[0, max_duration_s]`` audio utworu przez ``yt-dlp``.

        Args:
            video_id: identyfikator utworu YouTube.
            output_dir: katalog docelowy pobranego pliku audio.
            max_duration_s: maksymalna długość fragmentu w sekundach.

        Returns:
            :class:`YoutubeClip` opisujący pobrany fragment.

        Raises:
            RuntimeError: gdy biblioteka ``yt-dlp`` nie jest dostępna.
            Exception: błędy pobierania zgłaszane przez ``yt-dlp``.
        """
        try:
            import yt_dlp  # noqa: PLC0415 - leniwy import zewnętrznej zależności
        except ImportError as exc:  # pragma: no cover - środowisko bez yt-dlp
            raise RuntimeError(
                "Biblioteka 'yt-dlp' jest wymagana do pobierania fragmentów YouTube."
            ) from exc

        output_dir.mkdir(parents=True, exist_ok=True)
        clip_seconds = float(max(0, max_duration_s))
        output_path = output_dir / f"{video_id}.m4a"

        def _ranges(_info: object, _ydl: object) -> list[dict[str, float]]:
            # Pobranie wyłącznie wstępnego fragmentu [0, clip_seconds] (Wymaganie 1.6).
            return [{"start_time": 0.0, "end_time": clip_seconds}]

        options = {
            "format": "bestaudio/best",
            "outtmpl": str(output_dir / "%(id)s.%(ext)s"),
            "quiet": True,
            "noprogress": True,
            "download_ranges": _ranges,
            "force_keyframes_at_cuts": True,
        }

        with yt_dlp.YoutubeDL(options) as ydl:  # pragma: no cover - sieć/zewnętrzne
            ydl.download([f"https://www.youtube.com/watch?v={video_id}"])

        return YoutubeClip(
            video_id=video_id,
            path=output_path,
            duration_s=clip_seconds,
        )


class DatasetAcquirer:
    """Pozyskuje *Zbiór_Stylu* z lokalnego katalogu i tworzy *Manifest_Zbioru*.

    Klasa deleguje kontrolę zgodności ze SMF do
    :class:`~musicians_style.midi.parser.MidiParser`. Parser można wstrzyknąć
    przez konstruktor (ułatwia to testy izolowane - Wymaganie 1.3/1.4), a w razie
    jego braku tworzona jest domyślna instancja.

    Instancja jest bezstanowa względem pojedynczego pozyskania - tę samą instancję
    można wielokrotnie użyć dla różnych katalogów.
    """

    def __init__(
        self,
        parser: MidiParser | None = None,
        youtube_downloader: YoutubeAudioDownloader | None = None,
        audio_to_midi: AudioToMidiConverter | None = None,
    ) -> None:
        """Inicjalizuje akwizytor.

        Args:
            parser: instancja :class:`MidiParser` używana do walidacji struktury
                SMF. Gdy ``None``, tworzona jest domyślna instancja.
            youtube_downloader: komponent pobierający fragmenty audio z YouTube
                (Wymaganie 1.6). Gdy ``None``, używana jest domyślna
                implementacja :class:`YtDlpAudioDownloader`. Wstrzyknięcie atrapy
                pozwala testować :meth:`acquire_youtube` bez połączeń sieciowych.
            audio_to_midi: opcjonalny konwerter audio → MIDI. Konwersja audio na
                MIDI jest poza zakresem pracy, więc domyślnie ``None`` -
                w takim przypadku konwersja jest pomijana z wpisem w logu
                (log + skip).
        """
        self._parser = parser if parser is not None else MidiParser()
        self._youtube_downloader = youtube_downloader
        self._audio_to_midi = audio_to_midi
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

    def acquire_youtube(
        self,
        ids: list[str],
        max_duration_s: int = MAX_YOUTUBE_CLIP_S,
        artist_id: str | None = None,
        output_dir: Path | str | None = None,
    ) -> Manifest:
        """Pozyskuje fragmenty audio z YouTube i tworzy *Manifest_Zbioru* (Wymaganie 1.6).

        Dla każdego identyfikatora z ``ids`` pobierany jest **fragment audio o
        długości nie większej niż 15 sekund** (zgodnie z ograniczeniami praw
        autorskich opisanymi w korespondencji z promotorem). Żądana wartość
        ``max_duration_s`` jest przycinana do twardego limitu
        :data:`MAX_YOUTUBE_CLIP_S` - nawet jeśli wywołujący zażąda dłuższego
        fragmentu, pobrane zostanie nie więcej niż 15 sekund.

        Konwersja pobranego audio na MIDI pozostaje **poza zakresem** pracy
        (brak wiarygodnego komponentu). Gdy konwerter audio → MIDI nie został
        wstrzyknięty (:attr:`_audio_to_midi` is ``None``), konwersja jest
        pomijana z wpisem ostrzeżenia w logu (log + skip), a w manifeście
        zapisywany jest wpis opisujący pobrany fragment audio (źródło
        ``"youtube"``). Gdy konwerter jest dostępny, wynikowy plik MIDI jest
        walidowany tak jak plik lokalny (struktura SMF + długość).

        Pobieranie jest delegowane do wstrzykiwalnego komponentu
        :class:`YoutubeAudioDownloader`, dzięki czemu metodę można testować bez
        rzeczywistych połączeń sieciowych. Błąd pobierania pojedynczego
        identyfikatora skutkuje jego pominięciem z wpisem w logu i kontynuacją
        przetwarzania pozostałych (analogicznie do Wymagania 1.3 dla plików
        lokalnych).

        Args:
            ids: lista identyfikatorów utworów YouTube.
            max_duration_s: żądana maksymalna długość fragmentu w sekundach;
                przycinana do :data:`MAX_YOUTUBE_CLIP_S` (15 s).
            artist_id: identyfikator artysty; gdy ``None``, używana jest wartość
                ``"youtube"``.
            output_dir: katalog na pobrane fragmenty audio (i ewentualne pliki
                MIDI); gdy ``None``, używany jest podkatalog ``data/youtube/<artist_id>``.

        Returns:
            :class:`Manifest` z wpisami dla pomyślnie pobranych fragmentów.

        Raises:
            EmptyDatasetError: gdy żaden fragment nie został pomyślnie pozyskany
                (Wymaganie 1.5).
        """
        resolved_artist = artist_id if artist_id else YOUTUBE_SOURCE

        # Wymaganie 1.6: twardy limit 15 s niezależnie od żądania wywołującego.
        clip_seconds = self._clamp_clip_duration(max_duration_s)

        target_dir = (
            Path(output_dir)
            if output_dir is not None
            else Path("data") / YOUTUBE_SOURCE / resolved_artist
        )

        downloader = (
            self._youtube_downloader
            if self._youtube_downloader is not None
            else YtDlpAudioDownloader()
        )

        entries: list[FileEntry] = []
        n_download_failed = 0
        n_conversion_skipped = 0

        for video_id in ids:
            try:
                clip = downloader.download_clip(video_id, target_dir, clip_seconds)
            except Exception as exc:  # noqa: BLE001 - yt-dlp sygnalizuje błędy różnymi typami
                n_download_failed += 1
                self._log.warning(
                    "youtube clip rejected",
                    video_id=video_id,
                    reason=f"błąd pobierania fragmentu audio: {exc}",
                )
                continue

            # Wymaganie 1.6: defensywne przycięcie zaraportowanej długości do limitu.
            clip_duration = min(float(clip.duration_s), float(clip_seconds))

            if self._audio_to_midi is None:
                # Konwersja audio → MIDI poza zakresem pracy: log + skip.
                n_conversion_skipped += 1
                self._log.warning(
                    "audio-to-midi conversion skipped (out of scope)",
                    video_id=video_id,
                    audio_path=str(clip.path),
                    duration_s=clip_duration,
                    reason=(
                        "konwersja audio→MIDI jest poza zakresem pracy "
                        "(brak wiarygodnego komponentu); zapisano wpis audio"
                    ),
                )
                entries.append(
                    FileEntry(
                        path=str(clip.path),
                        sha256=self._safe_sha256(clip.path),
                        duration_s=clip_duration,
                        tracks=0,
                        source=YOUTUBE_SOURCE,
                    )
                )
                continue

            # Konwerter dostępny: konwertuj i waliduj wynik jak plik lokalny.
            try:
                midi_path = self._audio_to_midi.convert(clip.path, target_dir)
                duration_s, tracks = self._probe_local_file(Path(midi_path))
            except Exception as exc:  # noqa: BLE001 - konwerter/parser sygnalizują błędy różnymi typami
                n_conversion_skipped += 1
                self._log.warning(
                    "youtube clip rejected",
                    video_id=video_id,
                    reason=f"błąd konwersji audio→MIDI lub walidacji: {exc}",
                )
                continue

            entries.append(
                FileEntry(
                    path=str(midi_path),
                    sha256=self._safe_sha256(Path(midi_path)),
                    duration_s=min(duration_s, float(clip_seconds)),
                    tracks=tracks,
                    source=YOUTUBE_SOURCE,
                )
            )

        valid_count = len(entries)

        # Wymaganie 1.5: brak poprawnych fragmentów - wyjątek z opisem przyczyny.
        if valid_count == 0:
            reason = (
                f"Brak pozyskanych fragmentów YouTube dla artysty "
                f"'{resolved_artist}': zażądano {len(ids)} identyfikatorów, "
                f"{n_download_failed} nie udało się pobrać, "
                f"{n_conversion_skipped} pominięto na etapie konwersji audio→MIDI."
            )
            self._log.error(
                "empty dataset",
                artist_id=resolved_artist,
                reason=reason,
            )
            raise EmptyDatasetError(reason)

        # Wymaganie 1.2: liczba poprawnych fragmentów poniżej zalecanej liczności.
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
            "youtube dataset acquired",
            artist_id=resolved_artist,
            valid_files=valid_count,
            download_failed=n_download_failed,
            conversion_skipped=n_conversion_skipped,
            max_duration_s=clip_seconds,
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
    def _clamp_clip_duration(max_duration_s: int) -> int:
        """Przycina żądaną długość fragmentu do limitu praw autorskich (Wymaganie 1.6).

        Wynik mieści się w przedziale ``[0, MAX_YOUTUBE_CLIP_S]``. Wartości
        ujemne są traktowane jako ``0`` (brak fragmentu), a wartości większe niż
        :data:`MAX_YOUTUBE_CLIP_S` są przycinane do 15 sekund.
        """
        clamped = min(int(max_duration_s), MAX_YOUTUBE_CLIP_S)
        return max(clamped, 0)

    @staticmethod
    def _safe_sha256(path: Path) -> str:
        """Zwraca SHA-256 zawartości pliku lub pusty łańcuch, gdy plik jest niedostępny.

        Pozwala zbudować wpis manifestu nawet wtedy, gdy atrapa pobierania w
        testach nie utworzyła rzeczywistego pliku na dysku (brak twardej
        zależności od stanu systemu plików).
        """
        try:
            return compute_sha256(path)
        except OSError:
            return ""

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
