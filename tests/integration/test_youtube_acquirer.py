"""Testy integracyjne :meth:`DatasetAcquirer.acquire_youtube` (zadanie 5.3).

Weryfikują Wymaganie 1.6 - pozyskiwanie fragmentów audio z YouTube o długości
nie większej niż 15 sekund, z konwersją audio→MIDI uznaną za poza zakresem pracy
(log + skip). Pobieranie jest realizowane przez **atrapę** (mock) komponentu
:class:`YoutubeAudioDownloader`, dzięki czemu testy NIE wykonują rzeczywistych
połączeń sieciowych ani nie uruchamiają ``yt-dlp``.

Pokrywane scenariusze:

* **1.6** - długość każdego fragmentu jest przycinana do ``<= 15 s`` (twardy
  limit :data:`MAX_YOUTUBE_CLIP_S`), nawet gdy wywołujący zażąda więcej,
* konwersja audio→MIDI poza zakresem → wpis ostrzeżenia w logu + pominięcie
  konwersji + wpis audio w manifeście (źródło ``"youtube"``),
* błąd pobierania pojedynczego identyfikatora → pominięcie + log + kontynuacja,
* **1.5** - brak pozyskanych fragmentów → ``EmptyDatasetError`` (exit code 2),
* wstrzyknięty konwerter audio→MIDI → walidacja wynikowego pliku jak lokalnego,
* **1.2** - ostrzeżenie przy liczbie fragmentów < 30.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from mido import Message, MetaMessage, MidiFile, MidiTrack

from musicians_style.data.acquirer import (
    MAX_YOUTUBE_CLIP_S,
    MIN_RECOMMENDED_FILES,
    YOUTUBE_SOURCE,
    DatasetAcquirer,
    YoutubeClip,
)
from musicians_style.data.manifest import Manifest
from musicians_style.errors import EmptyDatasetError

_TICKS_PER_BEAT = 480
_TICKS_PER_SECOND = 960


# -- atrapy (mocki) ----------------------------------------------------------


class _RecordingLogger:
    """Minimalny logger zapisujący wywołania do listy (na potrzeby asercji)."""

    def __init__(self, sink: list[dict]) -> None:
        self._sink = sink

    def _record(self, level: str, event: str, **kwargs: object) -> None:
        self._sink.append({"level": level, "event": event, "kwargs": kwargs})

    def info(self, event: str, **kwargs: object) -> None:
        self._record("info", event, **kwargs)

    def warning(self, event: str, **kwargs: object) -> None:
        self._record("warning", event, **kwargs)

    def error(self, event: str, **kwargs: object) -> None:
        self._record("error", event, **kwargs)


class _FakeDownloader:
    """Atrapa pobierania YouTube - tworzy lokalny plik audio bez sieci.

    Zapisuje rzeczywistą długość fragmentu równą przekazanemu ``max_duration_s``,
    aby umożliwić weryfikację przycięcia do limitu praw autorskich (Wymaganie 1.6).
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def download_clip(
        self, video_id: str, output_dir: Path, max_duration_s: int
    ) -> YoutubeClip:
        self.calls.append((video_id, max_duration_s))
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"{video_id}.m4a"
        path.write_bytes(b"fake-audio-bytes")
        return YoutubeClip(
            video_id=video_id,
            path=path,
            duration_s=float(max_duration_s),
        )


class _OverlongDownloader:
    """Atrapa zwracająca długość przekraczającą żądany limit (test defensywnego cięcia)."""

    def download_clip(
        self, video_id: str, output_dir: Path, max_duration_s: int
    ) -> YoutubeClip:
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"{video_id}.m4a"
        path.write_bytes(b"fake-audio-bytes")
        # Złośliwie zwraca 999 s - akwizytor MUSI przyciąć do limitu.
        return YoutubeClip(video_id=video_id, path=path, duration_s=999.0)


class _FailingDownloader:
    """Atrapa zgłaszająca błąd dla wskazanych identyfikatorów."""

    def __init__(self, failing_ids: set[str]) -> None:
        self._failing = failing_ids

    def download_clip(
        self, video_id: str, output_dir: Path, max_duration_s: int
    ) -> YoutubeClip:
        if video_id in self._failing:
            raise RuntimeError(f"network error for {video_id}")
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"{video_id}.m4a"
        path.write_bytes(b"fake-audio-bytes")
        return YoutubeClip(
            video_id=video_id, path=path, duration_s=float(max_duration_s)
        )


def _write_midi(path: Path, duration_s: float) -> None:
    """Zapisuje poprawny plik SMF o zadanej długości (dla atrapy konwertera)."""
    mf = MidiFile(type=1, ticks_per_beat=_TICKS_PER_BEAT)
    conductor = MidiTrack()
    conductor.append(MetaMessage("time_signature", numerator=4, denominator=4, time=0))
    mf.tracks.append(conductor)
    note_ticks = max(1, int(round(duration_s * _TICKS_PER_SECOND)))
    note_track = MidiTrack()
    note_track.append(Message("note_on", note=60, velocity=100, time=0))
    note_track.append(Message("note_off", note=60, velocity=0, time=note_ticks))
    mf.tracks.append(note_track)
    mf.save(str(path))


class _FakeConverter:
    """Atrapa konwertera audio→MIDI tworząca poprawny plik MIDI o długości 10 s."""

    def convert(self, audio_path: Path, output_dir: Path) -> Path:
        midi_path = output_dir / (audio_path.stem + ".mid")
        _write_midi(midi_path, duration_s=10.0)
        return midi_path


# -- Wymaganie 1.6: domyślny limit 15 s i przekazanie go do pobierania -------


def test_default_max_duration_is_clamped_to_15s(tmp_path: Path) -> None:
    downloader = _FakeDownloader()
    acquirer = DatasetAcquirer(youtube_downloader=downloader)

    manifest = acquirer.acquire_youtube(
        ["vid1", "vid2"], artist_id="artist", output_dir=tmp_path
    )

    assert isinstance(manifest, Manifest)
    assert len(manifest.files) == 2
    # Limit przekazany do pobierania to dokładnie 15 s.
    assert {max_d for _, max_d in downloader.calls} == {MAX_YOUTUBE_CLIP_S}
    for entry in manifest.files:
        assert entry.source == YOUTUBE_SOURCE
        assert entry.duration_s <= MAX_YOUTUBE_CLIP_S


def test_requested_duration_above_limit_is_clamped(tmp_path: Path) -> None:
    downloader = _FakeDownloader()
    acquirer = DatasetAcquirer(youtube_downloader=downloader)

    acquirer.acquire_youtube(
        ["vid1"], max_duration_s=60, artist_id="artist", output_dir=tmp_path
    )

    # Mimo żądania 60 s, do pobierania trafia limit 15 s (Wymaganie 1.6).
    assert downloader.calls == [("vid1", MAX_YOUTUBE_CLIP_S)]


def test_reported_duration_defensively_clamped(tmp_path: Path) -> None:
    acquirer = DatasetAcquirer(youtube_downloader=_OverlongDownloader())

    manifest = acquirer.acquire_youtube(
        ["vid1"], artist_id="artist", output_dir=tmp_path
    )

    # Nawet gdy downloader zaraportuje 999 s, manifest nie przekracza 15 s.
    assert manifest.files[0].duration_s <= MAX_YOUTUBE_CLIP_S


def test_smaller_requested_duration_is_respected(tmp_path: Path) -> None:
    downloader = _FakeDownloader()
    acquirer = DatasetAcquirer(youtube_downloader=downloader)

    acquirer.acquire_youtube(
        ["vid1"], max_duration_s=10, artist_id="artist", output_dir=tmp_path
    )

    # Wartość mniejsza niż limit jest zachowana bez przycinania.
    assert downloader.calls == [("vid1", 10)]


# -- konwersja audio→MIDI poza zakresem: log + skip --------------------------


def test_audio_to_midi_conversion_skipped_logs_warning(tmp_path: Path) -> None:
    acquirer = DatasetAcquirer(youtube_downloader=_FakeDownloader())
    captured: list[dict] = []
    acquirer._log = _RecordingLogger(captured)  # type: ignore[attr-defined]

    manifest = acquirer.acquire_youtube(
        ["vid1"], artist_id="artist", output_dir=tmp_path
    )

    # Wpis audio trafia do manifestu (źródło youtube, brak ścieżek MIDI).
    assert len(manifest.files) == 1
    assert manifest.files[0].source == YOUTUBE_SOURCE
    assert manifest.files[0].tracks == 0
    # Zapisano ostrzeżenie o pominięciu konwersji (out of scope).
    skipped = [
        c
        for c in captured
        if c["event"] == "audio-to-midi conversion skipped (out of scope)"
    ]
    assert len(skipped) == 1
    assert skipped[0]["kwargs"]["video_id"] == "vid1"


# -- błąd pobierania pojedynczego identyfikatora: skip + kontynuacja ---------


def test_download_failure_is_skipped_and_continues(tmp_path: Path) -> None:
    acquirer = DatasetAcquirer(
        youtube_downloader=_FailingDownloader(failing_ids={"bad"})
    )
    captured: list[dict] = []
    acquirer._log = _RecordingLogger(captured)  # type: ignore[attr-defined]

    manifest = acquirer.acquire_youtube(
        ["good1", "bad", "good2"], artist_id="artist", output_dir=tmp_path
    )

    assert len(manifest.files) == 2
    rejections = [c for c in captured if c["event"] == "youtube clip rejected"]
    assert len(rejections) == 1
    assert rejections[0]["kwargs"]["video_id"] == "bad"


# -- Wymaganie 1.5: brak pozyskanych fragmentów -> EmptyDatasetError ----------


def test_all_downloads_fail_raises_empty_dataset_error(tmp_path: Path) -> None:
    acquirer = DatasetAcquirer(
        youtube_downloader=_FailingDownloader(failing_ids={"a", "b"})
    )

    with pytest.raises(EmptyDatasetError) as excinfo:
        acquirer.acquire_youtube(["a", "b"], artist_id="artist", output_dir=tmp_path)

    assert excinfo.value.exit_code == 2
    assert str(excinfo.value)


def test_empty_ids_raises_empty_dataset_error(tmp_path: Path) -> None:
    acquirer = DatasetAcquirer(youtube_downloader=_FakeDownloader())
    with pytest.raises(EmptyDatasetError):
        acquirer.acquire_youtube([], artist_id="artist", output_dir=tmp_path)


# -- wstrzyknięty konwerter audio→MIDI: walidacja wyniku jak lokalnego --------


def test_with_injected_converter_produces_midi_entries(tmp_path: Path) -> None:
    acquirer = DatasetAcquirer(
        youtube_downloader=_FakeDownloader(),
        audio_to_midi=_FakeConverter(),
    )

    manifest = acquirer.acquire_youtube(
        ["vid1", "vid2"], artist_id="artist", output_dir=tmp_path
    )

    assert len(manifest.files) == 2
    for entry in manifest.files:
        assert entry.source == YOUTUBE_SOURCE
        assert entry.path.endswith(".mid")
        assert entry.tracks >= 1
        assert len(entry.sha256) == 64
        assert entry.duration_s <= MAX_YOUTUBE_CLIP_S


# -- Wymaganie 1.2: ostrzeżenie przy < 30 fragmentach ------------------------


def test_below_recommended_size_warns(tmp_path: Path) -> None:
    acquirer = DatasetAcquirer(youtube_downloader=_FakeDownloader())
    captured: list[dict] = []
    acquirer._log = _RecordingLogger(captured)  # type: ignore[attr-defined]

    acquirer.acquire_youtube(["v1", "v2", "v3"], artist_id="artist", output_dir=tmp_path)

    warnings = [c for c in captured if c["event"] == "dataset below recommended size"]
    assert len(warnings) == 1
    assert warnings[0]["kwargs"]["min_required"] == MIN_RECOMMENDED_FILES
    assert warnings[0]["kwargs"]["valid_files"] == 3


def test_default_artist_id_is_youtube(tmp_path: Path) -> None:
    acquirer = DatasetAcquirer(youtube_downloader=_FakeDownloader())
    manifest = acquirer.acquire_youtube(["v1"], output_dir=tmp_path)
    assert manifest.artist_id == YOUTUBE_SOURCE
