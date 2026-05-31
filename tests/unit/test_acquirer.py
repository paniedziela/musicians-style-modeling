"""Testy jednostkowe :class:`musicians_style.data.acquirer.DatasetAcquirer` (zadanie 5.2).

Weryfikują kryteria akceptacji Wymagania 1 dla pozyskania *Zbioru_Stylu* z
lokalnego katalogu:

* **1.1 / 1.2** - akceptacja co najmniej 30 plików; ostrzeżenie przy mniejszej
  liczbie poprawnych plików (kontynuacja przetwarzania),
* **1.3** - pominięcie pliku niezgodnego ze SMF (log + kontynuacja),
* **1.4** - odrzucenie plików o czasie trwania poza przedziałem [5 s, 30 min],
* **1.5** - ``EmptyDatasetError`` gdy liczba poprawnych plików == 0,
* **1.7** - manifest zawiera wpisy wyłącznie dla poprawnych plików.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from mido import MidiFile, MidiTrack, Message, MetaMessage

from musicians_style.data.acquirer import (
    MAX_DURATION_S,
    MIN_DURATION_S,
    MIN_RECOMMENDED_FILES,
    DatasetAcquirer,
)
from musicians_style.data.manifest import Manifest
from musicians_style.errors import EmptyDatasetError

# Przy domyślnym tempie 120 BPM (500_000 µs/ćwierćnutę) i ticks_per_beat=480
# jeden tick trwa 0.5/480 s, więc liczba ticków = sekundy * 960.
_TICKS_PER_BEAT = 480
_TICKS_PER_SECOND = 960


def _write_midi(path: Path, duration_s: float, *, tracks: int = 1) -> None:
    """Zapisuje poprawny plik SMF o zadanej długości (w sekundach) i liczbie ścieżek."""
    mf = MidiFile(type=1, ticks_per_beat=_TICKS_PER_BEAT)

    conductor = MidiTrack()
    conductor.append(MetaMessage("time_signature", numerator=4, denominator=4, time=0))
    mf.tracks.append(conductor)

    note_ticks = max(1, int(round(duration_s * _TICKS_PER_SECOND)))
    note_track = MidiTrack()
    note_track.append(Message("note_on", note=60, velocity=100, time=0))
    note_track.append(Message("note_off", note=60, velocity=0, time=note_ticks))
    mf.tracks.append(note_track)

    # Dopełnienie do żądanej liczby ścieżek (puste ścieżki) - tylko format 1.
    while len(mf.tracks) < tracks:
        mf.tracks.append(MidiTrack())

    mf.save(str(path))


def _fill_directory(directory: Path, count: int, duration_s: float = 10.0) -> None:
    """Tworzy ``count`` poprawnych plików MIDI o danej długości w katalogu."""
    directory.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        _write_midi(directory / f"track_{i:03d}.mid", duration_s)


@pytest.fixture()
def acquirer() -> DatasetAcquirer:
    return DatasetAcquirer()


# -- Wymaganie 1.1 / 1.7: zbiór akceptowany, manifest poprawny --------------


def test_acquire_local_returns_manifest_with_valid_entries(
    tmp_path: Path, acquirer: DatasetAcquirer
) -> None:
    directory = tmp_path / "beatles"
    _fill_directory(directory, MIN_RECOMMENDED_FILES, duration_s=12.0)

    manifest = acquirer.acquire_local(directory)

    assert isinstance(manifest, Manifest)
    assert manifest.artist_id == "beatles"
    assert len(manifest.files) == MIN_RECOMMENDED_FILES
    entry = manifest.files[0]
    assert entry.source == "local"
    assert len(entry.sha256) == 64
    assert entry.tracks >= 1
    assert MIN_DURATION_S <= entry.duration_s <= MAX_DURATION_S


def test_artist_id_override(tmp_path: Path, acquirer: DatasetAcquirer) -> None:
    directory = tmp_path / "raw_dir"
    _fill_directory(directory, 3)
    manifest = acquirer.acquire_local(directory, artist_id="custom_artist")
    assert manifest.artist_id == "custom_artist"


# -- Wymaganie 1.2: ostrzeżenie przy < 30 poprawnych plikach -----------------


def test_below_recommended_size_warns_but_continues(
    tmp_path: Path, acquirer: DatasetAcquirer
) -> None:
    directory = tmp_path / "small"
    _fill_directory(directory, 29)

    captured: list[dict] = []
    acquirer._log = _RecordingLogger(captured)  # type: ignore[attr-defined]

    manifest = acquirer.acquire_local(directory)

    assert len(manifest.files) == 29  # kontynuacja mimo ostrzeżenia
    warnings = [c for c in captured if c["event"] == "dataset below recommended size"]
    assert len(warnings) == 1
    assert warnings[0]["kwargs"]["min_required"] == MIN_RECOMMENDED_FILES
    assert warnings[0]["kwargs"]["valid_files"] == 29


def test_at_or_above_recommended_size_does_not_warn(
    tmp_path: Path, acquirer: DatasetAcquirer
) -> None:
    directory = tmp_path / "enough"
    _fill_directory(directory, MIN_RECOMMENDED_FILES)

    captured: list[dict] = []
    acquirer._log = _RecordingLogger(captured)  # type: ignore[attr-defined]

    acquirer.acquire_local(directory)

    assert not [c for c in captured if c["event"] == "dataset below recommended size"]


# -- Wymaganie 1.3: pominięcie plików niezgodnych ze SMF ---------------------


def test_invalid_smf_files_are_skipped(
    tmp_path: Path, acquirer: DatasetAcquirer
) -> None:
    directory = tmp_path / "mixed"
    _fill_directory(directory, 3, duration_s=10.0)
    (directory / "broken.mid").write_bytes(b"not a midi file at all")
    (directory / "garbage.midi").write_bytes(b"\x00\x01\x02\x03")

    captured: list[dict] = []
    acquirer._log = _RecordingLogger(captured)  # type: ignore[attr-defined]

    manifest = acquirer.acquire_local(directory)

    assert len(manifest.files) == 3
    paths = {entry.path for entry in manifest.files}
    assert "broken.mid" not in paths and "garbage.midi" not in paths
    rejections = [c for c in captured if c["event"] == "file rejected"]
    assert {r["kwargs"]["file"] for r in rejections} == {"broken.mid", "garbage.midi"}


# -- Wymaganie 1.4: odrzucenie plików spoza przedziału długości --------------


def test_too_short_file_rejected(tmp_path: Path, acquirer: DatasetAcquirer) -> None:
    directory = tmp_path / "durations"
    _fill_directory(directory, 2, duration_s=10.0)
    _write_midi(directory / "too_short.mid", duration_s=2.0)

    manifest = acquirer.acquire_local(directory)
    paths = {entry.path for entry in manifest.files}
    assert "too_short.mid" not in paths
    assert len(manifest.files) == 2


def test_too_long_file_rejected(tmp_path: Path, acquirer: DatasetAcquirer) -> None:
    directory = tmp_path / "durations"
    _fill_directory(directory, 2, duration_s=10.0)
    _write_midi(directory / "too_long.mid", duration_s=MAX_DURATION_S + 60.0)

    manifest = acquirer.acquire_local(directory)
    paths = {entry.path for entry in manifest.files}
    assert "too_long.mid" not in paths
    assert len(manifest.files) == 2


def test_boundary_durations_accepted(
    tmp_path: Path, acquirer: DatasetAcquirer
) -> None:
    directory = tmp_path / "boundary"
    directory.mkdir()
    _write_midi(directory / "min.mid", duration_s=MIN_DURATION_S)
    _write_midi(directory / "mid.mid", duration_s=60.0)

    manifest = acquirer.acquire_local(directory)
    assert len(manifest.files) == 2


# -- Wymaganie 1.5: brak poprawnych plików -> EmptyDatasetError --------------


def test_empty_directory_raises_empty_dataset_error(
    tmp_path: Path, acquirer: DatasetAcquirer
) -> None:
    directory = tmp_path / "empty"
    directory.mkdir()
    with pytest.raises(EmptyDatasetError) as excinfo:
        acquirer.acquire_local(directory)
    assert str(excinfo.value)
    assert excinfo.value.exit_code == 2


def test_missing_directory_raises_empty_dataset_error(
    tmp_path: Path, acquirer: DatasetAcquirer
) -> None:
    with pytest.raises(EmptyDatasetError):
        acquirer.acquire_local(tmp_path / "does_not_exist")


def test_all_files_rejected_raises_empty_dataset_error(
    tmp_path: Path, acquirer: DatasetAcquirer
) -> None:
    directory = tmp_path / "all_bad"
    directory.mkdir()
    (directory / "broken1.mid").write_bytes(b"garbage")
    _write_midi(directory / "too_short.mid", duration_s=1.0)

    with pytest.raises(EmptyDatasetError) as excinfo:
        acquirer.acquire_local(directory)
    # Opis przyczyny wymienia liczbę kandydatów i powody odrzucenia (Wymaganie 1.5).
    assert "2" in str(excinfo.value)


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
