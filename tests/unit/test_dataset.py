"""Testy jednostkowe :mod:`musicians_style.training.dataset` (zadanie 9.1).

Weryfikują kontrakt datasetu pianoroll i agregatu wieloartystycznego z sekcji
*Pipeline_Treningu* (``design.md``) oraz Wymaganie 3.2:

* :class:`MultiArtistManifest` - deterministyczne mapowanie ``artysta → indeks
  Etykiety_Artysty`` (kolejność alfabetyczna), liczność plików, walidacja
  duplikatów i pustej kolekcji.
* :class:`PianorollDataset` - zwracanie par ``(pianoroll [1, T, P],
  Etykieta_Artysty one-hot [N])``, rozwijanie ścieżek względnych względem
  katalogu bazowego, kolejność próbek, tryb per-artysta (N == 1) i buforowanie.

Drobne pliki MIDI budowane są w katalogu tymczasowym przy użyciu ``mido``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from mido import MidiFile, MidiTrack, Message, MetaMessage

from musicians_style.data.manifest import FileEntry, Manifest
from musicians_style.midi.pianoroll import Pianoroll
from musicians_style.training.dataset import MultiArtistManifest, PianorollDataset

_TICKS_PER_BEAT = 480


def _write_midi(path: Path, pitch: int = 60, *, duration_ticks: int = 480) -> None:
    """Zapisuje minimalny poprawny plik SMF z pojedynczą nutą o danej wysokości."""
    mf = MidiFile(type=1, ticks_per_beat=_TICKS_PER_BEAT)
    conductor = MidiTrack()
    conductor.append(MetaMessage("time_signature", numerator=4, denominator=4, time=0))
    mf.tracks.append(conductor)
    notes = MidiTrack()
    notes.append(Message("note_on", note=pitch, velocity=100, time=0))
    notes.append(Message("note_off", note=pitch, velocity=0, time=duration_ticks))
    mf.tracks.append(notes)
    mf.save(str(path))


def _make_artist_dir(
    base: Path, artist_id: str, pitches: list[int]
) -> tuple[Path, Manifest]:
    """Tworzy katalog artysty z plikami MIDI i odpowiadający mu Manifest."""
    directory = base / artist_id
    directory.mkdir(parents=True, exist_ok=True)
    entries: list[FileEntry] = []
    for i, pitch in enumerate(pitches):
        name = f"track_{i:02d}.mid"
        _write_midi(directory / name, pitch=pitch)
        entries.append(
            FileEntry(
                path=name,
                sha256="0" * 64,
                duration_s=1.0,
                tracks=2,
                source="local",
            )
        )
    return directory, Manifest(artist_id=artist_id, files=entries)


# -- MultiArtistManifest -----------------------------------------------------


def test_multi_artist_manifest_orders_artists_alphabetically(tmp_path: Path) -> None:
    _, m_queen = _make_artist_dir(tmp_path, "queen", [60, 62])
    _, m_abba = _make_artist_dir(tmp_path, "abba", [64])
    _, m_beatles = _make_artist_dir(tmp_path, "beatles", [67, 69, 71])

    multi = MultiArtistManifest({"queen": m_queen, "abba": m_abba, "beatles": m_beatles})

    assert multi.artists == ["abba", "beatles", "queen"]
    assert multi.label_index("abba") == 0
    assert multi.label_index("beatles") == 1
    assert multi.label_index("queen") == 2
    assert multi.num_artists == 3
    assert multi.total_files == 1 + 3 + 2


def test_multi_artist_manifest_from_iterable(tmp_path: Path) -> None:
    _, m_a = _make_artist_dir(tmp_path, "alpha", [60])
    _, m_b = _make_artist_dir(tmp_path, "bravo", [62])

    multi = MultiArtistManifest([m_b, m_a])

    assert multi.artists == ["alpha", "bravo"]
    assert multi.num_artists == 2


def test_multi_artist_manifest_rejects_empty() -> None:
    with pytest.raises(ValueError):
        MultiArtistManifest([])


def test_multi_artist_manifest_rejects_duplicate_artist(tmp_path: Path) -> None:
    _, m_a = _make_artist_dir(tmp_path, "same", [60])
    m_dup = Manifest(artist_id="same", files=[])
    with pytest.raises(ValueError):
        MultiArtistManifest([m_a, m_dup])


def test_multi_artist_manifest_unknown_artist_label_raises(tmp_path: Path) -> None:
    _, m_a = _make_artist_dir(tmp_path, "alpha", [60])
    multi = MultiArtistManifest([m_a])
    with pytest.raises(KeyError):
        multi.label_index("ghost")


# -- PianorollDataset: tryb wieloartystyczny --------------------------------


def test_dataset_length_equals_total_files(tmp_path: Path) -> None:
    _, m_a = _make_artist_dir(tmp_path, "alpha", [60, 62])
    _, m_b = _make_artist_dir(tmp_path, "bravo", [64])
    multi = MultiArtistManifest([m_a, m_b])

    dataset = PianorollDataset(multi, roots=tmp_path)

    assert len(dataset) == 3


def test_dataset_item_shapes_and_label(tmp_path: Path) -> None:
    dir_a, m_a = _make_artist_dir(tmp_path, "alpha", [60])
    dir_b, m_b = _make_artist_dir(tmp_path, "bravo", [62])
    multi = MultiArtistManifest([m_a, m_b])

    pianoroll = Pianoroll()  # [64, 84]
    dataset = PianorollDataset(
        multi, roots={"alpha": dir_a, "bravo": dir_b}, pianoroll=pianoroll
    )

    roll, label = dataset[0]
    assert isinstance(roll, torch.Tensor)
    assert roll.dtype == torch.float32
    assert roll.shape == (1, pianoroll.window_steps, pianoroll.n_pitches)
    assert label.dtype == torch.float32
    assert label.shape == (2,)
    # Indeks 0 należy do artysty 'alpha' (etykieta 0).
    assert torch.equal(label, torch.tensor([1.0, 0.0]))


def test_dataset_labels_follow_artist_order(tmp_path: Path) -> None:
    dir_a, m_a = _make_artist_dir(tmp_path, "alpha", [60])
    dir_b, m_b = _make_artist_dir(tmp_path, "bravo", [62])
    multi = MultiArtistManifest([m_a, m_b])
    dataset = PianorollDataset(multi, roots={"alpha": dir_a, "bravo": dir_b})

    labels = [tuple(dataset[i][1].tolist()) for i in range(len(dataset))]
    assert labels == [(1.0, 0.0), (0.0, 1.0)]


def test_dataset_pianoroll_marks_expected_pitch(tmp_path: Path) -> None:
    # Plik zawiera nutę 60; oczekujemy aktywności tylko w kolumnie 60 - pitch_low.
    dir_a, m_a = _make_artist_dir(tmp_path, "alpha", [60])
    multi = MultiArtistManifest([m_a])
    pianoroll = Pianoroll()
    dataset = PianorollDataset(multi, roots={"alpha": dir_a}, pianoroll=pianoroll)

    roll, _ = dataset[0]
    active_cols = torch.nonzero(roll[0].sum(dim=0) > 0).flatten().tolist()
    assert active_cols == [60 - pianoroll.pitch_low]


# -- PianorollDataset: tryb per-artysta (Manifest) --------------------------


def test_dataset_single_manifest_is_one_hot_size_one(tmp_path: Path) -> None:
    dir_a, m_a = _make_artist_dir(tmp_path, "solo", [60, 64])
    dataset = PianorollDataset(m_a, roots=dir_a)

    assert dataset.num_artists == 1
    assert dataset.artists == ["solo"]
    assert len(dataset) == 2
    _, label = dataset[0]
    assert torch.equal(label, torch.tensor([1.0]))


# -- rozwiązywanie ścieżek i błędy ------------------------------------------


def test_dataset_per_artist_roots_mapping(tmp_path: Path) -> None:
    dir_a, m_a = _make_artist_dir(tmp_path / "a_root", "alpha", [60])
    dir_b, m_b = _make_artist_dir(tmp_path / "b_root", "bravo", [62])
    multi = MultiArtistManifest([m_a, m_b])

    dataset = PianorollDataset(
        multi, roots={"alpha": dir_a, "bravo": dir_b}
    )

    # Każdy element powinien się poprawnie sparsować z osobnego katalogu.
    for i in range(len(dataset)):
        roll, _ = dataset[i]
        assert roll.shape[0] == 1


def test_dataset_absolute_paths_used_as_is(tmp_path: Path) -> None:
    directory = tmp_path / "abs"
    directory.mkdir()
    abs_file = directory / "song.mid"
    _write_midi(abs_file, pitch=60)
    manifest = Manifest(
        artist_id="alpha",
        files=[
            FileEntry(
                path=str(abs_file),
                sha256="0" * 64,
                duration_s=1.0,
                tracks=2,
                source="local",
            )
        ],
    )
    dataset = PianorollDataset(manifest)  # brak roots - ścieżka bezwzględna
    roll, _ = dataset[0]
    assert roll.shape == (1, dataset._pianoroll.window_steps, dataset._pianoroll.n_pitches)


def test_dataset_index_out_of_range(tmp_path: Path) -> None:
    _, m_a = _make_artist_dir(tmp_path, "alpha", [60])
    dataset = PianorollDataset(m_a, roots=tmp_path)
    with pytest.raises(IndexError):
        _ = dataset[5]


def test_dataset_negative_index(tmp_path: Path) -> None:
    dir_a, m_a = _make_artist_dir(tmp_path, "alpha", [60, 62])
    dataset = PianorollDataset(m_a, roots=dir_a)
    last_pos = dataset[len(dataset) - 1]
    last_neg = dataset[-1]
    assert torch.equal(last_pos[0], last_neg[0])
    assert torch.equal(last_pos[1], last_neg[1])


def test_dataset_invalid_source_type() -> None:
    with pytest.raises(TypeError):
        PianorollDataset(object())  # type: ignore[arg-type]


# -- buforowanie -------------------------------------------------------------


def test_dataset_cache_returns_same_tensor(tmp_path: Path) -> None:
    dir_a, m_a = _make_artist_dir(tmp_path, "alpha", [60])
    dataset = PianorollDataset(m_a, roots=dir_a, cache=True)
    first, _ = dataset[0]
    second, _ = dataset[0]
    assert first is second  # ten sam obiekt z bufora


def test_dataset_works_with_dataloader(tmp_path: Path) -> None:
    from torch.utils.data import DataLoader

    dir_a, m_a = _make_artist_dir(tmp_path, "alpha", [60, 62])
    dir_b, m_b = _make_artist_dir(tmp_path, "bravo", [64, 67])
    multi = MultiArtistManifest([m_a, m_b])
    dataset = PianorollDataset(multi, roots={"alpha": dir_a, "bravo": dir_b})

    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    batch_rolls, batch_labels = next(iter(loader))
    assert batch_rolls.shape == (2, 1, dataset._pianoroll.window_steps, dataset._pianoroll.n_pitches)
    assert batch_labels.shape == (2, 2)
