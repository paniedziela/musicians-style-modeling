"""Testy jednostkowe agregacji cech zbioru (zadanie 3.7, Wymaganie 2.6).

Zakres :meth:`musicians_style.features.extractor.FeatureExtractor.extract_dataset`:

* obliczanie średniej, mediany i odchylenia standardowego każdej cechy jako
  *Wektorów_Cech* zbudowanych ze statystyk kolumnowych macierzy cech,
* obliczanie macierzy kowariancji o kształcie ``[42 × 42]`` (zgodnym z
  :data:`FEATURE_VECTOR_LENGTH`) używanej w odległości Mahalanobisa,
* obsługa przypadku pojedynczego pliku (kowariancja zerowa, bez ``NaN``),
* konwencja pola kategorycznego ``key`` (tonacja dominująca dla mean/median,
  placeholder dla std),
* rozwiązywanie względnych ścieżek ``FileEntry.path`` względem parametru ``root``,
* zgłoszenie :class:`EmptyDatasetError` dla pustego *Manifestu*,
* pominięcie pliku niezgodnego ze SMF bez przerywania agregacji,
* determinizm wielokrotnych wywołań (Wymaganie 2.4).

Pliki MIDI budowane są realnie przez :class:`MidiPrettyPrinter` w katalogu
tymczasowym i wczytywane przez :class:`MidiParser` (bez mockowania), dzięki czemu
test waliduje rzeczywisty potok parse → extract → agregacja.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from musicians_style.data.manifest import FileEntry, Manifest
from musicians_style.errors import EmptyDatasetError
from musicians_style.features.constants import NEUTRAL_KEY
from musicians_style.features.extractor import FeatureExtractor
from musicians_style.features.types import (
    FEATURE_VECTOR_LENGTH,
    AggregatedFeatures,
    FeatureVector,
)
from musicians_style.midi.printer import MidiPrettyPrinter
from musicians_style.midi.types import InternalRepr, MetaEvent, NoteEvent


@pytest.fixture()
def extractor() -> FeatureExtractor:
    return FeatureExtractor()


def _note(
    tick: int, pitch: int, duration: int, *, channel: int = 0, velocity: int = 80
) -> NoteEvent:
    return NoteEvent(
        tick=tick,
        channel=channel,
        pitch=pitch,
        velocity=velocity,
        duration_ticks=duration,
    )


def _tempo(bpm: float) -> MetaEvent:
    tempo_us = int(round(60_000_000.0 / bpm))
    return MetaEvent(tick=0, kind="tempo", payload={"tempo": tempo_us})


def _scale_repr(base_pitch: int, *, bpm: float = 120.0) -> InternalRepr:
    """Buduje krótki ślad melodyczny (gama 8 nut) od ``base_pitch``."""
    steps = [0, 2, 4, 5, 7, 9, 11, 12]
    notes = tuple(
        _note(i * 480, base_pitch + step, 480) for i, step in enumerate(steps)
    )
    return InternalRepr(
        ticks_per_beat=480, notes=notes, meta=(_tempo(bpm),), smf_format=1
    )


def _write_midi(repr_: InternalRepr, path: Path) -> None:
    MidiPrettyPrinter().write(repr_, path)


def _build_manifest(
    tmp_path: Path, reprs: list[InternalRepr], *, artist_id: str = "test_artist"
) -> Manifest:
    """Zapisuje pliki MIDI w ``tmp_path`` i buduje *Manifest* ze ścieżkami względnymi."""
    entries: list[FileEntry] = []
    for index, repr_ in enumerate(reprs):
        rel_name = f"track_{index:02d}.mid"
        _write_midi(repr_, tmp_path / rel_name)
        entries.append(
            FileEntry(
                path=rel_name,
                sha256="0" * 64,
                duration_s=4.0,
                tracks=2,
                source="local",
            )
        )
    return Manifest(artist_id=artist_id, files=entries)


# --------------------------------------------------------------------------- #
# Kształt i typy wyniku
# --------------------------------------------------------------------------- #
def test_returns_aggregated_features(
    extractor: FeatureExtractor, tmp_path: Path
) -> None:
    manifest = _build_manifest(
        tmp_path, [_scale_repr(60), _scale_repr(62), _scale_repr(64)]
    )
    result = extractor.extract_dataset(manifest, root=tmp_path)
    assert isinstance(result, AggregatedFeatures)
    assert isinstance(result.mean, FeatureVector)
    assert isinstance(result.median, FeatureVector)
    assert isinstance(result.std, FeatureVector)


def test_covariance_shape_matches_feature_length(
    extractor: FeatureExtractor, tmp_path: Path
) -> None:
    manifest = _build_manifest(
        tmp_path, [_scale_repr(60), _scale_repr(62), _scale_repr(64)]
    )
    result = extractor.extract_dataset(manifest, root=tmp_path)
    assert result.covariance.shape == (FEATURE_VECTOR_LENGTH, FEATURE_VECTOR_LENGTH)
    assert np.all(np.isfinite(result.covariance))


# --------------------------------------------------------------------------- #
# Poprawność statystyk kolumnowych
# --------------------------------------------------------------------------- #
def test_mean_median_std_match_manual_statistics(
    extractor: FeatureExtractor, tmp_path: Path
) -> None:
    reprs = [_scale_repr(60), _scale_repr(62), _scale_repr(64)]
    manifest = _build_manifest(tmp_path, reprs)
    result = extractor.extract_dataset(manifest, root=tmp_path)

    # Ręczne wyliczenie statystyk z tych samych Wektorów_Cech.
    parser_extractor = FeatureExtractor()
    matrix = np.vstack(
        [
            parser_extractor.extract(repr_).as_array()
            for repr_ in reprs
        ]
    )
    np.testing.assert_allclose(result.mean.as_array(), matrix.mean(axis=0))
    np.testing.assert_allclose(result.median.as_array(), np.median(matrix, axis=0))
    np.testing.assert_allclose(result.std.as_array(), matrix.std(axis=0, ddof=0))


def test_covariance_matches_numpy_cov(
    extractor: FeatureExtractor, tmp_path: Path
) -> None:
    reprs = [_scale_repr(60), _scale_repr(63), _scale_repr(67), _scale_repr(72)]
    manifest = _build_manifest(tmp_path, reprs)
    result = extractor.extract_dataset(manifest, root=tmp_path)

    matrix = np.vstack(
        [FeatureExtractor().extract(repr_).as_array() for repr_ in reprs]
    )
    expected = np.cov(matrix, rowvar=False)
    np.testing.assert_allclose(result.covariance, expected)


# --------------------------------------------------------------------------- #
# Konwencja pola kategorycznego key
# --------------------------------------------------------------------------- #
def test_mean_median_key_is_dominant_key(
    extractor: FeatureExtractor, tmp_path: Path
) -> None:
    # Trzy kopie tej samej gamy C-dur -> tonacja dominująca "C major".
    reprs = [_scale_repr(60), _scale_repr(60), _scale_repr(60)]
    manifest = _build_manifest(tmp_path, reprs)
    result = extractor.extract_dataset(manifest, root=tmp_path)
    assert result.mean.key == "C major"
    assert result.median.key == "C major"


def test_std_key_is_neutral_placeholder(
    extractor: FeatureExtractor, tmp_path: Path
) -> None:
    manifest = _build_manifest(tmp_path, [_scale_repr(60), _scale_repr(62)])
    result = extractor.extract_dataset(manifest, root=tmp_path)
    assert result.std.key == NEUTRAL_KEY


# --------------------------------------------------------------------------- #
# Przypadek pojedynczego pliku (Wymaganie: kowariancja bez NaN)
# --------------------------------------------------------------------------- #
def test_single_file_covariance_is_zero_matrix(
    extractor: FeatureExtractor, tmp_path: Path
) -> None:
    manifest = _build_manifest(tmp_path, [_scale_repr(60)])
    result = extractor.extract_dataset(manifest, root=tmp_path)
    assert result.covariance.shape == (FEATURE_VECTOR_LENGTH, FEATURE_VECTOR_LENGTH)
    assert np.all(np.isfinite(result.covariance))
    np.testing.assert_array_equal(
        result.covariance, np.zeros((FEATURE_VECTOR_LENGTH, FEATURE_VECTOR_LENGTH))
    )
    # Dla pojedynczego pliku mean == median == ten sam Wektor_Cech.
    np.testing.assert_allclose(result.mean.as_array(), result.median.as_array())


# --------------------------------------------------------------------------- #
# Rozwiązywanie ścieżek (root)
# --------------------------------------------------------------------------- #
def test_resolves_relative_paths_against_root(
    extractor: FeatureExtractor, tmp_path: Path
) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    manifest = _build_manifest(dataset_dir, [_scale_repr(60), _scale_repr(62)])
    # Ścieżki w manifeście są względne; root wskazuje katalog zbioru.
    result = extractor.extract_dataset(manifest, root=dataset_dir)
    assert isinstance(result, AggregatedFeatures)


def test_resolves_absolute_paths_without_root(
    extractor: FeatureExtractor, tmp_path: Path
) -> None:
    repr_ = _scale_repr(60)
    abs_path = tmp_path / "abs_track.mid"
    _write_midi(repr_, abs_path)
    manifest = Manifest(
        artist_id="abs_artist",
        files=[
            FileEntry(
                path=str(abs_path),
                sha256="0" * 64,
                duration_s=4.0,
                tracks=2,
                source="local",
            )
        ],
    )
    result = extractor.extract_dataset(manifest)  # root=None -> ścieżka bezwzględna
    assert isinstance(result, AggregatedFeatures)


# --------------------------------------------------------------------------- #
# Pusty Manifest (Wymaganie 1.5)
# --------------------------------------------------------------------------- #
def test_empty_manifest_raises(extractor: FeatureExtractor) -> None:
    manifest = Manifest(artist_id="empty_artist", files=[])
    with pytest.raises(EmptyDatasetError):
        extractor.extract_dataset(manifest)


# --------------------------------------------------------------------------- #
# Plik niezgodny ze SMF nie przerywa agregacji
# --------------------------------------------------------------------------- #
def test_invalid_file_skipped_uses_neutral_vector(
    extractor: FeatureExtractor, tmp_path: Path
) -> None:
    good = tmp_path / "good.mid"
    _write_midi(_scale_repr(60), good)
    broken = tmp_path / "broken.mid"
    broken.write_bytes(b"not a midi file at all")

    manifest = Manifest(
        artist_id="mixed",
        files=[
            FileEntry(path="good.mid", sha256="0" * 64, duration_s=4.0, tracks=2, source="local"),
            FileEntry(path="broken.mid", sha256="0" * 64, duration_s=4.0, tracks=1, source="local"),
        ],
    )
    # Agregacja nie zgłasza wyjątku - uszkodzony plik daje Wektor_Cech neutralny.
    result = extractor.extract_dataset(manifest, root=tmp_path)
    assert isinstance(result, AggregatedFeatures)
    assert result.covariance.shape == (FEATURE_VECTOR_LENGTH, FEATURE_VECTOR_LENGTH)


# --------------------------------------------------------------------------- #
# Determinizm (Wymaganie 2.4)
# --------------------------------------------------------------------------- #
def test_extract_dataset_is_deterministic(
    extractor: FeatureExtractor, tmp_path: Path
) -> None:
    manifest = _build_manifest(
        tmp_path, [_scale_repr(60), _scale_repr(62), _scale_repr(64)]
    )
    first = extractor.extract_dataset(manifest, root=tmp_path)
    second = extractor.extract_dataset(manifest, root=tmp_path)
    assert first == second
