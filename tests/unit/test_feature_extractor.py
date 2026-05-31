"""Testy jednostkowe :class:`musicians_style.features.extractor.FeatureExtractor`.

Zakres (zadanie 3.2, Wymagania 2.1, 2.2, 2.5, 2.7, 2.8):

* obliczanie poszczególnych cech (*tempo*, tonacja, histogramy, gęstość,
  długości nut, proporcja pauz),
* tempo domyślne 120 BPM przy braku meta-zdarzenia tempa (Wymaganie 2.2),
* wektor neutralny dla plików pustych / wyłącznie z pauzami (Wymaganie 2.7),
* częściowy *Wektor_Cech* przy degenerowanych danych liczbowych (Wymaganie 2.8),
* determinizm wielokrotnych wywołań (Wymaganie 2.4).
"""

from __future__ import annotations

import numpy as np
import pytest

from musicians_style.features.constants import NEUTRAL_FEATURE_VECTOR
from musicians_style.features.extractor import FeatureExtractor
from musicians_style.features.types import (
    INTERVAL_HISTOGRAM_BINS,
    PITCH_CLASS_BINS,
)
from musicians_style.midi.types import InternalRepr, MetaEvent, NoteEvent


@pytest.fixture()
def extractor() -> FeatureExtractor:
    return FeatureExtractor()


def _note(tick: int, pitch: int, duration: int, *, channel: int = 0, velocity: int = 80) -> NoteEvent:
    return NoteEvent(
        tick=tick, channel=channel, pitch=pitch, velocity=velocity, duration_ticks=duration
    )


def _tempo(bpm: float) -> MetaEvent:
    tempo_us = int(round(60_000_000.0 / bpm))
    return MetaEvent(tick=0, kind="tempo", payload={"tempo": tempo_us})


# --------------------------------------------------------------------------- #
# Pliki puste / wyłącznie pauzy (Wymaganie 2.7)
# --------------------------------------------------------------------------- #
def test_empty_repr_returns_neutral_vector(extractor: FeatureExtractor) -> None:
    result = extractor.extract(InternalRepr(ticks_per_beat=480))
    assert result == NEUTRAL_FEATURE_VECTOR


def test_empty_repr_with_meta_only_returns_neutral(extractor: FeatureExtractor) -> None:
    repr_ = InternalRepr(
        ticks_per_beat=480,
        notes=(),
        meta=(_tempo(90.0),),
        smf_format=1,
    )
    assert extractor.extract(repr_) == NEUTRAL_FEATURE_VECTOR


# --------------------------------------------------------------------------- #
# Tempo (Wymagania 2.1, 2.2)
# --------------------------------------------------------------------------- #
def test_tempo_read_from_meta_event(extractor: FeatureExtractor) -> None:
    repr_ = InternalRepr(
        ticks_per_beat=480,
        notes=(_note(0, 60, 480),),
        meta=(_tempo(90.0),),
        smf_format=1,
    )
    fv = extractor.extract(repr_)
    assert fv.tempo_bpm == pytest.approx(90.0, abs=1e-3)


def test_tempo_defaults_to_120_when_missing(extractor: FeatureExtractor) -> None:
    repr_ = InternalRepr(
        ticks_per_beat=480,
        notes=(_note(0, 60, 480),),
        smf_format=1,
    )
    fv = extractor.extract(repr_)
    assert fv.tempo_bpm == pytest.approx(120.0)


def test_tempo_averages_multiple_events(extractor: FeatureExtractor) -> None:
    repr_ = InternalRepr(
        ticks_per_beat=480,
        notes=(_note(0, 60, 480),),
        meta=(_tempo(60.0), MetaEvent(tick=480, kind="tempo", payload={"tempo": int(60_000_000.0 / 180.0)})),
        smf_format=1,
    )
    fv = extractor.extract(repr_)
    assert fv.tempo_bpm == pytest.approx(120.0, abs=1e-3)


def test_nonpositive_tempo_event_ignored(extractor: FeatureExtractor) -> None:
    repr_ = InternalRepr(
        ticks_per_beat=480,
        notes=(_note(0, 60, 480),),
        meta=(MetaEvent(tick=0, kind="tempo", payload={"tempo": 0}),),
        smf_format=1,
    )
    fv = extractor.extract(repr_)
    assert fv.tempo_bpm == pytest.approx(120.0)


# --------------------------------------------------------------------------- #
# Histogram klas wysokości (Wymaganie 2.1, 2.3)
# --------------------------------------------------------------------------- #
def test_pitch_class_histogram_normalized_to_one(extractor: FeatureExtractor) -> None:
    repr_ = InternalRepr(
        ticks_per_beat=480,
        notes=(_note(0, 60, 240), _note(240, 62, 240), _note(480, 64, 240)),
        smf_format=1,
    )
    fv = extractor.extract(repr_)
    assert fv.pitch_class_histogram.sum() == pytest.approx(1.0, abs=1e-9)


def test_pitch_class_histogram_counts_modulo_12(extractor: FeatureExtractor) -> None:
    # C4 (60) i C5 (72) trafiają do tej samej klasy 0.
    repr_ = InternalRepr(
        ticks_per_beat=480,
        notes=(_note(0, 60, 240), _note(240, 72, 240)),
        smf_format=1,
    )
    fv = extractor.extract(repr_)
    expected = np.zeros(PITCH_CLASS_BINS)
    expected[0] = 1.0
    np.testing.assert_allclose(fv.pitch_class_histogram, expected)


# --------------------------------------------------------------------------- #
# Histogram interwałów (Wymaganie 2.1)
# --------------------------------------------------------------------------- #
def test_interval_histogram_monophonic_intervals(extractor: FeatureExtractor) -> None:
    # ślad melodyczny: 60 -> 64 (+4) -> 62 (-2)
    repr_ = InternalRepr(
        ticks_per_beat=480,
        notes=(_note(0, 60, 240), _note(240, 64, 240), _note(480, 62, 240)),
        smf_format=1,
    )
    fv = extractor.extract(repr_)
    assert fv.interval_histogram.sum() == pytest.approx(1.0, abs=1e-9)
    # indeks interwału = interwał + 12
    assert fv.interval_histogram[4 + 12] == pytest.approx(0.5)
    assert fv.interval_histogram[-2 + 12] == pytest.approx(0.5)


def test_interval_histogram_single_note_all_zero(extractor: FeatureExtractor) -> None:
    repr_ = InternalRepr(
        ticks_per_beat=480,
        notes=(_note(0, 60, 480),),
        smf_format=1,
    )
    fv = extractor.extract(repr_)
    np.testing.assert_array_equal(
        fv.interval_histogram, np.zeros(INTERVAL_HISTOGRAM_BINS)
    )


def test_interval_histogram_clamps_large_intervals(extractor: FeatureExtractor) -> None:
    # skok 60 -> 90 to +30 półtonów, przycięty do +12.
    repr_ = InternalRepr(
        ticks_per_beat=480,
        notes=(_note(0, 60, 240), _note(240, 90, 240)),
        smf_format=1,
    )
    fv = extractor.extract(repr_)
    assert fv.interval_histogram[12 + 12] == pytest.approx(1.0)


def test_interval_uses_highest_note_per_onset(extractor: FeatureExtractor) -> None:
    # Akord na ticku 0 (60, 67): ślad bierze 67; potem 69 -> interwał +2.
    repr_ = InternalRepr(
        ticks_per_beat=480,
        notes=(_note(0, 60, 240), _note(0, 67, 240), _note(240, 69, 240)),
        smf_format=1,
    )
    fv = extractor.extract(repr_)
    assert fv.interval_histogram[2 + 12] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Tonacja (Krumhansl-Schmuckler)
# --------------------------------------------------------------------------- #
def test_key_detects_c_major(extractor: FeatureExtractor) -> None:
    # Gama C-dur - oczekiwana tonacja "C major".
    pitches = [60, 62, 64, 65, 67, 69, 71, 72]
    notes = tuple(_note(i * 240, p, 240) for i, p in enumerate(pitches))
    fv = extractor.extract(InternalRepr(ticks_per_beat=480, notes=notes, smf_format=1))
    assert fv.key == "C major"


def test_key_detects_a_minor(extractor: FeatureExtractor) -> None:
    # Materiał z wyraźną toniką A i tercją małą (C) - profil molowy A.
    pitches = [57, 57, 60, 60, 64, 62, 59, 57]
    notes = tuple(_note(i * 240, p, 240) for i, p in enumerate(pitches))
    fv = extractor.extract(InternalRepr(ticks_per_beat=480, notes=notes, smf_format=1))
    assert fv.key.endswith("minor")
    assert fv.key.startswith("A")


# --------------------------------------------------------------------------- #
# Gęstość, długości nut, proporcja pauz (Wymaganie 2.1)
# --------------------------------------------------------------------------- #
def test_note_density_per_second(extractor: FeatureExtractor) -> None:
    # 120 BPM, ticks_per_beat=480 -> 1 ćwierćnuta = 0.5 s. 4 nuty po ćwierćnucie
    # = 2 s całkowitego czasu -> 2 nuty/s.
    notes = tuple(_note(i * 480, 60 + i, 480) for i in range(4))
    fv = extractor.extract(InternalRepr(ticks_per_beat=480, notes=notes, smf_format=1))
    assert fv.note_density_per_s == pytest.approx(2.0, abs=1e-6)


def test_mean_and_std_note_duration(extractor: FeatureExtractor) -> None:
    # Dwie nuty: 480 ticków (0.5 s) i 960 ticków (1.0 s) przy 120 BPM.
    repr_ = InternalRepr(
        ticks_per_beat=480,
        notes=(_note(0, 60, 480), _note(960, 62, 960)),
        smf_format=1,
    )
    fv = extractor.extract(repr_)
    assert fv.mean_note_duration_s == pytest.approx(0.75, abs=1e-6)
    assert fv.std_note_duration_s == pytest.approx(0.25, abs=1e-6)


def test_rest_ratio_zero_for_continuous_notes(extractor: FeatureExtractor) -> None:
    # Dwie przylegające nuty wypełniają cały zakres czasowy - brak pauz.
    repr_ = InternalRepr(
        ticks_per_beat=480,
        notes=(_note(0, 60, 480), _note(480, 62, 480)),
        smf_format=1,
    )
    fv = extractor.extract(repr_)
    assert fv.rest_ratio == pytest.approx(0.0, abs=1e-9)


def test_rest_ratio_half_with_gap(extractor: FeatureExtractor) -> None:
    # Nuta 0..480, pauza 480..960, nuta 960..1440 -> aktywne 960/1440 = 2/3,
    # czyli rest_ratio = 1/3.
    repr_ = InternalRepr(
        ticks_per_beat=480,
        notes=(_note(0, 60, 480), _note(960, 62, 480)),
        smf_format=1,
    )
    fv = extractor.extract(repr_)
    assert fv.rest_ratio == pytest.approx(1.0 / 3.0, abs=1e-9)


def test_rest_ratio_in_unit_interval(extractor: FeatureExtractor) -> None:
    repr_ = InternalRepr(
        ticks_per_beat=480,
        notes=(_note(0, 60, 120), _note(900, 62, 60)),
        smf_format=1,
    )
    fv = extractor.extract(repr_)
    assert 0.0 <= fv.rest_ratio <= 1.0


def test_rest_ratio_overlapping_notes(extractor: FeatureExtractor) -> None:
    # Nakładające się nuty: 0..600 i 300..900 -> aktywne 0..900 = pełny zakres.
    repr_ = InternalRepr(
        ticks_per_beat=480,
        notes=(_note(0, 60, 600), _note(300, 64, 600)),
        smf_format=1,
    )
    fv = extractor.extract(repr_)
    assert fv.rest_ratio == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------------- #
# Determinizm (Wymaganie 2.4)
# --------------------------------------------------------------------------- #
def test_extract_is_deterministic(extractor: FeatureExtractor) -> None:
    repr_ = InternalRepr(
        ticks_per_beat=480,
        notes=(_note(0, 60, 240), _note(240, 64, 240), _note(480, 67, 240)),
        meta=(_tempo(100.0),),
        smf_format=1,
    )
    assert extractor.extract(repr_) == extractor.extract(repr_)


# --------------------------------------------------------------------------- #
# Częściowy błąd numeryczny (Wymaganie 2.8)
# --------------------------------------------------------------------------- #
def test_degenerate_ticks_per_beat_returns_partial_vector(
    extractor: FeatureExtractor,
) -> None:
    # ticks_per_beat = 0 -> seconds_per_tick = 0 -> note_density_per_s = inf,
    # które jest degenerowane i zastępowane wartością neutralną (Wymaganie 2.8).
    repr_ = InternalRepr(
        ticks_per_beat=0,
        notes=(_note(0, 60, 480), _note(480, 64, 480)),
        smf_format=1,
    )
    fv = extractor.extract(repr_)
    # Cechy nie mogą zawierać niefinitnych wartości - degeneracje zastąpione.
    assert np.isfinite(fv.tempo_bpm)
    assert np.isfinite(fv.note_density_per_s)
    assert np.isfinite(fv.mean_note_duration_s)
    assert 0.0 <= fv.rest_ratio <= 1.0
    # Histogramy nadal poprawne (zależne wyłącznie od wysokości).
    assert fv.pitch_class_histogram.sum() == pytest.approx(1.0, abs=1e-9)


def test_partial_vector_has_finite_features(extractor: FeatureExtractor) -> None:
    # Wszystkie nuty o zerowej długości w tym samym ticku -> zerowy zakres czasu.
    repr_ = InternalRepr(
        ticks_per_beat=480,
        notes=(_note(0, 60, 0), _note(0, 64, 0)),
        smf_format=1,
    )
    fv = extractor.extract(repr_)
    assert np.isfinite(fv.note_density_per_s)
    assert np.all(np.isfinite(fv.as_array()))


# --------------------------------------------------------------------------- #
# Stała długość wektora (Wymaganie 2.3)
# --------------------------------------------------------------------------- #
def test_feature_vector_constant_length(extractor: FeatureExtractor) -> None:
    short = InternalRepr(ticks_per_beat=480, notes=(_note(0, 60, 480),), smf_format=1)
    long = InternalRepr(
        ticks_per_beat=480,
        notes=tuple(_note(i * 120, 60 + (i % 12), 120) for i in range(200)),
        smf_format=1,
    )
    assert extractor.extract(short).as_array().shape == extractor.extract(long).as_array().shape
