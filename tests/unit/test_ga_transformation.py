"""Testy jednostkowe genotypu i transformacji *Algorytmu_Genetycznego* (zadanie 6.1).

Weryfikują kontrakt :class:`musicians_style.ga.types.Genome`,
:data:`musicians_style.ga.types.IDENTITY_GENOME` oraz czystej funkcji
:func:`musicians_style.ga.transformation.apply_transformation` opisany w sekcji
*Algorytm_Genetyczny* (``design.md``) i Wymaganiu 4.1:

* transpozycja przez ``round(transpose_semitones)`` z odrzuceniem nut spoza
  zakresu MIDI ``[0, 127]``,
* modyfikacja gęstości (``tick``) i długości nut (``duration_ticks``) przez
  przemnożenie,
* modyfikacja *velocity* z saturacją do ``[0, 127]``,
* niezmiennik transformacji tożsamościowej (Property 5, Wymaganie 11.7),
* czystość funkcji (brak modyfikacji wejścia, determinizm).
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from musicians_style.ga import IDENTITY_GENOME, Genome, apply_transformation
from musicians_style.midi.types import InternalRepr, MetaEvent, NoteEvent, event_key


def _sample_repr() -> InternalRepr:
    notes = (
        NoteEvent(tick=0, channel=0, pitch=60, velocity=100, duration_ticks=480),
        NoteEvent(tick=480, channel=1, pitch=64, velocity=80, duration_ticks=240),
        NoteEvent(tick=960, channel=0, pitch=67, velocity=40, duration_ticks=120),
    )
    meta = (
        MetaEvent(tick=0, kind="tempo", payload={"tempo": 500_000}),
        MetaEvent(tick=0, kind="time_signature", payload={"numerator": 4, "denominator": 4}),
    )
    return InternalRepr(ticks_per_beat=480, notes=notes, meta=meta, smf_format=1)


# -- Genome / IDENTITY_GENOME ------------------------------------------------


def test_genome_is_frozen() -> None:
    g = Genome(2.0, 1.5, 0.8, 10.0)
    with pytest.raises(FrozenInstanceError):
        g.transpose_semitones = 5.0  # type: ignore[misc]


def test_genome_value_equality_and_hashable() -> None:
    a = Genome(1.0, 1.0, 1.0, 0.0)
    b = Genome(1.0, 1.0, 1.0, 0.0)
    assert a == b
    assert hash(a) == hash(b)
    assert len({a, b}) == 1


def test_genome_allows_negative_parameters() -> None:
    # Wymaganie 4.1: parametry mogą przyjmować wartości ujemne.
    g = Genome(-5.0, -0.5, -1.0, -20.0)
    assert g.transpose_semitones == -5.0
    assert g.rhythm_density_factor == -0.5
    assert g.note_duration_factor == -1.0
    assert g.velocity_offset == -20.0


def test_identity_genome_values() -> None:
    assert IDENTITY_GENOME == Genome(0.0, 1.0, 1.0, 0.0)


# -- transformacja tożsamościowa (Property 5, Wymaganie 11.7) ----------------


def test_identity_transformation_preserves_repr() -> None:
    repr_ = _sample_repr()
    result = apply_transformation(repr_, IDENTITY_GENOME)
    assert result == repr_


def test_identity_transformation_preserves_empty_repr() -> None:
    repr_ = InternalRepr(ticks_per_beat=480)
    result = apply_transformation(repr_, IDENTITY_GENOME)
    assert result == repr_


# -- transpozycja ------------------------------------------------------------


def test_transposition_adds_rounded_semitones() -> None:
    repr_ = _sample_repr()
    result = apply_transformation(repr_, Genome(2.0, 1.0, 1.0, 0.0))
    pitches = sorted(n.pitch for n in result.notes)
    assert pitches == [62, 66, 69]


def test_transposition_rounds_to_nearest_integer() -> None:
    repr_ = _sample_repr()
    # 2.4 -> 2 (zaokrąglenie w dół do najbliższej liczby całkowitej).
    result = apply_transformation(repr_, Genome(2.4, 1.0, 1.0, 0.0))
    assert sorted(n.pitch for n in result.notes) == [62, 66, 69]


def test_transposition_negative() -> None:
    repr_ = _sample_repr()
    result = apply_transformation(repr_, Genome(-12.0, 1.0, 1.0, 0.0))
    assert sorted(n.pitch for n in result.notes) == [48, 52, 55]


def test_transposition_drops_notes_out_of_midi_range_high() -> None:
    repr_ = InternalRepr(
        ticks_per_beat=480,
        notes=(
            NoteEvent(tick=0, channel=0, pitch=120, velocity=100, duration_ticks=480),
            NoteEvent(tick=0, channel=0, pitch=60, velocity=100, duration_ticks=480),
        ),
        smf_format=1,
    )
    # +10 -> 130 (poza zakresem, odrzucone) oraz 70 (zachowane).
    result = apply_transformation(repr_, Genome(10.0, 1.0, 1.0, 0.0))
    assert [n.pitch for n in result.notes] == [70]


def test_transposition_drops_notes_out_of_midi_range_low() -> None:
    repr_ = InternalRepr(
        ticks_per_beat=480,
        notes=(
            NoteEvent(tick=0, channel=0, pitch=5, velocity=100, duration_ticks=480),
            NoteEvent(tick=0, channel=0, pitch=60, velocity=100, duration_ticks=480),
        ),
        smf_format=1,
    )
    # -10 -> -5 (odrzucone) oraz 50 (zachowane).
    result = apply_transformation(repr_, Genome(-10.0, 1.0, 1.0, 0.0))
    assert [n.pitch for n in result.notes] == [50]


def test_transposition_keeps_boundary_pitches() -> None:
    repr_ = InternalRepr(
        ticks_per_beat=480,
        notes=(
            NoteEvent(tick=0, channel=0, pitch=117, velocity=100, duration_ticks=480),
            NoteEvent(tick=0, channel=0, pitch=10, velocity=100, duration_ticks=480),
        ),
        smf_format=1,
    )
    # +10 -> 127 (granica włączna) i 20 - oba zachowane.
    result = apply_transformation(repr_, Genome(10.0, 1.0, 1.0, 0.0))
    assert sorted(n.pitch for n in result.notes) == [20, 127]


# -- gęstość rytmiczna i długość nut -----------------------------------------


def test_rhythm_density_scales_onsets() -> None:
    repr_ = _sample_repr()
    result = apply_transformation(repr_, Genome(0.0, 2.0, 1.0, 0.0))
    by_pitch = {n.pitch: n for n in result.notes}
    assert by_pitch[60].tick == 0
    assert by_pitch[64].tick == 960
    assert by_pitch[67].tick == 1920
    # Długości nut bez zmian.
    assert by_pitch[64].duration_ticks == 240


def test_note_duration_scales_durations() -> None:
    repr_ = _sample_repr()
    result = apply_transformation(repr_, Genome(0.0, 1.0, 0.5, 0.0))
    by_pitch = {n.pitch: n for n in result.notes}
    assert by_pitch[60].duration_ticks == 240
    assert by_pitch[64].duration_ticks == 120
    assert by_pitch[67].duration_ticks == 60
    # Czasy rozpoczęcia bez zmian.
    assert by_pitch[64].tick == 480


def test_density_and_duration_rounding() -> None:
    repr_ = InternalRepr(
        ticks_per_beat=480,
        notes=(NoteEvent(tick=100, channel=0, pitch=60, velocity=64, duration_ticks=100),),
        smf_format=1,
    )
    # tick: round(100 * 1.5) = 150, duration: round(100 * 1.5) = 150.
    result = apply_transformation(repr_, Genome(0.0, 1.5, 1.5, 0.0))
    note = result.notes[0]
    assert note.tick == 150
    assert note.duration_ticks == 150


def test_tick_and_duration_stay_non_negative() -> None:
    repr_ = _sample_repr()
    # Ujemne mnożniki nie powinny dawać ujemnych ticków/długości.
    result = apply_transformation(repr_, Genome(0.0, -1.0, -1.0, 0.0))
    for note in result.notes:
        assert note.tick >= 0
        assert note.duration_ticks >= 0


# -- dynamika (velocity) -----------------------------------------------------


def test_velocity_offset_added() -> None:
    repr_ = _sample_repr()
    result = apply_transformation(repr_, Genome(0.0, 1.0, 1.0, 10.0))
    by_pitch = {n.pitch: n for n in result.notes}
    assert by_pitch[60].velocity == 110
    assert by_pitch[64].velocity == 90
    assert by_pitch[67].velocity == 50


def test_velocity_saturates_high() -> None:
    repr_ = _sample_repr()
    result = apply_transformation(repr_, Genome(0.0, 1.0, 1.0, 100.0))
    assert all(n.velocity == 127 for n in result.notes)


def test_velocity_saturates_low() -> None:
    repr_ = _sample_repr()
    result = apply_transformation(repr_, Genome(0.0, 1.0, 1.0, -200.0))
    assert all(n.velocity == 0 for n in result.notes)


def test_velocity_offset_rounding() -> None:
    repr_ = InternalRepr(
        ticks_per_beat=480,
        notes=(NoteEvent(tick=0, channel=0, pitch=60, velocity=64, duration_ticks=480),),
        smf_format=1,
    )
    # round(64 + 1.5) = round(65.5) = 66 (bankers rounding to even -> 66).
    result = apply_transformation(repr_, Genome(0.0, 1.0, 1.0, 1.5))
    assert result.notes[0].velocity == 66


# -- metadane i czystość -----------------------------------------------------


def test_meta_and_global_params_preserved() -> None:
    repr_ = _sample_repr()
    result = apply_transformation(repr_, Genome(3.0, 2.0, 0.5, 20.0))
    assert result.meta == repr_.meta
    assert result.ticks_per_beat == repr_.ticks_per_beat
    assert result.smf_format == repr_.smf_format


def test_result_notes_sorted_by_event_key() -> None:
    repr_ = _sample_repr()
    result = apply_transformation(repr_, Genome(0.0, 0.0, 1.0, 0.0))
    keys = [event_key(n) for n in result.notes]
    assert keys == sorted(keys)


def test_function_is_pure_does_not_mutate_input() -> None:
    repr_ = _sample_repr()
    snapshot = _sample_repr()
    apply_transformation(repr_, Genome(5.0, 2.0, 0.5, 30.0))
    # Wejście pozostaje niezmienione (niemutowalne struktury).
    assert repr_ == snapshot


def test_function_is_deterministic() -> None:
    repr_ = _sample_repr()
    g = Genome(2.0, 1.3, 0.7, 15.0)
    assert apply_transformation(repr_, g) == apply_transformation(repr_, g)


def test_channel_preserved() -> None:
    repr_ = _sample_repr()
    result = apply_transformation(repr_, Genome(1.0, 1.0, 1.0, 0.0))
    channels = {n.pitch: n.channel for n in result.notes}
    # pitch 60->61 (ch0), 64->65 (ch1), 67->68 (ch0)
    assert channels == {61: 0, 65: 1, 68: 0}
