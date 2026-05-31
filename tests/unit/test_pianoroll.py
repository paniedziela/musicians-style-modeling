"""Testy jednostkowe :class:`musicians_style.midi.pianoroll.Pianoroll` (zadanie 2.7).

Weryfikują kontrakt konwertera Pianoroll z sekcji *Parser_MIDI i
Pretty_Printer_MIDI* (``design.md``) oraz Wymagania 5.3 i 5.7:

* ``from_internal`` produkuje binarną macierz ``[T × P]`` z krokiem szesnastki
  i zakresem wysokości z konfiguracji,
* ``to_internal`` rekonstruuje *Reprezentację_Wewnętrzną* zachowując tempo,
  metrum, kanały i *velocity* z *template* (Wymaganie 5.7),
* spójność z potokiem ``parse → Pianoroll → ... → write``.
"""

from __future__ import annotations

import numpy as np
import pytest

from musicians_style.midi.parser import MidiParser
from musicians_style.midi.pianoroll import Pianoroll
from musicians_style.midi.printer import MidiPrettyPrinter
from musicians_style.midi.types import InternalRepr, MetaEvent, NoteEvent, event_key


@pytest.fixture()
def converter() -> Pianoroll:
    # Domyślna geometria: pitch_range [24, 108) → 84 wysokości, krok szesnastki.
    return Pianoroll(pitch_range=(24, 108), steps_per_beat=4, window_steps=64)


def _sample_repr() -> InternalRepr:
    notes = (
        # C4 (60) od kroku 0 przez 1 ćwierćnutę = 4 kroki.
        NoteEvent(tick=0, channel=0, pitch=60, velocity=100, duration_ticks=480),
        # E4 (64) od kroku 4 (480 ticków) przez pół ćwierćnuty = 2 kroki.
        NoteEvent(tick=480, channel=1, pitch=64, velocity=80, duration_ticks=240),
    )
    meta = (
        MetaEvent(tick=0, kind="tempo", payload={"tempo": 500_000}),
        MetaEvent(tick=0, kind="time_signature", payload={"numerator": 3, "denominator": 4}),
        MetaEvent(tick=0, kind="key_signature", payload={"key": "C"}),
        MetaEvent(tick=0, kind="program_change", payload={"program": 1, "channel": 0}),
    )
    return InternalRepr(ticks_per_beat=480, notes=notes, meta=meta, smf_format=1)


# -- from_internal -----------------------------------------------------------


def test_from_internal_shape_and_dtype(converter: Pianoroll) -> None:
    roll = converter.from_internal(_sample_repr())
    assert roll.shape == (64, 84)
    assert roll.dtype == np.float32
    assert set(np.unique(roll)).issubset({0.0, 1.0})


def test_from_internal_marks_active_steps(converter: Pianoroll) -> None:
    roll = converter.from_internal(_sample_repr())
    # C4 (60) -> kolumna 60-24 = 36, kroki 0..4.
    assert roll[0:4, 36].tolist() == [1.0, 1.0, 1.0, 1.0]
    assert roll[4, 36] == 0.0
    # E4 (64) -> kolumna 40, kroki 4..6.
    assert roll[4:6, 40].tolist() == [1.0, 1.0]
    assert roll[6, 40] == 0.0


def test_from_internal_window_truncates(converter: Pianoroll) -> None:
    roll = converter.from_internal(_sample_repr(), window_steps=2)
    assert roll.shape == (2, 84)
    # C4 nadal aktywne w obu krokach, E4 (start krok 4) poza oknem.
    assert roll[0:2, 36].tolist() == [1.0, 1.0]
    assert roll[:, 40].sum() == 0.0


def test_from_internal_ignores_pitch_out_of_range(converter: Pianoroll) -> None:
    repr_ = InternalRepr(
        ticks_per_beat=480,
        notes=(
            NoteEvent(tick=0, channel=0, pitch=12, velocity=100, duration_ticks=480),
            NoteEvent(tick=0, channel=0, pitch=120, velocity=100, duration_ticks=480),
        ),
        smf_format=1,
    )
    roll = converter.from_internal(repr_)
    assert roll.sum() == 0.0


def test_from_internal_empty_repr(converter: Pianoroll) -> None:
    roll = converter.from_internal(InternalRepr(ticks_per_beat=480))
    assert roll.shape == (64, 84)
    assert roll.sum() == 0.0


def test_short_note_occupies_at_least_one_step(converter: Pianoroll) -> None:
    # Nuta krótsza niż krok szesnastki nadal zajmuje jeden krok.
    repr_ = InternalRepr(
        ticks_per_beat=480,
        notes=(NoteEvent(tick=0, channel=0, pitch=60, velocity=64, duration_ticks=10),),
        smf_format=1,
    )
    roll = converter.from_internal(repr_)
    assert roll[0, 36] == 1.0
    assert roll[1, 36] == 0.0


# -- to_internal -------------------------------------------------------------


def test_to_internal_preserves_template_metadata(converter: Pianoroll) -> None:
    template = _sample_repr()
    roll = converter.from_internal(template)
    reconstructed = converter.to_internal(roll, template)
    # Tempo, metrum, tonacja i ticks_per_beat pochodzą z template (Wymaganie 5.7).
    assert reconstructed.ticks_per_beat == template.ticks_per_beat
    assert reconstructed.smf_format == template.smf_format
    assert reconstructed.meta == template.meta


def test_to_internal_recovers_notes(converter: Pianoroll) -> None:
    template = _sample_repr()
    roll = converter.from_internal(template)
    reconstructed = converter.to_internal(roll, template)
    by_pitch = {n.pitch: n for n in reconstructed.notes}
    assert set(by_pitch) == {60, 64}
    # C4: start 0, długość 4 kroki = 480 ticków, velocity/channel z template.
    assert by_pitch[60].tick == 0
    assert by_pitch[60].duration_ticks == 480
    assert by_pitch[60].velocity == 100
    assert by_pitch[60].channel == 0
    # E4: start krok 4 = 480 ticków, długość 2 kroki = 240 ticków.
    assert by_pitch[64].tick == 480
    assert by_pitch[64].duration_ticks == 240
    assert by_pitch[64].velocity == 80
    assert by_pitch[64].channel == 1


def test_to_internal_sorted_notes(converter: Pianoroll) -> None:
    template = _sample_repr()
    reconstructed = converter.to_internal(converter.from_internal(template), template)
    keys = [event_key(n) for n in reconstructed.notes]
    assert keys == sorted(keys)


def test_to_internal_threshold_binarizes(converter: Pianoroll) -> None:
    template = _sample_repr()
    # Macierz ciągła (np. wyjście sigmoidu): tylko wartości >= 0.5 są aktywne.
    matrix = np.zeros((64, 84), dtype=np.float32)
    matrix[0:4, 36] = 0.9  # aktywne
    matrix[0:4, 37] = 0.3  # poniżej progu - pomijane
    reconstructed = converter.to_internal(matrix, template, threshold=0.5)
    pitches = {n.pitch for n in reconstructed.notes}
    assert pitches == {60}


def test_to_internal_uses_template_defaults_for_unknown_pitch(
    converter: Pianoroll,
) -> None:
    # Nuta o wysokości, której nie ma w template -> velocity mediana, kanał dominujący.
    template = InternalRepr(
        ticks_per_beat=480,
        notes=(
            NoteEvent(tick=0, channel=2, pitch=60, velocity=40, duration_ticks=240),
            NoteEvent(tick=0, channel=2, pitch=62, velocity=80, duration_ticks=240),
            NoteEvent(tick=0, channel=5, pitch=64, velocity=120, duration_ticks=240),
        ),
        smf_format=1,
    )
    matrix = np.zeros((64, 84), dtype=np.float32)
    matrix[0:2, 100 - 24] = 1.0  # G7 (100) - nieobecne w template
    reconstructed = converter.to_internal(matrix, template)
    note = next(n for n in reconstructed.notes if n.pitch == 100)
    assert note.velocity == 80  # mediana z [40, 80, 120]
    assert note.channel == 2  # kanał dominujący (2 występuje 2x)


def test_to_internal_rejects_non_2d(converter: Pianoroll) -> None:
    with pytest.raises(ValueError):
        converter.to_internal(np.zeros((64,), dtype=np.float32), _sample_repr())


# -- pipeline parse → Pianoroll → ... → write -------------------------------


def test_pipeline_roundtrip_quantized(converter: Pianoroll) -> None:
    """Spójny potok: write → parse → Pianoroll → to_internal → write → parse.

    Dla utworu skwantowanego do kroków szesnastkowych pełen obieg zachowuje
    wysokości, pozycje i długości nut (z dokładnością do kwantyzacji).
    """
    parser = MidiParser()
    printer = MidiPrettyPrinter()

    original = _sample_repr()
    parsed = parser.parse_bytes(printer.to_bytes(original))

    roll = converter.from_internal(parsed)
    reconstructed = converter.to_internal(roll, template=parsed)

    final = parser.parse_bytes(printer.to_bytes(reconstructed))

    assert {n.pitch for n in final.notes} == {60, 64}
    by_pitch = {n.pitch: n for n in final.notes}
    assert by_pitch[60].tick == 0
    assert by_pitch[60].duration_ticks == 480
    assert by_pitch[64].tick == 480
    assert by_pitch[64].duration_ticks == 240


# -- konstruktor / walidacja ------------------------------------------------


def test_invalid_pitch_range_raises() -> None:
    with pytest.raises(ValueError):
        Pianoroll(pitch_range=(108, 24))


def test_invalid_steps_per_beat_raises() -> None:
    with pytest.raises(ValueError):
        Pianoroll(steps_per_beat=0)


def test_n_pitches_property() -> None:
    assert Pianoroll(pitch_range=(24, 108)).n_pitches == 84
