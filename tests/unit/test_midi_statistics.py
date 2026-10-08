import pytest

from musicians_style.midi.statistics import polyphony
from musicians_style.midi.types import NoteEvent


def test_polyphony_counts_silence_and_excludes_zero_duration_notes():
    notes = (
        NoteEvent(0, 0, 60, 80, 240),
        NoteEvent(120, 0, 64, 80, 120),
        NoteEvent(480, 0, 67, 80, 240),
        NoteEvent(600, 0, 72, 80, 0),
    )
    mean, maximum = polyphony(iter(notes))
    assert mean == pytest.approx(600 / 720)
    assert maximum == 2


def test_adjacent_notes_do_not_overlap():
    assert polyphony((NoteEvent(0, 0, 60, 80, 120), NoteEvent(120, 0, 60, 80, 120))) == (1.0, 1)


def test_empty_and_zero_duration_polyphony():
    assert polyphony(()) == (0.0, 0)
    assert polyphony((NoteEvent(0, 0, 60, 80, 0),)) == (0.0, 0)
