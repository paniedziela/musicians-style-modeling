import pytest

from musicians_style.evaluation.content import content_metrics, semantic_midi_equal
from musicians_style.midi.types import InternalRepr, NoteEvent


def test_semantic_midi_equality_ignores_ambiguous_same_pitch_pairing():
    left = InternalRepr(480, (
        NoteEvent(0, 0, 60, 30, 480),
        NoteEvent(0, 0, 60, 90, 0),
    ))
    right = InternalRepr(480, (
        NoteEvent(0, 0, 60, 30, 0),
        NoteEvent(0, 0, 60, 90, 480),
    ))
    assert left != right
    assert semantic_midi_equal(left, right)


def test_content_metrics_measure_length_onsets_and_chroma():
    source = InternalRepr(480, (NoteEvent(0, 0, 60, 80, 480), NoteEvent(480, 0, 64, 80, 480)))
    metrics = content_metrics(source, source)
    assert metrics["length_error"] == 0.0
    assert metrics["onset_f1"] == 1.0
    assert metrics["melody_trigram_jaccard"] == 1.0
    assert metrics["chroma_cosine"] == pytest.approx(1.0)
