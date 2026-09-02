from __future__ import annotations

import pytest

from musicians_style.e4.experiment import E4Config, _split_ids
from musicians_style.e4.segmentation import encode_piece, stitch_segments
from musicians_style.midi.parser import MidiParser
from musicians_style.midi.printer import MidiPrettyPrinter
from musicians_style.midi.types import InternalRepr, MetaEvent, NoteEvent


def _score(notes: tuple[NoteEvent, ...], meta: tuple[MetaEvent, ...] = ()) -> InternalRepr:
    return InternalRepr(480, notes, meta, 1)


def test_long_mixed_meter_piece_stitches_every_window_without_truncation(tmp_path) -> None:
    # 4/4 + 3/4 + 6/8 gives nine bars: two complete windows and one padded one.
    meta = (
        MetaEvent(0, "time_signature", {"numerator": 4, "denominator": 4}),
        MetaEvent(4 * 1920, "time_signature", {"numerator": 3, "denominator": 4}),
        MetaEvent(4 * 1920 + 3 * 1440, "time_signature", {"numerator": 6, "denominator": 8}),
    )
    source = _score((NoteEvent(0, 0, 60, 90, 480), NoteEvent(4 * 1920 + 3 * 1440 + 2 * 1440 + 100, 0, 67, 90, 700)), meta)
    mapping = encode_piece(source)
    result = stitch_segments(mapping)
    assert mapping.segment_count == 3
    assert max(n.tick + n.duration_ticks for n in result.notes) == mapping.end_tick
    assert result.meta == source.meta
    output = tmp_path / "stitched.mid"
    MidiPrettyPrinter().write(result, output)
    assert MidiParser().parse(output) == result


def test_onset_frame_keeps_repeated_pitch_and_cross_boundary_sustain() -> None:
    # The first two notes repeat pitch 60; the third crosses the four-bar edge.
    source = _score((
        NoteEvent(0, 0, 60, 90, 480), NoteEvent(480, 0, 60, 90, 480),
        NoteEvent(3 * 1920 + 1680, 0, 64, 91, 960),
        NoteEvent(9 * 1920 - 480, 0, 72, 90, 480),
    ))
    mapping = encode_piece(source)
    assert mapping.segments[0].data[0, 0, 60 - 24] == 1
    assert mapping.segments[0].data[0, 4, 60 - 24] == 1
    assert mapping.segments[1].data[1, 0, 64 - 24] == 1
    result = stitch_segments(mapping)
    repeated = [n for n in result.notes if n.pitch == 60]
    crossing = [n for n in result.notes if n.pitch == 64]
    assert [n.tick for n in repeated] == [0, 480]
    assert len(crossing) == 1 and crossing[0].duration_ticks >= 960


def test_split_isolation_and_all_segments_are_counted() -> None:
    splits = {"repetitions": [{"repeat": 0, "folds": [{"fold": 0, "test": {"sample_ids": ["c"]}, "inner_folds": [{"fold": 0, "train": {"sample_ids": ["a"]}, "validation": {"sample_ids": ["b"]}}]}]}]}
    config = E4Config(None, None, None, None)  # paths are irrelevant to split extraction
    assert _split_ids(splits, config) == {"train": {"a"}, "validation": {"b"}, "test": {"c"}}
    source = _score((NoteEvent(8 * 1920, 0, 60, 90, 480),))
    mapping = encode_piece(source)
    assert mapping.segment_count == 3  # a later nonempty window must not be dropped


def test_split_extraction_rejects_overlap() -> None:
    splits = {"repetitions": [{"repeat": 0, "folds": [{"fold": 0, "test": {"sample_ids": ["a"]}, "inner_folds": [{"fold": 0, "train": {"sample_ids": ["a"]}, "validation": {"sample_ids": ["b"]}}]}]}]}
    with pytest.raises(ValueError, match="leakage"):
        _split_ids(splits, E4Config(None, None, None, None))
