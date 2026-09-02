from musicians_style.e3.structure import analyse_structure, build_bars
from musicians_style.midi.types import InternalRepr, MetaEvent, NoteEvent


def test_structure_marks_one_stable_skyline_note_per_onset(source_piece):
    structure = analyse_structure(source_piece)
    assert sum(note.melody for note in structure.notes) == 4
    assert [note.note.pitch for note in structure.notes if note.melody] == [72, 74, 76, 77]
    assert structure.source is source_piece


def test_meter_change_starts_a_new_bar_segment():
    piece = InternalRepr(
        480,
        (NoteEvent(0, 0, 60, 80, 2400),),
        (MetaEvent(1920, "time_signature", {"numerator": 3, "denominator": 4}),),
    )
    bars = build_bars(piece)
    assert [(bar.start_tick, bar.end_tick, bar.numerator) for bar in bars] == [
        (0, 1920, 4), (1920, 2400, 3)
    ]
