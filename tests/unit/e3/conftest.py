from __future__ import annotations

import pytest

from musicians_style.midi.types import InternalRepr, MetaEvent, NoteEvent


@pytest.fixture
def source_piece() -> InternalRepr:
    return InternalRepr(
        ticks_per_beat=480,
        notes=(
            NoteEvent(0, 0, 48, 70, 240), NoteEvent(0, 0, 72, 80, 480),
            NoteEvent(480, 0, 50, 70, 240), NoteEvent(480, 0, 74, 80, 480),
            NoteEvent(960, 0, 52, 70, 240), NoteEvent(960, 0, 76, 80, 480),
            NoteEvent(1440, 0, 53, 70, 240), NoteEvent(1440, 0, 77, 80, 480),
        ),
        meta=(MetaEvent(0, "time_signature", {"numerator": 4, "denominator": 4}),),
        smf_format=1,
    )
