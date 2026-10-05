"""Original protected notes survive every permitted synthetic pitch edit."""
from collections import Counter
import pytest
from hypothesis import given, strategies as st
from tests.property.profiles import property_settings
from musicians_style.accompaniment_search import PitchSource
from musicians_style.midi.types import InternalRepr,NoteEvent
from musicians_style.midi.printer import MidiPrettyPrinter
from musicians_style.midi.parser import MidiParser


def synthetic(pitches):
    piece=InternalRepr(480,tuple(n for i,pitch in enumerate(pitches) for n in (
        NoteEvent(i*480,0,pitch,64,200),NoteEvent(i*480,0,80,90,300))),(),0)
    return MidiPrettyPrinter().to_bytes(piece)


@given(st.lists(st.integers(min_value=20,max_value=65),min_size=5,max_size=12),st.integers(-4,4))
@pytest.mark.property
@property_settings(max_examples=200,deadline=None)
def test_property_original_protected_events_never_change(pitches,delta):
    source=PitchSource.from_bytes(synthetic(tuple(pitches)))
    key=source.accompaniment[0]
    _,output=source.move((),key,key.pitch+delta)
    old=Counter((n.tick,n.duration_ticks,n.pitch,n.velocity,n.channel) for key,n,_ in source.notes if key in source.protected)
    available=Counter((n.tick,n.duration_ticks,n.pitch,n.velocity,n.channel) for n in MidiParser().parse_bytes(output).notes)
    assert not old-available
    assert source.original==synthetic(tuple(pitches))
