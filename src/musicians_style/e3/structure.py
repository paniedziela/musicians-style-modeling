"""Historical E3 import path for the shared MIDI structure functions."""

from ..midi.structure import (
    analyse_structure,
    bar_for_tick,
    build_bars,
    piece_end_tick,
)

__all__ = ["analyse_structure", "bar_for_tick", "build_bars", "piece_end_tick"]
