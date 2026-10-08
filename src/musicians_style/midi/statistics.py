"""Small note-level statistics shared by features and evaluation."""

from collections.abc import Iterable

from .types import InternalRepr, NoteEvent


def polyphony(notes: Iterable[NoteEvent]) -> tuple[float, int]:
    """Return time-weighted mean and maximum over the positive-note span.

    Zero-duration notes do not sound. Note-offs precede note-ons at the same
    tick, so adjacent notes do not overlap. Silence inside the span counts.
    """
    boundaries = sorted(
        boundary
        for note in notes
        if note.duration_ticks > 0
        for boundary in ((note.tick, 1), (note.tick + note.duration_ticks, -1))
    )
    if not boundaries:
        return 0.0, 0
    active = maximum = area = 0
    previous = boundaries[0][0]
    for tick, change in boundaries:
        area += active * (tick - previous)
        active += change
        maximum = max(maximum, active)
        previous = tick
    span = boundaries[-1][0] - boundaries[0][0]
    return float(area / span), maximum


def max_polyphony(piece: InternalRepr) -> int:
    return polyphony(piece.notes)[1]


def mean_polyphony(piece: InternalRepr) -> float:
    return polyphony(piece.notes)[0]
