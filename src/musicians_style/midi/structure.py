"""Meter-aware bars and deterministic Skyline melody masks."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from bisect import bisect_right

from .types import InternalRepr, NoteEvent, event_key


@dataclass(frozen=True, order=True)
class NoteId:
    """Stable identity of a note, including an ordinal for exact duplicates."""

    tick: int
    channel: int
    pitch: int
    velocity: int
    duration_ticks: int
    duplicate: int = 0


@dataclass(frozen=True)
class IndexedNote:
    note_id: NoteId
    note: NoteEvent
    bar_index: int
    melody: bool = False


@dataclass(frozen=True)
class Bar:
    index: int
    start_tick: int
    end_tick: int
    numerator: int
    denominator: int

    @property
    def length_ticks(self) -> int:
        return self.end_tick - self.start_tick


@dataclass(frozen=True)
class PieceStructure:
    source: InternalRepr
    bars: tuple[Bar, ...]
    notes: tuple[IndexedNote, ...]
    ambiguous_melody_onsets: tuple[int, ...] = field(default_factory=tuple)

    @property
    def melody_ids(self) -> frozenset[NoteId]:
        return frozenset(item.note_id for item in self.notes if item.melody)


def piece_end_tick(repr_: InternalRepr) -> int:
    return max((note.tick + note.duration_ticks for note in repr_.notes), default=0)


def _time_signatures(repr_: InternalRepr) -> list[tuple[int, int, int]]:
    signatures: dict[int, tuple[int, int]] = {0: (4, 4)}
    for event in sorted(repr_.meta, key=lambda item: item.tick):
        if event.kind != "time_signature":
            continue
        numerator = int(event.payload.get("numerator", 4))
        denominator = int(event.payload.get("denominator", 4))
        if numerator <= 0 or denominator <= 0 or denominator & (denominator - 1):
            raise ValueError(f"invalid time signature at tick {event.tick}")
        signatures[int(event.tick)] = (numerator, denominator)
    return [(tick, *signature) for tick, signature in sorted(signatures.items())]


def build_bars(repr_: InternalRepr, *, end_tick: int | None = None) -> tuple[Bar, ...]:
    """Partition the score into bars; a meter change starts a new segment."""
    if repr_.ticks_per_beat <= 0:
        raise ValueError("ticks_per_beat must be positive")
    final_tick = piece_end_tick(repr_) if end_tick is None else int(end_tick)
    if final_tick < 0:
        raise ValueError("end_tick must be non-negative")
    signatures = _time_signatures(repr_)
    bars: list[Bar] = []
    for position, (start, numerator, denominator) in enumerate(signatures):
        if start > final_tick:
            break
        segment_end = signatures[position + 1][0] if position + 1 < len(signatures) else final_tick
        segment_end = min(segment_end, final_tick)
        ticks = repr_.ticks_per_beat * numerator * 4
        if ticks % denominator:
            raise ValueError("time signature cannot be represented exactly in ticks")
        bar_length = ticks // denominator
        cursor = start
        while cursor < segment_end:
            boundary = min(cursor + bar_length, segment_end)
            bars.append(Bar(len(bars), cursor, boundary, numerator, denominator))
            cursor = boundary
    if not bars and final_tick == 0:
        numerator, denominator = signatures[0][1:]
        length = repr_.ticks_per_beat * numerator * 4 // denominator
        bars.append(Bar(0, 0, length, numerator, denominator))
    return tuple(bars)


def _stable_notes(notes: tuple[NoteEvent, ...]) -> list[tuple[NoteId, NoteEvent]]:
    counts: dict[tuple[int, int, int, int, int], int] = defaultdict(int)
    result = []
    for note in sorted(notes, key=lambda n: (*event_key(n), n.duration_ticks)):
        key = (note.tick, note.channel, note.pitch, note.velocity, note.duration_ticks)
        ordinal = counts[key]
        counts[key] += 1
        result.append((NoteId(*key, ordinal), note))
    return result


def analyse_structure(repr_: InternalRepr) -> PieceStructure:
    """Build bars and select exactly one highest note at every onset."""
    bars = build_bars(repr_)
    stable = _stable_notes(repr_.notes)
    by_tick: dict[int, list[tuple[NoteId, NoteEvent]]] = defaultdict(list)
    for pair in stable:
        by_tick[pair[1].tick].append(pair)
    melody_ids: set[NoteId] = set()
    ambiguous: list[int] = []
    for tick, onset in sorted(by_tick.items()):
        top_pitch = max(note.pitch for _, note in onset)
        candidates = sorted(note_id for note_id, note in onset if note.pitch == top_pitch)
        melody_ids.add(candidates[0])
        if len(candidates) > 1:
            ambiguous.append(tick)

    indexed: list[IndexedNote] = []
    starts = [bar.start_tick for bar in bars]
    for note_id, note in stable:
        bar_index = min(len(bars) - 1, max(0, bisect_right(starts, note.tick) - 1))
        containing = bars[bar_index] if bars else None
        if containing is None:
            raise ValueError("cannot assign note to a bar")
        indexed.append(IndexedNote(note_id, note, containing.index, note_id in melody_ids))
    return PieceStructure(repr_, bars, tuple(indexed), tuple(ambiguous))


def bar_for_tick(structure: PieceStructure, tick: int) -> Bar:
    for bar in structure.bars:
        if bar.start_tick <= tick < bar.end_tick:
            return bar
    return structure.bars[-1]
