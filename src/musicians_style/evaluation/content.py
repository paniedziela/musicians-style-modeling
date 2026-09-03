"""Shared, representation-level content and MIDI-validity measurements.

The exact pairing of simultaneous, overlapping MIDI notes with the same channel
and pitch is not observable in the event stream.  For that reason semantic
round-trip comparison uses note-on and note-off multisets instead of dataclass
equality of reconstructed ``NoteEvent`` objects.
"""

from __future__ import annotations

from bisect import bisect_right
from collections import Counter, defaultdict
from typing import Any, Sequence

import numpy as np

from ..midi.types import InternalRepr


def end_tick(piece: InternalRepr) -> int:
    return max((note.tick + note.duration_ticks for note in piece.notes), default=0)


def semantic_midi_equal(left: InternalRepr, right: InternalRepr) -> bool:
    """Compare the observable MIDI events while ignoring ambiguous note pairing."""
    if (
        left.ticks_per_beat != right.ticks_per_beat
        or left.smf_format != right.smf_format
        or left.meta != right.meta
    ):
        return False

    def event_multisets(piece: InternalRepr) -> tuple[Counter[tuple[int, ...]], Counter[tuple[int, ...]]]:
        note_ons: Counter[tuple[int, ...]] = Counter()
        note_offs: Counter[tuple[int, ...]] = Counter()
        for note in piece.notes:
            note_ons[(note.tick, note.channel, note.pitch, note.velocity)] += 1
            note_offs[(note.tick + note.duration_ticks, note.channel, note.pitch)] += 1
        return note_ons, note_offs

    return event_multisets(left) == event_multisets(right)


def onset_f1(before: InternalRepr, after: InternalRepr, tolerance_beats: float = 1 / 16) -> float:
    before_ticks = sorted({note.tick / before.ticks_per_beat for note in before.notes})
    after_ticks = sorted({note.tick / after.ticks_per_beat for note in after.notes})
    used: set[int] = set()
    matches = 0
    for tick in before_ticks:
        candidates = [
            (abs(tick - other), index)
            for index, other in enumerate(after_ticks)
            if index not in used and abs(tick - other) <= tolerance_beats
        ]
        if candidates:
            _, index = min(candidates)
            used.add(index)
            matches += 1
    if not before_ticks and not after_ticks:
        return 1.0
    precision = matches / len(after_ticks) if after_ticks else 0.0
    recall = matches / len(before_ticks) if before_ticks else 0.0
    return float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0


def melody_contour(piece: InternalRepr) -> tuple[int, ...]:
    by_tick: dict[int, list[int]] = defaultdict(list)
    for note in piece.notes:
        by_tick[note.tick].append(note.pitch)
    melody = [max(by_tick[tick]) for tick in sorted(by_tick)]
    return tuple(int(np.sign(right - left)) for left, right in zip(melody, melody[1:]))


def multiset_ngram_jaccard(left: Sequence[Any], right: Sequence[Any], n: int = 3) -> float:
    def grams(values: Sequence[Any]) -> Counter[tuple[Any, ...]]:
        return Counter(tuple(values[index : index + n]) for index in range(max(0, len(values) - n + 1)))

    first, second = grams(left), grams(right)
    if not first and not second:
        return 1.0
    union = sum((first | second).values())
    return float(sum((first & second).values()) / union) if union else 1.0


def max_polyphony(piece: InternalRepr) -> int:
    boundaries = [
        boundary
        for note in piece.notes
        if note.duration_ticks > 0
        for boundary in ((note.tick, 1), (note.tick + note.duration_ticks, -1))
    ]
    active = maximum = 0
    for _, change in sorted(boundaries, key=lambda item: (item[0], item[1])):
        active += change
        maximum = max(maximum, active)
    return maximum


def mean_polyphony(piece: InternalRepr) -> float:
    boundaries = [
        boundary
        for note in piece.notes
        if note.duration_ticks > 0
        for boundary in ((note.tick, 1), (note.tick + note.duration_ticks, -1))
    ]
    if not boundaries:
        return 0.0
    active = area = 0
    previous = min(tick for tick, _ in boundaries)
    for tick, change in sorted(boundaries, key=lambda item: (item[0], item[1])):
        area += active * (tick - previous)
        active += change
        previous = tick
    span = max(tick for tick, _ in boundaries) - min(tick for tick, _ in boundaries)
    return float(area / span) if span else 0.0


def chroma_cosine(left: InternalRepr, right: InternalRepr) -> float:
    def chroma(piece: InternalRepr) -> np.ndarray:
        values = np.zeros(12, dtype=float)
        for note in piece.notes:
            values[note.pitch % 12] += max(0, note.duration_ticks)
        return values

    first, second = chroma(left), chroma(right)
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if not denominator:
        return 1.0 if not first.any() and not second.any() else 0.0
    return float(np.dot(first, second) / denominator)


def _bar_intervals(piece: InternalRepr) -> tuple[tuple[int, int], ...]:
    """Return meter-aware bar intervals without depending on the E3 package."""
    final_tick = end_tick(piece)
    signatures: dict[int, tuple[int, int]] = {0: (4, 4)}
    for event in sorted(piece.meta, key=lambda item: item.tick):
        if event.kind == "time_signature":
            signatures[int(event.tick)] = (
                int(event.payload.get("numerator", 4)),
                int(event.payload.get("denominator", 4)),
            )
    changes = [(tick, *signature) for tick, signature in sorted(signatures.items())]
    bars: list[tuple[int, int]] = []
    for position, (start, numerator, denominator) in enumerate(changes):
        if start > final_tick:
            break
        segment_end = changes[position + 1][0] if position + 1 < len(changes) else final_tick
        segment_end = min(segment_end, final_tick)
        numerator_ticks = piece.ticks_per_beat * numerator * 4
        if numerator <= 0 or denominator <= 0 or denominator & (denominator - 1) or numerator_ticks % denominator:
            raise ValueError(f"invalid time signature at tick {start}")
        bar_length = numerator_ticks // denominator
        cursor = start
        while cursor < segment_end:
            boundary = min(cursor + bar_length, segment_end)
            bars.append((cursor, boundary))
            cursor = boundary
    if not bars and final_tick == 0:
        numerator, denominator = changes[0][1:]
        bars.append((0, piece.ticks_per_beat * numerator * 4 // denominator))
    return tuple(bars)


def empty_bar_ratio(piece: InternalRepr) -> float:
    bars = _bar_intervals(piece)
    starts = tuple(start for start, _ in bars)
    occupied: set[int] = set()
    for note in piece.notes:
        note_end = note.tick + note.duration_ticks
        if note_end <= note.tick:
            continue
        index = max(0, bisect_right(starts, note.tick) - 1)
        while index < len(bars) and bars[index][0] < note_end:
            bar_start, bar_end = bars[index]
            if note.tick < bar_end and note_end > bar_start:
                occupied.add(index)
            index += 1
    return (len(bars) - len(occupied)) / max(1, len(bars))


def content_metrics(
    source: InternalRepr,
    output: InternalRepr,
    *,
    onset_tolerance_beats: float = 1 / 16,
) -> dict[str, Any]:
    source_end, output_end = end_tick(source), end_tick(output)
    source_count, output_count = len(source.notes), len(output.notes)
    source_bars, output_bars = _bar_intervals(source), _bar_intervals(output)
    contour_similarity = multiset_ngram_jaccard(melody_contour(source), melody_contour(output))
    return {
        "nonempty": bool(output.notes),
        "end_tick_input": source_end,
        "end_tick_output": output_end,
        "length_error_ticks": abs(output_end - source_end),
        "length_error": abs(output_end - source_end) / max(1, source_end),
        "bar_count_input": len(source_bars),
        "bar_count_output": len(output_bars),
        "bar_count_preserved": len(source_bars) == len(output_bars),
        "empty_bar_ratio_input": empty_bar_ratio(source),
        "empty_bar_ratio_output": empty_bar_ratio(output),
        "note_count_input": source_count,
        "note_count_output": output_count,
        "note_count_ratio": output_count / max(1, source_count),
        "onset_f1": onset_f1(source, output, onset_tolerance_beats),
        "melody_trigram_jaccard": contour_similarity,
        "chroma_cosine": chroma_cosine(source, output),
        "max_polyphony_input": max_polyphony(source),
        "max_polyphony_output": max_polyphony(output),
        "mean_polyphony_input": mean_polyphony(source),
        "mean_polyphony_output": mean_polyphony(output),
        "zero_duration_ratio_input": sum(n.duration_ticks <= 0 for n in source.notes) / max(1, source_count),
        "zero_duration_ratio_output": sum(n.duration_ticks <= 0 for n in output.notes) / max(1, output_count),
        "ticks_per_beat_preserved": source.ticks_per_beat == output.ticks_per_beat,
        "smf_format_preserved": source.smf_format == output.smf_format,
        "meta_preserved": source.meta == output.meta,
    }
