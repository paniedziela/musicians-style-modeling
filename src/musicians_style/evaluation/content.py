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
from ..midi.structure import Bar, build_bars, piece_end_tick as end_tick
from ..midi.statistics import max_polyphony, mean_polyphony, polyphony


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


def empty_bar_ratio(piece: InternalRepr) -> float:
    return _empty_bar_ratio(piece, build_bars(piece))


def _empty_bar_ratio(piece: InternalRepr, bars: tuple[Bar, ...]) -> float:
    starts = tuple(bar.start_tick for bar in bars)
    occupied: set[int] = set()
    for note in piece.notes:
        note_end = note.tick + note.duration_ticks
        if note_end <= note.tick:
            continue
        index = max(0, bisect_right(starts, note.tick) - 1)
        while index < len(bars) and bars[index].start_tick < note_end:
            bar = bars[index]
            if note.tick < bar.end_tick and note_end > bar.start_tick:
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
    source_bars, output_bars = build_bars(source), build_bars(output)
    source_mean, source_max = polyphony(source.notes)
    output_mean, output_max = polyphony(output.notes)
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
        "empty_bar_ratio_input": _empty_bar_ratio(source, source_bars),
        "empty_bar_ratio_output": _empty_bar_ratio(output, output_bars),
        "note_count_input": source_count,
        "note_count_output": output_count,
        "note_count_ratio": output_count / max(1, source_count),
        "onset_f1": onset_f1(source, output, onset_tolerance_beats),
        "melody_trigram_jaccard": contour_similarity,
        "chroma_cosine": chroma_cosine(source, output),
        "max_polyphony_input": source_max,
        "max_polyphony_output": output_max,
        "mean_polyphony_input": source_mean,
        "mean_polyphony_output": output_mean,
        "zero_duration_ratio_input": sum(n.duration_ticks <= 0 for n in source.notes) / max(1, source_count),
        "zero_duration_ratio_output": sum(n.duration_ticks <= 0 for n in output.notes) / max(1, output_count),
        "ticks_per_beat_preserved": source.ticks_per_beat == output.ticks_per_beat,
        "smf_format_preserved": source.smf_format == output.smf_format,
        "meta_preserved": source.meta == output.meta,
    }
