"""Four-bar, meter-aware onset/frame representation used by E4.

This module deliberately has no dependency on the historical one-window
``PianorollDataset``.  A segment always retains its exact bar boundaries, so a
64-step tensor can be mapped back to score ticks under changing meter.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
from bisect import bisect_right

import numpy as np

from ..e3.structure import build_bars, piece_end_tick
from ..e3.types import Bar
from ..midi.types import InternalRepr, NoteEvent, event_key

STEPS_PER_BAR = 16
STEPS_PER_SEGMENT = 64
PITCH_LOW = 24
PITCH_HIGH = 108


@dataclass(frozen=True)
class Segment:
    """A single non-overlapping four-bar window and its model input."""
    index: int
    start_bar: int
    bars: tuple[Bar, ...]
    data: np.ndarray  # [onset/frame, time, pitch]
    mask: np.ndarray  # [time]

    @property
    def start_tick(self) -> int:
        return self.bars[0].start_tick

    @property
    def end_tick(self) -> int:
        return self.bars[-1].end_tick

    @property
    def nonempty(self) -> bool:
        return bool(self.data[:, self.mask.astype(bool), :].any())


@dataclass(frozen=True)
class SegmentMap:
    """Mapping between an entire :class:`InternalRepr` and E4 segments."""
    source: InternalRepr
    bars: tuple[Bar, ...]
    segments: tuple[Segment, ...]
    pitch_low: int = PITCH_LOW
    pitch_high: int = PITCH_HIGH
    end_tick: int = 0

    @property
    def segment_count(self) -> int:
        return len(self.segments)


def _boundaries(bar: Bar) -> tuple[int, ...]:
    # Integer interpolation avoids floats and makes endpoint mapping exact.
    return tuple(bar.start_tick + round(bar.length_ticks * step / STEPS_PER_BAR) for step in range(STEPS_PER_BAR + 1))


def _step_for_tick(bar: Bar, tick: int) -> int:
    bounds = _boundaries(bar)
    if tick <= bounds[0]:
        return 0
    if tick >= bounds[-1]:
        return STEPS_PER_BAR - 1
    return min(range(STEPS_PER_BAR), key=lambda step: (abs(bounds[step] - tick), step))


def _bar_for_tick(bars: tuple[Bar, ...], tick: int, starts: tuple[int, ...] | None = None) -> int | None:
    starts = starts if starts is not None else tuple(bar.start_tick for bar in bars)
    position = bisect_right(starts, tick) - 1
    if position >= 0 and position < len(bars) and tick < bars[position].end_tick:
        return position
    return None


def _local_step(segment: Segment, tick: int) -> int | None:
    for offset, bar in enumerate(segment.bars):
        if bar.start_tick <= tick < bar.end_tick:
            return offset * STEPS_PER_BAR + _step_for_tick(bar, tick)
    return None


def _frame_steps(segment: Segment, start: int, end: int) -> Iterable[int]:
    """Yield all cells whose score-time interval overlaps the note."""
    for offset, bar in enumerate(segment.bars):
        for step in range(STEPS_PER_BAR):
            left, right = _boundaries(bar)[step : step + 2]
            if left < end and right > start:
                yield offset * STEPS_PER_BAR + step


def encode_piece(
    source: InternalRepr, *, pitch_low: int = PITCH_LOW, pitch_high: int = PITCH_HIGH
) -> SegmentMap:
    """Encode every bar, including padded and empty terminal windows."""
    if pitch_high <= pitch_low:
        raise ValueError("pitch_high must exceed pitch_low")
    end_tick = piece_end_tick(source)
    bars = build_bars(source, end_tick=end_tick)
    segments: list[Segment] = []
    bar_starts = tuple(bar.start_tick for bar in bars)
    for index, begin in enumerate(range(0, len(bars), 4)):
        selected = bars[begin : begin + 4]
        data = np.zeros((2, STEPS_PER_SEGMENT, pitch_high - pitch_low), dtype=np.uint8)
        mask = np.zeros(STEPS_PER_SEGMENT, dtype=np.uint8)
        mask[: len(selected) * STEPS_PER_BAR] = 1
        segment = Segment(index, begin, selected, data, mask)
        segments.append(segment)
    # Iterate notes first.  The previous window-first formulation multiplied
    # every note by every segment of a long score, making the audit needlessly
    # quadratic in piece length.
    for note in source.notes:
        if not pitch_low <= note.pitch < pitch_high or note.duration_ticks <= 0:
            continue
        note_end = note.tick + note.duration_ticks
        pitch = note.pitch - pitch_low
        first_bar = _bar_for_tick(bars, note.tick, bar_starts)
        last_bar = _bar_for_tick(bars, max(note.tick, note_end - 1), bar_starts)
        if first_bar is None or last_bar is None:
            continue
        for segment in segments[first_bar // 4 : last_bar // 4 + 1]:
            local_onset = _local_step(segment, note.tick)
            if local_onset is not None:
                segment.data[0, local_onset, pitch] = 1
            for local_frame in _frame_steps(segment, note.tick, note_end):
                segment.data[1, local_frame, pitch] = 1
    return SegmentMap(source, bars, tuple(segments), pitch_low, pitch_high, end_tick)


def _cell_start(segment: Segment, local_step: int) -> int:
    bar = segment.bars[local_step // STEPS_PER_BAR]
    return _boundaries(bar)[local_step % STEPS_PER_BAR]


def _cell_end(segment: Segment, local_step: int) -> int:
    bar = segment.bars[local_step // STEPS_PER_BAR]
    return _boundaries(bar)[local_step % STEPS_PER_BAR + 1]


def quantization_errors(segment_map: SegmentMap) -> list[tuple[str, float, float]]:
    """Return ``(meter, onset_error, end_error)`` for in-range source notes."""
    errors: list[tuple[str, float, float]] = []
    bar_starts = tuple(bar.start_tick for bar in segment_map.bars)
    for note in segment_map.source.notes:
        if not segment_map.pitch_low <= note.pitch < segment_map.pitch_high or note.duration_ticks <= 0:
            continue
        start_bar = _bar_for_tick(segment_map.bars, note.tick, bar_starts)
        end_bar = _bar_for_tick(segment_map.bars, max(note.tick, note.tick + note.duration_ticks - 1), bar_starts)
        if start_bar is None or end_bar is None:
            continue
        start = segment_map.bars[start_bar]
        end = segment_map.bars[end_bar]
        onset = _boundaries(start)[_step_for_tick(start, note.tick)]
        end_bounds = _boundaries(end)
        decoded_end = next((value for value in end_bounds if value >= note.tick + note.duration_ticks), end_bounds[-1])
        errors.append((f"{start.numerator}/{start.denominator}", float(onset - note.tick), float(decoded_end - (note.tick + note.duration_ticks))))
    return errors


def stitch_segments(segment_map: SegmentMap, segments: Iterable[Segment] | None = None) -> InternalRepr:
    """Decode all windows into one full score, retaining meta and out-of-range notes.

    A frame at a window boundary can sustain an already sounding note.  A new
    onset always closes that note first, hence repeated pitches remain distinct.
    """
    provided = tuple(segments if segments is not None else segment_map.segments)
    by_index = {segment.index: segment for segment in provided}
    if set(by_index) != set(range(len(segment_map.segments))):
        raise ValueError("stitch requires exactly one segment for every SegmentMap window")
    grid: list[tuple[int, int, np.ndarray, np.ndarray]] = []
    for expected in segment_map.segments:
        actual = by_index[expected.index]
        if actual.bars != expected.bars or actual.data.shape != expected.data.shape or actual.mask.shape != expected.mask.shape:
            raise ValueError("segment geometry does not match SegmentMap")
        for local in np.flatnonzero(actual.mask):
            grid.append((_cell_start(actual, int(local)), _cell_end(actual, int(local)), actual.data[0, local], actual.data[1, local]))
    grid.sort(key=lambda item: item[0])
    source_by_pitch = {pitch: [n for n in segment_map.source.notes if n.pitch == pitch] for pitch in range(segment_map.pitch_low, segment_map.pitch_high)}
    notes: list[NoteEvent] = [n for n in segment_map.source.notes if not segment_map.pitch_low <= n.pitch < segment_map.pitch_high]
    active: dict[int, tuple[int, int, int]] = {}  # pitch -> start, channel, velocity
    for start, end, onsets, frames in grid:
        for offset in range(segment_map.pitch_high - segment_map.pitch_low):
            pitch = segment_map.pitch_low + offset
            onset, frame = bool(onsets[offset]), bool(frames[offset])
            if onset:
                if pitch in active:
                    note_start, channel, velocity = active.pop(pitch)
                    if start > note_start:
                        notes.append(NoteEvent(note_start, channel, pitch, velocity, start - note_start))
                candidates = source_by_pitch[pitch]
                nearest = min(candidates, key=lambda n: (abs(n.tick - start), n.tick, n.channel)) if candidates else None
                active[pitch] = (start, nearest.channel if nearest else 0, nearest.velocity if nearest else 64)
            if not frame and pitch in active:
                note_start, channel, velocity = active.pop(pitch)
                if start > note_start:
                    notes.append(NoteEvent(note_start, channel, pitch, velocity, start - note_start))
        # Close after examining all pitch cells: frame is active throughout this cell.
        for pitch, (note_start, channel, velocity) in tuple(active.items()):
            offset = pitch - segment_map.pitch_low
            if not bool(frames[offset]):
                continue
            # defer closure to next cell; final closure is handled below
    for pitch, (note_start, channel, velocity) in active.items():
        if segment_map.end_tick > note_start:
            notes.append(NoteEvent(note_start, channel, pitch, velocity, segment_map.end_tick - note_start))
    return InternalRepr(segment_map.source.ticks_per_beat, tuple(sorted(notes, key=event_key)), segment_map.source.meta, segment_map.source.smf_format)
