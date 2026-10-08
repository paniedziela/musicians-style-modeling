"""Deterministic, bar-local E3 transformations."""

from __future__ import annotations

import hashlib
from dataclasses import replace

import numpy as np

from ..midi.types import InternalRepr, NoteEvent, event_key
from .profile import DURATION_EDGES, TargetProfile
from ..midi.structure import analyse_structure
from .types import E3Genome, PieceStructure


def _unit(seed: int, stage: str, key: object) -> float:
    digest = hashlib.sha256(f"{seed}|{stage}|{key!r}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def _clamp_strength(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def _preferred_bin(histogram: np.ndarray, current: int, key: object) -> int:
    positive = np.flatnonzero(histogram > 0)
    if not positive.size:
        return current
    # Stable tie breaking favours a nearby, frequent target position.
    return min(
        (int(index) for index in positive),
        key=lambda index: (-float(histogram[index]), abs(index - current), repr(key), index),
    )


def _duration_target_ticks(profile: TargetProfile, ticks_per_beat: int, current: int) -> int:
    target_bin = _preferred_bin(profile.duration_histogram, 0, current)
    low = float(DURATION_EDGES[target_bin])
    high = float(DURATION_EDGES[target_bin + 1])
    beats = low * 1.5 if not np.isfinite(high) else (low + high) / 2.0
    return max(1, int(round(beats * ticks_per_beat)))


def apply_transformation(
    source: InternalRepr,
    genome: E3Genome,
    profile: TargetProfile,
    *,
    seed: int,
    structure: PieceStructure | None = None,
) -> InternalRepr:
    """Apply the five E3 genes without modifying protected melody timing."""
    if genome == E3Genome():
        return source
    structure = structure or analyse_structure(source)
    transpose = int(round(genome.transpose_semitones))
    if any(not 0 <= item.note.pitch + transpose <= 127 for item in structure.notes):
        transpose = 0
    rhythm = _clamp_strength(genome.rhythm_strength)
    duration = _clamp_strength(genome.duration_strength)
    thin = _clamp_strength(genome.thin_strength)
    double = _clamp_strength(genome.double_strength)

    transformed: dict[object, NoteEvent] = {}
    by_onset: dict[int, list] = {}
    for item in structure.notes:
        by_onset.setdefault(item.note.tick, []).append(item)

    for onset_tick, onset in sorted(by_onset.items()):
        bar = structure.bars[onset[0].bar_index]
        position = (onset_tick - bar.start_tick) / max(1, bar.length_ticks)
        current_bin = min(15, int(position * 16))
        desired_bin = _preferred_bin(profile.onset_histogram, current_bin, onset_tick)
        desired_tick = bar.start_tick + int(round((desired_bin + 0.5) * bar.length_ticks / 16.0))
        moved_tick = int(round(onset_tick + rhythm * (desired_tick - onset_tick)))
        moved_tick = min(bar.end_tick - 1, max(bar.start_tick, moved_tick))

        accompaniment = [item for item in onset if not item.melody]
        texture_mode = "thin" if thin >= double else "double"
        removable = sorted(accompaniment, key=lambda item: (item.note.pitch, item.note_id))
        remove_count = int(round(len(removable) * thin)) if texture_mode == "thin" and len(onset) > 1 else 0
        remove_count = min(remove_count, max(0, len(onset) - 1))
        removed = {item.note_id for item in removable[:remove_count]}

        for item in onset:
            note = item.note
            if item.note_id in removed:
                continue
            tick = note.tick if item.melody else moved_tick
            target_duration = _duration_target_ticks(profile, source.ticks_per_beat, note.duration_ticks)
            if item.melody:
                new_duration = note.duration_ticks
            else:
                new_duration = int(round(note.duration_ticks + duration * (target_duration - note.duration_ticks)))
                new_duration = max(1, min(new_duration, bar.end_tick - tick))
            transformed[item.note_id] = replace(
                note, tick=tick, pitch=note.pitch + transpose, duration_ticks=new_duration
            )

        if texture_mode == "double" and accompaniment and _unit(seed, "double", onset_tick) < double:
            room = max(0, profile.max_polyphony - len(onset))
            if room:
                source_item = min(accompaniment, key=lambda item: (item.note.pitch, item.note_id))
                base = transformed.get(source_item.note_id)
                if base is not None:
                    melody_pitch = max(item.note.pitch + transpose for item in onset if item.melody)
                    choices = [pitch for pitch in (base.pitch - 12, base.pitch + 12) if 0 <= pitch <= 127 and pitch < melody_pitch]
                    if choices:
                        duplicate = replace(base, pitch=max(choices))
                        transformed[(source_item.note_id, "octave")] = duplicate

    notes = tuple(sorted(transformed.values(), key=event_key))
    return InternalRepr(source.ticks_per_beat, notes, source.meta, source.smf_format)
