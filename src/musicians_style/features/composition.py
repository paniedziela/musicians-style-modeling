"""Tempo-invariant custom93 descriptors, independent of experiment runners.

Feature order, units and numerical conventions retain the E1b contract.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np

from ..midi.types import InternalRepr, NoteEvent
from ..midi.statistics import polyphony

COMPOSITION_FEATURES_SCHEMA_VERSION = "e1.3.0"
FEATURE_GROUPS = ("pitch", "melody", "rhythm", "texture", "harmony", "structure")


def _spec(name: str, group: str, unit: str) -> dict[str, str]:
    return {"name": name, "group": group, "unit": unit}


FEATURE_SPECS: tuple[dict[str, str], ...] = tuple(
    [
        _spec("pitch_range", "pitch", "semitones"),
        _spec("pitch_mean", "pitch", "midi_note"),
        _spec("pitch_std", "pitch", "semitones"),
        _spec("pitch_entropy", "pitch", "bits"),
    ]
    + [_spec(f"pitch_class_{pc}", "pitch", "note_fraction") for pc in range(12)]
    + [
        _spec("directed_interval_mean", "melody", "semitones"),
        _spec("directed_interval_std", "melody", "semitones"),
        _spec("absolute_interval_mean", "melody", "semitones"),
        _spec("absolute_interval_std", "melody", "semitones"),
        _spec("ascending_interval_ratio", "melody", "interval_fraction"),
        _spec("descending_interval_ratio", "melody", "interval_fraction"),
        _spec("repeated_interval_ratio", "melody", "interval_fraction"),
        _spec("stepwise_interval_ratio", "melody", "interval_fraction"),
        _spec("leap_interval_ratio", "melody", "interval_fraction"),
    ]
    + [
        _spec(f"absolute_interval_class_{interval}", "melody", "interval_fraction")
        for interval in range(13)
    ]
    + [
        _spec(f"contour_bigram_{left}_{right}", "melody", "bigram_fraction")
        for left in ("down", "same", "up")
        for right in ("down", "same", "up")
    ]
    + [
        _spec("duration_mean", "rhythm", "quarter_note_beats"),
        _spec("duration_std", "rhythm", "quarter_note_beats"),
        _spec("duration_median", "rhythm", "quarter_note_beats"),
        _spec("duration_cv", "rhythm", "ratio"),
    ]
    + [
        _spec(f"duration_bin_{name}", "rhythm", "note_fraction")
        for name in ("lt_1_4", "1_4_to_1_2", "1_2_to_1", "1_to_2", "ge_2")
    ]
    + [
        _spec("ioi_mean", "rhythm", "quarter_note_beats"),
        _spec("ioi_std", "rhythm", "quarter_note_beats"),
        _spec("ioi_cv", "rhythm", "ratio"),
        _spec("downbeat_onset_ratio", "rhythm", "onset_fraction"),
        _spec("metrical_pulse_onset_ratio", "rhythm", "onset_fraction"),
        _spec("offbeat_onset_ratio", "rhythm", "onset_fraction"),
        _spec("syncopated_note_ratio", "rhythm", "note_fraction"),
    ]
    + [
        _spec("mean_polyphony", "texture", "simultaneous_notes"),
        _spec("max_polyphony", "texture", "simultaneous_notes"),
        _spec("chord_onset_ratio", "texture", "onset_fraction"),
        _spec("chord_size_mean", "texture", "notes_per_onset"),
        _spec("chord_size_std", "texture", "notes_per_onset"),
        _spec("chord_size_max", "texture", "notes_per_onset"),
        _spec("monophonic_onset_ratio", "texture", "onset_fraction"),
    ]
    + [_spec(f"duration_chroma_{pc}", "harmony", "duration_fraction") for pc in range(12)]
    + [
        _spec("chroma_entropy", "harmony", "bits"),
        _spec("tonal_clarity", "harmony", "correlation"),
        _spec("chord_consonance_ratio", "harmony", "pitch_pair_fraction"),
        _spec("successive_chroma_jaccard", "harmony", "ratio"),
        _spec("bass_motion_mean", "harmony", "semitones"),
    ]
    + [
        _spec("pitch_trigram_repeat_ratio", "structure", "trigram_fraction"),
        _spec("pitch_trigram_vocabulary_ratio", "structure", "ratio"),
        _spec("rhythm_trigram_repeat_ratio", "structure", "trigram_fraction"),
        _spec("rhythm_trigram_vocabulary_ratio", "structure", "ratio"),
        _spec("pitch_best_lag_similarity", "structure", "ratio"),
        _spec("rhythm_best_lag_similarity", "structure", "ratio"),
    ]
)


def _distribution(values: Iterable[int], size: int) -> np.ndarray:
    counts = np.zeros(size, dtype=np.float64)
    for value in values:
        counts[int(value)] += 1.0
    total = float(counts.sum())
    return counts / total if total else counts


def _entropy(distribution: np.ndarray) -> float:
    positive = distribution[distribution > 0.0]
    return float(-(positive * np.log2(positive)).sum()) if positive.size else 0.0


def _safe_stats(values: Sequence[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(array.std())


def _ratio(mask: np.ndarray) -> float:
    return float(mask.mean()) if mask.size else 0.0


def _onsets(notes: Sequence[NoteEvent]) -> list[tuple[int, tuple[NoteEvent, ...]]]:
    grouped: dict[int, list[NoteEvent]] = {}
    for note in notes:
        grouped.setdefault(note.tick, []).append(note)
    return [
        (tick, tuple(sorted(grouped[tick], key=lambda note: (note.pitch, note.channel))))
        for tick in sorted(grouped)
    ]


def _meter_at(repr_: InternalRepr, tick: int) -> tuple[int, int, int]:
    """Return the active signature and its segment start tick (default 4/4)."""
    active = (0, 4, 4)
    for event in repr_.meta:
        if event.kind != "time_signature" or event.tick > tick:
            continue
        numerator = int(event.payload.get("numerator", 4))
        denominator = int(event.payload.get("denominator", 4))
        if numerator > 0 and denominator > 0:
            active = (event.tick, numerator, denominator)
    return active


def _metrical_positions(
    repr_: InternalRepr, ticks: Sequence[int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bar_positions: list[float] = []
    pulse_positions: list[float] = []
    pulse_lengths: list[float] = []
    for tick in ticks:
        segment_tick, numerator, denominator = _meter_at(repr_, tick)
        pulse = 4.0 / denominator
        bar = numerator * pulse
        elapsed = (tick - segment_tick) / repr_.ticks_per_beat
        bar_positions.append(elapsed % bar)
        pulse_positions.append(elapsed % pulse)
        pulse_lengths.append(pulse)
    return np.asarray(bar_positions), np.asarray(pulse_positions), np.asarray(pulse_lengths)


def _tonal_clarity(chroma: np.ndarray) -> float:
    # Krumhansl-Kessler key profiles; only the maximum correlation is retained.
    major = np.asarray([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor = np.asarray([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    if not chroma.any() or float(np.std(chroma)) == 0.0:
        return 0.0
    values = []
    for profile in (major, minor):
        for shift in range(12):
            values.append(float(np.corrcoef(chroma, np.roll(profile, shift))[0, 1]))
    return max(values)


def _ngram_stats(sequence: Sequence[Any], size: int = 3) -> tuple[float, float]:
    if len(sequence) < size:
        return 0.0, 0.0
    grams = [tuple(sequence[index : index + size]) for index in range(len(sequence) - size + 1)]
    unique = len(set(grams))
    return float(1.0 - unique / len(grams)), float(unique / len(grams))


def _best_lag_similarity(sequence: Sequence[Any], max_lag: int = 8) -> float:
    if len(sequence) < 2:
        return 0.0
    return max(
        float(np.mean(np.asarray(sequence[lag:]) == np.asarray(sequence[:-lag])))
        for lag in range(1, min(max_lag, len(sequence) - 1) + 1)
    )


def extract_composition_features(repr_: InternalRepr) -> np.ndarray:
    """Extract score-only descriptors measured in beats or meter-relative units."""
    if repr_.ticks_per_beat <= 0:
        raise ValueError("ticks_per_beat must be positive")
    notes = tuple(sorted(repr_.notes, key=lambda note: (note.tick, note.pitch, note.channel)))
    if not notes:
        raise ValueError("composition feature extraction requires at least one note")
    onsets = _onsets(notes)
    onset_ticks = [tick for tick, _ in onsets]
    melody = np.asarray([max(note.pitch for note in chord) for _, chord in onsets], dtype=float)
    pitches = np.asarray([note.pitch for note in notes], dtype=float)

    pitch_classes = _distribution((note.pitch % 12 for note in notes), 12)
    values: list[float] = [
        float(pitches.max() - pitches.min()), float(pitches.mean()), float(pitches.std()),
        _entropy(pitch_classes), *pitch_classes.tolist(),
    ]

    intervals = np.diff(melody)
    absolute = np.abs(intervals)
    interval_mean, interval_std = _safe_stats(intervals.tolist())
    absolute_mean, absolute_std = _safe_stats(absolute.tolist())
    interval_classes = _distribution((min(int(value), 12) for value in absolute), 13)
    contour = np.sign(intervals).astype(int)
    contour_bigrams = _distribution(
        (
            (int(contour[index]) + 1) * 3 + int(contour[index + 1]) + 1
            for index in range(max(0, len(contour) - 1))
        ),
        9,
    )
    values.extend([
        interval_mean, interval_std, absolute_mean, absolute_std,
        _ratio(intervals > 0), _ratio(intervals < 0), _ratio(intervals == 0),
        _ratio(absolute <= 2), _ratio(absolute >= 5),
        *interval_classes.tolist(), *contour_bigrams.tolist(),
    ])

    durations = np.asarray([note.duration_ticks / repr_.ticks_per_beat for note in notes])
    duration_mean = float(durations.mean())
    duration_bins = np.histogram(durations, bins=[-np.inf, 0.25, 0.5, 1.0, 2.0, np.inf])[0]
    duration_bins = duration_bins / len(durations)
    onset_beats = np.asarray(onset_ticks, dtype=float) / repr_.ticks_per_beat
    iois = np.diff(onset_beats)
    ioi_mean, ioi_std = _safe_stats(iois.tolist())
    bar_positions, pulse_positions, pulse_lengths = _metrical_positions(repr_, onset_ticks)
    tolerance = 1e-7
    half_pulse_distance = np.abs(pulse_positions - pulse_lengths / 2.0)
    # Notes sharing an onset also share its metrical position.
    onset_indices = np.repeat(np.arange(len(onsets)), [len(chord) for _, chord in onsets])
    note_positions = pulse_positions[onset_indices]
    note_pulses = pulse_lengths[onset_indices]
    syncopated = (note_positions > tolerance) & (
        durations > note_pulses - note_positions + tolerance
    )
    values.extend([
        duration_mean, float(durations.std()), float(np.median(durations)),
        float(durations.std() / duration_mean) if duration_mean else 0.0,
        *duration_bins.tolist(), ioi_mean, ioi_std, ioi_std / ioi_mean if ioi_mean else 0.0,
        _ratio(np.abs(bar_positions) <= tolerance),
        _ratio(np.abs(pulse_positions) <= tolerance),
        _ratio(half_pulse_distance <= tolerance),
        float(np.mean(syncopated)),
    ])

    chord_sizes = np.asarray([len(chord) for _, chord in onsets], dtype=float)
    mean_polyphony, max_polyphony = polyphony(notes)
    values.extend([
        mean_polyphony, float(max_polyphony), _ratio(chord_sizes >= 2),
        float(chord_sizes.mean()), float(chord_sizes.std()), float(chord_sizes.max()),
        _ratio(chord_sizes == 1),
    ])

    duration_chroma = np.zeros(12, dtype=float)
    for note, duration in zip(notes, durations):
        duration_chroma[note.pitch % 12] += duration
    duration_chroma /= duration_chroma.sum() if duration_chroma.sum() else 1.0
    pitch_pairs: list[int] = []
    chroma_sets: list[set[int]] = []
    basses: list[int] = []
    consonant_classes = {0, 3, 4, 5, 7, 8, 9}
    for _, chord in onsets:
        chroma_sets.append({note.pitch % 12 for note in chord})
        basses.append(min(note.pitch for note in chord))
        for left in range(len(chord)):
            for right in range(left + 1, len(chord)):
                pitch_pairs.append(abs(chord[right].pitch - chord[left].pitch) % 12)
    jaccards = [
        len(left & right) / len(left | right)
        for left, right in zip(chroma_sets, chroma_sets[1:]) if left | right
    ]
    values.extend([
        *duration_chroma.tolist(),
        _entropy(duration_chroma),
        _tonal_clarity(duration_chroma),
        float(np.mean([value in consonant_classes for value in pitch_pairs]))
        if pitch_pairs
        else 0.0,
        float(np.mean(jaccards)) if jaccards else 0.0,
        float(np.mean(np.abs(np.diff(basses)))) if len(basses) > 1 else 0.0,
    ])

    pitch_sequence = [int(value) % 12 for value in melody]
    rhythm_sequence = [int(round(value * 8)) for value in iois]
    pitch_repeat, pitch_vocabulary = _ngram_stats(pitch_sequence)
    rhythm_repeat, rhythm_vocabulary = _ngram_stats(rhythm_sequence)
    values.extend([
        pitch_repeat, pitch_vocabulary, rhythm_repeat, rhythm_vocabulary,
        _best_lag_similarity(pitch_sequence), _best_lag_similarity(rhythm_sequence),
    ])

    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (len(FEATURE_SPECS),) or not np.isfinite(vector).all():
        raise ValueError(f"invalid composition feature vector: shape={vector.shape}")
    return vector
