"""Leakage-safe fold-local target profiles for E3."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from ..midi.types import InternalRepr
from ..midi.statistics import max_polyphony
from ..midi.structure import analyse_structure

GROUPS = ("pitch", "rhythm", "texture")
DURATION_EDGES = np.asarray([0.0, 0.25, 0.5, 1.0, 2.0, 4.0, np.inf])


@dataclass(frozen=True)
class TargetProfile:
    composer: str
    feature_names: tuple[str, ...]
    feature_groups: tuple[str, ...]
    mean: np.ndarray
    std: np.ndarray
    onset_histogram: np.ndarray
    duration_histogram: np.ndarray
    pitch_class_histogram: np.ndarray
    interval_histogram: np.ndarray
    chord_size_histogram: np.ndarray
    max_polyphony: int
    train_sample_ids: tuple[str, ...]
    train_group_ids: tuple[str, ...]
    train_sha256: tuple[str, ...]
    fingerprint: str


def _normalise(counts: np.ndarray) -> np.ndarray:
    total = float(counts.sum())
    return counts / total if total else np.zeros_like(counts, dtype=float)


def style_vector(repr_: InternalRepr) -> tuple[np.ndarray, tuple[str, ...], tuple[str, ...]]:
    """Features intentionally controllable by the E3 operators."""
    structure = analyse_structure(repr_)
    accompaniment = [item for item in structure.notes if not item.melody]
    onset_counts = np.zeros(16, dtype=float)
    durations: list[float] = []
    pitch_classes = np.zeros(12, dtype=float)
    intervals = np.zeros(25, dtype=float)
    by_tick: dict[int, list[int]] = {}
    melody_by_tick = {item.note.tick: item.note.pitch for item in structure.notes if item.melody}
    for item in accompaniment:
        bar = structure.bars[item.bar_index]
        position = (item.note.tick - bar.start_tick) / max(1, bar.length_ticks)
        onset_counts[min(15, int(position * 16))] += 1
        durations.append(item.note.duration_ticks / repr_.ticks_per_beat)
        pitch_classes[item.note.pitch % 12] += 1
        delta = item.note.pitch - melody_by_tick.get(item.note.tick, item.note.pitch)
        intervals[min(24, max(0, delta + 12))] += 1
        by_tick.setdefault(item.note.tick, []).append(item.note.pitch)
    duration_hist = np.histogram(durations, bins=DURATION_EDGES)[0].astype(float)
    chord_sizes = np.zeros(8, dtype=float)
    full_onsets: dict[int, int] = {}
    for item in structure.notes:
        full_onsets[item.note.tick] = full_onsets.get(item.note.tick, 0) + 1
    for size in full_onsets.values():
        chord_sizes[min(7, max(0, size - 1))] += 1
    parts = [_normalise(onset_counts), _normalise(duration_hist), _normalise(pitch_classes),
             _normalise(intervals), _normalise(chord_sizes)]
    names = tuple(
        [f"onset_bin_{i}" for i in range(16)]
        + [f"duration_bin_{i}" for i in range(6)]
        + [f"pitch_class_{i}" for i in range(12)]
        + [f"melody_interval_{i - 12}" for i in range(25)]
        + [f"chord_size_{i + 1}" for i in range(8)]
    )
    groups = tuple(["rhythm"] * 22 + ["pitch"] * 37 + ["texture"] * 8)
    return np.concatenate(parts), names, groups


def build_target_profile(
    composer: str,
    rows: Sequence[Mapping[str, Any]],
    representations: Mapping[str, InternalRepr],
    *,
    forbidden_rows: Sequence[Mapping[str, Any]] = (),
) -> TargetProfile:
    """Build a target profile and prove sample/group/SHA disjointness first."""
    selected = sorted((row for row in rows if row.get("composer") == composer), key=lambda r: str(r["sample_id"]))
    if not selected:
        raise ValueError(f"no training samples for target composer {composer!r}")
    for field in ("sample_id", "group_id", "sha256"):
        train_values = {str(row.get(field)) for row in selected if row.get(field)}
        forbidden = {str(row.get(field)) for row in forbidden_rows if row.get(field)}
        overlap = train_values & forbidden
        if overlap:
            raise ValueError(f"{field} leakage in target profile: {sorted(overlap)}")
    vectors = []
    names: tuple[str, ...] = ()
    groups: tuple[str, ...] = ()
    for row in selected:
        sample_id = str(row["sample_id"])
        if sample_id not in representations:
            raise ValueError(f"missing representation for {sample_id}")
        vector, names, groups = style_vector(representations[sample_id])
        vectors.append(vector)
    matrix = np.asarray(vectors, dtype=float)
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    ids = tuple(str(row["sample_id"]) for row in selected)
    group_ids = tuple(sorted({str(row["group_id"]) for row in selected}))
    hashes = tuple(sorted({str(row["sha256"]) for row in selected}))
    fingerprint = hashlib.sha256(json.dumps({"sample_ids": ids, "group_ids": group_ids, "sha256": hashes}, sort_keys=True).encode()).hexdigest()
    slices = (16, 22, 34, 59, 67)
    return TargetProfile(
        composer=composer, feature_names=names, feature_groups=groups, mean=mean, std=std,
        onset_histogram=_normalise(matrix[:, :slices[0]].sum(axis=0)),
        duration_histogram=_normalise(matrix[:, slices[0]:slices[1]].sum(axis=0)),
        pitch_class_histogram=_normalise(matrix[:, slices[1]:slices[2]].sum(axis=0)),
        interval_histogram=_normalise(matrix[:, slices[2]:slices[3]].sum(axis=0)),
        chord_size_histogram=_normalise(matrix[:, slices[3]:slices[4]].sum(axis=0)),
        max_polyphony=max(1, max(max_polyphony(representations[str(row["sample_id"])]) for row in selected)),
        train_sample_ids=ids, train_group_ids=group_ids, train_sha256=hashes, fingerprint=fingerprint,
    )
