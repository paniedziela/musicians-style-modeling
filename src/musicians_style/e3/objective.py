"""Grouped style objective and hard feasibility constraints."""

from __future__ import annotations

from collections import Counter

import numpy as np

from ..midi.parser import MidiParser
from ..midi.printer import MidiPrettyPrinter
from ..midi.types import InternalRepr
from ..evaluation.content import max_polyphony, semantic_midi_equal
from .profile import GROUPS, TargetProfile, style_vector
from ..midi.structure import analyse_structure, piece_end_tick
from .types import CandidateEvaluation, ConstraintReport, E3Genome, PieceStructure


def validate_constraints(
    source: InternalRepr,
    output: InternalRepr,
    profile: TargetProfile,
    genome: E3Genome,
    *,
    source_structure: PieceStructure | None = None,
    roundtrip: bool = True,
) -> ConstraintReport:
    violations: list[str] = []
    if not output.notes:
        violations.append("empty_midi")
    valid_roundtrip = True
    if roundtrip:
        try:
            reparsed = MidiParser().parse_bytes(MidiPrettyPrinter().to_bytes(output))
            valid_roundtrip = semantic_midi_equal(reparsed, output)
        except Exception:  # validation boundary: any serializer/parser failure is infeasible
            valid_roundtrip = False
        if not valid_roundtrip:
            violations.append("roundtrip")
    if output.ticks_per_beat != source.ticks_per_beat:
        violations.append("ticks_per_beat")
    if output.smf_format != source.smf_format:
        violations.append("smf_format")
    if output.meta != source.meta:
        violations.append("meta")
    transpose = int(round(genome.transpose_semitones))
    source_structure = source_structure or analyse_structure(source)
    protected = Counter(
        (item.note.tick, item.note.duration_ticks, item.note.pitch + transpose, item.note.channel, item.note.velocity)
        for item in source_structure.notes if item.melody
    )
    available = Counter(
        (note.tick, note.duration_ticks, note.pitch, note.channel, note.velocity)
        for note in output.notes
    )
    if any(available[key] < count for key, count in protected.items()):
        violations.append("melody")
    source_end = piece_end_tick(source)
    output_end = piece_end_tick(output)
    length_error = abs(output_end - source_end) / max(1, source_end)
    quantisation = max(1, source.ticks_per_beat // 16)
    if abs(output_end - source_end) > quantisation or length_error > 0.05:
        violations.append("length")
    ratio = len(output.notes) / max(1, len(source.notes))
    if not 0.9 <= ratio <= 1.1:
        violations.append("note_count")
    polyphony = max_polyphony(output)
    # Identity is the guaranteed feasible fallback.  A source that already
    # exceeds the target corpus maximum must not fail before it is modified;
    # transformations may not make that pre-existing excess worse.
    if polyphony > max(profile.max_polyphony, max_polyphony(source)):
        violations.append("polyphony")
    zero_before = sum(note.duration_ticks <= 0 for note in source.notes)
    zero_after = sum(note.duration_ticks <= 0 for note in output.notes)
    if zero_after > zero_before:
        violations.append("zero_duration_growth")
    return ConstraintReport(
        feasible=not violations,
        violations=tuple(violations),
        violation_score=float(len(violations) + length_error + abs(ratio - 1.0)),
        roundtrip_valid=valid_roundtrip,
        length_error=length_error,
        note_count_ratio=ratio,
        max_polyphony=polyphony,
    )


def grouped_distance(vector: np.ndarray, profile: TargetProfile) -> dict[str, float]:
    z = (vector - profile.mean) / profile.std
    return {
        group: float(np.sqrt(np.mean(np.square(z[np.asarray(profile.feature_groups) == group]))))
        for group in GROUPS
    }


class E3Objective:
    def __init__(self, source: InternalRepr, profile: TargetProfile, *, roundtrip: bool = True) -> None:
        self.source = source
        self.profile = profile
        self.roundtrip = roundtrip
        self.structure = analyse_structure(source)
        self.input_distances = grouped_distance(style_vector(source)[0], profile)

    def evaluate(self, genome: E3Genome, output: InternalRepr) -> CandidateEvaluation:
        distances = grouped_distance(style_vector(output)[0], self.profile)
        gains = {group: self.input_distances[group] - distances[group] for group in GROUPS}
        report = validate_constraints(
            self.source, output, self.profile, genome,
            source_structure=self.structure, roundtrip=self.roundtrip,
        )
        return CandidateEvaluation(genome, output, float(np.mean(list(gains.values()))), gains, report)


def ranking_key(value: CandidateEvaluation) -> tuple[float, float]:
    """Deb ordering: feasible first, then objective/violation magnitude."""
    return (1.0, value.style_gain) if value.constraints.feasible else (0.0, -value.constraints.violation_score)
