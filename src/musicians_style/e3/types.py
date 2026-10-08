"""Small, immutable data contracts used by E3."""

from __future__ import annotations

from dataclasses import dataclass

from ..midi.types import InternalRepr
# Retain old import paths, including for historical pickles.
from ..midi.structure import Bar, IndexedNote, NoteId, PieceStructure


@dataclass(frozen=True)
class E3Genome:
    transpose_semitones: int = 0
    rhythm_strength: float = 0.0
    duration_strength: float = 0.0
    thin_strength: float = 0.0
    double_strength: float = 0.0


IDENTITY_GENOME = E3Genome()


@dataclass(frozen=True)
class ConstraintReport:
    feasible: bool
    violations: tuple[str, ...]
    violation_score: float
    roundtrip_valid: bool = True
    length_error: float = 0.0
    note_count_ratio: float = 1.0
    max_polyphony: int = 0


@dataclass(frozen=True)
class CandidateEvaluation:
    genome: E3Genome
    output: InternalRepr
    style_gain: float
    group_gains: dict[str, float]
    constraints: ConstraintReport

