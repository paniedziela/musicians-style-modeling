"""Small, immutable data contracts used by E3."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..midi.types import InternalRepr, NoteEvent


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

