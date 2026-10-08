"""L0103-inspired accompaniment pitch moves. No frozen E3 mutation semantics."""
from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass, replace
import copy
import io
import math
import random
import time
from types import SimpleNamespace

import mido
import numpy as np
from sklearn.pipeline import Pipeline

from .content_metrics import observe_midi, measure_content
from .e3.objective import validate_constraints
from .e3.structure import analyse_structure
from .e3.types import NoteId, E3Genome
from .features.composition import extract_composition_features
from .evaluation.content import max_polyphony
from .midi.parser import MidiParser
from .style_metrics import LOGISTIC


class MoveRejected(ValueError):
    """An explicit, reportable infeasible move."""


def _messages(data):
    midi = mido.MidiFile(file=io.BytesIO(data))
    return midi, tuple(tuple(msg.dict() for msg in track) for track in midi.tracks)


@dataclass(frozen=True)
class PitchSource:
    original: bytes
    notes: tuple  # (original NoteId, immutable NoteEvent, on/off raw positions)
    protected: frozenset[NoteId]
    editable: bool

    @classmethod
    def from_bytes(cls, data: bytes):
        data = bytes(data)
        observation = observe_midi(data)
        structure = analyse_structure(observation.piece)
        midi, _ = _messages(data)
        locations = defaultdict(deque)
        for track_index, track in enumerate(midi.tracks):
            active = defaultdict(deque)
            tick = 0
            for ordinal, msg in enumerate(track):
                tick += msg.time
                if msg.type == 'note_on' and msg.velocity > 0:
                    active[msg.channel, msg.note].append((tick, msg.velocity, ordinal))
                elif msg.type in ('note_on', 'note_off'):
                    queue = active[msg.channel, msg.note]
                    if queue:
                        onset, velocity, attack = queue.popleft()
                        key = (onset, msg.channel, msg.note, velocity, tick-onset)
                        locations[key].append(((track_index, attack), (track_index, ordinal)))
        records = []
        for item in structure.notes:
            n = item.note
            key = (n.tick, n.channel, n.pitch, n.velocity, n.duration_ticks)
            positions = locations[key].popleft() if locations[key] else None
            records.append((item.note_id, n, positions))
        # Unmatched events disable edits, but identity always preserves original bytes.
        return cls(data, tuple(records), structure.melody_ids,
                   not observation.unmatched and all(r[2] is not None for r in records))

    @property
    def piece(self):
        # Never expose the mutable metadata payloads as persistent source state.
        return MidiParser().parse_bytes(self.original)

    @property
    def accompaniment(self):
        return tuple(r[0] for r in self.notes if r[0] not in self.protected)

    def canonical(self, pitches=()):
        items = list(pitches.items()) if hasattr(pitches, 'items') else list(pitches)
        if len({key for key, _ in items}) != len(items):
            raise MoveRejected('duplicate_note_id')
        original = {key: note.pitch for key, note, _ in self.notes}
        clean = []
        for key, pitch in items:
            if key not in original:
                raise MoveRejected('unknown_note_id')
            if isinstance(pitch, bool) or not isinstance(pitch, int):
                raise MoveRejected('integer_pitch_required')
            if key in self.protected and pitch != original[key]:
                raise MoveRejected('protected_event')
            if not 0 <= pitch <= 127:
                raise MoveRejected('pitch_range')
            if abs(pitch-original[key]) > 4:
                raise MoveRejected('original_pitch_radius')
            if pitch != original[key]:
                clean.append((key, pitch))
        return tuple(sorted(clean))

    def serialize(self, pitches=(), *, changed_fraction=.20):
        if not math.isfinite(changed_fraction) or not 0 <= changed_fraction <= 1:
            raise ValueError('changed_fraction must be in [0,1]')
        state = self.canonical(pitches)
        if not state:
            return self.original
        if not self.editable:
            raise MoveRejected('unmatched_or_unmapped_source')
        if len(state) > math.floor(len(self.accompaniment)*changed_fraction):
            raise MoveRejected('changed_note_ceiling')
        edits = dict(state)
        expected = tuple((key, replace(note, pitch=edits.get(key, note.pitch)))
                         for key, note, _ in self.notes)
        # Only newly introduced same-channel/pitch collision pairs are forbidden.
        def collisions(rows):
            groups = defaultdict(list)
            for key, note in rows:
                groups[note.channel, note.pitch].append((key, note))
            found = set()
            for group in groups.values():
                group.sort(key=lambda row: (row[1].tick, row[0]))
                for i, (key, note) in enumerate(group):
                    for other_key, other in group[i+1:]:
                        if other.tick > note.tick+note.duration_ticks:
                            break
                        if other.tick == note.tick or other.tick < note.tick+note.duration_ticks:
                            found.add(frozenset((key, other_key)))
            return found
        original = tuple((key, note) for key, note, _ in self.notes)
        if collisions(expected)-collisions(original):
            raise MoveRejected('same_channel_pitch_overlap')
        protected = [n for key, n in original if key in self.protected]
        for key, note in expected:
            if key in edits and any(note.pitch > melody.pitch and
                    (note.tick == melody.tick or note.tick <= melody.tick < note.tick+note.duration_ticks)
                    for melody in protected):
                raise MoveRejected('overtops_protected_attack')
        midi, before = _messages(self.original)
        changed_positions = {}
        for key, _, positions in self.notes:
            if key in edits:
                for track, ordinal in positions:
                    midi.tracks[track][ordinal] = midi.tracks[track][ordinal].copy(note=edits[key])
                    changed_positions[track, ordinal] = edits[key]
        buffer = io.BytesIO()
        midi.save(file=buffer)
        data = buffer.getvalue()
        restored, after = _messages(data)
        if (restored.type, restored.ticks_per_beat, len(after)) != (midi.type, midi.ticks_per_beat, len(before)):
            raise MoveRejected('raw_structure')
        for track, (old, new) in enumerate(zip(before, after)):
            if len(old) != len(new):
                raise MoveRejected('raw_event_count')
            for ordinal, (left, right) in enumerate(zip(old, new)):
                allowed = dict(left)
                if (track, ordinal) in changed_positions:
                    allowed['note'] = changed_positions[track, ordinal]
                if allowed != right:
                    raise MoveRejected('raw_message_invariant')
        piece = MidiParser().parse_bytes(data)
        if Counter(piece.notes) != Counter(n for _, n in expected):
            raise MoveRejected('roundtrip_note_lineage')
        source = self.piece
        constraints = validate_constraints(source, piece,
            SimpleNamespace(max_polyphony=max_polyphony(source)), E3Genome(),
            source_structure=analyse_structure(source), roundtrip=False)
        if not constraints.feasible:
            raise MoveRejected('e3_constraints:'+','.join(constraints.violations))
        # Original on/off positions already give stronger protection; V2-03 is additional evidence.
        content = measure_content(observe_midi(self.original), observe_midi(data))
        exact = content['v2_exact_pitch']
        if exact['event_identity_status'] != 'passed' or exact['duration_status'] in ('failed','undefined'):
            raise MoveRejected('protected_content')
        if exact['protected_velocity_status'] != 'passed' or not all(content['structural_technical'].values()):
            raise MoveRejected('content_structure')
        return data

    def move(self, state, note_id, pitch, *, changed_fraction=.20):
        if note_id in self.protected:
            raise MoveRejected('protected_event')
        values = dict(self.canonical(state))
        values[note_id] = pitch
        candidate = self.canonical(values)
        return candidate, self.serialize(candidate, changed_fraction=changed_fraction)

    def note_table(self, state=()):
        edits = dict(self.canonical(state))
        return [dict(note_id=repr(key), protected=key in self.protected,
            tick=n.tick, duration=n.duration_ticks, velocity=n.velocity, channel=n.channel,
            before=n.pitch, after=edits.get(key,n.pitch)) for key,n,_ in self.notes]


class Logistic93Objective:
    """Read-only DEVELOPMENT target probability; no fit, external metric or E3 scorer."""
    def __init__(self, fitted: Pipeline, target: str):
        if not isinstance(fitted, Pipeline) or [name for name,_ in fitted.steps] != ['variance','scale','model']:
            raise ValueError('frozen logistic93 pipeline required')
        if fitted.n_features_in_ != 93 or any(fitted[-1].get_params()[k] != v for k,v in LOGISTIC.items()):
            raise ValueError('frozen logistic93 configuration required')
        if target not in fitted.classes_:
            raise ValueError('unknown target')
        self._fitted = copy.deepcopy(fitted)
        self._index = list(fitted.classes_).index(target)

    def __call__(self, data: bytes):
        vector = extract_composition_features(MidiParser().parse_bytes(data))
        value = float(self._fitted.predict_proba(vector.reshape(1,-1))[0,self._index])
        if not np.isfinite(value):
            raise ValueError('undefined logistic93 affinity')
        return value


@dataclass(frozen=True)
class SearchResult:
    state: tuple
    midi: bytes
    affinity: float
    gain: float
    history: tuple
    accounting: dict


def local_search(source: PitchSource, objective, *, seed=1729, proposals=512, changed_fraction=.20):
    """Budget includes identity; full memory, strict improvement, stable original IDs."""
    if isinstance(proposals,bool) or not isinstance(proposals,int) or proposals < 1:
        raise ValueError('positive proposal budget required')
    # Validate configuration even if there are no accompaniment notes.
    source.serialize((),changed_fraction=changed_fraction)
    started = time.perf_counter()
    baseline = float(objective(source.original))
    if not math.isfinite(baseline):
        raise ValueError('nonfinite identity affinity')
    rng = random.Random(seed)
    state, best, data = (), baseline, source.original
    visited = {()}
    history = [dict(proposal=0,status='identity',affinity=baseline,state=())]
    counts = Counter(proposals=1,unique_evaluations=1,accepted=0,cache_hits=0,rejections=0)
    reasons = Counter()
    for index in range(1,proposals):
        counts['proposals'] += 1
        if not source.accompaniment:
            counts['rejections'] += 1
            reasons['no_accompaniment'] += 1
            history.append(dict(proposal=index,status='rejected',reason='no_accompaniment'))
            continue
        key = rng.choice(source.accompaniment)
        pitch = key.pitch+rng.choice((-4,-3,-2,-1,0,1,2,3,4))
        raw = dict(state)
        raw[key] = pitch
        # Rejected valid-map states also enter visited memory.
        try:
            candidate = source.canonical(raw)
            if candidate in visited:
                counts['cache_hits'] += 1
                history.append(dict(proposal=index,status='visited'))
                continue
            visited.add(candidate)
            output = source.serialize(candidate,changed_fraction=changed_fraction)
            value = float(objective(output))
            if not math.isfinite(value):
                raise ValueError('nonfinite affinity')
            counts['unique_evaluations'] += 1
            accepted = value > best+1e-12
            if accepted:
                state,best,data = candidate,value,output
                counts['accepted'] += 1
            history.append(dict(proposal=index,status='accepted' if accepted else 'nonimproving',
                                affinity=value,state=[dict(note_id=asdict(key),pitch=pitch) for key,pitch in candidate]))
        except Exception as exc:
            counts['rejections'] += 1
            reason = str(exc) if isinstance(exc,MoveRejected) else type(exc).__name__+':'+str(exc)
            reasons[reason] += 1
            history.append(dict(proposal=index,status='rejected',reason=reason))
    return SearchResult(state,data,best,best-baseline,tuple(history),dict(counts,
        rejection_reasons=dict(reasons),visited_states=len(visited),seed=seed,
        stop_reason='proposal_budget',runtime_seconds=time.perf_counter()-started))
