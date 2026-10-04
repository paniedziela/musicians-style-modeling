"""V2-03 measurements; frozen experiments never import this module.

Compare the original E3 Skyline mask, not a freshly selected output melody.
Observable on/off events and their order take precedence over FIFO note tuples.
No aggregate content score or musicological melody claim is made.
"""

from __future__ import annotations

import io
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mido

from .e3.structure import analyse_structure
from .evaluation.content import end_tick, max_polyphony, mean_polyphony
from .midi.parser import MidiParser
from .midi.types import InternalRepr


CONTRACT = {
    "version": "v2-03.content.1",
    "selector": "frozen e3.structure.analyse_structure: one highest source note per onset; original deterministic NoteId tie break",
    "melodic_identity": ["absolute_pitch", "onset_tick", "note_off_tick", "duration_ticks", "protected_event_order"],
    "structural_technical": ["meter", "essential_metadata", "smf_format", "ticks_per_beat", "protected_channels"],
    "current_transformation": ["protected_note_on_velocity"],
    "historical_e3": "recorded integer global transposition in [-6,6]; identity relative to that shift; original representation permits reordering at identical ticks",
    "v2_exact_pitch": "zero shift; original absolute pitches",
    "comparison": "multiplicity-aware inclusion of original protected on/off events; accompaniment may change",
    "order": "chronological order and observable within-track simultaneous order; cross-track ties and nonunique occurrence mapping are ambiguous",
    "duration": "paired duration is ambiguous when same-channel/pitch voices overlap; report semantic on/off retention independently",
    "statuses": ["passed", "failed", "ambiguous", "undefined"],
    "undefined": "empty source, asynchronous SMF 2, nonpositive resolution, unmatched raw note events, invalid historical shift",
    "metadata_scope": "frozen parser tempo/time_signature/key_signature/program_change; other raw metadata changes recorded as diagnostics",
    "limits": "MIDI cannot identify indistinguishable voices; event inclusion cannot prove causal lineage. Skyline is an operational proxy. No harmonic/phrase tolerance or aggregate scalar is invented.",
}


@dataclass(frozen=True)
class MidiObservation:
    piece: InternalRepr
    # tick, kind (on/off), channel, pitch, velocity, track, ordinal
    events: tuple[tuple[int, str, int, int, int, int, int], ...]
    raw_format: int
    raw_resolution: int
    unmatched: tuple[tuple[Any, ...], ...]
    other_metadata: tuple[tuple[Any, ...], ...]


def observe_midi(source: Path | bytes) -> MidiObservation:
    """Read bytes once, keeping raw events that the frozen parser may normalize."""
    data = source if isinstance(source, bytes) else source.read_bytes()
    piece = MidiParser().parse_bytes(data)
    midi = mido.MidiFile(file=io.BytesIO(data))
    events, unmatched, metadata = [], [], []
    for track_index, track in enumerate(midi.tracks):
        tick = 0
        active: Counter[tuple[int, int]] = Counter()
        for ordinal, message in enumerate(track):
            tick += message.time
            if message.type in {"note_on", "note_off"}:
                kind = "on" if message.type == "note_on" and message.velocity > 0 else "off"
                key = (message.channel, message.note)
                events.append((tick, kind, *key, message.velocity, track_index, ordinal))
                if kind == "on":
                    active[key] += 1
                elif active[key]:
                    active[key] -= 1
                else:
                    unmatched.append((track_index, tick, "unmatched_off", *key))
            elif message.is_meta and message.type not in {"set_tempo", "time_signature", "key_signature", "end_of_track"}:
                payload = message.dict()
                payload.pop("time", None)
                metadata.append((tick, repr(sorted(payload.items()))))
        unmatched.extend((track_index, tick, "unclosed_on", *key, count) for key, count in active.items() if count)
    return MidiObservation(piece, tuple(sorted(events, key=lambda e: (e[0], e[5], e[6]))),
                           midi.type, midi.ticks_per_beat, tuple(unmatched), tuple(sorted(metadata)))


def _event_key(event: tuple[Any, ...]) -> tuple[Any, ...]:
    return event[:4]


def _ambiguous_ranges(observation: MidiObservation) -> dict[tuple[int, int], list[tuple[int, int]]]:
    """Connected voice components with concurrent same-channel/pitch attacks."""
    active: Counter[tuple[int, int]] = Counter()
    starts = {}
    ambiguous = set()
    ranges: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for tick, kind, channel, pitch, _, _, _ in observation.events:
        key = (channel, pitch)
        if kind == "on":
            if active[key]:
                ambiguous.add(key)
            else:
                starts[key] = tick
            active[key] += 1
        else:
            active[key] = max(0, active[key] - 1)
            if not active[key] and key in ambiguous:
                ranges[key].append((starts[key], tick))
                ambiguous.remove(key)
    return ranges


def _ambiguous_onset(ranges: dict, channel: int, pitch: int, tick: int) -> bool:
    return any(start <= tick < end or start == tick == end for start, end in ranges.get((channel, pitch), ()))


def _order(source: MidiObservation, output: MidiObservation, expected: Counter, shift: int) -> str:
    """Compare identifiable protected events without inventing cross-track order."""
    positions = []
    ambiguous = False
    identifiable = set(expected)
    mappings = []
    for observation, delta in ((source, 0), (output, shift)):
        found: dict[tuple[Any, ...], list[tuple[int, int]]] = defaultdict(list)
        for event in observation.events:
            found[_event_key(event)].append((event[5], event[6]))
        mapping = {}
        for tick, kind, channel, pitch in expected:
            key = (tick, kind, channel, pitch + delta)
            candidates = found[key]
            if len(candidates) < expected[(tick, kind, channel, pitch)]:
                return "failed"
            if len(candidates) != 1 or expected[(tick, kind, channel, pitch)] != 1:
                ambiguous = True
                identifiable.discard((tick, kind, channel, pitch))
                continue
            mapping[(tick, kind, channel, pitch)] = candidates[0]
        mappings.append(mapping)
    # Compare the identifiable subset even if other occurrences are ambiguous.
    for mapping in mappings:
        mapped = [(tick, *mapping[(tick, kind, channel, pitch)], kind, channel, pitch)
                  for tick, kind, channel, pitch in identifiable]
        by_tick: dict[int, set[int]] = defaultdict(set)
        for tick, track, *_ in mapped:
            by_tick[tick].add(track)
        if any(len(tracks) > 1 for tracks in by_tick.values()):
            ambiguous = True
            mapped = [e for e in mapped if len(by_tick[e[0]]) == 1]
        positions.append([ (e[0], *e[3:]) for e in sorted(mapped)])
    # Cross-track ticks must be removed in both observations, rather than
    # interpreting their arbitrary track-number merge as a failed ordering.
    first, second = positions
    shared = set(first) & set(second)
    if [e for e in first if e in shared] != [e for e in second if e in shared]:
        return "failed"
    return "ambiguous" if ambiguous else "passed"


def measure_content(source: MidiObservation, output: MidiObservation, *, historical_shift: int | None = None) -> dict[str, Any]:
    """Return separately named policies/invariants and ambiguity diagnostics."""
    before, after = source.piece, output.piece
    structure_error = None
    try:
        structure = analyse_structure(before) if before.notes and source.raw_resolution > 0 else None
    except ValueError as exc:
        structure, structure_error = None, str(exc)
    selected = [item.note for item in structure.notes if item.melody] if structure else []
    expected = Counter(event for n in selected for event in
                       ((n.tick, "on", n.channel, n.pitch), (n.tick + n.duration_ticks, "off", n.channel, n.pitch)))
    available = Counter(_event_key(e) for e in output.events)
    output_ons = Counter((e[0], e[2], e[3], e[4]) for e in output.events if e[1] == "on")
    output_highest: dict[int, int] = {}
    for tick, kind, channel, pitch, *_ in output.events:
        if kind == "on":
            output_highest[tick] = max(output_highest.get(tick, -1), pitch)
    source_overlap, output_overlap = _ambiguous_ranges(source), _ambiguous_ranges(output)
    try:
        output_structure = analyse_structure(after) if after.notes else None
    except ValueError:
        output_structure = None
    output_skyline = Counter((n.note.tick, n.note.channel, n.note.pitch, n.note.duration_ticks)
                             for n in output_structure.notes if n.melody) if output_structure else Counter()
    undefined = []
    if structure_error:
        undefined.append("invalid_source_structure: " + structure_error)
    if not selected:
        undefined.append("empty_source")
    if source.raw_format == 2 or output.raw_format == 2:
        undefined.append("asynchronous_smf_2")
    if source.raw_resolution <= 0 or output.raw_resolution <= 0:
        undefined.append("nonpositive_resolution")
    if source.unmatched or output.unmatched:
        undefined.append("unmatched_note_events")

    def policy(shift: int, *, valid: bool = True, historical: bool = False) -> dict[str, Any]:
        reasons = undefined + ([] if valid else ["invalid_historical_shift"])
        missing = []
        for (tick, kind, channel, pitch), count in sorted(expected.items()):
            actual = available[(tick, kind, channel, pitch + shift)]
            if actual < count:
                missing.append({"tick": tick, "kind": kind, "channel": channel, "pitch": pitch + shift, "missing": count - actual})
        on_ok = not any(e["kind"] == "on" for e in missing)
        off_ok = not any(e["kind"] == "off" for e in missing)
        missing_on = sum(e["missing"] for e in missing if e["kind"] == "on")
        missing_off = sum(e["missing"] for e in missing if e["kind"] == "off")
        literal_order = _order(source, output, expected, shift) if not reasons else "undefined"
        chronological_order = "undefined" if reasons else "passed" if not missing else "failed"
        order = chronological_order if historical else literal_order
        overlap_notes = [n for n in selected if _ambiguous_onset(source_overlap, n.channel, n.pitch, n.tick)
                         or _ambiguous_onset(output_overlap, n.channel, n.pitch + shift, n.tick)]
        duration = "failed" if not on_ok or not off_ok else "ambiguous" if overlap_notes else "passed"
        semantic = "undefined" if reasons else "passed" if not missing else "failed"
        status = "undefined" if reasons else "failed" if semantic == "failed" or order == "failed" else "ambiguous" if duration == "ambiguous" or order == "ambiguous" else "passed"
        velocities = Counter((n.tick, n.channel, n.pitch + shift, n.velocity) for n in selected)
        velocity_ok = all(output_ons[key] >= count for key, count in velocities.items())
        # Tuple result retained only as a diagnostic of arbitrary FIFO pairing.
        tuples = Counter((n.tick, n.channel, n.pitch, n.duration_ticks) for n in after.notes)
        tuple_ok = all(tuples[(n.tick, n.channel, n.pitch + shift, n.duration_ticks)] >= 1 for n in selected)
        skyline_ok = all(output_skyline[(n.tick, n.channel, n.pitch + shift, n.duration_ticks)] >= 1 for n in selected)
        higher = [{"tick": n.tick, "protected_pitch": n.pitch + shift,
                   "highest_output_pitch": output_highest[n.tick]}
                  for n in selected if output_highest.get(n.tick, -1) > n.pitch + shift]
        return {"shift": shift, "status": status, "event_identity_status": semantic,
                "pitch_onset_status": "undefined" if reasons else "passed" if on_ok else "failed",
                "note_off_status": "undefined" if reasons else "passed" if off_ok else "failed",
                "duration_status": "undefined" if reasons else duration, "event_order_status": order,
                "chronological_order_status": chronological_order, "literal_event_order_status": literal_order,
                "protected_velocity_status": "undefined" if reasons or not on_ok else "passed" if velocity_ok else "failed",
                "fifo_tuple_equal": tuple_ok, "reselected_output_skyline_equal": skyline_ok,
                "pairing_ambiguous_protected_count": len(overlap_notes),
                "on_event_retention_fraction": None if reasons else 1 - missing_on / len(selected),
                "off_event_retention_fraction": None if reasons else 1 - missing_off / len(selected),
                "missing_events": missing, "higher_note_at_protected_onset": higher, "undefined_reasons": reasons}

    meta = lambda piece, kind: tuple(m for m in piece.meta if m.kind == kind)
    result = {"contract_version": CONTRACT["version"], "protected_note_count": len(selected),
              "source_selector_tie_onsets": list(structure.ambiguous_melody_onsets) if structure else [],
              "v2_exact_pitch": policy(0),
              "structural_technical": {
                  "meter_preserved": meta(before, "time_signature") == meta(after, "time_signature"),
                  "essential_metadata_preserved": before.meta == after.meta,
                  "smf_format_preserved": source.raw_format == output.raw_format,
                  "resolution_preserved": source.raw_resolution == output.raw_resolution,
                  "channel_set_preserved": {e[2] for e in source.events} == {e[2] for e in output.events},
                  "protected_channel_set_retained": {n.channel for n in selected} <= {e[2] for e in output.events} if selected else None,
              },
              "validity_structure": {"source_note_count": len(before.notes), "output_note_count": len(after.notes),
                  "source_end_tick": end_tick(before), "output_end_tick": end_tick(after),
                  "source_max_polyphony": max_polyphony(before), "output_max_polyphony": max_polyphony(after),
                  "source_mean_polyphony": mean_polyphony(before), "output_mean_polyphony": mean_polyphony(after),
                  "output_nonempty": bool(after.notes), "source_unmatched_events": list(source.unmatched),
                  "output_unmatched_events": list(output.unmatched)},
              "other_metadata_preserved": source.other_metadata == output.other_metadata}
    if historical_shift is not None:
        valid = isinstance(historical_shift, int) and not isinstance(historical_shift, bool) and -6 <= historical_shift <= 6
        result["historical_e3"] = policy(historical_shift if valid else 0, valid=valid, historical=True)
        original = result["historical_e3"]
        original["original_tuple_protection_status"] = "undefined" if original["undefined_reasons"] else "passed" if original["fifo_tuple_equal"] and original["protected_velocity_status"] == "passed" else "failed"
    # Velocity has its own category; it never enters melodic identity status.
    result["current_transformation"] = {"protected_velocity_exact": result["v2_exact_pitch"]["protected_velocity_status"]}
    if "historical_e3" in result:
        result["current_transformation"]["protected_velocity_historical_e3"] = result["historical_e3"]["protected_velocity_status"]
    return result
