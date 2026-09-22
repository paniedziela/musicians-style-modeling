"""Explicit performance-event policy at the standalone inference boundary.

The historical note representation and experiment algorithms remain unchanged.
Control events are transported beside it, never included in note-based metrics.
"""
from collections import Counter
import io

import mido

POLICIES = ("preserve", "score-only", "strict")
CORE_EVENTS = {"note_on", "note_off", "program_change", "set_tempo", "time_signature", "key_signature"}
PERFORMANCE_EVENTS = {"control_change", "pitchwheel", "aftertouch"}


def performance_events(raw, policy="preserve"):
    if policy not in POLICIES:
        raise ValueError("Nieznany tryb MIDI: wybierz preserve, score-only lub strict.")
    events, removed = [], Counter()
    for track_index, track in enumerate(raw.tracks):
        tick = 0
        for index, message in enumerate(track):
            tick += message.time
            if message.is_meta or message.type in CORE_EVENTS:
                continue
            if policy == "score-only":
                removed[message.type] += 1
            elif policy == "preserve" and message.type in PERFORMANCE_EVENTS:
                events.append((tick, track_index, index, message.copy(time=0)))
            else:
                raise ValueError(f"Nieobsługiwane zdarzenie {message.type} w trybie {policy}. "
                                 "Wybierz jawny tryb score-only, aby usunąć zdarzenia wykonawcze; "
                                 "control_change, pitchwheel i aftertouch obsługuje tryb preserve.")
    events.sort(key=lambda item: item[:3])
    info = {"policy": policy, "preserved": dict(Counter(event[3].type for event in events)),
            "removed": dict(removed),
            "controllers": sorted({event[3].control for event in events if event[3].type == "control_change"}),
            "metrics_include_performance_events": False}
    return events, info


def performance_warnings(info):
    def counts(values):
        return ", ".join(f"{name}={count}" for name, count in sorted(values.items()))

    warnings = []
    if info["preserved"]:
        warnings.append("Preserved performance events at their original channels, values, and absolute ticks: "
                        + counts(info["preserved"]) + ". The algorithm and note-based metrics do not model "
                        "the sustain pedal (CC64), other controllers, or pitch bend, so their interaction with "
                        "transformed notes may affect playback.")
        warnings.append("When tracks are merged, same-tick performance events are written before notes. "
                        "Controller and program-change order follows the source track order.")
    if info["removed"]:
        warnings.append("The score-only policy removed performance events: " + counts(info["removed"])
                        + ". Sustain-pedal activity was not converted into note durations, so playback may change.")
    return warnings


def export_performance(note_bytes, original_bytes, policy="preserve"):
    """Merge performance + original program ordering onto the note writer's track.

    Keeping bank-select and program changes in one ordered stream avoids moving
    all bank selects ahead of all program changes at the same tick.
    """
    raw = mido.MidiFile(file=io.BytesIO(original_bytes))
    events, info = performance_events(raw, policy)
    if not events:
        return note_bytes, info
    combined = list(events)
    for track_index, track in enumerate(raw.tracks):
        tick = 0
        for index, message in enumerate(track):
            tick += message.time
            if message.type == "program_change":
                combined.append((tick, track_index, index, message.copy(time=0)))
    combined.sort(key=lambda item: item[:3])
    output = mido.MidiFile(file=io.BytesIO(note_bytes))
    note_track = output.tracks[0 if output.type == 0 else 1]
    timeline = [(tick, 1, index, message) for index, (tick, _, _, message) in enumerate(combined)]
    tick = 0
    for index, message in enumerate(note_track):
        tick += message.time
        if message.type in {"end_of_track", "program_change"}:
            continue
        timeline.append((tick, 0 if message.is_meta else 2, index, message.copy(time=0)))
    note_track.clear()
    previous = 0
    for tick, _, _, message in sorted(timeline, key=lambda event: event[:3]):
        note_track.append(message.copy(time=tick-previous))
        previous = tick
    buffer = io.BytesIO()
    output.save(file=buffer)
    encoded = buffer.getvalue()
    # Verify against the file itself, not the internal representation which ignores CC.
    restored, _ = performance_events(mido.MidiFile(file=io.BytesIO(encoded)), "preserve")
    signature = lambda values: [(tick, message.bytes()) for tick, _, _, message in values]
    if signature(events) != signature(restored):
        raise ValueError("Eksport zmienił zdarzenia wykonawcze MIDI.")
    return encoded, info
