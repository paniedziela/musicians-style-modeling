"""Synthetic contract checks only; real 600-output audit stays outside pytest."""

from dataclasses import replace

import mido
import pytest

from musicians_style.content_metrics import measure_content, observe_midi
from musicians_style.midi.printer import MidiPrettyPrinter
from musicians_style.midi.types import InternalRepr, MetaEvent, NoteEvent


def piece(*notes):
    return InternalRepr(480, notes or (NoteEvent(0, 0, 72, 80, 120), NoteEvent(240, 0, 74, 80, 120)),
                        (MetaEvent(0, "time_signature", {"numerator": 4, "denominator": 4}),))


def observe(value):
    return observe_midi(MidiPrettyPrinter().to_bytes(value))


def compare(before, after, **kwargs):
    return measure_content(observe(before), observe(after), **kwargs)


def test_identity_is_exact_and_categories_are_separate():
    result = compare(piece(), piece())
    assert result["v2_exact_pitch"]["status"] == "passed"
    assert all(result["structural_technical"].values())
    assert result["current_transformation"]["protected_velocity_exact"] == "passed"


@pytest.mark.parametrize("field,value,failed_field", [
    ("pitch", 73, "pitch_onset_status"), ("tick", 1, "pitch_onset_status"),
    ("duration_ticks", 121, "note_off_status"), ("channel", 1, "pitch_onset_status"),
])
def test_protected_corruption(field, value, failed_field):
    before = piece()
    after = replace(before, notes=(replace(before.notes[0], **{field: value}), before.notes[1]))
    policy = compare(before, after)["v2_exact_pitch"]
    assert policy["status"] == "failed"
    assert policy[failed_field] == "failed"
    assert policy["missing_events"]
    assert policy["on_event_retention_fraction"] < 1 or policy["off_event_retention_fraction"] < 1


def test_meter_change_does_not_become_melody_damage():
    before = piece()
    after = replace(before, meta=(MetaEvent(0, "time_signature", {"numerator": 3, "denominator": 4}),))
    result = compare(before, after)
    assert result["v2_exact_pitch"]["status"] == "passed"
    assert result["structural_technical"]["meter_preserved"] is False


@pytest.mark.parametrize("field,value,check", [("ticks_per_beat", 960, "resolution_preserved"), ("smf_format", 0, "smf_format_preserved")])
def test_technical_change_is_reported_separately(field, value, check):
    before = piece()
    result = compare(before, replace(before, **{field: value}))
    assert result["structural_technical"][check] is False
    assert result["v2_exact_pitch"]["event_identity_status"] == "passed"


def test_velocity_is_current_transformation_invariant():
    before = piece()
    after = replace(before, notes=tuple(replace(n, velocity=70) for n in before.notes))
    result = compare(before, after)
    assert result["v2_exact_pitch"]["status"] == "passed"
    assert result["current_transformation"]["protected_velocity_exact"] == "failed"


def test_original_shift_policy_and_exact_policy_are_evaluated_separately():
    before = piece()
    after = replace(before, notes=tuple(replace(n, pitch=n.pitch + 3) for n in before.notes))
    result = compare(before, after, historical_shift=3)
    assert result["historical_e3"]["status"] == "passed"
    assert result["v2_exact_pitch"]["status"] == "failed"
    assert compare(before, after, historical_shift=2)["historical_e3"]["status"] == "failed"


@pytest.mark.parametrize("shift", [7, -7, 1.5, True])
def test_invalid_original_shift_is_undefined(shift):
    result = compare(piece(), piece(), historical_shift=shift)
    assert result["historical_e3"]["status"] == "undefined"
    assert result["v2_exact_pitch"]["status"] == "passed"


def test_overlap_repairing_does_not_create_false_note_off_failure():
    # Original Skyline is 72@0 until 200, then 80@100. Moving accompaniment
    # 72@100 to 50 creates an earlier off which FIFO assigns to 72@0.
    before = piece(NoteEvent(0, 0, 72, 80, 200), NoteEvent(100, 0, 72, 70, 200), NoteEvent(100, 0, 80, 80, 100))
    after = piece(NoteEvent(0, 0, 72, 80, 200), NoteEvent(50, 0, 72, 70, 100), NoteEvent(100, 0, 80, 80, 100))
    result = compare(before, after)["v2_exact_pitch"]
    assert result["event_identity_status"] == "passed"
    assert result["note_off_status"] == "passed"
    assert result["fifo_tuple_equal"] is False
    assert result["duration_status"] == "ambiguous"
    assert result["status"] == "ambiguous"


def test_overlap_does_not_hide_a_missing_protected_off():
    before = piece(NoteEvent(0, 0, 72, 80, 200), NoteEvent(100, 0, 72, 80, 200))
    after = replace(before, notes=(replace(before.notes[0], duration_ticks=201), before.notes[1]))
    assert compare(before, after)["v2_exact_pitch"]["note_off_status"] == "failed"


def test_added_high_note_does_not_reselect_protected_melody():
    before = piece()
    after = replace(before, notes=before.notes + (NoteEvent(0, 0, 90, 80, 120),))
    policy = compare(before, after)["v2_exact_pitch"]
    assert policy["event_identity_status"] == "passed"
    assert policy["reselected_output_skyline_equal"] is False
    assert policy["higher_note_at_protected_onset"] == [{"tick": 0, "protected_pitch": 72, "highest_output_pitch": 90}]


def test_source_skyline_tie_is_recorded_with_original_deterministic_choice():
    before = piece(NoteEvent(0, 0, 72, 80, 120), NoteEvent(0, 1, 72, 80, 120))
    result = compare(before, before)
    assert result["protected_note_count"] == 1
    assert result["source_selector_tie_onsets"] == [0]


def test_duplicate_occurrence_order_is_ambiguous_and_counts_are_not_lost():
    before = piece(NoteEvent(0, 0, 72, 80, 120), NoteEvent(0, 0, 72, 80, 120))
    result = compare(before, before)["v2_exact_pitch"]
    assert result["event_identity_status"] == "passed"
    assert result["event_order_status"] == "ambiguous"


def test_empty_source_is_undefined():
    empty = InternalRepr(480)
    assert compare(empty, empty)["v2_exact_pitch"]["status"] == "undefined"
    assert compare(empty, empty)["v2_exact_pitch"]["on_event_retention_fraction"] is None


def test_unclosed_raw_note_is_undefined_even_when_parser_drops_it():
    observation = observe(piece())
    bad = replace(observation, unmatched=((0, 1000, "unclosed_on", 0, 50, 1),))
    assert measure_content(bad, observation)["v2_exact_pitch"]["status"] == "undefined"


def test_smf2_is_undefined_despite_parser_normalization():
    observation = observe(piece())
    assert measure_content(replace(observation, raw_format=2), observation)["v2_exact_pitch"]["status"] == "undefined"


def test_reversed_observable_simultaneous_order_is_not_ignored():
    before = piece(NoteEvent(0, 0, 72, 80, 240), NoteEvent(240, 0, 74, 80, 120))
    observation = observe(before)
    events = list(observation.events)
    tied = [i for i, e in enumerate(events) if e[0] == 240]
    first, second = tied
    events[first] = (*events[first][:6], observation.events[second][6])
    events[second] = (*events[second][:6], observation.events[first][6])
    altered = replace(observation, events=tuple(sorted(events, key=lambda e: (e[0], e[5], e[6]))))
    policy = measure_content(observation, altered)["v2_exact_pitch"]
    assert policy["event_identity_status"] == "passed"
    assert policy["event_order_status"] == "failed"
    assert policy["status"] == "failed"
    historical = measure_content(observation, altered, historical_shift=0)["historical_e3"]
    assert historical["literal_event_order_status"] == "failed"
    assert historical["event_order_status"] == "passed"
    assert historical["original_tuple_protection_status"] == "passed"
    assert historical["status"] == "passed"


def test_cross_track_ties_have_no_observable_total_order():
    observation = observe(piece(NoteEvent(0, 0, 72, 80, 240), NoteEvent(240, 0, 74, 80, 120)))
    events = tuple((*e[:5], 2 if e[0] == 240 and e[1] == "on" else e[5], e[6]) for e in observation.events)
    multi = replace(observation, events=events)
    assert measure_content(multi, multi)["v2_exact_pitch"]["event_order_status"] == "ambiguous"


def test_zero_velocity_note_on_is_observed_as_note_off(tmp_path):
    midi = mido.MidiFile()
    midi.tracks.append(mido.MidiTrack([
        mido.Message("note_on", note=72, velocity=80),
        mido.Message("note_on", note=72, velocity=0, time=120),
    ]))
    path = tmp_path / "zero-off.mid"
    midi.save(path)
    observation = observe_midi(path)
    assert [e[1] for e in observation.events] == ["on", "off"]
    assert not observation.unmatched
    assert measure_content(observation, observation)["v2_exact_pitch"]["status"] == "passed"


def test_overlap_elsewhere_does_not_make_an_isolated_protected_duration_ambiguous():
    before = piece(NoteEvent(0, 0, 72, 80, 20), NoteEvent(100, 0, 72, 80, 100),
                   NoteEvent(150, 0, 72, 80, 100), NoteEvent(100, 0, 80, 80, 20), NoteEvent(150, 0, 82, 80, 20))
    policy = compare(before, before)["v2_exact_pitch"]
    assert policy["pairing_ambiguous_protected_count"] == 0
    assert policy["duration_status"] == "passed"


def test_missing_exact_pitch_does_not_falsely_claim_velocity_changed():
    before = piece()
    after = replace(before, notes=tuple(replace(n, pitch=n.pitch + 1) for n in before.notes))
    result = compare(before, after, historical_shift=1)
    assert result["current_transformation"]["protected_velocity_exact"] == "undefined"
    assert result["current_transformation"]["protected_velocity_historical_e3"] == "passed"


def test_raw_unmatched_off_is_not_discarded(tmp_path):
    midi = mido.MidiFile()
    midi.tracks.append(mido.MidiTrack([mido.Message("note_off", note=72, time=10)]))
    path = tmp_path / "unmatched.mid"
    midi.save(path)
    observation = observe_midi(path)
    assert observation.unmatched == ((0, 10, "unmatched_off", 0, 72),)
    assert "unmatched_note_events" in measure_content(observation, observation)["v2_exact_pitch"]["undefined_reasons"]


def test_raw_unclosed_on_is_not_discarded(tmp_path):
    midi = mido.MidiFile()
    midi.tracks.append(mido.MidiTrack([mido.Message("note_on", note=72, velocity=80)]))
    path = tmp_path / "unclosed.mid"
    midi.save(path)
    observation = observe_midi(path)
    assert observation.unmatched == ((0, 0, "unclosed_on", 0, 72, 1),)


def test_order_failure_on_identifiable_events_survives_other_ambiguity():
    before = piece(NoteEvent(0, 0, 72, 80, 240), NoteEvent(240, 0, 74, 80, 120),
                   NoteEvent(480, 0, 76, 80, 120), NoteEvent(480, 0, 76, 80, 120))
    observation = observe(before)
    events = list(observation.events)
    first, second = [i for i, e in enumerate(events) if e[0] == 240]
    events[first] = (*events[first][:6], observation.events[second][6])
    events[second] = (*events[second][:6], observation.events[first][6])
    altered = replace(observation, events=tuple(sorted(events, key=lambda e: (e[0], e[5], e[6]))))
    assert measure_content(observation, altered)["v2_exact_pitch"]["event_order_status"] == "failed"
