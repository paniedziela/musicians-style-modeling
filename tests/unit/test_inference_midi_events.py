import io
from dataclasses import replace

import mido
import pytest

from musicians_style.e3.algorithm import SearchConfig
from musicians_style.e3.inference import infer, validate_input
from musicians_style.midi.inference_io import export_performance, performance_events
from musicians_style.midi.parser import MidiParser
from musicians_style.midi.printer import MidiPrettyPrinter
from tests.unit.e3.test_inference import inputs, piece


def with_controls(path, smf=1):
    MidiPrettyPrinter().write(replace(piece(), smf_format=smf), path)
    raw = mido.MidiFile(path)
    # Include same-tick bank/program ordering and a sustain-pedal release after the last note.
    events = [mido.Message("control_change", channel=0, control=0, value=0),
              mido.Message("program_change", channel=0, program=0),
              mido.Message("control_change", channel=0, control=7, value=90),
              mido.Message("control_change", channel=0, control=64, value=127, time=120),
              mido.Message("pitchwheel", channel=0, pitch=100, time=480),
              mido.Message("aftertouch", channel=0, value=30, time=60),
              mido.Message("control_change", channel=0, control=64, value=0, time=6000)]
    if smf == 1:
        raw.tracks.append(mido.MidiTrack(events))
    else:
        # Merge using absolute timing, not naive insertion of delta times.
        raw.tracks[0] = mido.merge_tracks([raw.tracks[0], mido.MidiTrack(events)])
    raw.save(path)
    return raw


@pytest.mark.parametrize("smf", [0, 1])
def test_preserves_events_and_note_timing(inputs, tmp_path, smf):
    source, profiles = inputs
    raw = with_controls(source, smf)
    original_piece, warnings = validate_input(source.read_bytes())
    report = infer(source, "Beethoven", tmp_path / "result.mid", profiles_dir=profiles,
                   config=SearchConfig(generations=1, population_size=4))
    result = mido.MidiFile(tmp_path / "result.mid")
    signature = lambda mid: [(tick, msg.bytes()) for tick, _, _, msg in performance_events(mid)[0]]
    assert signature(result) == signature(raw)
    assert result.type == smf
    assert report["midi_processing"]["controllers"] == [0, 7, 64]
    assert report["midi_processing"]["preserved"]["control_change"] == 4
    assert report["midi_processing"]["removed"] == {}
    assert report["roundtrip_valid"]
    # Bypass with an identity note export as an independent timing check.
    encoded, _ = export_performance(MidiPrettyPrinter().to_bytes(original_piece), source.read_bytes())
    assert MidiParser().parse_bytes(encoded) == original_piece


def test_explicit_score_only_removes_sysex_without_shifting_notes(inputs, tmp_path):
    source, profiles = inputs
    raw = with_controls(source)
    raw.tracks[-1].append(mido.Message("sysex", data=(0x7E, 0x7F, 0x09, 0x01), time=120))
    raw.save(source)
    with pytest.raises(ValueError, match="sysex"):
        validate_input(source.read_bytes())
    piece_before = MidiParser().parse(source)
    encoded, info = export_performance(MidiPrettyPrinter().to_bytes(piece_before), source.read_bytes(), "score-only")
    assert info["removed"]["sysex"] == 1
    assert info["removed"]["control_change"] == 4
    assert not performance_events(mido.MidiFile(file=io.BytesIO(encoded)))[0]
    assert MidiParser().parse_bytes(encoded) == piece_before
    report = infer(source, "Beethoven", tmp_path / "out.mid", profiles_dir=profiles,
                   midi_policy="score-only", config=SearchConfig(generations=0, population_size=2))
    assert report["midi_processing"]["removed"]["sysex"] == 1


def test_score_only_identity_is_not_reported_as_transfer(inputs, tmp_path):
    source, profiles = inputs
    mono = replace(piece(), notes=tuple(n for n in piece().notes if n.pitch >= 72))
    MidiPrettyPrinter().write(mono, source)
    raw = mido.MidiFile(source)
    raw.tracks[0].insert(0, mido.Message("control_change", control=7, value=80))
    raw.save(source)
    report = infer(source, "Beethoven", tmp_path / "out.mid", profiles_dir=profiles,
                   midi_policy="score-only", config=SearchConfig(generations=0, population_size=2))
    assert report["status"] == "normalized_only"
    assert report["note_material_unchanged"]
