import hashlib
import json
from dataclasses import asdict, replace

import mido
import numpy as np
import pytest

from musicians_style.e3.__main__ import main
from musicians_style.e3.algorithm import SearchConfig
from musicians_style.e3.inference import infer, validate_input, load_profile, json_bytes, SCHEMA
from musicians_style.e3.profile import build_target_profile
from musicians_style.midi.parser import MidiParser
from musicians_style.midi.printer import MidiPrettyPrinter
from musicians_style.midi.types import InternalRepr, NoteEvent, MetaEvent


def piece():
    notes = tuple(NoteEvent(tick=i * 480, channel=0, pitch=p, velocity=80, duration_ticks=360)
                  for i in range(12) for p in (48 + i % 4, 60 + i % 4, 72 + i % 4))
    return InternalRepr(480, notes, (MetaEvent(0, "key_signature", {"key": "C"}),), 1)


@pytest.fixture
def inputs(tmp_path):
    source = tmp_path / "input.mid"
    MidiPrettyPrinter().write(piece(), source)
    training = replace(piece(), notes=tuple(replace(n, pitch=n.pitch + 2) for n in piece().notes))
    profile = build_target_profile("Beethoven", [{"sample_id": "train", "group_id": "train",
        "composer": "Beethoven", "sha256": hashlib.sha256(MidiPrettyPrinter().to_bytes(training)).hexdigest()}],
        {"train": training})
    directory = tmp_path / "profiles"
    directory.mkdir()
    values = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in asdict(profile).items()}
    (directory / "profile.json").write_bytes(json_bytes({"schema": SCHEMA, "profile": values,
                                                        "provenance": {"scope": "synthetic unit fixture"}}))
    return source, directory


def test_cli_roundtrip_determinism_and_report(inputs, tmp_path, capsys):
    source, directory = inputs
    args = ["infer", "--input", str(source), "--target", "Beethoven", "--profiles-dir", str(directory),
            "--generations", "2", "--population-size", "8"]
    assert main(args + ["--output", str(tmp_path / "one.mid")]) == 0
    assert main(args + ["--output", str(tmp_path / "two.mid")]) == 0
    assert (tmp_path / "one.mid").read_bytes() == (tmp_path / "two.mid").read_bytes()
    assert MidiParser().parse(tmp_path / "one.mid").notes
    report = json.loads((tmp_path / "one.json").read_text(encoding="utf-8"))
    assert report["roundtrip_valid"]
    assert report["configuration_label"] == "custom/demo search"
    assert main(args + ["--output", str(tmp_path / "one.mid")]) == 1


def test_identity_explicit(inputs, tmp_path):
    source, directory = inputs
    # Every onset is protected; no pitch-class/accompaniment benefit exists.
    mono = replace(piece(), notes=tuple(n for n in piece().notes if n.pitch >= 72))
    MidiPrettyPrinter().write(mono, source)
    report = infer(source, "Beethoven", tmp_path / "out.mid", profiles_dir=directory,
                   config=SearchConfig(generations=0, population_size=2))
    assert report["status"] == "unchanged"
    assert any("unchanged" in warning for warning in report["warnings"])


def test_missing_corrupt_and_unknown(inputs, tmp_path):
    source, directory = inputs
    with pytest.raises(ValueError, match="Brak profilu"):
        infer(source, "Unknown", tmp_path / "out.mid", profiles_dir=directory)
    with pytest.raises(FileNotFoundError):
        infer(tmp_path / "absent.mid", "Beethoven", tmp_path / "out.mid", profiles_dir=directory)
    source.write_bytes(b"not midi")
    with pytest.raises(ValueError, match="Niepoprawny"):
        infer(source, "Beethoven", tmp_path / "out.mid", profiles_dir=directory)


def test_profile_schema_validation(inputs):
    _, directory = inputs
    path = directory / "profile.json"
    data = json.loads(path.read_text())
    data["profile"]["std"][0] = 0
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="odchyleń"):
        load_profile(path)


@pytest.mark.parametrize("kind", ["control_change", "pitchwheel", "sysex"])
def test_rejects_lossy_events(kind, tmp_path):
    path = tmp_path / "control.mid"
    MidiPrettyPrinter().write(piece(), path)
    raw = mido.MidiFile(path)
    raw.tracks[0].insert(0, mido.Message(kind))
    raw.save(path)
    with pytest.raises(ValueError, match="Nieobsługiwane zdarzenie"):
        validate_input(path.read_bytes(), midi_policy="strict")


def test_no_feasible_output_not_saved(inputs, tmp_path, monkeypatch):
    from musicians_style.e3.algorithm import E3GeneticAlgorithm
    original = E3GeneticAlgorithm.run
    def invalid(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        return replace(result, evaluation=replace(result.evaluation,
            constraints=replace(result.evaluation.constraints, feasible=False)))
    monkeypatch.setattr(E3GeneticAlgorithm, "run", invalid)
    source, directory = inputs
    with pytest.raises(ValueError, match="Brak dopuszczalnego"):
        infer(source, "Beethoven", tmp_path / "out.mid", profiles_dir=directory,
              config=SearchConfig(generations=0, population_size=2))
    assert not (tmp_path / "out.mid").exists()


def test_key_signature_export_follows_transposition(inputs, tmp_path, monkeypatch):
    from musicians_style.e3.algorithm import E3GeneticAlgorithm, SearchResult
    from musicians_style.e3.objective import E3Objective
    from musicians_style.e3.types import E3Genome
    from musicians_style.e3.transformation import apply_transformation
    def fixed(self, source, profile, **kwargs):
        genome = E3Genome(transpose_semitones=2)
        output = apply_transformation(source, genome, profile, seed=1729)
        evaluation = E3Objective(source, profile).evaluate(genome, output)
        return SearchResult(genome, output, evaluation, (), "test", 1, 0)
    monkeypatch.setattr(E3GeneticAlgorithm, "run", fixed)
    source, directory = inputs
    report = infer(source, "Beethoven", tmp_path / "out.mid", profiles_dir=directory)
    assert report["transpose_semitones"] == 2
    assert report["content"]["meta_preserved"] is False
    assert MidiParser().parse(tmp_path / "out.mid").meta[0].payload["key"] == "D"


def test_training_input_rejected(inputs, tmp_path):
    source, directory = inputs
    training = replace(piece(), notes=tuple(replace(n, pitch=n.pitch + 2) for n in piece().notes))
    MidiPrettyPrinter().write(training, source)
    with pytest.raises(ValueError, match="korpusu treningowego"):
        infer(source, "Beethoven", tmp_path / "out.mid", profiles_dir=directory)


@pytest.mark.regression
def test_web_http_uses_same_service_and_download(inputs, tmp_path):
    import threading
    import time
    import urllib.request
    import urllib.error
    from http.server import ThreadingHTTPServer
    from musicians_style.listener.server import Library, make_handler
    source, directory = inputs
    library = Library(tmp_path, tmp_path / "missing.sf2", tmp_path / "cache",
                      profiles=directory, inference_workspace=tmp_path / "jobs")
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(library))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    def get(path):
        with urllib.request.urlopen(base + path, timeout=10) as response:
            return response.read()
    def post(value):
        request = urllib.request.Request(base + "/api/inference", data=json.dumps(value).encode(),
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=10) as response:
            return json.load(response)
    try:
        assert json.loads(get("/api/inference/styles"))["styles"] == ["Beethoven"]
        with pytest.raises(urllib.error.HTTPError) as error:
            post({"input_id": "input.mid", "target": "Beethoven", "output": "../bad.mid"})
        assert error.value.code == 400
        job = post({"input_id": "input.mid", "target": "Beethoven", "generations": 2, "population_size": 8})
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            job = json.loads(get("/api/inference/status?id=" + job["id"]))
            if job["state"] != "running":
                break
            time.sleep(0.02)
        assert job["state"] == "completed", job
        infer(source, "Beethoven", tmp_path / "cli.mid", profiles_dir=directory,
              config=SearchConfig(generations=2, population_size=8))
        assert get("/api/download?id=" + job["output_id"]) == (tmp_path / "cli.mid").read_bytes()
        assert json.loads(get("/api/midi?id=" + job["output_id"]))["notes"]
        assert job["report"]["roundtrip_valid"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
