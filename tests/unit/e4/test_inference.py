from dataclasses import replace
import json
import time

import pytest
import torch

from musicians_style.e4.experiment import E4Config, SCHEMA_VERSION, _geometry
from musicians_style.e4.model import ConditionalGenerator
from musicians_style.e4.inference import infer, checkpoint_info
from musicians_style.midi.printer import MidiPrettyPrinter
from musicians_style.listener.server import Library
from tests.unit.e3.test_inference import piece
from tests.unit.test_inference_midi_events import with_controls


@pytest.fixture
def checkpoint(tmp_path):
    model = ConditionalGenerator()
    with torch.no_grad():
        model.decoder.weight.zero_()
        model.decoder.bias.zero_()
    path = tmp_path / "fixture.pt"
    torch.save({"schema_version": SCHEMA_VERSION, "geometry": _geometry(E4Config(None,None,None,None)),
                "composer_to_index": {"Bach": 0, "Beethoven": 1, "Chopin": 2}, "generator": model.state_dict(),
                "thresholds": {"onset": .5, "frame": .5}, "epoch": 0}, path)
    return path


def test_e4_roundtrip_progress_and_controls(checkpoint, tmp_path):
    source = tmp_path / "in.mid"
    with_controls(source)
    progress = []
    result = infer(source, "Chopin", tmp_path / "out.mid", checkpoint=checkpoint, progress=progress.append)
    assert result["method"] == "E4.6"
    assert result["experimental"] and result["roundtrip_valid"]
    assert progress[-1]["segment"] == result["segments"]
    assert result["midi_processing"]["preserved"]["control_change"] == 4
    assert result["change_origin"] in {"identity", "representation_only"}
    assert result["model"]["epoch"] == 0  # synthetic fixture, not a trained checkpoint
    assert "style_gain" not in result


def test_e4_rejects_incompatible_checkpoint_and_missing_thresholds(checkpoint):
    state = torch.load(checkpoint, weights_only=True)
    state["schema_version"] = "e4.5.0"
    torch.save(state, checkpoint)
    with pytest.raises(ValueError, match="zgodnym"):
        checkpoint_info(checkpoint)
    state["schema_version"] = SCHEMA_VERSION
    state["thresholds"] = {}
    torch.save(state, checkpoint)
    with pytest.raises(ValueError, match="progów"):
        checkpoint_info(checkpoint)


def test_e4_rejects_multichannel(checkpoint, tmp_path):
    source = tmp_path / "in.mid"
    notes = tuple(replace(n, channel=i % 2) for i, n in enumerate(piece().notes))
    MidiPrettyPrinter().write(replace(piece(), notes=notes), source)
    with pytest.raises(ValueError, match="jednego kanału"):
        infer(source, "Bach", tmp_path / "out.mid", checkpoint=checkpoint)


def test_e4_web_matches_service_and_missing_e3_does_not_block(checkpoint, tmp_path):
    source = tmp_path / "in.mid"
    with_controls(source)
    library = Library(tmp_path, tmp_path / "font.sf2", tmp_path / "cache", e4_checkpoint=checkpoint)
    methods = library.inference.methods()["methods"]
    assert next(item for item in methods if item["id"] == "e4")["styles"] == ["Bach", "Beethoven", "Chopin"]
    job = library.inference.start({"input_id": "in.mid", "target": "Bach", "method": "e4"})
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        job = library.inference.status(job["id"])
        if job["state"] != "running":
            break
        time.sleep(.02)
    assert job["state"] == "completed", job
    infer(source, "Bach", tmp_path / "cli.mid", checkpoint=checkpoint)
    assert library.resolve(job["output_id"]).read_bytes() == (tmp_path / "cli.mid").read_bytes()


def test_unavailable_checkpoint_keeps_e3_option(tmp_path):
    library = Library(tmp_path, tmp_path / "font.sf2", tmp_path / "cache")
    catalog = library.inference.methods()
    assert "e4" in catalog["unavailable"]
    assert [method["id"] for method in catalog["methods"]] == ["e3"]
