"""MIDI timing, library boundaries, synthesis cache and browser HTTP contract."""
import hashlib
import io
import json
import threading
import urllib.error
import urllib.request
import wave
from http.server import ThreadingHTTPServer
from pathlib import Path

import mido
import numpy as np
import pretty_midi
import pytest

from musicians_style.listener.server import Library, make_handler


def midi_bytes():
    midi = mido.MidiFile()
    track = mido.MidiTrack()
    midi.tracks.append(track)
    track.extend([
        mido.MetaMessage("set_tempo", tempo=500000),
        mido.Message("note_on", note=60, velocity=80),
        mido.Message("note_off", note=60, time=480),
        mido.MetaMessage("set_tempo", tempo=1000000),
        mido.Message("note_on", note=64, velocity=90),
        mido.Message("note_off", note=64, time=480),
    ])
    buffer = io.BytesIO()
    midi.save(file=buffer)
    return buffer.getvalue()


@pytest.fixture
def library(tmp_path):
    root = tmp_path / "library"
    root.mkdir()
    (root / "piece.mid").write_bytes(midi_bytes())
    font = tmp_path / "piano.sf2"
    font.touch()
    return Library(root, font, tmp_path / "cache")


def test_timing_includes_tempo_changes(library):
    data = library.describe("piece.mid")
    assert data["duration"] == pytest.approx(1.5)
    assert data["notes"][0][:3] == [0, 0.5, 60]
    assert data["notes"][1][:3] == [0.5, 1.5, 64]
    assert data["path"] == "piece.mid"
    assert data["sha256"] == hashlib.sha256(midi_bytes()).hexdigest()


@pytest.mark.parametrize("file_id", ["../outside.mid", "piece.txt", "upload:unknown"])
def test_library_rejects_paths_outside_root(library, file_id):
    with pytest.raises((ValueError, FileNotFoundError)):
        library.resolve(file_id)


def test_upload_validates_and_preserves_filename(library):
    data = library.upload(midi_bytes(), "../mój utwór.mid")
    assert data["name"] == "mój utwór.mid"
    assert library.resolve(data["id"]).read_bytes() == midi_bytes()
    with pytest.raises(ValueError, match="odczytać"):
        library.upload(b"invalid MIDI", "bad.mid")


def test_catalog_handles_experiment_metadata_and_original(library):
    task = library.root / "experiments" / "e3_test" / "tasks" / "task1"
    task.mkdir(parents=True)
    (task / "output.mid").write_bytes(midi_bytes())
    (task / "result.json").write_text(json.dumps({"source_id": "work1", "target_composer": "Bach"}))
    inputs = task.parents[1] / "inputs"
    inputs.mkdir()
    (inputs / "manifest.json").write_text(json.dumps({
        "dataset_root": ".", "samples": [{"sample_id": "work1", "score_path": "piece.mid"}]}))
    output = next(e for e in library.catalog() if e["target"])
    assert output["experiment"] == "e3_test"
    assert output["folder"] == "experiments/e3_test/tasks/task1"
    assert output["path"] == "experiments/e3_test/tasks/task1/output.mid"
    assert output["original"] == "piece.mid"


def test_render_cache_and_soundfont_changes(library, monkeypatch):
    calls = []
    def synth(self, **kwargs):
        calls.append(kwargs)
        return np.array([0, 2, -2, 0], dtype=float)
    monkeypatch.setattr(pretty_midi.PrettyMIDI, "fluidsynth", synth)
    path = library.render("piece.mid")
    assert library.render("piece.mid") == path
    assert len(calls) == 1
    with wave.open(str(path)) as audio:
        assert audio.getframerate() == 22050
        assert audio.getnframes() == 4
    library.soundfont.write_bytes(b"updated")
    assert library.render("piece.mid") != path
    assert len(calls) == 2


@pytest.fixture
def server(library):
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(library))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()
    thread.join()


@pytest.mark.regression
def test_browser_routes_upload_and_byte_ranges(server, monkeypatch):
    monkeypatch.setattr(pretty_midi.PrettyMIDI, "fluidsynth", lambda self, **kw: np.zeros(100))
    with urllib.request.urlopen(server + "/") as response:
        assert b'lang="pl"' in response.read()
    with urllib.request.urlopen(server + "/details.css") as response:
        assert response.headers["Content-Type"].startswith("text/css")
        assert b".comparison.warning" in response.read()
    with urllib.request.urlopen(server + "/api/library") as response:
        assert len(json.load(response)["files"]) == 1
    upload = urllib.request.Request(server + "/api/upload?name=custom.mid", data=midi_bytes(), headers={"Content-Type": "audio/midi"})
    with urllib.request.urlopen(upload) as response:
        assert json.load(response)["name"] == "custom.mid"
    request = urllib.request.Request(server + "/api/audio?id=piece.mid", headers={"Range": "bytes=0-43"})
    with urllib.request.urlopen(request) as response:
        assert response.status == 206
        assert response.headers["Content-Range"] == "bytes 0-43/244"
        assert response.read().startswith(b"RIFF")
    request = urllib.request.Request(server + "/api/audio?id=piece.mid", headers={"Range": "bytes=999-"})
    with pytest.raises(urllib.error.HTTPError) as error:
        urllib.request.urlopen(request)
    assert error.value.code == 416


@pytest.mark.regression
def test_bad_midi_returns_readable_api_error(server, library):
    (library.root / "bad.mid").write_bytes(b"not MIDI")
    with pytest.raises(urllib.error.HTTPError) as error:
        urllib.request.urlopen(server + "/api/midi?id=bad.mid")
    assert error.value.code == 400
    assert "odczytać" in json.load(error.value)["error"]
