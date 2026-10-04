"""Synthetic feasibility checks; real corpus audit remains an explicit tool artifact."""

from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import mido
import pytest

from musicians_style.e1.composition_features import FEATURE_SPECS, extract_composition_features
from musicians_style.feature_backends import custom93
from musicians_style.feature_backends.musif import musical_family, project_row
from musicians_style.feature_feasibility import (
    aligned_cache, attempt, audit_custom93, compare_attempts, musif_attempt, run_feasibility,
    sanitize_midi, select_pilot,
)
from musicians_style.midi.parser import MidiParser
from musicians_style.provenance import sha256_file


def pilot_inputs():
    samples, train, test = [], [], []
    for composer in ("Bach", "Beethoven", "Chopin"):
        for group in ("04", "02", "01", "03", "00test"):
            for suffix in ("z", "a"):
                sample = dict(sample_id=f"{composer}-{group}-{suffix}", group_id=f"{composer}-{group}",
                              composer=composer, score_path="score.mid", sha256="hash", validation_status="accepted")
                samples.append(sample)
                (test if group == "00test" else train).append(sample["sample_id"])
    return {"samples": samples}, {"repetitions": [{"repeat": 0, "folds": [{"fold": 0,
            "train": {"sample_ids": train}, "test": {"sample_ids": test}}]}]}


def test_predeclared_lexical_selection_uses_training_work_groups_only():
    manifest, splits = pilot_inputs()
    selected = select_pilot(manifest, splits)
    assert len(selected) == 9
    assert len({r["group_id"] for r in selected}) == 9
    assert all(r["sample_id"].endswith("-a") and "00test" not in r["sample_id"] for r in selected)
    assert [r["group_id"].split("-")[-1] for r in selected] == ["01", "02", "03"] * 3
    manifest["samples"].reverse()
    splits["repetitions"][0]["folds"][0]["train"]["sample_ids"].reverse()
    assert select_pilot(manifest, splits) == selected


@pytest.mark.parametrize("corruption", ["sample_overlap", "work_overlap", "unknown", "duplicate", "too_few"])
def test_selection_rejects_invalid_or_leaking_splits(corruption):
    manifest, splits = pilot_inputs()
    fold = splits["repetitions"][0]["folds"][0]
    if corruption == "sample_overlap":
        fold["train"]["sample_ids"].append(fold["test"]["sample_ids"][0])
    elif corruption == "work_overlap":
        manifest["samples"][-1]["group_id"] = "Chopin-01"
    elif corruption == "unknown":
        fold["train"]["sample_ids"].append("unknown")
    elif corruption == "duplicate":
        manifest["samples"].append(copy.deepcopy(manifest["samples"][0]))
    else:
        fold["train"]["sample_ids"] = [s for s in fold["train"]["sample_ids"] if not s.startswith("Bach-") or "-01-" in s]
    with pytest.raises(ValueError):
        select_pilot(manifest, splits)


def midi_file(path: Path, name="Bach confidential title") -> Path:
    midi = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    track.extend([mido.MetaMessage("track_name", name=name, time=0),
                  mido.MetaMessage("text", text=name, time=30),
                  mido.MetaMessage("set_tempo", tempo=500000, time=20),
                  mido.Message("note_on", note=60, velocity=80, time=10),
                  mido.Message("note_off", note=60, time=480),
                  mido.MetaMessage("lyrics", text=name, time=10),
                  mido.MetaMessage("end_of_track", time=20)])
    midi.save(path)
    return path


def musical_events(path):
    events = []
    for i, track in enumerate(mido.MidiFile(path).tracks):
        tick = 0
        for msg in track:
            tick += msg.time
            if not msg.is_meta or msg.type in {"set_tempo", "end_of_track"}:
                events.append((i, tick, msg.copy(time=0)))
    return events


def test_metadata_sanitation_preserves_tick_timing_and_original_bytes(tmp_path):
    source = midi_file(tmp_path / "Bach.mid")
    before = source.read_bytes()
    target = tmp_path / "input.mid"
    diagnostics = sanitize_midi(source, target)
    assert source.read_bytes() == before
    assert musical_events(source) == musical_events(target)
    assert diagnostics["removed_meta_messages"] == {"track_name": 1, "text": 1, "lyrics": 1}
    assert b"confidential" not in target.read_bytes()


def test_custom93_wraps_exact_frozen_contract_and_ignores_descriptive_metadata(tmp_path):
    source = midi_file(tmp_path / "Bach_label.mid")
    other = midi_file(tmp_path / "Chopin_label.mid", name="Chopin other metadata")
    result = custom93.extract(source)
    assert result["schema"] == list(FEATURE_SPECS)
    assert len(result["values"]) == 93
    assert result["values"] == extract_composition_features(MidiParser().parse(source)).tolist()
    assert custom93.extract(other) == result
    assert set(result) == {"schema", "values", "diagnostics"}


def test_musif_projection_excludes_even_numeric_identifiers_and_metadata():
    raw = {"Id": 9, "WindowId": 1, "FileName": 42, "Composer": 7, "Title": 3,
           "Score_Composer": 12, "PartBach_Notes": 1, "Score_Notes": 100,
           "Score_IntervalM3_Per": .2, "Score_Degree#4_Count": 3}
    result = project_row(raw, set())
    assert [s["name"] for s in result["schema"]] == ["Score_Degree#4_Count", "Score_IntervalM3_Per", "Score_Notes"]
    assert result["values"] == [3., .2, 100.]
    assert set(result["diagnostics"]["excluded"]) == set(raw) - {s["name"] for s in result["schema"]}
    assert musical_family("Score_IntervalBach3_Per", set()) is None


def test_missing_nonfinite_and_empty_results_are_explicit():
    result = project_row({"Score_Notes": 1., "Score_RhythmInt": None, "Score_Density": float("inf"), "Measures": float("nan")}, set())
    assert result["diagnostics"]["missing"] == ["Measures", "Score_RhythmInt"]
    assert result["diagnostics"]["nonfinite"] == ["Score_Density"]
    assert result["values"].count(None) == 3
    json.dumps(result, allow_nan=False)
    with pytest.raises(ValueError, match="no finite"):
        project_row({"Id": 1, "Score_Notes": None}, set())
    with pytest.raises(ValueError, match="nonnumeric"):
        project_row({"Score_Notes": "composer", "Measures": 2}, set())


@pytest.mark.parametrize("field", ["schema", "values", "diagnostics", "status"])
def test_determinism_checks_values_full_schema_quality_and_outcomes(field):
    left = {"status": "success", **project_row({"Score_Notes": 5.}, set()), "extraction_seconds": 1}
    right = copy.deepcopy(left)
    right["extraction_seconds"] = 999
    assert compare_attempts(left, right)["passed"]
    if field == "schema":
        right[field][0]["group"] = "wrong"
    elif field == "values":
        right[field][0] += 1
    elif field == "diagnostics":
        right[field]["missing"].append("missing")
    else:
        right[field] = "failure"
    assert not compare_attempts(left, right)["passed"]


def test_failure_records_compare_stable_failure_type_without_runtime_or_path_noise(tmp_path):
    first = attempt(custom93.extract, tmp_path / "missing.mid")
    second = attempt(custom93.extract, tmp_path / "also_missing.mid")
    assert first["status"] == "failure"
    assert compare_attempts(first, second)["passed"]
    second["failure"]["type"] = "OtherFailure"
    assert not compare_attempts(first, second)["passed"]


def test_cache_audit_is_explicit_exact_and_hash_sensitive(tmp_path):
    source = midi_file(tmp_path / "score.mid")
    row = dict(sample_id="s", score_path="score.mid", sha256=sha256_file(source))
    cache = {"features_schema_version": "e1.3.0", "feature_contract": list(FEATURE_SPECS), "samples": [{**row, "values": custom93.extract(source)["values"]}]}
    assert audit_custom93([row], cache, tmp_path)["passed"]
    cache["samples"][0]["values"][0] += 1e-12
    failed = audit_custom93([row], cache, tmp_path)
    assert not failed["passed"] and failed["samples"][0]["differing_indices"] == [0]
    cache["samples"][0]["sha256"] = "wrong"
    assert not audit_custom93([row], cache, tmp_path)["samples"][0]["hash_equal"]
    cache["features_schema_version"] = "wrong"
    assert not audit_custom93([row], cache, tmp_path)["schema_version_equal"]


def test_missing_worker_is_failure_with_persisted_record(tmp_path):
    source = midi_file(tmp_path / "score.mid")
    result = musif_attempt(Path(__file__).resolve().parents[2], tmp_path / "missing_python", source, tmp_path / "attempt", 1)
    assert result["status"] == "failure"
    assert json.loads((tmp_path / "attempt/attempt.json").read_text())["failure"]["type"] == "FileNotFoundError"


def test_musif_union_cache_records_absent_columns_without_imputation():
    selected = [{"sample_id": "a"}, {"sample_id": "b"}]
    records = [{"sample_id": sample["sample_id"], "repetition": rep, "status": "success",
                **project_row({"Score_Notes": 1, **({"Measures": 4} if sample["sample_id"] == "a" else {})}, set())}
               for rep in (0, 1) for sample in selected]
    cache = aligned_cache(records, selected)
    assert cache["repetitions"][0]["model_features"] == [[4., 1.], [None, 1.]]
    assert cache["repetitions"][0]["absent_from_sample_schema"] == [[], ["Measures"]]
    assert cache["repetitions"][0]["model_features"] == cache["repetitions"][1]["model_features"]


def test_custom93_aligned_cache_preserves_frozen_names_order_and_values(tmp_path):
    result = custom93.extract(midi_file(tmp_path / "s.mid"))
    rows = [{"backend": "custom93", "sample_id": "s", "repetition": rep, "status": "success", **result} for rep in (0, 1)]
    cache = aligned_cache(rows, [{"sample_id": "s"}])
    assert cache["schema"] == list(FEATURE_SPECS)
    assert cache["repetitions"][0]["model_features"] == [result["values"]]


@pytest.mark.parametrize("ambient", ["stale", "foreign", "missing"])
def test_entry_guard_rejects_bad_import_before_backend_loading(tmp_path, ambient):
    checkout = tmp_path / "checkout"
    source = checkout / "src/musicians_style"
    source.mkdir(parents=True)
    repository = Path(__file__).resolve().parents[2]
    shutil.copyfile(repository / "src/musicians_style/provenance.py", source / "provenance.py")
    (checkout / "tools").mkdir()
    shutil.copyfile(repository / "tools/feature_feasibility.py", checkout / "tools/feature_feasibility.py")
    ambient_root = checkout / ".venv/Lib/site-packages" if ambient == "stale" else tmp_path / ambient / "src"
    if ambient != "missing":
        package = ambient_root / "musicians_style"
        package.mkdir(parents=True)
        (package / "__init__.py").touch()
        (package / "feature_feasibility.py").write_text("raise RuntimeError('backend loaded before guard')")
    output = tmp_path / "diagnostic"
    result = subprocess.run([sys.executable, "-B", "-S", str(checkout / "tools/feature_feasibility.py"),
                             "--output", str(output), "--musif-python", "unused"],
                            env=dict(os.environ, PYTHONPATH=str(ambient_root)), cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 2, result.stderr
    assert not json.loads((output / "provenance.json").read_text())["import"]["passed"]
    assert not (output / "pilot_manifest.json").exists()


@pytest.mark.parametrize("problem", [None, "missing_source", "hash_mismatch"])
def test_synthetic_pipeline_records_all_attempts_failures_and_read_only_cache_audit(tmp_path, monkeypatch, problem):
    repository = Path(__file__).resolve().parents[2]
    data, results = tmp_path / "data", tmp_path / "results"
    derived = data / "derived/e1_asap"
    dataset = data / "corpus"
    derived.mkdir(parents=True)
    dataset.mkdir()
    results.mkdir()
    manifest, splits = pilot_inputs()
    manifest["dataset_root"] = "historical/corpus"
    cache = {"features_schema_version": "e1.3.0", "feature_contract": list(FEATURE_SPECS), "samples": []}
    for row in manifest["samples"]:
        row["score_path"] = row["sample_id"] + ".mid"
        path = midi_file(dataset / row["score_path"])
        row["sha256"] = sha256_file(path)
        cache["samples"].append({**row, "values": custom93.extract(path)["values"]})
    for name, payload in (("manifest", manifest), ("splits", splits), ("composition_features", cache)):
        (derived / (name + ".json")).write_text(json.dumps(payload), encoding="utf-8")
    victim = dataset / select_pilot(manifest, splits)[0]["score_path"]
    if problem == "missing_source":
        victim.unlink()
    elif problem == "hash_mismatch":
        victim.write_bytes(b"changed MIDI")
    before = {p: p.read_bytes() for p in data.rglob("*") if p.is_file()}
    output = results / "pilot"
    calls = []
    def failure_worker(checkout, python, source, directory, timeout):
        declared = json.loads((output / "pilot_manifest.json").read_text())
        assert len(declared["samples"]) == 9  # persisted before the first attempt
        calls.append(source)
        return {"status": "failure", "failure": {"type": "SyntheticBackendFailure", "message": "explicit"}}
    monkeypatch.setattr("musicians_style.feature_feasibility.musif_attempt", failure_worker)
    configuration = {"data": data, "results": results, "literature": tmp_path}
    result = run_feasibility(repository, output, tmp_path / "isolated/python", configuration=configuration)
    assert result["passed"] == (problem is None)
    assert result["attempts"] == 36 and len(calls) == (18 if problem is None else 16)
    rows = json.loads((output / "attempts.json").read_text())
    assert len(rows) == 36
    assert len(json.loads((output / "failures.json").read_text())) == (18 if problem is None else 20)
    assert len(json.loads((output / "custom93_equivalence_audit.json").read_text())["samples"]) == 30
    assert all(p.read_bytes() == b for p, b in before.items())
    assert not result["frozen_changes"]
    with pytest.raises(FileExistsError):
        run_feasibility(repository, output, tmp_path / "isolated/python", configuration=configuration)
    forbidden = results / "e3_asap/new-audit"
    forbidden.parent.mkdir()
    with pytest.raises(ValueError, match="frozen"):
        run_feasibility(repository, forbidden, tmp_path / "isolated/python", configuration=configuration)
    assert not forbidden.exists()


def test_project_interpreter_cannot_be_used_as_isolated_musif_environment(tmp_path):
    with pytest.raises(ValueError, match="separate isolated"):
        run_feasibility(tmp_path, tmp_path / "output", Path(sys.executable))
