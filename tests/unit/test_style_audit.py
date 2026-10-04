from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

from musicians_style.style_audit import align_tasks, agreement, cluster_summary, movement, run_style_audit, validate_splits


def test_cluster_weights_denominators_and_repeated_observations():
    rows = [dict(group_id=g, composer=c, x=x) for g, c, x in
            [("a", "A", 1), ("a", "A", 1), ("b", "A", 0), ("c", "B", 0), ("d", "C", None)]]
    s = cluster_summary(rows, "x")
    assert s["defined"] == 4 and s["undefined"] == 1 and s["groups"] == 3
    assert s["group_mean"] == pytest.approx(1 / 3)
    assert s["row_mean"] == .5
    assert cluster_summary(rows, "x", stratified=True)["group_mean"] == .25
    assert s == cluster_summary(rows, "x")


def test_correlations_cluster_and_undefined():
    rows = [dict(group_id=str(i), x=i, y=2 * i) for i in range(5)]
    r = agreement(rows + [dict(group_id="missing", x=None, y=3)], "x", "y")
    assert r["complete_pairs"] == 5 and r["undefined_pairs"] == 1
    assert r["group_coefficients"]["pearson"] == pytest.approx(1)
    assert r["group_coefficients"]["spearman"] == pytest.approx(1)
    assert r["undefined_bootstrap_draws"] > 0
    assert agreement([dict(group_id=str(i), x=1., y=i) for i in range(4)], "x", "y")["undefined_reason"]


def test_movement_direction_identity_negative_and_missing():
    before, after = {"A": .8, "B": .2}, {"A": .9, "B": .1}
    assert movement(before, after, "A", "B")["delta"] == pytest.approx(-.1)
    assert movement(before, after, "A", "B")["negative"] == 1
    assert movement(before, before, "A", "B")["null"] == 1
    assert movement(before, {"A": None, "B": None}, "A", "B")["delta"] is None


@pytest.fixture
def split_fixture():
    rows = [dict(sample_id=str(i), group_id="g" + str(i), sha256="h" + str(i), composer=chr(65 + i % 3)) for i in range(6)]
    folds = []
    for i in range(2):
        test = [str(j) for j in range(6) if j % 2 == i]
        train = sorted(set(r["sample_id"] for r in rows) - set(test))
        folds.append(dict(fold=i, train=dict(sample_ids=train), test=dict(sample_ids=test), inner_folds=[
            dict(train=dict(sample_ids=train[1:]), validation=dict(sample_ids=train[:1])),
            dict(train=dict(sample_ids=train[:1]), validation=dict(sample_ids=train[1:]))]))
    return rows, dict(repetitions=[dict(repeat=0, folds=folds)])


@pytest.mark.parametrize("field", ["sample_id", "group_id", "sha256"])
def test_splits_fail_forbidden_overlap(split_fixture, field):
    rows, splits = split_fixture
    rows[1][field] = rows[0][field]
    with pytest.raises(ValueError):
        validate_splits(rows, splits)


def test_inner_overlap_rejected_even_without_selection(split_fixture):
    rows, splits = split_fixture
    inner = splits["repetitions"][0]["folds"][0]["inner_folds"][0]
    inner["train"]["sample_ids"].extend(inner["validation"]["sample_ids"])
    with pytest.raises(ValueError, match="sample_id"):
        validate_splits(rows, splits)


def tasks(split_fixture):
    rows, splits = split_fixture
    folds = validate_splits(rows, splits)
    result = []
    for _, f in folds:
        for sid in f["test"]["sample_ids"]:
            r = next(r for r in rows if r["sample_id"] == sid)
            for c in {r["composer"] for r in rows} - {r["composer"]}:
                result.append(dict(task_id=sid + c, source_id=sid, source_group_id=r["group_id"], source_composer=r["composer"],
                    target_composer=c, fold=f["fold"], seed=1729, phase="full", status="completed",
                    profile_train_ids=sorted(s for s in f["train"]["sample_ids"] if next(r for r in rows if r["sample_id"] == s)["composer"] == c)))
    return rows, folds, result


def test_alignment_is_exact_and_order_independent(split_fixture):
    rows, folds, t = tasks(split_fixture)
    assert len(align_tasks(t, list(reversed(t)), rows, folds)) == 12


@pytest.mark.parametrize("problem", ["task_id", "source_group_id", "fold", "profile_train_ids", "missing", "duplicate", "training_hash"])
def test_alignment_rejects_metadata_and_coverage_changes(split_fixture, problem):
    rows, folds, left = tasks(split_fixture)
    right = deepcopy(left)
    if problem == "missing":
        right.pop()
    elif problem == "duplicate":
        right.append(right[0])
    elif problem == "training_hash":
        train_id = next(f["train"]["sample_ids"][0] for _, f in folds if f["fold"] == right[0]["fold"])
        right[0]["output_sha256"] = next(r["sha256"] for r in rows if r["sample_id"] == train_id)
    else:
        right[0][problem] = [] if problem == "profile_train_ids" else "wrong"
    with pytest.raises(ValueError):
        align_tasks(left, right, rows, folds)


def test_protected_destination_rejected_before_creation(tmp_path):
    target = tmp_path / "src/audit"
    with pytest.raises(ValueError, match="destination"):
        run_style_audit(tmp_path, target)
    assert not target.exists()


@pytest.mark.parametrize("ambient", ["stale", "foreign", "missing"])
def test_entry_guard_before_style_import(tmp_path, ambient):
    checkout = tmp_path / "checkout"
    source = checkout / "src/musicians_style"
    source.mkdir(parents=True)
    repo = Path(__file__).resolve().parents[2]
    shutil.copyfile(repo / "src/musicians_style/provenance.py", source / "provenance.py")
    (checkout / "tools").mkdir()
    shutil.copyfile(repo / "tools/style_audit.py", checkout / "tools/style_audit.py")
    ambient_root = checkout / ".venv/Lib/site-packages" if ambient == "stale" else tmp_path / ambient / "src"
    if ambient != "missing":
        package = ambient_root / "musicians_style"
        package.mkdir(parents=True)
        (package / "__init__.py").touch()
        (package / "style_audit.py").write_text("raise RuntimeError('loaded before guard')")
    output = tmp_path / "diagnostic"
    result = subprocess.run([sys.executable, "-B", "-S", str(checkout / "tools/style_audit.py"), "--output", str(output)],
                            env=dict(os.environ, PYTHONPATH=str(ambient_root)), cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 2, result.stderr
    assert not json.loads((output / "provenance.json").read_text())["import"]["passed"]
    assert not (output / "fit_manifest.json").exists()


@pytest.fixture
def mini_corpus(tmp_path):
    from musicians_style.content_audit import run_content_audit
    from musicians_style.e1.composition_features import FEATURE_SPECS, COMPOSITION_FEATURES_SCHEMA_VERSION, extract_composition_features
    from musicians_style.e3.profile import build_target_profile
    from musicians_style.midi.printer import MidiPrettyPrinter
    from musicians_style.midi.types import InternalRepr, NoteEvent
    from musicians_style.provenance import sha256_file, write_json
    rows, representations, cached = [], {}, []
    for i in range(15):
        sid, composer = str(i), chr(65 + i % 3)
        piece = InternalRepr(480, tuple(NoteEvent(t + i * 3, 0, p + i, 80, d + i * 5) for t, p, d in
                                       [(0, 40, 240), (0, 60, 120), (240, 55, 600), (500, 70, 120)]))
        path = tmp_path / f"datasets/corpus/{sid}.mid"
        MidiPrettyPrinter().write(piece, path)
        r = dict(sample_id=sid, group_id=sid, sha256=sha256_file(path), composer=composer,
                 score_path=f"{sid}.mid", validation_status="accepted")
        rows.append(r)
        representations[sid] = piece
        cached.append(dict(sample_id=sid, sha256=r["sha256"], values=extract_composition_features(piece).tolist()))
    folds = []
    for i in range(5):
        test = [r["sample_id"] for r in rows if int(r["sample_id"]) // 3 == i]
        train = [r["sample_id"] for r in rows if r["sample_id"] not in test]
        inner = []
        for j in range(3):
            val = train[j::3]
            inner.append(dict(train=dict(sample_ids=[s for s in train if s not in val]), validation=dict(sample_ids=val)))
        folds.append(dict(fold=i, train=dict(sample_ids=train), test=dict(sample_ids=test), inner_folds=inner))
    derived = tmp_path / "datasets/derived/e1_asap"
    derived.mkdir(parents=True)
    payloads = dict(manifest=dict(dataset_root="datasets/corpus", samples=rows), splits=dict(repetitions=[dict(repeat=0, folds=folds)]),
                    composition_features=dict(feature_contract=list(FEATURE_SPECS), features_schema_version=COMPOSITION_FEATURES_SCHEMA_VERSION, samples=cached))
    for name, value in payloads.items():
        write_json(derived / (name + ".json"), value)
    for name in ("e2_asap", "e3_asap"):
        run = tmp_path / "experiments" / name
        (run / "inputs").mkdir(parents=True)
        for k, v in payloads.items():
            write_json(run / "inputs" / (k + ".json"), v)
        records = []
        for fold in folds:
            train = [r for r in rows if r["sample_id"] in fold["train"]["sample_ids"]]
            forbidden = [r for r in rows if r["sample_id"] in fold["test"]["sample_ids"]]
            profiles = {c: build_target_profile(c, train, representations, forbidden_rows=forbidden) for c in ("A", "B", "C")}
            for r in forbidden:
                for c in {"A", "B", "C"} - {r["composer"]}:
                    task = r["sample_id"] + c
                    output = run / (task + ".mid")
                    output.write_bytes((tmp_path / "datasets/corpus" / r["score_path"]).read_bytes())
                    records.append(dict(task_id=task, source_id=r["sample_id"], source_group_id=r["group_id"], source_composer=r["composer"],
                        target_composer=c, fold=fold["fold"], seed=1729, phase="full", status="completed", output_path=output.name,
                        output_sha256=sha256_file(output), profile_train_ids=list(profiles[c].train_sample_ids), profile_fingerprint=profiles[c].fingerprint,
                        best_genome=dict(transpose_semitones=0), style_gain=0., delta_p_target=0.))
        write_json(run / "results.json", dict(results=records))
    previous = tmp_path / "experiments/research_v2_03_2026-10-04/content_verified"
    assert run_content_audit(tmp_path, previous, expected_counts=(15, 30), provenance={"import": {"passed": True}})["passed"]
    return tmp_path


@pytest.mark.integration
def test_complete_synthetic_audit_read_only_and_model_provenance(mini_corpus):
    from musicians_style.provenance import sha256_file
    before = {str(p): sha256_file(p) for p in mini_corpus.rglob("*") if p.is_file()}
    output = mini_corpus / "audit"
    result = run_style_audit(mini_corpus, output, expected_counts=(15, 30), expected_folds=5, provenance={"import": {"passed": True}})
    assert result["passed"], result
    assert result["real_rows"] == 15 and result["output_rows"] == 90
    assert all(sha256_file(Path(p)) == digest for p, digest in before.items())
    fits = json.loads((output / "fit_manifest.json").read_text())
    assert len(fits) == 5
    assert all(r["train_samples"] == 12 for r in fits)
    summary = json.loads((output / "summary.json").read_text())
    for m in ("rms67", "gaussian67", "logistic93", "onset_duration", "time_pitch"):
        assert summary["movement"]["identity:all"]["measures"][m]["null"]["count"] == 30
    with pytest.raises(FileExistsError):
        run_style_audit(mini_corpus, output)


@pytest.mark.integration
@pytest.mark.parametrize("problem", ["hash", "snapshot", "cache", "missing"])
def test_audit_failures_leave_explicit_diagnostics_without_fits(mini_corpus, problem):
    if problem in {"hash", "missing"}:
        p = mini_corpus / "datasets/corpus/0.mid"
        p.unlink() if problem == "missing" else p.write_bytes(b"broken")
    else:
        p = mini_corpus / ("experiments/e3_asap/inputs/manifest.json" if problem == "snapshot" else "datasets/derived/e1_asap/composition_features.json")
        payload = json.loads(p.read_text())
        payload["samples"][0]["values" if problem == "cache" else "composer"] = [] if problem == "cache" else "wrong"
        p.write_text(json.dumps(payload))
    output = mini_corpus / "bad-audit"
    result = run_style_audit(mini_corpus, output, expected_counts=(15, 30), expected_folds=5, provenance={"import": {"passed": True}})
    assert not result["passed"] and result["failures"]
    assert not (output / "fit_manifest.json").exists()
