import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from musicians_style.asset_paths import resolve_asset_roots
from musicians_style.provenance import sha256_file, write_json
from musicians_style.research_audit import EvidenceAudit, run_audit


@pytest.fixture
def evidence(tmp_path):
    derived = tmp_path / "datasets/derived/e1_asap"
    derived.mkdir(parents=True)
    rows = []
    for i in range(4):
        midi = tmp_path / f"datasets/corpus/{i}/midi_score.mid"
        midi.parent.mkdir(parents=True)
        midi.write_bytes(f"fixture-midi-{i}".encode())
        midi.with_name("xml_score.musicxml").write_text("<score/>")
        rows.append(dict(sample_id=str(i), group_id=str(i), composer="fixture",
                         score_path=f"{i}/midi_score.mid", sha256=sha256_file(midi), validation_status="accepted"))
    write_json(derived / "manifest.json", dict(dataset_root="datasets/corpus", samples=rows))
    folds = []
    for train, test in [(["0", "1"], ["2", "3"]), (["2", "3"], ["0", "1"])]:
        inner = [dict(train=dict(sample_ids=[train[0]]), validation=dict(sample_ids=[train[1]])),
                 dict(train=dict(sample_ids=[train[1]]), validation=dict(sample_ids=[train[0]]))]
        folds.append(dict(train=dict(sample_ids=train), test=dict(sample_ids=test), inner_folds=inner))
    write_json(derived / "splits.json", dict(dataset=dict(sample_count=4, group_count=4), repetitions=[dict(folds=folds)]))
    write_json(derived / "composition_features.json", dict(schema="fixture"))
    for name in ("e2_asap", "e3_asap"):
        run = tmp_path / "experiments" / name
        (run / "inputs").mkdir(parents=True)
        for filename in ("manifest.json", "splits.json", "composition_features.json"):
            shutil.copyfile(derived / filename, run / "inputs" / filename)
        output = run / "output.mid"
        output.write_bytes(b"frozen-output")
        write_json(run / "results.json", dict(results=[dict(task_id="fixture", output_path="output.mid", output_sha256=sha256_file(output))]))
        write_json(run / "run_manifest.json", {})
    e4 = tmp_path / "experiments/e4_asap_v3"
    (e4 / "validation_midi").mkdir(parents=True)
    (e4 / "validation_midi/output.mid").write_bytes(b"validation-output")
    (e4 / "best.pt").write_bytes(b"checkpoint")
    digest = sha256_file(e4 / "best.pt")
    write_json(e4 / "validation.json", dict(checkpoint_sha256=digest, summary=dict(passed=False), records=[dict(sample_id="0", output_path="old/root/output.mid")]))
    write_json(e4 / "status.json", dict(status="validation_no_go", checkpoint_sha256=digest))
    e1 = tmp_path / "experiments/e1_asap_e1b_2026-09-02_005705_449734"
    e1.mkdir()
    write_json(e1 / "run_manifest.json", dict(code_commit="historical-reference"))
    (tmp_path / "Literatura").mkdir()
    return tmp_path


def baseline(root):
    return EvidenceAudit(resolve_asset_roots(root, environ={})).baseline(dict(samples=4, groups=4, e2=1, e3=1, e4=1))


def test_baseline_verification_is_read_only(evidence):
    before = {str(p): sha256_file(p) for p in evidence.rglob("*") if p.is_file()}
    result = baseline(evidence)
    assert result["passed"]
    assert any(c["status"] == "unavailable" for c in result["checks"])
    assert before == {str(p): sha256_file(p) for p in evidence.rglob("*") if p.is_file()}


@pytest.mark.parametrize("problem", ["hash", "missing", "snapshot", "split", "unrecorded_hash", "checkpoint"])
def test_baseline_preserves_explicit_failures(evidence, problem):
    if problem == "hash":
        (evidence / "datasets/corpus/0/midi_score.mid").write_bytes(b"changed")
    elif problem == "missing":
        (evidence / "experiments/e3_asap/output.mid").unlink()
    elif problem == "snapshot":
        write_json(evidence / "experiments/e2_asap/inputs/composition_features.json", dict(schema="conflict"))
    elif problem == "split":
        p = evidence / "datasets/derived/e1_asap/splits.json"
        value = json.loads(p.read_text())
        value["repetitions"][0]["folds"][0]["test"]["sample_ids"].append("0")
        write_json(p, value)
    elif problem == "unrecorded_hash":
        p = evidence / "experiments/e2_asap/results.json"
        value = json.loads(p.read_text())
        del value["results"][0]["output_sha256"]
        write_json(p, value)
    else:
        (evidence / "experiments/e4_asap_v3/best.pt").write_bytes(b"changed")
    result = baseline(evidence)
    assert not result["passed"]
    assert any(c["status"] == "failed" for c in result["checks"])


def test_existing_output_is_rejected_without_touching_it(tmp_path):
    sentinel = tmp_path / "sentinel"
    sentinel.write_text("keep")
    with pytest.raises(FileExistsError):
        run_audit(tmp_path, tmp_path)
    assert sentinel.read_text() == "keep"


@pytest.mark.parametrize("ambient", ["correct", "stale", "other", "missing"])
def test_entry_point_checks_real_subprocess_import_before_research(tmp_path, ambient):
    checkout = tmp_path / "checkout"
    source = checkout / "src/musicians_style"
    source.mkdir(parents=True)
    (source / "__init__.py").touch()
    repository = Path(__file__).resolve().parents[2]
    for filename in ("asset_paths.py", "provenance.py", "research_audit.py"):
        shutil.copyfile(repository / "src/musicians_style" / filename, source / filename)
    (checkout / "tools").mkdir()
    shutil.copyfile(repository / "tools/research_audit.py", checkout / "tools/research_audit.py")
    for folder in ("datasets", "experiments", "Literatura"):
        (checkout / folder).mkdir()
    ambient_root = {"correct": checkout / "src", "stale": checkout / ".venv/Lib/site-packages",
                    "other": tmp_path / "other-worktree/src", "missing": tmp_path / "absent"}[ambient]
    if ambient not in {"correct", "missing"}:
        foreign = ambient_root / "musicians_style"
        foreign.mkdir(parents=True)
        # If research code loads after this invalid import, the test must fail.
        (foreign / "__init__.py").touch()
        (foreign / "research_audit.py").write_text("raise RuntimeError('research loaded too early')")
    environment = dict(os.environ, PYTHONPATH=str(ambient_root))
    output = tmp_path / "audit"
    completed = subprocess.run([sys.executable, "-B", "-S", str(checkout / "tools/research_audit.py"), "--output", str(output)],
                               cwd=tmp_path, env=environment, capture_output=True, text=True)
    record = json.loads((output / "provenance.json").read_text())
    assert record["import"]["passed"] == (ambient == "correct")
    assert completed.returncode == (0 if ambient == "correct" else 2), completed.stderr
    assert (output / "report.md").is_file()
    assert (output / "asset_inventory.json").exists() == (ambient == "correct")


def test_malformed_baseline_produces_diagnostic_artifact(evidence):
    write_json(evidence / "datasets/derived/e1_asap/manifest.json", dict(samples=[dict(validation_status="accepted")]))
    result = run_audit(evidence, evidence / "new-audit", scope="baseline", provenance={"import": dict(passed=True, resolved_source_path="fixture")})
    assert not result["passed"]
    assert any(c["check"] == "historical_schema_readability" for c in result["checks"])
    assert (evidence / "new-audit/audit.json").is_file()
