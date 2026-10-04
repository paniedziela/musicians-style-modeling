import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from musicians_style.content_audit import clustered_fraction, clustered_rate, run_content_audit
from musicians_style.midi.printer import MidiPrettyPrinter
from musicians_style.midi.types import InternalRepr, NoteEvent
from musicians_style.provenance import sha256_file, write_json


@pytest.fixture
def corpus(tmp_path):
    rows = []
    for name in ("A", "B", "C"):
        path = tmp_path / f"datasets/corpus/{name}/midi_score.mid"
        MidiPrettyPrinter().write(InternalRepr(480, (NoteEvent(0, 0, 72, 80, 120),)), path)
        rows.append(dict(sample_id=name, group_id=name, composer=name, score_path=f"{name}/midi_score.mid",
                         validation_status="accepted", sha256=sha256_file(path)))
    (tmp_path / "datasets/derived/e1_asap").mkdir(parents=True)
    write_json(tmp_path / "datasets/derived/e1_asap/manifest.json", dict(dataset_root="datasets/corpus", samples=rows))
    for experiment in ("e2_asap", "e3_asap"):
        run = tmp_path / "experiments" / experiment
        records = []
        for source in rows:
            for target in rows:
                if source == target:
                    continue
                output = run / f"{source['sample_id']}-{target['sample_id']}.mid"
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_bytes((tmp_path / "datasets/corpus" / source["score_path"]).read_bytes())
                records.append(dict(task_id=output.stem, source_id=source["sample_id"], source_composer=source["composer"],
                                    source_group_id=source["group_id"], target_composer=target["composer"], status="completed",
                                    output_path=output.name, output_sha256=sha256_file(output), best_genome=dict(transpose_semitones=0)))
        write_json(run / "results.json", dict(results=records))
    return tmp_path


def audit(root, name="audit"):
    return run_content_audit(root, root / name, expected_counts=(3, 6),
                             provenance={"import": {"passed": True}})


def test_synthetic_audit_complete_read_only_and_rejects_existing_destination(corpus):
    before = {str(p): sha256_file(p) for p in corpus.rglob("*") if p.is_file()}
    result = audit(corpus)
    assert result["passed"]
    assert result["records"] == 15
    assert all(sha256_file(Path(p)) == h for p, h in before.items())
    assert len(json.loads((corpus / "audit/per_output.json").read_text())) == 15
    with pytest.raises(FileExistsError):
        audit(corpus)


@pytest.mark.parametrize("problem", ["missing", "hash", "invalid_midi", "unknown_source", "coverage"])
def test_scientific_failures_are_records_not_silent_exclusions(corpus, problem):
    path = corpus / "experiments/e3_asap/results.json"
    payload = json.loads(path.read_text())
    row = payload["results"][0]
    output = path.parent / row["output_path"]
    if problem == "missing":
        output.unlink()
    elif problem in {"hash", "invalid_midi"}:
        output.write_bytes(b"broken")
        if problem == "invalid_midi":
            row["output_sha256"] = sha256_file(output)
    elif problem == "unknown_source":
        row["source_id"] = "unknown"
    else:
        row["target_composer"] = row["source_composer"]
    write_json(path, payload)
    result = audit(corpus)
    assert not result["passed"]
    rows = json.loads((corpus / "audit/per_output.json").read_text())
    assert len(rows) == 15
    if problem != "coverage":
        assert sum(r["audit_status"] == "error" for r in rows) == 1


def test_frozen_destination_is_rejected_before_creation(corpus):
    target = corpus / "experiments/e3_asap/new-audit"
    with pytest.raises(ValueError, match="frozen"):
        run_content_audit(corpus, target)
    assert not target.exists()


def test_group_weighting_and_undefined_denominators():
    rows = [dict(source_group_id=group, measurement={"policy": {"event_identity_status": status}})
            for group, status in [("a", "passed"), ("a", "passed"), ("b", "failed"), ("c", "undefined")]]
    result = clustered_rate(rows, "policy")
    assert result["group_mean"] == 0.5
    assert result["output_rate"] == pytest.approx(2 / 3)
    assert result["defined_outputs"] == 3
    assert result["statuses"]["undefined"] == 1
    assert result == clustered_rate(rows, "policy")


def test_partial_retention_uses_equal_group_weights():
    rows = [dict(source_group_id=group, measurement={"policy": {"fraction": value}})
            for group, value in [("a", 1.0), ("a", 0.5), ("b", 0.0), ("c", None)]]
    result = clustered_fraction(rows, "policy", "fraction")
    assert result["defined_outputs"] == 3
    assert result["groups"] == 2
    assert result["group_mean"] == 0.375
    assert result == clustered_fraction(rows, "policy", "fraction")


@pytest.mark.parametrize("ambient", ["stale", "foreign", "missing"])
def test_content_entry_point_rejects_bad_import_before_content_loading(tmp_path, ambient):
    checkout = tmp_path / "checkout"
    source = checkout / "src/musicians_style"
    source.mkdir(parents=True)
    repository = Path(__file__).resolve().parents[2]
    shutil.copyfile(repository / "src/musicians_style/provenance.py", source / "provenance.py")
    (checkout / "tools").mkdir()
    shutil.copyfile(repository / "tools/content_audit.py", checkout / "tools/content_audit.py")
    ambient_root = checkout / ".venv/Lib/site-packages" if ambient == "stale" else tmp_path / ambient / "src"
    if ambient != "missing":
        package = ambient_root / "musicians_style"
        package.mkdir(parents=True)
        (package / "__init__.py").touch()
        (package / "content_audit.py").write_text("raise RuntimeError('content loaded before guard')")
    output = tmp_path / "diagnostic"
    result = subprocess.run([sys.executable, "-B", "-S", str(checkout / "tools/content_audit.py"), "--output", str(output)],
                            env=dict(os.environ, PYTHONPATH=str(ambient_root)), cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 2, result.stderr
    assert not json.loads((output / "provenance.json").read_text())["import"]["passed"]
    assert not (output / "per_output.json").exists()
