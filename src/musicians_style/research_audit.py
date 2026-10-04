"""Opt-in filesystem/hash audits. No feature extraction, fitting or MIDI rewriting."""

from __future__ import annotations

import ast
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .asset_paths import resolve_asset_roots
from .provenance import collect_provenance, fingerprint, sha256_file, write_json


PAPERS = ("L0027", "L0034", "L0067", "L0074", "L0093", "L0098", "L0103", "L0150", "L0185", "L0207", "L0208", "L0209", "L0230")


def test_inventory(checkout: Path) -> dict[str, Any]:
    """Inspect test tiers without importing or executing the tests."""
    records = []
    for path in sorted((checkout / "tests").rglob("test_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        module_markers = set()
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "pytestmark" for t in node.targets):
                module_markers.update(
                    n.attr for n in ast.walk(node.value)
                    if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Attribute) and n.value.attr == "mark"
                )
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or not node.name.startswith("test_"):
                continue
            markers = module_markers | {
                n.attr for decorator in node.decorator_list for n in ast.walk(decorator)
                if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Attribute) and n.value.attr == "mark"
            }
            relative = path.relative_to(checkout).as_posix()
            if "/property/" in relative:
                tier = "property"
            elif "/integration/" in relative or "integration" in markers:
                tier = "integration"
            elif "regression" in markers or "slow" in markers:
                tier = "regression"
            else:
                tier = "fast"
            records.append({"file": relative, "test": node.name, "tier": tier, "markers": sorted(markers)})
    return {"method": "static AST; parametrized case counts require pytest collection", "counts": dict(Counter(r["tier"] for r in records)), "tests": records}


class EvidenceAudit:
    def __init__(self, roots: dict[str, dict[str, str]]) -> None:
        self.data = Path(roots["data"]["path"])
        self.results = Path(roots["results"]["path"])
        self.checks: list[dict[str, Any]] = []
        self.files: dict[str, str] = {}
        self.observations: dict[str, Any] = {}

    def check(self, name: str, passed: bool | None, details: Any = None) -> None:
        self.checks.append({"check": name, "status": "unavailable" if passed is None else "passed" if passed else "failed", "details": details})

    def hash_file(self, path: Path, expected: str | None = None) -> bool:
        try:
            digest = sha256_file(path)
            self.files[str(path)] = digest
            return expected is None or digest == expected
        except OSError:
            return False

    def read(self, path: Path) -> dict[str, Any]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("expected JSON object")
            self.hash_file(path)
            return payload
        except (OSError, ValueError) as exc:
            self.check(str(path), False, str(exc))
            return {}

    def baseline(self, expected_counts: dict[str, int] | None = None) -> dict[str, Any]:
        counts = {"samples": 150, "groups": 87, "e2": 300, "e3": 300, "e4": 86}
        counts.update(expected_counts or {})
        derived = self.data / "derived" / "e1_asap"
        manifest = self.read(derived / "manifest.json")
        splits = self.read(derived / "splits.json")
        rows = [r for r in manifest.get("samples", []) if r.get("validation_status") == "accepted"]
        by_id = {r["sample_id"]: r for r in rows}
        self.check("canonical_counts", len(rows) == counts["samples"] and len({r["group_id"] for r in rows}) == counts["groups"], {"samples": len(rows), "groups": len({r["group_id"] for r in rows}), "by_composer": dict(Counter(r["composer"] for r in rows))})
        self.check("unique_sample_ids", len(by_id) == len(rows))
        # Historical dataset_root includes the old 'datasets/' prefix or an absolute root.
        # Only its dataset directory name is rebased; stored manifests stay untouched.
        dataset = self.data / Path(manifest.get("dataset_root", "asap-dataset-1.2")).name
        self.observations["historical_dataset_root_mapping"] = {"recorded": manifest.get("dataset_root"), "resolved": str(dataset)}
        missing_xml, bad_midi = [], []
        for row in rows:
            midi = (dataset / row["score_path"]).resolve()
            if not row.get("sha256") or not midi.is_relative_to(dataset.resolve()) or not self.hash_file(midi, row.get("sha256")):
                bad_midi.append(row["sample_id"])
            xml = midi.with_name("xml_score.musicxml")
            if not xml.is_relative_to(dataset.resolve()) or not self.hash_file(xml):
                missing_xml.append(row["sample_id"])
        self.check("canonical_midi_hashes", not bad_midi, {"checked": len(rows), "failures": bad_midi})
        self.check("matching_xml", not missing_xml, {"checked": len(rows), "failures": missing_xml})
        split_failures = []
        disjoint_checks = 0
        for rep in splits.get("repetitions", []):
            seen_test = []
            for fold in rep.get("folds", []):
                train, test = fold["train"]["sample_ids"], fold["test"]["sample_ids"]
                seen_test.extend(test)
                inner_seen = []
                pairs = [(train, test)]
                if set(train) | set(test) != set(by_id):
                    split_failures.append("outer coverage")
                for inner in fold.get("inner_folds", []):
                    a, b = inner["train"]["sample_ids"], inner["validation"]["sample_ids"]
                    inner_seen.extend(b)
                    if set(a) | set(b) != set(train):
                        split_failures.append("inner coverage")
                    pairs.append((a, b))
                if sorted(inner_seen) != sorted(train):
                    split_failures.append("inner validation repetition")
                for a, b in pairs:
                    if len(set(a)) != len(a) or len(set(b)) != len(b):
                        split_failures.append("duplicate partition identity")
                    for key in ("sample_id", "group_id", "sha256"):
                        disjoint_checks += 1
                        try:
                            if {by_id[i][key] for i in a} & {by_id[i][key] for i in b}:
                                split_failures.append(key + " leakage")
                        except KeyError as exc:
                            split_failures.append("unknown/missing split identity: " + str(exc))
            if sorted(seen_test) != sorted(by_id):
                split_failures.append("outer test repetition")
        self.check("split_integrity", bool(splits.get("repetitions")) and not split_failures, {"disjoint_checks": disjoint_checks, "failures": split_failures})
        self.check("split_declared_counts", splits.get("dataset", {}).get("sample_count") == len(rows) and splits.get("dataset", {}).get("group_count") == len({r["group_id"] for r in rows}))
        for name, count_key in (("e2_asap", "e2"), ("e3_asap", "e3")):
            run = self.results / name
            result = self.read(run / "results.json")
            outputs = result.get("results", [])
            failures = []
            for row in outputs:
                raw = Path(row.get("output_path", ""))
                output = (run / raw).resolve()
                if not row.get("output_sha256") or not output.is_relative_to(run.resolve()) or not self.hash_file(output, row.get("output_sha256")):
                    failures.append(row.get("task_id"))
            self.check(name + "_outputs", len(outputs) == counts[count_key] and not failures, {"checked": len(outputs), "failures": failures})
            for filename in ("manifest.json", "splits.json", "composition_features.json"):
                snapshot, original = self.read(run / "inputs" / filename), self.read(derived / filename)
                self.check(name + "_snapshot_" + filename, bool(snapshot) and snapshot == original)
            run_manifest = self.read(run / "run_manifest.json")
            for item, record in run_manifest.get("inputs", {}).items():
                filename = "composition_features.json" if item == "composition_features" else item + ".json"
                expected = record.get("sha256")
                self.check(name + "_recorded_input_hash_" + item, bool(expected) and self.hash_file(run / "inputs" / filename, expected))
            declared_hash = run_manifest.get("full_results_sha256")
            if declared_hash:
                self.check(name + "_results_hash", self.hash_file(run / "results.json", declared_hash))
            self.check(name + "_historical_code_commit", None, {"recorded": run_manifest.get("code_commit"), "limitation": "A commit reference is not a complete historical code/environment snapshot."})
        e4 = self.results / "e4_asap_v3"
        validation, status = self.read(e4 / "validation.json"), self.read(e4 / "status.json")
        records = validation.get("records", [])
        missing = [r.get("sample_id") for r in records if not self.hash_file(e4 / "validation_midi" / Path(r.get("output_path", "")).name)]
        self.check("e4_validation_outputs", len(records) == counts["e4"] and not missing, {"checked": len(records), "missing": missing})
        expected_checkpoint = validation.get("checkpoint_sha256")
        self.check("e4_checkpoint_hash", bool(expected_checkpoint) and self.hash_file(e4 / "best.pt", expected_checkpoint))
        self.check("e4_status_checkpoint_consistency", bool(expected_checkpoint) and status.get("checkpoint_sha256") == expected_checkpoint)
        self.check("e4_validation_no_go", status.get("status") == "validation_no_go" and validation.get("summary", {}).get("passed") is False)
        outer_files = [str(p) for p in e4.glob("*test*")]
        self.check("e4_no_outer_test_artifact", not outer_files, {"files": outer_files, "limitation": "Artifact absence does not prove that data was never inspected."})
        # Preserve original contradictory commit references, rather than repairing them.
        e1 = self.read(self.results / "e1_asap_e1b_2026-09-02_005705_449734" / "run_manifest.json")
        self.observations["historical_commits"] = {"canonical_manifest": manifest.get("code_commit"), "e1_run": e1.get("code_commit"), "note": "Different references / anonymized history require reconciliation; no provenance is synthesized."}
        return {"schema": "research_audit.baseline.1", "passed": not any(c["status"] == "failed" for c in self.checks), "checks": self.checks, "observations": self.observations, "evidence_sha256": self.files}


def run_audit(checkout: Path, output: Path, *, scope: str = "inventory", configuration: dict[str, str | None] | None = None, provenance: dict[str, Any] | None = None) -> dict[str, Any]:
    """Reserve a fresh directory; failures remain inspectable and never overwrite evidence."""
    if scope not in {"inventory", "baseline"}:
        raise ValueError("scope must be inventory or baseline")
    checkout, output = checkout.resolve(), output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    provenance = provenance or collect_provenance(checkout)
    roots = resolve_asset_roots(checkout, configuration)
    provenance["roots"] = roots
    provenance["configuration_fingerprint"] = fingerprint({"scope": scope, "roots": roots})
    provenance["configuration_sha256"] = {
        str(p.relative_to(checkout)): sha256_file(p)
        for p in sorted((checkout / "configs").glob("*.yaml"))
    }
    provenance["audit_source_sha256"] = {
        str(p.relative_to(checkout)): sha256_file(p)
        for p in [checkout / "src/musicians_style" / name
                  for name in ("asset_paths.py", "provenance.py", "research_audit.py")]
        + [checkout / "tools/research_audit.py"] if p.is_file()
    }
    write_json(output / "provenance.json", provenance)
    if not provenance["import"]["passed"]:
        result = {"passed": False, "scope": scope, "error": "checkout import guard failed; no dataset inspection performed"}
        write_json(output / "audit.json", result)
        (output / "report.md").write_text("# Research audit failed\n\n" + result["error"] + ".\nUse an editable installation or set PYTHONPATH to this checkout's src and retry in a fresh directory.\n", encoding="utf-8")
        return result
    data, results, literature = (Path(roots[k]["path"]) for k in ("data", "results", "literature"))
    inventory = {
        "roots": roots,
        "root_exists": {k: Path(v["path"]).is_dir() for k, v in roots.items()},
        "score_midi_count": sum(1 for _ in data.rglob("midi_score.mid")),
        "score_xml_count": sum(1 for _ in data.rglob("xml_score.musicxml")),
        "experiment_directories": sorted(p.name for p in results.iterdir() if p.is_dir()) if results.is_dir() else [],
        "selected_papers": {p: {"path": str(literature / "selected" / (p + ".pdf")), "sha256": sha256_file(literature / "selected" / (p + ".pdf")) if (literature / "selected" / (p + ".pdf")).is_file() else None} for p in PAPERS},
    }
    write_json(output / "asset_inventory.json", inventory)
    write_json(output / "test_inventory.json", test_inventory(checkout))
    frozen = {str(p.relative_to(checkout)): sha256_file(p) for name in ("e1", "e2", "e3", "e4") for p in sorted((checkout / "src" / "musicians_style" / name).rglob("*.py"))}
    provenance["frozen_experiment_code_sha256"] = frozen
    write_json(output / "provenance.json", provenance)
    if scope == "baseline":
        audit = EvidenceAudit(roots)
        try:
            result = audit.baseline()
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            audit.check("historical_schema_readability", False, f"{type(exc).__name__}: {exc}")
            result = {"schema": "research_audit.baseline.1", "passed": False,
                      "checks": audit.checks, "observations": audit.observations,
                      "evidence_sha256": audit.files}
    else:
        result = {"schema": "research_audit.inventory.1", "passed": all(inventory["root_exists"].values()), "checks": [], "note": "Baseline hashes/splits were not verified in inventory scope."}
    result["scope"] = scope
    write_json(output / "audit.json", result)
    lines = ["# Research audit", "", f"Scope: {scope}; passed: {result['passed']}", "", f"Imported source: {provenance['import']['resolved_source_path']}", "", f"Score MIDI/XML inventory: {inventory['score_midi_count']}/{inventory['score_xml_count']}", "", "## Checks", ""]
    lines += [f"- {c['status']}: {c['check']} — {json.dumps(c['details'], ensure_ascii=False)}" for c in result.get("checks", [])]
    lines += ["", "Historical commit references are evidence, not reconstructed code snapshots. No scientific computation or historical output rewriting was performed.", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return result
