"""Explicit V2-02 pilot and cache audit, without model fitting."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import mido

from .asset_paths import resolve_asset_roots
from .content_audit import frozen_files
from .feature_backends import custom93
from .provenance import collect_provenance, fingerprint, sha256_file, write_json

CONTRACT = "v2-02.feasibility.1"
COMPOSERS = ("Bach", "Beethoven", "Chopin")


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def available_hash(path: Path) -> str | None:
    """Missing/unreadable pilot input must become a record, not abort coverage."""
    try:
        return sha256_file(path)
    except OSError:
        return None


def select_pilot(manifest: dict, splits: dict) -> list[dict]:
    """First three lexical train work groups per composer, one lexical sample each."""
    repeat = next(r for r in splits["repetitions"] if r["repeat"] == 0)
    fold = next(f for f in repeat["folds"] if f["fold"] == 0)
    accepted = [r for r in manifest["samples"] if r["validation_status"] == "accepted"]
    by_id = {r["sample_id"]: r for r in accepted}
    train, test = set(fold["train"]["sample_ids"]), set(fold["test"]["sample_ids"])
    if len(by_id) != len(accepted) or not train <= by_id.keys() or not test <= by_id.keys():
        raise ValueError("duplicate or unknown split sample identities")
    if train & test or {by_id[s]["group_id"] for s in train} & {by_id[s]["group_id"] for s in test}:
        raise ValueError("outer training/test sample or work overlap")
    selected = []
    for composer in COMPOSERS:
        candidates = [by_id[s] for s in train if by_id[s]["composer"] == composer]
        groups = sorted({r["group_id"] for r in candidates})[:3]
        if len(groups) != 3:
            raise ValueError(f"insufficient training work groups for {composer}")
        for group in groups:
            row = min((r for r in candidates if r["group_id"] == group), key=lambda r: r["sample_id"])
            selected.append({k: row[k] for k in ("sample_id", "group_id", "composer", "score_path", "sha256")})
    return selected


def sanitize_midi(source: Path, destination: Path) -> dict:
    """Neutral MIDI input: retain musical messages, remove descriptive metadata."""
    midi = mido.MidiFile(source)
    removed = Counter()
    keep_meta = {"set_tempo", "time_signature", "key_signature", "end_of_track",
                 "channel_prefix", "midi_port", "smpte_offset"}
    for track in midi.tracks:
        retained, pending = [], 0
        for msg in track:
            pending += msg.time
            if msg.is_meta and msg.type not in keep_meta:
                removed[msg.type] += 1
            else:
                retained.append(msg.copy(time=pending))
                pending = 0
        if pending:
            retained.append(mido.MetaMessage("end_of_track", time=pending))
        track[:] = retained
    midi.save(destination)
    return {"removed_meta_messages": dict(removed), "input_sha256": sha256_file(destination)}


def attempt(extractor, path: Path) -> dict:
    started = time.perf_counter()
    try:
        result = {"status": "success", **extractor(path)}
    except Exception as exc:
        result = {"status": "failure", "failure": {"type": type(exc).__name__, "message": str(exc)}}
    result["extraction_seconds"] = time.perf_counter() - started
    return result


def compare_attempts(left: dict, right: dict) -> dict:
    """Compare outcomes and full schemas/values, excluding runtime and path diagnostics."""
    status = left["status"] == right["status"]
    if left["status"] == right["status"] == "success":
        schema = left["schema"] == right["schema"]
        values = left["values"] == right["values"]
        quality = left["diagnostics"] == right["diagnostics"]
        return {"passed": schema and values and quality, "status_equal": status,
                "schema_equal": schema, "values_equal": values, "diagnostics_equal": quality}
    failure = status and left.get("failure", {}).get("type") == right.get("failure", {}).get("type")
    return {"passed": failure, "status_equal": status, "failure_type_equal": failure}


def musif_attempt(checkout: Path, python: Path, source: Path, directory: Path, timeout: float) -> dict:
    directory.mkdir()
    started = time.perf_counter()
    try:
        sanitation = sanitize_midi(source, directory / "input.mid")
        environment = dict(os.environ, PYTHONPATH=str(checkout / "src"), PYTHONDONTWRITEBYTECODE="1")
        command = [str(python), "-B", str(checkout / "tools/feature_backend_worker.py"),
                   "--input", "input.mid", "--output", "result.json"]
        with (directory / "worker.log").open("w", encoding="utf-8") as log:
            process = subprocess.run(command, cwd=directory, env=environment, stdout=log,
                                     stderr=subprocess.STDOUT, timeout=timeout, check=False)
        result = read(directory / "result.json") if (directory / "result.json").exists() else {
            "status": "failure", "failure": {"type": "WorkerExit", "message": f"exit={process.returncode}; see worker.log"}}
        result["worker_exit_code"] = process.returncode
        if process.returncode and result["status"] == "success":
            result = {"status": "failure", "failure": {"type": "WorkerExit", "message": str(process.returncode)}}
        result["sanitation"] = sanitation
    except Exception as exc:
        result = {"status": "failure", "failure": {"type": type(exc).__name__, "message": str(exc)}}
    result["wall_seconds"] = time.perf_counter() - started
    write_json(directory / "attempt.json", result)
    return result


def audit_custom93(samples: list[dict], cache: dict, dataset: Path) -> dict:
    """Dataset-wide equivalence is an explicit scientific audit, never a pytest dataset case."""
    started = time.perf_counter()
    cached = {r["sample_id"]: r for r in cache["samples"]}
    ids = [r["sample_id"] for r in samples]
    schema_equal = cache["feature_contract"] == [dict(s) for s in custom93.FEATURE_SPECS]
    version_equal = cache.get("features_schema_version") == custom93.COMPOSITION_FEATURES_SCHEMA_VERSION
    coverage = len(cached) == len(cache["samples"]) and len(set(ids)) == len(ids) and set(ids) == set(cached)
    rows = []
    for row in sorted(samples, key=lambda r: r["sample_id"]):
        item = {"sample_id": row["sample_id"]}
        try:
            source = dataset / row["score_path"]
            actual = sha256_file(source)
            result = custom93.extract(source)
            reference = cached[row["sample_id"]]
            hash_equal = actual == row["sha256"] == reference["sha256"]
            value_equal = result["values"] == reference["values"]
            differences = [i for i, (a, b) in enumerate(zip(result["values"], reference["values"])) if a != b]
            item.update(passed=hash_equal and value_equal, hash_equal=hash_equal,
                        values_exact_equal=value_equal, differing_indices=differences)
        except Exception as exc:
            item.update(passed=False, failure={"type": type(exc).__name__, "message": str(exc)})
        rows.append(item)
    return {"schema": CONTRACT, "passed": bool(rows) and schema_equal and version_equal and coverage and all(r["passed"] for r in rows),
            "schema_exact_equal": schema_equal, "schema_version_equal": version_equal, "coverage_equal": coverage, "samples": rows,
            "runtime_seconds": time.perf_counter() - started, "comparison": "exact float64 values, names/order/group/unit and source hashes; no tolerance"}


def summarize_backend(records: list[dict]) -> dict:
    success = [r for r in records if r["status"] == "success"]
    schemas = [r["schema"] for r in success]
    union = sorted({s["name"] for schema in schemas for s in schema})
    intersection = set.intersection(*({s["name"] for s in schema} for schema in schemas)) if schemas else set()
    mappings = [{s["name"]: v for s, v in zip(r["schema"], r["values"])} for r in success]
    finite_intersection = [name for name in intersection if all(m[name] is not None for m in mappings)]
    constant = [name for name in finite_intersection if len({m[name] for m in mappings}) == 1]
    return {"attempts": len(records), "successes": len(success), "failures": len(records) - len(success),
            "failure_types": dict(Counter(r["failure"]["type"] for r in records if r["status"] == "failure")),
            "feature_counts": sorted({len(s) for s in schemas}), "union_feature_count": len(union),
            "raw_feature_counts": sorted({r.get("raw_feature_count", len(r["schema"])) for r in success}),
            "excluded_column_counts": sorted({len(r["diagnostics"]["excluded"]) for r in success}),
            "intersection_feature_count": len(intersection), "finite_intersection_feature_count": len(finite_intersection),
            "constant_finite_intersection_feature_count": len(constant), "constant_finite_intersection_names": sorted(constant),
            "families": dict(Counter(s["group"] for s in {s["name"]: s for schema in schemas for s in schema}.values())),
            "complete_cases": sum(not r["diagnostics"]["missing"] and not r["diagnostics"]["nonfinite"] for r in success),
            "wall_seconds": sum(r.get("wall_seconds", r.get("extraction_seconds", 0)) for r in records),
            "extraction_seconds": sum(r.get("extraction_seconds", 0) for r in records),
            "union_schema_names": union}


def aligned_cache(records: list[dict], selected: list[dict]) -> dict:
    """Numeric matrices with sidecar identities; pilot union performs no fitting."""
    schemas = {s["name"]: s for r in records if r["status"] == "success" for s in r["schema"]}
    # The custom93 model matrix must retain the frozen contract's exact order.
    names = [s["name"] for s in custom93.FEATURE_SPECS] if records and records[0].get("backend") == "custom93" else sorted(schemas)
    if names and not schemas:
        schemas = {s["name"]: dict(s) for s in custom93.FEATURE_SPECS}
    repetitions = []
    for repetition in (0, 1):
        matrix, statuses, absent = [], [], []
        for sample in selected:
            row = next(r for r in records if r["repetition"] == repetition and r["sample_id"] == sample["sample_id"])
            statuses.append(row["status"])
            mapping = {s["name"]: value for s, value in zip(row.get("schema", []), row.get("values", []))}
            matrix.append([mapping.get(name) for name in names])
            absent.append([name for name in names if name not in mapping])
        repetitions.append({"repetition": repetition, "model_features": matrix, "statuses": statuses,
                            "absent_from_sample_schema": absent})
    return {"schema": [schemas[name] for name in names], "repetitions": repetitions,
            "row_mapping": "pilot_manifest.json samples list order; no IDs, labels or metadata in model_features",
            "note": "pilot union inventory only; nulls remain explicit, no vocabulary fitting or imputation"}


def run_feasibility(checkout: Path, output: Path, musif_python: Path, *, timeout: float = 180,
                    configuration: dict | None = None, provenance: dict | None = None) -> dict:
    checkout, output, musif_python = checkout.resolve(), output.resolve(), musif_python.resolve()
    if musif_python == Path(sys.executable).resolve():
        raise ValueError("use a separate isolated musif interpreter, not the project environment")
    roots = resolve_asset_roots(checkout, configuration)
    data = Path(roots["data"]["path"])
    derived = data / "derived/e1_asap"
    # Resolve every input and persist selection before any backend extraction.
    manifest, splits, cache = (read(derived / (n + ".json")) for n in ("manifest", "splits", "composition_features"))
    selected = select_pilot(manifest, splits)
    dataset = data / Path(manifest["dataset_root"]).name
    sources = [dataset / r["score_path"] for r in manifest["samples"] if r["validation_status"] == "accepted"]
    protected = set(frozen_files(checkout, roots)) | {p.resolve() for p in sources}
    protected.update(p.resolve() for p in derived.glob("*.json"))
    previous = Path(roots["results"]["path"]) / "research_v2_03_2026-10-04"
    protected.update(p.resolve() for p in previous.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    protected.update(checkout / p for p in ("src/musicians_style/content_metrics.py", "src/musicians_style/content_audit.py", "tools/content_audit.py", "docs/research/V2_03_COMPLETION.md"))
    forbidden = [dataset, derived, checkout / "src", checkout / "configs", previous]
    results_root = Path(roots["results"]["path"])
    forbidden += [p for p in results_root.iterdir() if p.is_dir() and p.name.startswith(("e1_", "e2_", "e3_", "e4_"))] if results_root.is_dir() else []
    if any(output == p or output in p.parents for p in protected) or any(output.is_relative_to(p.resolve()) for p in forbidden):
        raise ValueError("output must not contain or be inside frozen/input locations")
    output.mkdir(parents=True, exist_ok=False)
    before = {str(p): available_hash(p) for p in sorted(protected)}
    write_json(output / "frozen_sha256.json", before)
    provenance = provenance or collect_provenance(checkout)
    provenance.update(roots=roots, dataset_mapping={"recorded": manifest["dataset_root"], "resolved": str(dataset)},
                      input_sha256={n: sha256_file(derived / (n + ".json")) for n in ("manifest", "splits", "composition_features")},
                      musif_python=str(musif_python), worker_timeout_seconds=timeout)
    provenance["new_source_sha256"] = {str(p.relative_to(checkout)): sha256_file(p) for p in
        [checkout / "src/musicians_style/feature_feasibility.py", checkout / "tools/feature_feasibility.py", checkout / "tools/feature_backend_worker.py",
         checkout / "requirements/research-v2-02-musif.lock.txt",
         *sorted((checkout / "src/musicians_style/feature_backends").glob("*.py"))]}
    write_json(output / "provenance.json", provenance)
    pilot = {"schema": CONTRACT, "repeat": 0, "outer_fold": 0, "partition": "train",
             "selection": "first three lexical work groups per composer; first lexical sample per group",
             "samples": selected, "input_sha256": provenance["input_sha256"]}
    write_json(output / "pilot_manifest.json", pilot)
    from .feature_backends.musif import CONFIG, GLOBAL_FIELDS
    write_json(output / "backend_contract.json", {"schema": CONTRACT, "custom93": "frozen e1.3.0; names/order/semantics unchanged",
               "musif_version": "1.2.4", "musif_config": CONFIG, "musif_global_allowlist": GLOBAL_FIELDS,
               "musif_scope": "reviewed score-level numeric musical fields; bounded interval/scale-degree patterns; no part/sound/family headers or category encoding",
               "input": "MIDI only, neutral input.mid; descriptive meta removed while preserving absolute ticks",
               "missing": "explicit nullable features; no learned transformations", "repetitions": 2})
    attempts, comparisons = [], []
    for repetition in (0, 1):
        for index, sample in enumerate(selected):
            source = dataset / sample["score_path"]
            valid_hash = available_hash(source) == sample["sha256"]
            for backend in ("custom93", "musif"):
                if not valid_hash:
                    result = {"status": "failure", "failure": {"type": "SourceHashMismatch", "message": "missing or changed predeclared MIDI"}}
                elif backend == "custom93":
                    result = attempt(custom93.extract, source)
                else:
                    result = musif_attempt(checkout, musif_python, source, output / f"musif_r{repetition}_{index:02d}", timeout)
                attempts.append({"sample_id": sample["sample_id"], "backend": backend, "repetition": repetition, **result})
                write_json(output / "attempts.json", attempts)
                print(f"{backend} repeat={repetition} sample={index + 1}/9 status={result['status']}", flush=True)
    for backend in ("custom93", "musif"):
        for sample in selected:
            pair = [r for r in attempts if r["backend"] == backend and r["sample_id"] == sample["sample_id"]]
            comparisons.append({"backend": backend, "sample_id": sample["sample_id"], **compare_attempts(*pair)})
    summary = {backend: summarize_backend([r for r in attempts if r["backend"] == backend]) for backend in ("custom93", "musif")}
    schemas = {backend: {"per_attempt": [{"sample_id": r["sample_id"], "repetition": r["repetition"], "schema": r.get("schema", []), "status": r["status"]} for r in attempts if r["backend"] == backend],
                         "schema_fingerprints": sorted({fingerprint(r["schema"]) for r in attempts if r["backend"] == backend and r["status"] == "success"})} for backend in summary}
    write_json(output / "schemas.json", schemas)
    write_json(output / "feature_cache.json", {backend: aligned_cache([r for r in attempts if r["backend"] == backend], selected) for backend in summary})
    write_json(output / "failures.json", [r for r in attempts if r["status"] == "failure"])
    write_json(output / "determinism.json", {"passed": all(r["passed"] for r in comparisons), "comparisons": comparisons,
               "complete_pilot_schema_equal": {b: [r.get("schema") for r in attempts if r["backend"] == b and r["repetition"] == 0] == [r.get("schema") for r in attempts if r["backend"] == b and r["repetition"] == 1] for b in summary}})
    write_json(output / "summary.json", summary)
    audit = audit_custom93([r for r in manifest["samples"] if r["validation_status"] == "accepted"], cache, dataset)
    write_json(output / "custom93_equivalence_audit.json", audit)
    after_paths = set(frozen_files(checkout, roots)) | set(protected)
    after_paths.update(p.resolve() for p in derived.glob("*.json"))
    after_paths.update(p.resolve() for p in previous.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    changes = [str(p) for p in sorted(after_paths) if str(p) not in before or before[str(p)] != available_hash(p)]
    outcome = {"schema": CONTRACT, "passed": len(attempts) == 36 and audit["passed"] and not changes and all(before.values()) and all(r["passed"] for r in comparisons),
               "attempts": len(attempts), "predeclared_samples": len(selected), "frozen_files": len(before), "frozen_changes": changes,
               "unavailable_frozen_inputs": [p for p, h in before.items() if h is None],
               "custom93_equivalence_passed": audit["passed"], "determinism_passed": all(r["passed"] for r in comparisons),
               "note": "audit completeness/determinism is separate from musif feasibility; missing values are explicit nulls, no imputation or classification"}
    write_json(output / "audit.json", outcome)
    lines = ["# V2-02 MIDI feature feasibility", "", f"Audit passed: {outcome['passed']}; nine train samples, two repetitions per backend.", "",
             "| Backend | Successful attempts | Failures | Feature counts | Union count | Complete cases | Wall seconds |", "|---|---:|---:|---|---:|---:|---:|"]
    for backend, s in summary.items():
        lines.append(f"| {backend} | {s['successes']}/18 | {s['failures']} | {s['feature_counts']} | {s['union_feature_count']} | {s['complete_cases']} | {s['wall_seconds']:.3f} |")
    lines += ["", f"custom93 exact cache equivalence: {audit['passed']} ({len(audit['samples'])} samples, {audit['runtime_seconds']:.3f} seconds).",
              f"Frozen/input/V2-03 preservation: {len(before)} files; {len(changes)} changes.", "",
              "See schemas.json, attempts.json, failures.json and determinism.json for full schemas, null/missing/nonfinite values, exclusions, individual times and explicit failures.",
              "No MusicXML substitution, learned imputation/selection, full E1c fitting, V2-04 or V2-05. Feature union is descriptive pilot inventory, not a fitted model schema. Stop for review.", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return outcome
