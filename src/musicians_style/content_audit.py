"""Explicit V2-03 scientific audit of frozen E2/E3 outputs and identities."""

from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .asset_paths import resolve_asset_roots
from .content_metrics import CONTRACT, measure_content, observe_midi
from .provenance import collect_provenance, fingerprint, sha256_file, write_json


def frozen_files(checkout: Path, roots: dict[str, Any]) -> list[Path]:
    """Include complete frozen run trees, including reports and task artifacts."""
    directories = [checkout / "configs"]
    directories += [checkout / "src/musicians_style" / name for name in
                    ("e1", "e2", "e3", "e4", "midi", "evaluation", "ga", "features")]
    results = Path(roots["results"]["path"])
    directories += [p for p in results.iterdir() if p.is_dir() and p.name.startswith(("e1_", "e2_", "e3_", "e4_"))] if results.is_dir() else []
    files = {p.resolve() for directory in directories for p in directory.rglob("*")
             if p.is_file() and "__pycache__" not in p.parts}
    files.update(p.resolve() for p in (checkout / "src/musicians_style/config.py",) if p.is_file())
    return sorted(files)


def clustered_rate(rows: list[dict[str, Any]], policy: str, *, seed: int = 1729, draws: int = 2000) -> dict[str, Any]:
    """Equal-work-group mean and percentile bootstrap, with undefined excluded."""
    groups: dict[str, list[float]] = defaultdict(list)
    statuses = Counter()
    for row in rows:
        value = row.get("measurement", {}).get(policy, {}).get("event_identity_status", "undefined")
        statuses[value] += 1
        if value in {"passed", "failed"}:
            groups[row["source_group_id"]].append(float(value == "passed"))
    if not groups:
        return {"statuses": dict(statuses), "defined_outputs": 0, "groups": 0, "group_mean": None, "ci95": None}
    values = np.asarray([np.mean(groups[key]) for key in sorted(groups)])
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(axis=1)
    return {"statuses": dict(statuses), "defined_outputs": sum(map(len, groups.values())), "groups": len(groups),
            "output_rate": statuses["passed"] / sum(map(len, groups.values())),
            "group_mean": float(values.mean()), "ci95": np.quantile(means, [0.025, 0.975]).tolist()}


def clustered_fraction(rows: list[dict[str, Any]], policy: str, field: str) -> dict[str, Any]:
    groups: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = row.get("measurement", {}).get(policy, {}).get(field)
        if value is not None:
            groups[row["source_group_id"]].append(value)
    if not groups:
        return {"defined_outputs": 0, "groups": 0, "group_mean": None, "ci95": None}
    values = np.asarray([np.mean(groups[key]) for key in sorted(groups)])
    rng = np.random.default_rng(1729)
    means = values[rng.integers(0, len(values), size=(2000, len(values)))].mean(axis=1)
    return {"defined_outputs": sum(map(len, groups.values())), "groups": len(groups),
            "group_mean": float(values.mean()), "ci95": np.quantile(means, [0.025, 0.975]).tolist()}


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        buckets[row["experiment"] + ":" + row["source_composer"] + "->" + row["target_composer"]].append(row)
        buckets[row["experiment"] + ":all"].append(row)
    summaries = {}
    for key, rows in sorted(buckets.items()):
        policies = ["v2_exact_pitch"] + (["historical_e3"] if key.startswith("E3:") else [])
        summary = {"outputs": len(rows), "measurement_errors": sum(r["audit_status"] != "measured" for r in rows), "policies": {}}
        for policy in policies:
            measurements = [r["measurement"][policy] for r in rows if policy in r.get("measurement", {})]
            summary["policies"][policy] = {
                "event_retention_clustered": clustered_rate(rows, policy),
                "on_event_fraction_clustered": clustered_fraction(rows, policy, "on_event_retention_fraction"),
                "off_event_fraction_clustered": clustered_fraction(rows, policy, "off_event_retention_fraction"),
                "full_policy_statuses": dict(Counter(m["status"] for m in measurements)),
                "duration_statuses": dict(Counter(m["duration_status"] for m in measurements)),
                "order_statuses": dict(Counter(m["event_order_status"] for m in measurements)),
                "literal_order_diagnostic_statuses": dict(Counter(m["literal_event_order_status"] for m in measurements)),
                "original_tuple_protection_statuses": dict(Counter(m["original_tuple_protection_status"] for m in measurements if "original_tuple_protection_status" in m)),
                "fifo_tuple_failures": sum(not m["fifo_tuple_equal"] for m in measurements),
                "reselected_output_skyline_failures": sum(not m["reselected_output_skyline_equal"] for m in measurements),
                "pairing_ambiguous_outputs": sum(m["pairing_ambiguous_protected_count"] > 0 for m in measurements),
                "higher_note_outputs": sum(bool(m["higher_note_at_protected_onset"]) for m in measurements),
                "protected_velocity_statuses": dict(Counter(m["protected_velocity_status"] for m in measurements)),
            }
        summary["structural_technical_failures"] = dict(Counter(
            name for r in rows for name, value in r.get("measurement", {}).get("structural_technical", {}).items() if value is False))
        summary["selector_tie_outputs"] = sum(bool(r.get("measurement", {}).get("source_selector_tie_onsets")) for r in rows)
        summaries[key] = summary
    return {"method": {"unit": "work group", "estimand": "equal-work-group mean of within-group observable protected event retention rates and per-output retained on/off fractions",
                       "ci": "percentile cluster bootstrap, 2000 draws, seed 1729; descriptive, no multiplicity adjustment",
                       "undefined": "excluded from rates with explicit denominator/status counts; measurement errors count as undefined",
                       "caution": "event retention does not imply identified voice durations or musicological melody; identities are source-self references, not serialized regenerations"},
            "directions": summaries}


def run_content_audit(checkout: Path, output: Path, *, configuration: dict[str, str | None] | None = None,
                      provenance: dict[str, Any] | None = None, expected_counts: tuple[int, int] = (150, 300)) -> dict[str, Any]:
    started = time.perf_counter()
    checkout, output = checkout.resolve(), output.resolve()
    roots = resolve_asset_roots(checkout, configuration)
    frozen = frozen_files(checkout, roots)
    protected_dirs = [checkout / "src", checkout / "configs", Path(roots["data"]["path"]), Path(roots["literature"]["path"])]
    protected_dirs += [Path(roots["results"]["path"]) / name for name in ("e2_asap", "e3_asap")]
    if any(output.is_relative_to(p.resolve()) or p.resolve().is_relative_to(output) for p in protected_dirs) or any(output.is_relative_to(p.parent) for p in frozen):
        raise ValueError("audit destination overlaps frozen inputs")
    output.mkdir(parents=True, exist_ok=False)
    provenance = provenance or collect_provenance(checkout)
    provenance["roots"] = roots
    provenance["configuration_fingerprint"] = fingerprint({"contract": CONTRACT, "roots": roots, "expected_counts": expected_counts})
    provenance["audit_source_sha256"] = {str(p.relative_to(checkout)): sha256_file(p) for p in
        [checkout / "src/musicians_style" / name for name in ("content_metrics.py", "content_audit.py", "asset_paths.py", "provenance.py")]
        + [checkout / "tools/content_audit.py"] if p.is_file()}
    write_json(output / "provenance.json", provenance)
    write_json(output / "metric_contract.json", CONTRACT)
    if not provenance["import"]["passed"]:
        result = {"passed": False, "error": "checkout import guard failed"}
        write_json(output / "audit.json", result)
        return result
    hashes = {str(p): sha256_file(p) for p in frozen}
    records, errors, checks = [], [], []
    observations = {}

    def read_json(path: Path) -> Any:
        hashes[str(path.resolve())] = sha256_file(path)
        return json.loads(path.read_text(encoding="utf-8"))

    def load(path: Path, expected: str) -> Any:
        path = path.resolve()
        digest = sha256_file(path)
        hashes[str(path)] = digest
        if not expected or digest != expected:
            raise ValueError("missing or mismatched recorded SHA256: " + str(path))
        if path not in observations:
            observations[path] = observe_midi(path)
        return observations[path]

    try:
        manifest = read_json(Path(roots["data"]["path"]) / "derived/e1_asap/manifest.json")
        samples = [r for r in manifest["samples"] if r.get("validation_status") == "accepted"]
        by_id = {r["sample_id"]: r for r in samples}
        checks.append({"name": "canonical_source_count_and_uniqueness", "passed": len(samples) == len(by_id) == expected_counts[0]})
        dataset = Path(roots["data"]["path"]) / Path(manifest["dataset_root"]).name

        def source_path(sample: dict[str, Any]) -> Path:
            path = (dataset / sample["score_path"]).resolve()
            if not path.is_relative_to(dataset.resolve()):
                raise ValueError("source escapes dataset")
            return path

        for sample in sorted(samples, key=lambda r: r["sample_id"]):
            row = {"experiment": "identity", "task_id": "identity:" + sample["sample_id"], "source_id": sample["sample_id"],
                   "source_group_id": sample["group_id"], "source_composer": sample["composer"], "target_composer": sample["composer"]}
            try:
                path = source_path(sample)
                observation = load(path, sample["sha256"])
                row.update(audit_status="measured", source_path=str(path), source_sha256=sample["sha256"],
                           measurement=measure_content(observation, observation))
            except Exception as exc:  # preserve one explicit failure record per reference
                row.update(audit_status="error", error=f"{type(exc).__name__}: {exc}")
            records.append(row)
        for experiment, folder in (("E2", "e2_asap"), ("E3", "e3_asap")):
            run = Path(roots["results"]["path"]) / folder
            results = read_json(run / "results.json")["results"]
            checks.append({"name": experiment + "_count_unique_tasks", "passed": len(results) == expected_counts[1] and len({r["task_id"] for r in results}) == len(results)})
            pairs = Counter((r["source_id"], r["target_composer"]) for r in results)
            composers = {s["composer"] for s in samples}
            expected_pairs = {(s["sample_id"], target) for s in samples for target in composers - {s["composer"]}}
            checks.append({"name": experiment + "_direction_coverage", "passed": set(pairs) == expected_pairs and all(n == 1 for n in pairs.values())})
            for original in sorted(results, key=lambda r: r["task_id"]):
                row = {k: original[k] for k in ("task_id", "source_id", "source_group_id", "source_composer", "target_composer")}
                row["experiment"] = experiment
                try:
                    sample = by_id[row["source_id"]]
                    if row["source_group_id"] != sample["group_id"] or row["source_composer"] != sample["composer"] or original.get("status") != "completed":
                        raise ValueError("source identity or completion status mismatch")
                    path = (run / original["output_path"].replace("\\", "/")).resolve()
                    if not path.is_relative_to(run.resolve()):
                        raise ValueError("output escapes frozen run")
                    before = load(source_path(sample), sample["sha256"])
                    after = load(path, original["output_sha256"])
                    shift = original["best_genome"]["transpose_semitones"] if experiment == "E3" else None
                    row.update(audit_status="measured", source_path=str(source_path(sample)), source_sha256=sample["sha256"],
                               output_path=str(path), output_sha256=original["output_sha256"], recorded_transposition=shift,
                               measurement=measure_content(before, after, historical_shift=shift))
                except Exception as exc:  # audit boundary; never silently exclude an output
                    row.update(audit_status="error", error=f"{type(exc).__name__}: {exc}")
                records.append(row)
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    changed = []
    for path, digest in hashes.items():
        try:
            if sha256_file(Path(path)) != digest:
                changed.append(path)
        except OSError:
            changed.append(path)
    new_frozen = sorted(set(map(str, frozen_files(checkout, roots))) - set(hashes))
    checks.append({"name": "frozen_inputs_byte_identical", "passed": not changed and not new_frozen,
                   "files": len(hashes), "changed_or_missing": changed, "added": new_frozen})
    checks.append({"name": "all_references_and_outputs_measured", "passed": len(records) == expected_counts[0] + 2 * expected_counts[1] and all(r["audit_status"] == "measured" for r in records)})
    result = {"schema": "v2-03.audit.1", "passed": not errors and all(c["passed"] for c in checks), "checks": checks,
              "errors": errors, "records": len(records), "elapsed_seconds": time.perf_counter() - started,
              "note": "Audit completion/integrity status; content damage and ambiguity are findings, not audit execution failures."}
    summary = summarize(records)
    write_json(output / "per_output.json", records)
    write_json(output / "summary.json", summary)
    write_json(output / "audit.json", result)
    write_json(output / "frozen_sha256.json", hashes)
    ambiguities = [r for r in records if r.get("audit_status") == "error" or
                   r.get("measurement", {}).get("source_selector_tie_onsets") or
                   any(m.get("status") in {"ambiguous", "undefined"} or m.get("higher_note_at_protected_onset")
                       for k, m in r.get("measurement", {}).items() if k in {"v2_exact_pitch", "historical_e3"})]
    write_json(output / "ambiguities.json", ambiguities)
    lines = ["# V2-03 content audit", "", f"Audit completed with integrity checks: {result['passed']}; records: {len(records)}.", "",
             "Original source Skyline is protected; output Skyline is only a diagnostic. Historical E3 uses its recorded allowed shift; V2 uses zero shift.", "",
             "| Experiment / direction | Outputs | Policy | Observable events passed / failed / undefined | Full policy statuses | Work-group mean [95% CI] |", "|---|---:|---|---|---|---|"]
    for key, bucket in summary["directions"].items():
        for policy, measurement in bucket["policies"].items():
            rate = measurement["event_retention_clustered"]
            lines.append(f"| {key} | {bucket['outputs']} | {policy} | {rate['statuses']} | {measurement['full_policy_statuses']} | {rate['group_mean']} {rate['ci95']} |")
    lines += ["", "Full-policy ambiguity is retained when MIDI voice pairing or simultaneous event occurrence/order cannot be identified; it is not classified as damage.",
              "On/off retention proves observable event inclusion, not a unique note duration or causal voice lineage. FIFO tuple failures are diagnostics only.",
              "Historical E3 uses its original same-tick permutation-equivalent representation: chronological order and original tuple/velocity protection are reported. Literal simultaneous serialization order is a separate diagnostic and a V2 requirement; it never retroactively changes E3's interpretation.",
              "Meter/essential metadata/format/resolution/channels are structural checks; protected velocity is a separate current-transformation invariant.",
              "The bootstrap weights work groups equally, excludes explicitly undefined values, and is descriptive. No aggregate content scalar, listening claim, training or output regeneration.",
              "Other metadata excludes end_of_track and track allocation; frozen parser essential metadata scope is recorded in metric_contract.json.", "",
              "See summary.json for direction counts, duration/order statuses, structural failures, velocity, selector ties and higher-note cases; per_output.json and ambiguities.json retain all details.", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return result
