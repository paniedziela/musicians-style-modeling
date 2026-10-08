"""Evaluate frozen E2/E3 MIDIs with the existing jSymbolic style proxy.

Run with the project environment. Re-running reuses extraction_cache.json;
all classifier fits and summaries are recomputed from the frozen repeat-0 folds.
"""

from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tempfile import TemporaryDirectory

from musicians_style.asset_paths import resolve_asset_roots
from musicians_style.external_classifier import fit_external, score_external
from musicians_style.feature_backends.jsymbolic import numeric_schema
from musicians_style.jsymbolic_runtime import JSymbolicRuntime
from musicians_style.provenance import sha256_file, write_json
from musicians_style.style_audit import (
    align_tasks,
    cluster_summary,
    movement,
    validate_splits,
)


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def compact_extraction(record):
    fields = ("status", "input_sha256", "values", "diagnostics", "failure")
    return {name: record[name] for name in fields if name in record}


def extract_one(runtime, path, output):
    # Runtime sanitation and Java exports are temporary; input MIDIs are read only.
    with TemporaryDirectory(prefix="jsymbolic_", dir=output) as temporary:
        record = runtime.extract(path, Path(temporary) / "extraction")
        return compact_extraction(record)


def summarize(rows):
    complete = [
        row
        for row in rows
        if row["E2_delta"] is not None and row["E3_delta"] is not None
    ]

    def comparisons(selected):
        return {
            label: cluster_summary(selected, field)
            for label, field in (
                ("E2_minus_identity", "E2_delta"),
                ("E3_minus_identity", "E3_delta"),
                ("E3_minus_E2", "paired_delta"),
            )
        }

    directions = sorted({(row["composer"], row["target"]) for row in rows})
    direction_summaries = {}
    for source, target in directions:
        direction_rows = [
            row
            for row in complete
            if row["composer"] == source and row["target"] == target
        ]
        direction_summaries[f"{source}→{target}"] = comparisons(direction_rows)

    return {
        "tasks": len(rows),
        "complete_pairs": len(complete),
        "undefined_pairs": len(rows) - len(complete),
        "overall": comparisons(complete),
        "directions": direction_summaries,
    }


def report(summary, args):
    def estimate(value):
        if value["group_mean"] is None:
            return "undefined"
        lower, upper = value["ci95"]
        return f"{value['group_mean']:+.4g} [{lower:+.4g}, {upper:+.4g}]"

    coverage = summary["extractions"]
    lines = [
        "# jSymbolic external evaluation of frozen E2/E3",
        "",
        (
            f"{summary['sources']} accepted sources; 300 E2 and 300 E3 outputs. "
            f"{summary['complete_pairs']}/300 source–target pairs scored in all comparisons; "
            f"{len(coverage['failures'])} extraction failures."
        ),
        "",
        (
            "The existing 638-column jSymbolic 2.2 projection and logistic classifier "
            "are reused. Five models use the existing repeat-0 grouped outer folds; "
            "imputation, variance filtering and scaling are fitted on each training fold "
            "only. The held-out source and both outputs use the same model. Identity "
            "is the unchanged source score (movement zero)."
        ),
        "",
        (
            "Target movement is p(target|output) − p(target|source). Values below are "
            "equal-work means with the existing 2,000-draw work bootstrap 95% intervals "
            "(seed 1729); each work averages its source pieces and, overall, both targets. "
            "All comparisons use the same complete pairs. Row-weighted means are also "
            "retained in summary.json."
        ),
        "",
        "| Direction | Pairs | Works | E2 − identity | E3 − identity | E3 − E2 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    comparisons = [("Overall", summary["overall"])]
    comparisons += list(summary["directions"].items())
    for label, values in comparisons:
        counts = values["E2_minus_identity"]
        cells = [
            estimate(values[name])
            for name in ("E2_minus_identity", "E3_minus_identity", "E3_minus_E2")
        ]
        lines.append(
            f"| {label} | {counts['defined']} | {counts['groups']} | "
            + " | ".join(cells)
            + " |"
        )
    paired = summary["overall"]["E3_minus_E2"]
    if paired["ci95"] is not None:
        lower, upper = paired["ci95"]
        if lower > 0:
            conclusion = "E3 has greater overall target-style movement than E2."
        elif upper < 0:
            conclusion = "E2 has greater overall target-style movement than E3."
        else:
            conclusion = (
                "The overall E3−E2 interval includes zero; this evaluator does "
                "not establish an overall advantage for either method."
            )
        lines += ["", conclusion]
    lines += [
        "",
        (
            "These are descriptive classifier-proxy results conditional on the fixed "
            "folds and fits, not evidence of authorship, musical quality or content "
            "preservation. Direction intervals are descriptive, without new tests or "
            "decision thresholds. The existing jSymbolic metadata sanitation also "
            "applies to these inputs."
        ),
        "",
        (
            "Reproduce with the pinned project environment (Python 3.10, numpy 1.26.4, "
            "scikit-learn 1.5.1, mido 1.3.2) from the repository root:"
        ),
        "",
        "```powershell",
        (
            ".\\.venv\\Scripts\\python.exe tools/evaluate_e2_e3_jsymbolic.py "
            f'--distribution "{args.distribution}" --java "{args.java}" '
            f'--output "{args.output}"'
        ),
        "```",
        "",
        (
            "The extraction cache retains the pinned backend/runtime, feature schema "
            "and nullable values for all 750 inputs. Re-running reuses it and refits "
            "the five classifiers. Optional --source-cache reuses existing Track A "
            "source extractions. per_task.csv contains target probabilities and paired "
            "movements; fits.json records training/test IDs, parameters and fit warnings."
        ),
        "",
    ]
    return "\n".join(lines)


def parse_arguments(checkout):
    """Parse command options and resolve the input and output directories."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distribution", required=True, type=Path)
    parser.add_argument("--java", required=True, type=Path)
    parser.add_argument(
        "--output", type=Path, default=Path("experiments/e2_e3_jsymbolic")
    )
    parser.add_argument("--source-cache", type=Path)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--data-root")
    parser.add_argument("--results-root")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")

    roots = resolve_asset_roots(
        checkout, {"data": args.data_root, "results": args.results_root}
    )
    data = Path(roots["data"]["path"])
    results = Path(roots["results"]["path"])
    args.output = args.output.resolve()
    # Keep every write in a separate results child, outside existing experiments.
    if args.output.parent != results or args.output.name in ("e2_asap", "e3_asap"):
        parser.error("--output must be a separate direct child of the results root")
    if args.output.exists() and not (args.output / "extraction_cache.json").is_file():
        parser.error("existing output is not an extraction cache from this script")
    return args, data, results


def load_corpus(data, results):
    """Load and align the frozen corpus, checking each MIDI against its saved hash."""
    derived = data / "derived/e1_asap"
    manifest = read(derived / "manifest.json")
    samples = [
        row for row in manifest["samples"] if row["validation_status"] == "accepted"
    ]
    all_folds = validate_splits(samples, read(derived / "splits.json"))
    folds = [fold for repeat, fold in all_folds if repeat == 0]
    frozen = {
        name: read(results / folder / "results.json")["results"]
        for name, folder in (("E2", "e2_asap"), ("E3", "e3_asap"))
    }
    repeat_zero_folds = [(0, fold) for fold in folds]
    pairs = align_tasks(frozen["E2"], frozen["E3"], samples, repeat_zero_folds)
    if len(samples) != 150 or len(pairs) != 300:
        raise ValueError("expected the frozen 150-source/300-pair corpus")

    dataset = (data / Path(manifest["dataset_root"]).name).resolve()
    inputs = {}
    for sample in samples:
        sample_id = sample["sample_id"]
        path = (dataset / sample["score_path"]).resolve()
        if not path.is_relative_to(dataset) or sha256_file(path) != sample["sha256"]:
            raise ValueError("source path/hash mismatch: " + sample_id)
        inputs[sample_id] = (path, sample["sha256"])

    for e2, e3 in pairs:
        for name, row in (("E2", e2), ("E3", e3)):
            folder = "e2_asap" if name == "E2" else "e3_asap"
            base = (results / folder).resolve()
            relative_path = row["output_path"].replace("\\", "/")
            path = (base / relative_path).resolve()
            if not path.is_relative_to(base):
                raise ValueError("output path outside frozen experiment")
            if sha256_file(path) != row["output_sha256"]:
                raise ValueError("frozen output mismatch: " + row["task_id"])
            key = name + ":" + row["task_id"]
            inputs[key] = (path, row["output_sha256"])
    return samples, folds, pairs, inputs


def load_extractions(runtime, samples, inputs, output, source_cache, workers):
    """Reuse compatible cached features and extract only missing inputs."""
    schema = numeric_schema(runtime.expected)
    cache_path = output / "extraction_cache.json"
    if cache_path.exists():
        cache = read(cache_path)
        backend_matches = cache["backend"]["lock"] == runtime.lock
        if not backend_matches or cache["schema"] != schema:
            raise ValueError("cached extraction backend/schema mismatch")
    else:
        cache = {"backend": runtime.verify(), "schema": schema, "records": {}}
    records = cache["records"]

    if source_cache:
        previous = read(source_cache)["external"]
        for sample in samples:
            sample_id = sample["sample_id"]
            record = previous.get(sample_id)
            if sample_id in records or not record:
                continue
            if record["status"] != "success":
                continue
            if record["runtime"]["lock"] != runtime.lock:
                continue
            if record["schema"] != schema:
                continue
            if record["input_sha256"] != inputs[sample_id][1]:
                continue
            records[sample_id] = compact_extraction(record)

    for key, record in records.items():
        if key not in inputs or record.get("input_sha256") != inputs[key][1]:
            raise ValueError("cached input mismatch: " + key)
    output.mkdir(parents=True, exist_ok=True)
    write_json(cache_path, cache)
    pending = [(key, path) for key, (path, _) in inputs.items() if key not in records]
    print(f"Cached {len(records)}/750; extracting {len(pending)} inputs", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(extract_one, runtime, path, output): key
            for key, path in pending
        }
        for count, future in enumerate(as_completed(futures), 1):
            key = futures[future]
            records[key] = future.result()
            if count % 10 == 0 or count == len(pending):
                write_json(cache_path, cache)
                failures = sum(r["status"] != "success" for r in records.values())
                print(
                    f"Extracted {count}/{len(pending)}; failures={failures}", flush=True
                )
    return records, schema


def score_folds(samples, folds, pairs, records, schema):
    """Fit each classifier on its training fold and score held-out inputs."""
    by_id = {sample["sample_id"]: sample for sample in samples}
    attempts = {key: dict(record, schema=schema) for key, record in records.items()}
    scores = {}
    fits = []
    for fold in folds:
        train = [by_id[sample_id] for sample_id in fold["train"]["sample_ids"]]
        test = [by_id[sample_id] for sample_id in fold["test"]["sample_ids"]]
        pipeline, metadata = fit_external(train, test, attempts)
        score_ids = list(fold["test"]["sample_ids"])
        for e2, e3 in pairs:
            if e2["fold"] != fold["fold"]:
                continue
            score_ids.append("E2:" + e2["task_id"])
            score_ids.append("E3:" + e3["task_id"])
        predictions = score_external(pipeline, schema, attempts, score_ids)
        for prediction in predictions:
            scores[prediction["sample_id"]] = prediction
        fits.append(
            {
                "fold": fold["fold"],
                "train_ids": metadata["successful_training_ids"],
                "test_ids": fold["test"]["sample_ids"],
                "failed_training_ids": metadata["failed_training_ids"],
                "parameters": metadata["parameters"],
                "warnings": metadata["warnings"],
            }
        )
        print(f"Scored fold {fold['fold']}", flush=True)
    return scores, fits


def build_task_rows(samples, pairs, scores):
    """Pair identity, E2 and E3 target probabilities and movements for each task."""
    by_id = {sample["sample_id"]: sample for sample in samples}
    rows = []
    for e2, e3 in pairs:
        sample_id = e2["source_id"]
        target = e2["target_composer"]
        source = by_id[sample_id]
        before = scores[sample_id].get("affinities")
        row = {
            "task_id": e2["task_id"],
            "sample_id": sample_id,
            "group_id": source["group_id"],
            "composer": source["composer"],
            "target": target,
            "fold": e2["fold"],
            "identity_target": before[target] if before else None,
            "identity_delta": 0.0 if before else None,
        }
        for name, frozen_row in (("E2", e2), ("E3", e3)):
            key = name + ":" + frozen_row["task_id"]
            after = scores[key].get("affinities")
            row[name + "_target"] = after[target] if after else None
            delta = None
            if before and after:
                delta = movement(before, after, source["composer"], target)["delta"]
            row[name + "_delta"] = delta

        e2_delta = row["E2_delta"]
        e3_delta = row["E3_delta"]
        row["paired_delta"] = None
        if e2_delta is not None and e3_delta is not None:
            row["paired_delta"] = e3_delta - e2_delta
        rows.append(row)
    return rows


def write_results(checkout, args, samples, rows, records, scores, fits):
    """Write the existing fit, task, summary and report artifacts."""
    write_json(args.output / "fits.json", fits)
    with (args.output / "per_task.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows)
    summary["sources"] = len(samples)
    summary["extractions"] = {
        "inputs": len(records),
        "failures": {
            key: record["failure"]
            for key, record in records.items()
            if record["status"] != "success"
        },
        "score_failures": {
            key: result["failure"]
            for key, result in scores.items()
            if result["status"] != "success"
        },
    }
    write_json(args.output / "summary.json", summary)
    summary_path = checkout / "docs/results/E2_E3_jSymbolic.md"
    summary_path.write_text(report(summary, args), encoding="utf-8")


def main():
    checkout = Path(__file__).resolve().parents[1]
    args, data, results = parse_arguments(checkout)
    samples, folds, pairs, inputs = load_corpus(data, results)
    runtime = JSymbolicRuntime(checkout, args.distribution, args.java)
    records, schema = load_extractions(
        runtime, samples, inputs, args.output, args.source_cache, args.workers
    )
    scores, fits = score_folds(samples, folds, pairs, records, schema)
    rows = build_task_rows(samples, pairs, scores)
    write_results(checkout, args, samples, rows, records, scores, fits)
    print(f"Results: {args.output}", flush=True)


if __name__ == "__main__":
    main()
