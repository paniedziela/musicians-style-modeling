"""Open-set robustness analysis for experiment E1-open."""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
import subprocess
from collections import defaultdict
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import sklearn
from sklearn.base import clone
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from .asap import E1Config, build_e1_manifest
from .classification import fixed_e1_estimator
from .composition_features import build_composition_feature_cache

OPEN_MANIFEST_FILENAME = "open_manifest.json"
OPEN_QUALITY_REPORT_FILENAME = "open_quality_report.json"
OPEN_FEATURES_FILENAME = "open_composition_features.json"
OPEN_RESULTS_FILENAME = "e1_open_results.json"
OPEN_PREDICTIONS_FILENAME = "e1_open_predictions.json"
OPEN_RESULTS_SCHEMA_VERSION = "e1.open.1.0"
ProgressCallback = Callable[[dict[str, Any]], None]


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _notify(callback: ProgressCallback | None, event: dict[str, Any]) -> None:
    if callback is None:
        return
    try:
        callback({"timestamp_utc": datetime.now(timezone.utc).isoformat(), **event})
    except Exception:
        # Progress output is auxiliary and must not invalidate computed metrics.
        return


def _validate_open_config(config: E1Config) -> None:
    if not config.open_calibration_composers or not config.open_test_composers:
        raise ValueError("open_set requires calibration_composers and test_composers")
    if set(config.open_models) - {"logistic_regression", "random_forest"}:
        raise ValueError("E1-open supports fixed logistic_regression and random_forest")


def build_open_data(config: E1Config) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Audit all known and unknown composers and build their E1b feature cache."""
    _validate_open_config(config)
    all_composers = (
        *config.composers,
        *config.open_calibration_composers,
        *config.open_test_composers,
    )
    audit_config = replace(
        config,
        composers=tuple(all_composers),
        minimum_samples_per_class=config.open_minimum_samples_per_composer,
        schema_version="e1.open.data.1.0",
    )
    manifest, report = build_e1_manifest(audit_config)
    role_by_composer = {
        **{composer: "known" for composer in config.composers},
        **{
            composer: "unknown_calibration"
            for composer in config.open_calibration_composers
        },
        **{composer: "unknown_test" for composer in config.open_test_composers},
    }
    for sample in manifest["samples"]:
        sample["open_role"] = role_by_composer[sample["composer"]]
    manifest["open_set_partition"] = {
        "known_composers": list(config.composers),
        "calibration_composers": list(config.open_calibration_composers),
        "test_composers": list(config.open_test_composers),
    }
    cache = build_composition_feature_cache(
        manifest, dataset_root=config.dataset_root
    )
    for row in cache["samples"]:
        row["open_role"] = role_by_composer[row["composer"]]
    cache["open_set_partition"] = manifest["open_set_partition"]
    return manifest, report, cache


def write_open_data(config: E1Config) -> tuple[Path, Path, Path, dict[str, Any]]:
    manifest, report, cache = build_open_data(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = config.output_dir / OPEN_MANIFEST_FILENAME
    report_path = config.output_dir / OPEN_QUALITY_REPORT_FILENAME
    cache_path = config.output_dir / OPEN_FEATURES_FILENAME
    for path, payload in (
        (manifest_path, manifest),
        (report_path, report),
        (cache_path, cache),
    ):
        _atomic_json(path, payload)
    return manifest_path, report_path, cache_path, report


def _indices(sample_ids: Sequence[str], positions: dict[str, int]) -> np.ndarray:
    return np.asarray([positions[sample_id] for sample_id in sample_ids], dtype=int)


def _aligned_probabilities(estimator: Any, x: np.ndarray, labels: list[str]) -> np.ndarray:
    raw = estimator.predict_proba(x)
    result = np.zeros((len(x), len(labels)), dtype=np.float64)
    columns = {str(label): index for index, label in enumerate(estimator.classes_)}
    for target_index, label in enumerate(labels):
        result[:, target_index] = raw[:, columns[label]]
    return result


def _aggregate_unknown(
    rows: list[dict[str, Any]],
    probability_lists: dict[str, list[np.ndarray]],
    labels: list[str],
    *,
    source: str,
    fold_by_id: dict[str, int],
) -> list[dict[str, Any]]:
    by_id = {row["sample_id"]: row for row in rows}
    result = []
    for sample_id in sorted(probability_lists):
        probabilities = np.mean(probability_lists[sample_id], axis=0)
        best = int(np.argmax(probabilities))
        sample = by_id[sample_id]
        result.append(
            {
                "sample_id": sample_id,
                "composer": sample["composer"],
                "title": sample.get("title"),
                "group_id": sample["group_id"],
                "sha256": sample["sha256"],
                "source": source,
                "fold": fold_by_id[sample_id],
                "known_score": float(probabilities[best]),
                "predicted_composer": labels[best],
                "class_probabilities": {
                    label: float(probabilities[index])
                    for index, label in enumerate(labels)
                },
            }
        )
    return result


def _unknown_fold(sample_id: str, fold_numbers: Sequence[int], seed: int) -> int:
    """Assign each unknown sample to exactly one model without using its label."""
    digest = hashlib.sha256(f"{seed}:{sample_id}".encode("utf-8")).digest()
    return int(fold_numbers[int.from_bytes(digest[:8], "big") % len(fold_numbers)])


def _choose_threshold(
    known_scores: np.ndarray,
    unknown_scores: np.ndarray,
    *,
    target_known_tpr: float,
) -> tuple[float, dict[str, float]]:
    if not len(known_scores) or not len(unknown_scores):
        raise ValueError("threshold calibration requires known and unknown samples")
    candidates = np.unique(np.concatenate((known_scores, unknown_scores)))
    candidates = np.concatenate(
        ([np.nextafter(candidates.min(), -np.inf)], candidates)
    )
    feasible = []
    for threshold in candidates:
        known_recall = float(np.mean(known_scores >= threshold))
        unknown_recall = float(np.mean(unknown_scores < threshold))
        if known_recall + 1e-12 >= target_known_tpr:
            feasible.append(
                (
                    0.5 * (known_recall + unknown_recall),
                    unknown_recall,
                    float(threshold),
                    known_recall,
                )
            )
    best = max(feasible, key=lambda item: (item[0], item[1], item[2]))
    return best[2], {
        "balanced_accuracy": best[0],
        "known_recall": best[3],
        "unknown_recall": best[1],
    }


def _fpr_at_tpr(y_known: np.ndarray, scores: np.ndarray, target: float) -> float:
    fpr, tpr, _ = roc_curve(y_known, scores)
    eligible = fpr[tpr >= target]
    return float(eligible.min()) if eligible.size else 1.0


def _evaluate(
    known_rows: list[dict[str, Any]],
    unknown_rows: list[dict[str, Any]],
    *,
    threshold: float,
    known_labels: list[str],
    target_known_tpr: float,
) -> dict[str, Any]:
    all_rows = [*known_rows, *unknown_rows]
    y_known = np.asarray([1] * len(known_rows) + [0] * len(unknown_rows))
    scores = np.asarray([row["known_score"] for row in all_rows])
    known_accepted = np.asarray(
        [row["known_score"] >= threshold for row in known_rows], dtype=bool
    )
    unknown_rejected = np.asarray(
        [row["known_score"] < threshold for row in unknown_rows], dtype=bool
    )
    accepted_known_rows = [
        row for row, accepted in zip(known_rows, known_accepted) if accepted
    ]
    accepted_balanced_accuracy = (
        float(
            recall_score(
                [row["composer"] for row in accepted_known_rows],
                [row["predicted_composer"] for row in accepted_known_rows],
                labels=known_labels,
                average="macro",
                zero_division=0,
            )
        )
        if accepted_known_rows
        else 0.0
    )
    open_truth = [row["composer"] for row in known_rows] + ["unknown"] * len(
        unknown_rows
    )
    open_predicted = [
        row["predicted_composer"] if row["known_score"] >= threshold else "unknown"
        for row in all_rows
    ]
    labels = [*known_labels, "unknown"]
    return {
        "known_count": len(known_rows),
        "unknown_count": len(unknown_rows),
        "auroc_known_vs_unknown": float(roc_auc_score(y_known, scores)),
        "auprc_known_vs_unknown": float(average_precision_score(y_known, scores)),
        "false_positive_rate_at_target_known_tpr": _fpr_at_tpr(
            y_known, scores, target_known_tpr
        ),
        "target_known_tpr": target_known_tpr,
        "known_recall_at_threshold": float(known_accepted.mean()),
        "unknown_recall_at_threshold": float(unknown_rejected.mean()),
        "known_unknown_balanced_accuracy_at_threshold": float(
            0.5 * (known_accepted.mean() + unknown_rejected.mean())
        ),
        "coverage": float(
            np.mean([row["known_score"] >= threshold for row in all_rows])
        ),
        "accepted_known_balanced_accuracy": accepted_balanced_accuracy,
        "open_set_confusion_matrix": confusion_matrix(
            open_truth, open_predicted, labels=labels
        ).tolist(),
        "open_set_labels": labels,
    }


def _run_scenario(
    matrix: np.ndarray,
    rows: list[dict[str, Any]],
    splits: dict[str, Any],
    *,
    known_composers: tuple[str, ...],
    calibration_unknown_composers: tuple[str, ...],
    test_unknown_composers: tuple[str, ...],
    model_name: str,
    known_calibration_folds: tuple[int, ...],
    target_known_tpr: float,
    scenario: str,
    progress_callback: ProgressCallback | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    labels = sorted(known_composers)
    positions = {row["sample_id"]: index for index, row in enumerate(rows)}
    metadata = {row["sample_id"]: row for row in rows}
    calibration_unknown = [
        row for row in rows if row["composer"] in calibration_unknown_composers
    ]
    test_unknown = [row for row in rows if row["composer"] in test_unknown_composers]
    calibration_probabilities: dict[str, list[np.ndarray]] = defaultdict(list)
    test_probabilities: dict[str, list[np.ndarray]] = defaultdict(list)
    known_oof: list[dict[str, Any]] = []
    repetition = splits["repetitions"][0]
    seed = int(repetition["seed"])
    fold_numbers = [int(fold["fold"]) for fold in repetition["folds"]]
    unknown_fold_by_id = {
        row["sample_id"]: _unknown_fold(row["sample_id"], fold_numbers, seed)
        for row in [*calibration_unknown, *test_unknown]
    }
    for position, fold in enumerate(repetition["folds"], start=1):
        train_ids = [
            sample_id
            for sample_id in fold["train"]["sample_ids"]
            if metadata[sample_id]["composer"] in known_composers
        ]
        test_ids = [
            sample_id
            for sample_id in fold["test"]["sample_ids"]
            if metadata[sample_id]["composer"] in known_composers
        ]
        _notify(
            progress_callback,
            {
                "event": "open_fit_started",
                "scenario": scenario,
                "model": model_name,
                "position": position,
                "total_fits": len(repetition["folds"]),
                "fold": int(fold["fold"]),
            },
        )
        estimator = clone(fixed_e1_estimator(model_name, seed + int(fold["fold"])))
        train_idx = _indices(train_ids, positions)
        fitted = estimator.fit(
            matrix[train_idx],
            np.asarray([metadata[sample_id]["composer"] for sample_id in train_ids]),
        )
        test_idx = _indices(test_ids, positions)
        known_probabilities = _aligned_probabilities(fitted, matrix[test_idx], labels)
        for sample_id, probabilities in zip(test_ids, known_probabilities):
            best = int(np.argmax(probabilities))
            sample = metadata[sample_id]
            known_oof.append(
                {
                    "sample_id": sample_id,
                    "composer": sample["composer"],
                    "title": sample.get("title"),
                    "group_id": sample["group_id"],
                    "sha256": sample["sha256"],
                    "source": "known_oof",
                    "fold": int(fold["fold"]),
                    "known_score": float(probabilities[best]),
                    "predicted_composer": labels[best],
                    "class_probabilities": {
                        label: float(probabilities[index])
                        for index, label in enumerate(labels)
                    },
                }
            )
        for selected, collector in (
            (calibration_unknown, calibration_probabilities),
            (test_unknown, test_probabilities),
        ):
            selected = [
                row
                for row in selected
                if unknown_fold_by_id[row["sample_id"]] == int(fold["fold"])
            ]
            if not selected:
                continue
            selected_idx = _indices(
                [row["sample_id"] for row in selected], positions
            )
            probabilities = _aligned_probabilities(
                fitted, matrix[selected_idx], labels
            )
            for sample, values in zip(selected, probabilities):
                collector[sample["sample_id"]].append(values)
        _notify(
            progress_callback,
            {
                "event": "open_fit_completed",
                "scenario": scenario,
                "model": model_name,
                "position": position,
                "total_fits": len(repetition["folds"]),
                "fold": int(fold["fold"]),
            },
        )
    calibration_unknown_rows = _aggregate_unknown(
        calibration_unknown,
        calibration_probabilities,
        labels,
        source="unknown_calibration",
        fold_by_id=unknown_fold_by_id,
    )
    test_unknown_rows = _aggregate_unknown(
        test_unknown,
        test_probabilities,
        labels,
        source="unknown_test",
        fold_by_id=unknown_fold_by_id,
    )
    calibration_known_rows = [
        row for row in known_oof if row["fold"] in known_calibration_folds
    ]
    evaluation_known_rows = [
        row for row in known_oof if row["fold"] not in known_calibration_folds
    ]
    threshold, threshold_metrics = _choose_threshold(
        np.asarray([row["known_score"] for row in calibration_known_rows]),
        np.asarray([row["known_score"] for row in calibration_unknown_rows]),
        target_known_tpr=target_known_tpr,
    )
    for row in [*known_oof, *calibration_unknown_rows, *test_unknown_rows]:
        row["model"] = model_name
        row["scenario"] = scenario
        row["threshold"] = threshold
        row["accepted_as_known"] = row["known_score"] >= threshold
        row["open_prediction"] = (
            row["predicted_composer"] if row["accepted_as_known"] else "unknown"
        )
    summary = {
        "scenario": scenario,
        "model": model_name,
        "known_composers": labels,
        "calibration_unknown_composers": list(calibration_unknown_composers),
        "test_unknown_composers": list(test_unknown_composers),
        "threshold": threshold,
        "threshold_selection": {
            "objective": "maximum known-vs-unknown balanced accuracy subject to target known TPR",
            "known_calibration_folds": list(known_calibration_folds),
            "target_known_tpr": target_known_tpr,
            **threshold_metrics,
        },
        "calibration_metrics": _evaluate(
            calibration_known_rows,
            calibration_unknown_rows,
            threshold=threshold,
            known_labels=labels,
            target_known_tpr=target_known_tpr,
        ),
        "test_metrics": _evaluate(
            evaluation_known_rows,
            test_unknown_rows,
            threshold=threshold,
            known_labels=labels,
            target_known_tpr=target_known_tpr,
        ),
    }
    return summary, [*known_oof, *calibration_unknown_rows, *test_unknown_rows]


def run_e1_open(
    feature_cache: dict[str, Any],
    splits: dict[str, Any],
    *,
    known_composers: tuple[str, ...],
    calibration_composers: tuple[str, ...],
    test_composers: tuple[str, ...],
    model_names: tuple[str, ...] = ("logistic_regression", "random_forest"),
    known_calibration_folds: tuple[int, ...] = (0, 1),
    target_known_tpr: float = 0.95,
    include_rotations: bool = True,
    progress_callback: ProgressCallback | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate confidence rejection without training an artificial other class."""
    variants = feature_cache.get("variants", {})
    if "composition_full" not in variants:
        raise ValueError("E1-open requires the composition_full variant")
    all_composers = set(known_composers) | set(calibration_composers) | set(
        test_composers
    )
    available = {row["composer"] for row in feature_cache.get("samples", [])}
    missing = all_composers - available
    if missing:
        raise ValueError(f"feature cache lacks open-set composers: {sorted(missing)}")
    unknown_models = set(model_names) - {"logistic_regression", "random_forest"}
    if unknown_models:
        raise ValueError(f"unsupported E1-open models: {sorted(unknown_models)}")
    rows = feature_cache["samples"]
    indices = np.asarray(variants["composition_full"]["indices"], dtype=int)
    matrix = np.asarray([row["values"] for row in rows], dtype=np.float64)[:, indices]
    summaries = []
    predictions = []
    for model_name in model_names:
        summary, rows_out = _run_scenario(
            matrix,
            rows,
            splits,
            known_composers=known_composers,
            calibration_unknown_composers=calibration_composers,
            test_unknown_composers=test_composers,
            model_name=model_name,
            known_calibration_folds=known_calibration_folds,
            target_known_tpr=target_known_tpr,
            scenario="external_unknowns",
            progress_callback=progress_callback,
        )
        summaries.append(summary)
        predictions.extend(rows_out)
        if include_rotations:
            for held_out in known_composers:
                rotation_known = tuple(
                    composer for composer in known_composers if composer != held_out
                )
                summary, rows_out = _run_scenario(
                    matrix,
                    rows,
                    splits,
                    known_composers=rotation_known,
                    calibration_unknown_composers=calibration_composers,
                    test_unknown_composers=(held_out,),
                    model_name=model_name,
                    known_calibration_folds=known_calibration_folds,
                    target_known_tpr=target_known_tpr,
                    scenario=f"leave_{held_out.lower()}_unknown",
                    progress_callback=progress_callback,
                )
                summaries.append(summary)
                predictions.extend(rows_out)
    result = {
        "results_schema_version": OPEN_RESULTS_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "code_commit": _git_commit(),
        "feature_schema_version": feature_cache.get("features_schema_version"),
        "splits_schema_version": splits.get("splits_schema_version"),
        "protocol": {
            "closed_set_training_only": True,
            "feature_variant": "composition_full",
            "split_repeat": int(splits["repetitions"][0]["repeat"]),
            "models": list(model_names),
            "known_composers": list(known_composers),
            "calibration_composers": list(calibration_composers),
            "test_composers": list(test_composers),
            "known_calibration_folds": list(known_calibration_folds),
            "target_known_tpr": target_known_tpr,
            "include_rotations": include_rotations,
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "summaries": summaries,
    }
    return result, predictions


def write_e1_open_results(
    feature_cache_path: Path | str,
    splits_path: Path | str,
    output_dir: Path | str,
    *,
    config: E1Config,
    config_path: Path | str | None = None,
    manifest_path: Path | str | None = None,
    progress_callback: ProgressCallback | None = None,
    include_rotations: bool = True,
) -> tuple[Path, Path, dict[str, Any]]:
    """Run E1-open and persist inputs, results, predictions, and run status."""
    cache_source = Path(feature_cache_path).resolve()
    splits_source = Path(splits_path).resolve()
    cache = json.loads(cache_source.read_text(encoding="utf-8"))
    splits = json.loads(splits_source.read_text(encoding="utf-8"))
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    snapshots = destination / "inputs"
    snapshots.mkdir(parents=True, exist_ok=True)
    sources = {"feature_cache": cache_source, "splits": splits_source}
    if manifest_path is not None:
        sources["manifest"] = Path(manifest_path).resolve()
    inputs = {}
    for name, source in sources.items():
        snapshot = snapshots / f"{name}{source.suffix or '.json'}"
        shutil.copyfile(source, snapshot)
        inputs[name] = {
            "source_path": str(source),
            "snapshot_path": snapshot.relative_to(destination).as_posix(),
            "sha256": _sha256(source),
        }
    if config_path is not None:
        source = Path(config_path).resolve()
        snapshot = destination / f"config_used{source.suffix or '.yaml'}"
        shutil.copyfile(source, snapshot)
        inputs["config"] = {
            "source_path": str(source),
            "snapshot_path": snapshot.relative_to(destination).as_posix(),
            "sha256": _sha256(source),
        }
    run_manifest_path = destination / "run_manifest.json"
    run_manifest = {
        "run_schema_version": "e1.open.run.1.0",
        "experiment": "E1-open",
        "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "code_commit": _git_commit(),
        "inputs": inputs,
    }
    _atomic_json(run_manifest_path, run_manifest)
    try:
        result, predictions = run_e1_open(
            cache,
            splits,
            known_composers=config.composers,
            calibration_composers=config.open_calibration_composers,
            test_composers=config.open_test_composers,
            model_names=config.open_models,
            known_calibration_folds=config.open_known_calibration_folds,
            target_known_tpr=config.open_target_known_tpr,
            include_rotations=include_rotations,
            progress_callback=progress_callback,
        )
    except BaseException as exc:
        run_manifest.update(
            {
                "status": "failed",
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        _atomic_json(run_manifest_path, run_manifest)
        raise
    result["provenance"] = {"inputs": inputs, "run_manifest": "run_manifest.json"}
    result_path = destination / OPEN_RESULTS_FILENAME
    predictions_path = destination / OPEN_PREDICTIONS_FILENAME
    _atomic_json(result_path, result)
    _atomic_json(predictions_path, predictions)
    run_manifest.update(
        {
            "status": "completed",
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "outputs": {
                "results": {
                    "path": result_path.name,
                    "sha256": _sha256(result_path),
                },
                "predictions": {
                    "path": predictions_path.name,
                    "sha256": _sha256(predictions_path),
                },
            },
        }
    )
    _atomic_json(run_manifest_path, run_manifest)
    return result_path, predictions_path, result
