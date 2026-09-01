"""Leakage-safe nested classification and reporting for E1a and E1b."""

from __future__ import annotations

import json
import hashlib
import platform
import shutil
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import numpy as np
import sklearn
from sklearn.base import BaseEstimator, clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RESULTS_FILENAME = "e1a_results.json"
PREDICTIONS_FILENAME = "e1a_predictions.json"
RESULTS_SCHEMA_VERSION = "e1.2.1"
RUN_MANIFEST_FILENAME = "run_manifest.json"
ProgressCallback = Callable[[dict[str, Any]], None]

E1B_RESULTS_FILENAME = "e1b_results.json"
E1B_PREDICTIONS_FILENAME = "e1b_predictions.json"
E1B_RESULTS_SCHEMA_VERSION = "e1.3.0"


def _notify(callback: ProgressCallback | None, event: dict[str, Any]) -> None:
    """Report progress without allowing a presentation failure to abort training."""
    if callback is None:
        return
    try:
        callback({"timestamp_utc": datetime.now(timezone.utc).isoformat(), **event})
    except Exception:
        # Console/file progress is auxiliary; metrics must remain authoritative.
        return


def _models(seed: int) -> dict[str, tuple[BaseEstimator, dict[str, list[Any]] | None]]:
    return {
        "dummy_most_frequent": (DummyClassifier(strategy="most_frequent"), None),
        "logistic_regression": (
            Pipeline(
                [
                    ("variance", VarianceThreshold()),
                    ("scale", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            class_weight="balanced", max_iter=5000, random_state=seed
                        ),
                    ),
                ]
            ),
            {"model__C": [0.01, 0.1, 1.0, 10.0]},
        ),
        "random_forest": (
            Pipeline(
                [
                    ("variance", VarianceThreshold()),
                    (
                        "model",
                        RandomForestClassifier(
                            n_estimators=300,
                            class_weight="balanced",
                            random_state=seed,
                            n_jobs=1,
                        ),
                    ),
                ]
            ),
            {
                "model__max_features": ["sqrt", 0.5],
                "model__min_samples_leaf": [1, 2],
            },
        ),
    }


def _index(ids: list[str], positions: dict[str, int]) -> np.ndarray:
    try:
        return np.asarray([positions[sample_id] for sample_id in ids], dtype=int)
    except KeyError as exc:
        raise ValueError(f"split references unknown sample_id: {exc.args[0]}") from exc


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        )
        return result.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _one_sample_per_group(rows: list[dict[str, Any]], seed: int) -> set[str]:
    """Choose one sample from every group deterministically for a sensitivity run."""
    grouped: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        grouped[row["group_id"]].append(row["sample_id"])
    rng = np.random.default_rng(seed)
    return {
        sorted(grouped[group_id])[int(rng.integers(len(grouped[group_id])))]
        for group_id in sorted(grouped)
    }


def _filter_ids(ids: list[str], eligible: set[str]) -> list[str]:
    return [sample_id for sample_id in ids if sample_id in eligible]


def _metrics(y_true: list[str], y_pred: list[str], labels: list[str]) -> dict[str, Any]:
    return {
        # Fixed labels keep fold scores comparable even if grouped splitting leaves
        # a very small validation fold without one of the composers.
        "balanced_accuracy": float(
            recall_score(
                y_true, y_pred, labels=labels, average="macro", zero_division=0
            )
        ),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "labels": labels,
    }


def _permutation_importance_for_fold(
    estimator: BaseEstimator,
    x: np.ndarray,
    y: np.ndarray,
    *,
    feature_names: list[str],
    feature_groups: dict[str, str],
    labels: list[str],
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    """Measure held-out importance without using it for model selection."""

    def scorer(fitted: BaseEstimator, values: np.ndarray, truth: np.ndarray) -> float:
        return float(
            recall_score(
                truth,
                fitted.predict(values),
                labels=labels,
                average="macro",
                zero_division=0,
            )
        )

    measured = permutation_importance(
        estimator,
        x,
        y,
        scoring=scorer,
        n_repeats=repeats,
        random_state=seed,
        n_jobs=1,
    )
    features = [
        {
            "feature": name,
            "group": feature_groups.get(name, "unknown"),
            "importance_mean": float(measured.importances_mean[index]),
            "importance_std": float(measured.importances_std[index]),
        }
        for index, name in enumerate(feature_names)
    ]
    groups: dict[str, list[int]] = defaultdict(list)
    for index, name in enumerate(feature_names):
        groups[feature_groups.get(name, "unknown")].append(index)
    group_rows = []
    for group, indices in sorted(groups.items()):
        # Sum within each repeat before computing dispersion so the group total
        # remains in balanced-accuracy points.
        totals = measured.importances[np.asarray(indices)].sum(axis=0)
        group_rows.append(
            {
                "group": group,
                "importance_mean": float(totals.mean()),
                "importance_std": float(totals.std()),
            }
        )
    return {"features": features, "groups": group_rows}


def _summarize_importance(folds: list[dict[str, Any]]) -> dict[str, Any]:
    feature_values: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    group_values: dict[tuple[str, str], list[float]] = defaultdict(list)
    for fold in folds:
        model = fold["model"]
        for row in fold["features"]:
            feature_values[(model, row["group"], row["feature"])].append(
                row["importance_mean"]
            )
        for row in fold["groups"]:
            group_values[(model, row["group"])].append(row["importance_mean"])
    return {
        "by_feature": [
            {
                "model": model,
                "group": group,
                "feature": feature,
                "mean_across_folds": float(np.mean(values)),
                "std_across_folds": float(np.std(values)),
                "fold_count": len(values),
            }
            for (model, group, feature), values in sorted(feature_values.items())
        ],
        "by_group": [
            {
                "model": model,
                "group": group,
                "mean_across_folds": float(np.mean(values)),
                "std_across_folds": float(np.std(values)),
                "fold_count": len(values),
            }
            for (model, group), values in sorted(group_values.items())
        ],
    }


def _error_and_form_analysis(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [
        row
        for row in predictions
        if row["analysis"] == "all_samples" and row.get("form")
    ]
    if not rows:
        return {}
    by_form: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    confusions: Counter[tuple[str, str, str, str]] = Counter()
    sample_errors: Counter[tuple[str, str, str, str, str]] = Counter()
    for row in rows:
        key = (row["variant"], row["model"], row["form"])
        by_form[key].append(row)
        if row["true_composer"] != row["predicted_composer"]:
            confusions[
                (
                    row["variant"],
                    row["model"],
                    row["true_composer"],
                    row["predicted_composer"],
                )
            ] += 1
            sample_errors[
                (
                    row["variant"],
                    row["model"],
                    row["sample_id"],
                    row["title"],
                    row["form"],
                )
            ] += 1
    return {
        "by_form": [
            {
                "variant": variant,
                "model": model,
                "form": form,
                "prediction_count": len(selected),
                "accuracy": float(
                    np.mean(
                        [
                            row["true_composer"] == row["predicted_composer"]
                            for row in selected
                        ]
                    )
                ),
            }
            for (variant, model, form), selected in sorted(by_form.items())
        ],
        "confusion_pairs": [
            {
                "variant": variant,
                "model": model,
                "true_composer": truth,
                "predicted_composer": predicted,
                "count": count,
            }
            for (variant, model, truth, predicted), count in sorted(confusions.items())
        ],
        "misclassified_samples": [
            {
                "variant": variant,
                "model": model,
                "sample_id": sample_id,
                "title": title,
                "form": form,
                "error_count_across_repeats": count,
            }
            for (variant, model, sample_id, title, form), count in sorted(
                sample_errors.items(), key=lambda item: (-item[1], item[0])
            )
        ],
    }


def _cluster_bootstrap_ci(
    rows: list[dict[str, Any]], *, seed: int, samples: int
) -> list[float]:
    rng = np.random.default_rng(seed)
    clusters: dict[tuple[int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        clusters[(row["repeat"], row["true_composer"], row["group_id"])].append(row)
    strata: dict[tuple[int, str], list[tuple[int, str, str]]] = defaultdict(list)
    for key in sorted(clusters):
        strata[(key[0], key[1])].append(key)
    scores: list[float] = []
    for _ in range(samples):
        selected: list[dict[str, Any]] = []
        for keys in strata.values():
            chosen = rng.integers(0, len(keys), size=len(keys))
            selected.extend(
                row for index in chosen for row in clusters[keys[int(index)]]
            )
        scores.append(
            float(balanced_accuracy_score(
                [row["true_composer"] for row in selected],
                [row["predicted_composer"] for row in selected],
            ))
        )
    return [float(value) for value in np.percentile(scores, [2.5, 97.5])]


def _prediction_label_association_p_value(
    rows: list[dict[str, Any]], *, seed: int, permutations: int
) -> float:
    observed = balanced_accuracy_score(
        [row["true_composer"] for row in rows],
        [row["predicted_composer"] for row in rows],
    )
    rng = np.random.default_rng(seed)
    by_repeat: dict[int, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_repeat[row["repeat"]][row["group_id"]].append(row)
    exceedances = 0
    for _ in range(permutations):
        permuted_true: list[str] = []
        predicted: list[str] = []
        for repeat in sorted(by_repeat):
            groups = by_repeat[repeat]
            keys = sorted(groups)
            labels = [groups[key][0]["true_composer"] for key in keys]
            shuffled = rng.permutation(labels)
            mapping = dict(zip(keys, shuffled))
            for key in keys:
                for row in groups[key]:
                    permuted_true.append(str(mapping[key]))
                    predicted.append(row["predicted_composer"])
        if balanced_accuracy_score(permuted_true, predicted) >= observed:
            exceedances += 1
    return float((exceedances + 1) / (permutations + 1))


def validate_oof_predictions(
    feature_cache: dict[str, Any],
    splits: dict[str, Any],
    predictions: list[dict[str, Any]],
    *,
    model_names: tuple[str, ...],
    analysis_modes: tuple[str, ...],
) -> dict[str, Any]:
    """Fail closed unless every expected OOF prediction exists exactly once."""
    metadata = {row["sample_id"]: row for row in feature_cache["samples"]}
    expected: set[tuple[str, str, str, int, str]] = set()
    fold_lookup: dict[tuple[int, str], int] = {}
    for repetition in splits["repetitions"]:
        repeat = int(repetition["repeat"])
        for fold in repetition["folds"]:
            for sample_id in fold["test"]["sample_ids"]:
                fold_lookup[(repeat, sample_id)] = int(fold["fold"])
        for analysis in analysis_modes:
            eligible = (
                set(metadata)
                if analysis == "all_samples"
                else _one_sample_per_group(feature_cache["samples"], int(repetition["seed"]))
            )
            for variant in feature_cache["variants"]:
                for model in model_names:
                    expected.update(
                        (analysis, variant, model, repeat, sample_id)
                        for sample_id in eligible
                    )

    actual: set[tuple[str, str, str, int, str]] = set()
    errors: list[str] = []
    for row in predictions:
        key = (
            row["analysis"], row["variant"], row["model"], int(row["repeat"]), row["sample_id"]
        )
        if key in actual:
            errors.append(f"duplicate OOF prediction: {key}")
        actual.add(key)
        sample = metadata.get(row["sample_id"])
        if sample is None:
            errors.append(f"unknown sample in predictions: {row['sample_id']}")
            continue
        if row["true_composer"] != sample["composer"]:
            errors.append(f"composer mismatch: {row['sample_id']}")
        if row["group_id"] != sample["group_id"] or row["sha256"] != sample["sha256"]:
            errors.append(f"metadata mismatch: {row['sample_id']}")
        if int(row["fold"]) != fold_lookup.get((int(row["repeat"]), row["sample_id"])):
            errors.append(f"fold mismatch: {row['sample_id']}")
    missing = expected - actual
    unexpected = actual - expected
    if missing:
        errors.append(f"missing {len(missing)} expected OOF predictions")
    if unexpected:
        errors.append(f"found {len(unexpected)} unexpected OOF predictions")
    if errors:
        raise ValueError("OOF validation failed: " + "; ".join(errors[:10]))
    return {
        "passed": True,
        "prediction_count": len(predictions),
        "expected_prediction_count": len(expected),
        "unique_prediction_keys": len(actual),
    }


def _fixed_permutation_estimator(model_name: str, seed: int) -> BaseEstimator:
    estimator, _ = _models(seed)[model_name]
    if model_name == "logistic_regression":
        estimator.set_params(model__C=1.0)
    elif model_name == "random_forest":
        estimator.set_params(model__max_features="sqrt", model__min_samples_leaf=1)
    return estimator


def _retrained_group_permutation_tests(
    feature_cache: dict[str, Any],
    splits: dict[str, Any],
    *,
    permutations: int,
    model_names: tuple[str, ...],
    progress_callback: ProgressCallback | None,
) -> list[dict[str, Any]]:
    """Retrain fixed pipelines after group-level label permutations on repeat 0."""
    if permutations <= 0:
        return []
    rows = feature_cache["samples"]
    ids = [row["sample_id"] for row in rows]
    positions = {sample_id: index for index, sample_id in enumerate(ids)}
    matrix = np.asarray([row["values"] for row in rows], dtype=np.float64)
    target = np.asarray([row["composer"] for row in rows])
    groups = np.asarray([row["group_id"] for row in rows])
    repetition = splits["repetitions"][0]
    seed = int(repetition["seed"])
    folds = [
        (
            _index(fold["train"]["sample_ids"], positions),
            _index(fold["test"]["sample_ids"], positions),
        )
        for fold in repetition["folds"]
    ]
    group_ids = sorted(set(groups.tolist()))
    group_labels = {group: str(target[np.flatnonzero(groups == group)[0]]) for group in group_ids}
    tested_models = tuple(name for name in model_names if name != "dummy_most_frequent")
    results: list[dict[str, Any]] = []
    for variant_offset, (variant, specification) in enumerate(feature_cache["variants"].items()):
        x = matrix[:, np.asarray(specification["indices"], dtype=int)]
        for model_offset, model_name in enumerate(tested_models):
            estimator = _fixed_permutation_estimator(model_name, seed)

            def cross_validated_score(y: np.ndarray) -> float:
                truth: list[str] = []
                predicted: list[str] = []
                for train_idx, test_idx in folds:
                    fitted = clone(estimator).fit(x[train_idx], y[train_idx])
                    truth.extend(y[test_idx].tolist())
                    predicted.extend(fitted.predict(x[test_idx]).tolist())
                return float(balanced_accuracy_score(truth, predicted))

            observed = cross_validated_score(target)
            rng = np.random.default_rng(900001 + variant_offset * 1009 + model_offset)
            exceedances = 0
            null_scores: list[float] = []
            for permutation in range(permutations):
                shuffled = rng.permutation([group_labels[group] for group in group_ids])
                mapping = dict(zip(group_ids, shuffled))
                permuted = np.asarray([mapping[group] for group in groups])
                score = cross_validated_score(permuted)
                null_scores.append(score)
                exceedances += int(score >= observed)
                _notify(
                    progress_callback,
                    {
                        "event": "permutation_completed",
                        "variant": variant,
                        "model": model_name,
                        "position": permutation + 1,
                        "total_permutations": permutations,
                        "null_balanced_accuracy": score,
                    },
                )
            results.append(
                {
                    "variant": variant,
                    "model": model_name,
                    "repeat": int(repetition["repeat"]),
                    "seed": seed,
                    "hyperparameters": (
                        {"model__C": 1.0}
                        if model_name == "logistic_regression"
                        else {"model__max_features": "sqrt", "model__min_samples_leaf": 1}
                    ),
                    "observed_balanced_accuracy": observed,
                    "permutations": permutations,
                    "p_value": float((exceedances + 1) / (permutations + 1)),
                    "null_mean": float(np.mean(null_scores)),
                    "null_std": float(np.std(null_scores)),
                }
            )
    return results


def run_e1a(
    feature_cache: dict[str, Any],
    splits: dict[str, Any],
    *,
    bootstrap_samples: int = 2000,
    permutations: int = 999,
    retraining_permutations: int = 0,
    include_group_sensitivity: bool = True,
    model_names: tuple[str, ...] | None = None,
    permutation_importance_repeats: int = 0,
    progress_callback: ProgressCallback | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate cache variants with precomputed outer and inner grouped folds."""
    if (
        bootstrap_samples < 1
        or permutations < 1
        or retraining_permutations < 0
        or permutation_importance_repeats < 0
    ):
        raise ValueError("bootstrap/permutation counts are outside their valid ranges")
    rows = feature_cache.get("samples", [])
    ids = [row["sample_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("feature cache sample_id values are not unique")
    positions = {sample_id: index for index, sample_id in enumerate(ids)}
    matrix = np.asarray([row["values"] for row in rows], dtype=np.float64)
    target = np.asarray([row["composer"] for row in rows])
    labels = sorted(set(target.tolist()))
    metadata = {row["sample_id"]: row for row in rows}
    predictions: list[dict[str, Any]] = []
    fold_results: list[dict[str, Any]] = []
    importance_folds: list[dict[str, Any]] = []
    importance_variant = feature_cache.get("importance_variant")
    feature_groups = {
        row["name"]: row["group"] for row in feature_cache.get("feature_contract", [])
    }
    selected_model_names = model_names or tuple(_models(0))
    unknown = set(selected_model_names) - set(_models(0))
    if unknown:
        raise ValueError(f"unknown model names: {sorted(unknown)}")
    analysis_modes = (
        ("all_samples", "one_sample_per_group")
        if include_group_sensitivity
        else ("all_samples",)
    )
    fold_count = sum(len(repetition["folds"]) for repetition in splits["repetitions"])
    total_fits = (
        len(analysis_modes)
        * len(feature_cache["variants"])
        * fold_count
        * len(selected_model_names)
    )
    completed_fits = 0
    _notify(progress_callback, {"event": "run_started", "total_fits": total_fits})

    sensitivity_selections: list[dict[str, Any]] = []
    for analysis in analysis_modes:
        for variant, specification in feature_cache["variants"].items():
            columns = np.asarray(specification["indices"], dtype=int)
            x = matrix[:, columns]
            for repetition in splits["repetitions"]:
                repeat = int(repetition["repeat"])
                seed = int(repetition["seed"])
                eligible = (
                    set(ids)
                    if analysis == "all_samples"
                    else _one_sample_per_group(rows, seed)
                )
                if analysis == "one_sample_per_group" and variant == next(iter(feature_cache["variants"])):
                    sensitivity_selections.append(
                        {"repeat": repeat, "seed": seed, "sample_ids": sorted(eligible)}
                    )
                available = _models(seed)
                for fold in repetition["folds"]:
                    train_ids = _filter_ids(fold["train"]["sample_ids"], eligible)
                    test_ids = _filter_ids(fold["test"]["sample_ids"], eligible)
                    train_idx = _index(train_ids, positions)
                    test_idx = _index(test_ids, positions)
                    train_positions = {
                        sample_id: index for index, sample_id in enumerate(train_ids)
                    }
                    inner_cv = [
                        (
                            _index(_filter_ids(inner["train"]["sample_ids"], eligible), train_positions),
                            _index(_filter_ids(inner["validation"]["sample_ids"], eligible), train_positions),
                        )
                        for inner in fold["inner_folds"]
                    ]
                    for model_name in selected_model_names:
                        position = completed_fits + 1
                        progress_fields = {
                            "position": position,
                            "total_fits": total_fits,
                            "analysis": analysis,
                            "variant": variant,
                            "model": model_name,
                            "repeat": repeat,
                            "fold": int(fold["fold"]),
                        }
                        _notify(progress_callback, {"event": "fit_started", **progress_fields})
                        started = perf_counter()
                        estimator, grid = available[model_name]
                        if grid is None:
                            fitted = estimator.fit(x[train_idx], target[train_idx])
                            best_params: dict[str, Any] = {}
                            inner_score = None
                        else:
                            search = GridSearchCV(
                                estimator,
                                grid,
                                scoring="balanced_accuracy",
                                cv=inner_cv,
                                n_jobs=1,
                                refit=True,
                                error_score="raise",
                            )
                            fitted = search.fit(x[train_idx], target[train_idx])
                            best_params = {
                                key: _json_value(value)
                                for key, value in search.best_params_.items()
                            }
                            inner_score = float(search.best_score_)
                        predicted = fitted.predict(x[test_idx]).tolist()
                        truth = target[test_idx].tolist()
                        fold_metrics = _metrics(truth, predicted, labels)
                        if (
                            permutation_importance_repeats > 0
                            and variant == importance_variant
                            and analysis == "all_samples"
                            and model_name != "dummy_most_frequent"
                        ):
                            importance = _permutation_importance_for_fold(
                                fitted,
                                x[test_idx],
                                target[test_idx],
                                feature_names=list(specification["feature_names"]),
                                feature_groups=feature_groups,
                                labels=labels,
                                repeats=permutation_importance_repeats,
                                seed=(
                                    seed
                                    + int(fold["fold"]) * 1009
                                    + selected_model_names.index(model_name)
                                ),
                            )
                            importance_folds.append(
                                {
                                    "variant": variant,
                                    "model": model_name,
                                    "repeat": repeat,
                                    "fold": int(fold["fold"]),
                                    **importance,
                                }
                            )
                        fold_results.append(
                            {
                                "analysis": analysis,
                                "variant": variant,
                                "model": model_name,
                                "repeat": repeat,
                                "seed": seed,
                                "fold": int(fold["fold"]),
                                "best_params": best_params,
                                "inner_balanced_accuracy": inner_score,
                                **fold_metrics,
                            }
                        )
                        for index, prediction in zip(test_idx, predicted):
                            sample_id = ids[int(index)]
                            prediction_row = {
                                "analysis": analysis,
                                "variant": variant,
                                "model": model_name,
                                "repeat": repeat,
                                "fold": int(fold["fold"]),
                                "sample_id": sample_id,
                                "group_id": metadata[sample_id]["group_id"],
                                "sha256": metadata[sample_id]["sha256"],
                                "true_composer": str(target[int(index)]),
                                "predicted_composer": str(prediction),
                            }
                            for field in ("title", "form"):
                                if field in metadata[sample_id]:
                                    prediction_row[field] = metadata[sample_id][field]
                            predictions.append(prediction_row)
                        completed_fits += 1
                        _notify(
                            progress_callback,
                            {
                                "event": "fit_completed",
                                **progress_fields,
                                "elapsed_seconds": perf_counter() - started,
                                "balanced_accuracy": fold_metrics["balanced_accuracy"],
                            },
                        )

    summaries: list[dict[str, Any]] = []
    combinations = sorted(
        {(row["analysis"], row["variant"], row["model"]) for row in predictions}
    )
    for offset, (analysis, variant, model) in enumerate(combinations):
        selected = [
            row for row in predictions
            if row["analysis"] == analysis and row["variant"] == variant and row["model"] == model
        ]
        repeat_metrics = []
        for repeat in sorted({row["repeat"] for row in selected}):
            repeated = [row for row in selected if row["repeat"] == repeat]
            repeat_metrics.append({"repeat": repeat, **_metrics(
                [row["true_composer"] for row in repeated],
                [row["predicted_composer"] for row in repeated], labels
            )})
        summary = _metrics(
            [row["true_composer"] for row in selected],
            [row["predicted_composer"] for row in selected], labels
        )
        summaries.append(
            {
                "analysis": analysis,
                "variant": variant,
                "model": model,
                **summary,
                "balanced_accuracy_95_ci_clustered": _cluster_bootstrap_ci(
                    selected, seed=104729 + offset, samples=bootstrap_samples
                ),
                "prediction_label_association_permutation_p_value": _prediction_label_association_p_value(
                    selected, seed=130363 + offset, permutations=permutations
                ),
                "repeat_metrics": repeat_metrics,
            }
        )
    oof_validation = validate_oof_predictions(
        feature_cache,
        splits,
        predictions,
        model_names=selected_model_names,
        analysis_modes=analysis_modes,
    )
    retrained_tests = _retrained_group_permutation_tests(
        feature_cache,
        splits,
        permutations=retraining_permutations,
        model_names=selected_model_names,
        progress_callback=progress_callback,
    )
    result = {
        "results_schema_version": RESULTS_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "feature_schema_version": feature_cache.get("features_schema_version"),
        "splits_schema_version": splits.get("splits_schema_version"),
        "protocol": {
            "outer_splits": splits.get("outer_splits"),
            "inner_splits": splits.get("inner_splits"),
            "seeds": splits.get("seeds"),
            "primary_metric": "balanced_accuracy",
            "secondary_metric": "macro_f1",
            "bootstrap_samples": bootstrap_samples,
            "label_permutations": permutations,
            "label_permutation_method": "fixed OOF prediction-label association control",
            "retraining_permutations": retraining_permutations,
            "retraining_permutation_method": "group labels, fixed pipelines, first predeclared repeat",
            "analysis_modes": list(analysis_modes),
            "permutation_importance_repeats": permutation_importance_repeats,
            "permutation_importance_scope": (
                "outer test folds, all_samples, non-dummy models, importance_variant only"
                if permutation_importance_repeats
                else None
            ),
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "summaries": summaries,
        "fold_results": fold_results,
        "sensitivity_selections": sensitivity_selections,
        "oof_validation": oof_validation,
        "retrained_group_permutation_tests": retrained_tests,
        "permutation_importance": {
            "variant": importance_variant,
            "folds": importance_folds,
            **_summarize_importance(importance_folds),
        },
        "error_and_form_analysis": _error_and_form_analysis(predictions),
    }
    _notify(progress_callback, {"event": "run_completed", "total_fits": total_fits})
    return result, predictions


def run_e1b(
    feature_cache: dict[str, Any],
    splits: dict[str, Any],
    *,
    permutation_importance_repeats: int = 10,
    **kwargs: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run the E1.3 protocol on composition features and their group ablations."""
    if feature_cache.get("features_schema_version") != "e1.3.0":
        raise ValueError("run_e1b requires an e1.3.0 composition feature cache")
    result, predictions = run_e1a(
        feature_cache,
        splits,
        permutation_importance_repeats=permutation_importance_repeats,
        **kwargs,
    )
    result["results_schema_version"] = E1B_RESULTS_SCHEMA_VERSION
    result["experiment"] = "E1.3"
    return result, predictions


def _write_classification_results(
    feature_cache_path: Path | str,
    splits_path: Path | str,
    output_dir: Path | str,
    *,
    experiment: str,
    results_filename: str,
    predictions_filename: str,
    runner: Callable[..., tuple[dict[str, Any], list[dict[str, Any]]]],
    config_path: Path | str | None = None,
    manifest_path: Path | str | None = None,
    **kwargs: Any,
) -> tuple[Path, Path, dict[str, Any]]:
    """Atomically run and persist a classification stage with provenance."""
    cache_path = Path(feature_cache_path).resolve()
    split_path = Path(splits_path).resolve()
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    splits = json.loads(split_path.read_text(encoding="utf-8"))
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    snapshot_dir = destination / "inputs"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    source_inputs: dict[str, Path] = {
        "feature_cache": cache_path,
        "splits": split_path,
    }
    if manifest_path is not None:
        source_inputs["manifest"] = Path(manifest_path).resolve()
    inputs: dict[str, dict[str, str]] = {}
    for name, source in source_inputs.items():
        snapshot = snapshot_dir / f"{name}{source.suffix or '.json'}"
        shutil.copyfile(source, snapshot)
        inputs[name] = {
            "source_path": str(source),
            "snapshot_path": snapshot.relative_to(destination).as_posix(),
            "sha256": _sha256(source),
        }
    if config_path is not None:
        source_config = Path(config_path).resolve()
        config_snapshot = destination / f"config_used{source_config.suffix or '.yaml'}"
        shutil.copyfile(source_config, config_snapshot)
        inputs["config"] = {
            "source_path": str(source_config),
            "snapshot_path": config_snapshot.relative_to(destination).as_posix(),
            "sha256": _sha256(source_config),
        }
    run_manifest_path = destination / RUN_MANIFEST_FILENAME
    run_manifest = {
        "run_schema_version": "e1.run.1.0",
        "experiment": experiment,
        "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "code_commit": _git_commit(),
        "inputs": inputs,
        "parameters": {
            key: value for key, value in kwargs.items() if key != "progress_callback"
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    _atomic_json(run_manifest_path, run_manifest)
    try:
        result, predictions = runner(cache, splits, **kwargs)
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
    result["provenance"] = {
        "code_commit": run_manifest["code_commit"],
        "inputs": inputs,
        "run_manifest": RUN_MANIFEST_FILENAME,
    }
    result_path = destination / results_filename
    predictions_path = destination / predictions_filename
    for path, payload in ((result_path, result), (predictions_path, predictions)):
        _atomic_json(path, payload)
    run_manifest.update(
        {
            "status": "completed",
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "outputs": {
                "results": {"path": results_filename, "sha256": _sha256(result_path)},
                "predictions": {
                    "path": predictions_filename,
                    "sha256": _sha256(predictions_path),
                },
            },
            "oof_validation": result["oof_validation"],
        }
    )
    _atomic_json(run_manifest_path, run_manifest)
    return result_path, predictions_path, result


def write_e1a_results(
    feature_cache_path: Path | str,
    splits_path: Path | str,
    output_dir: Path | str,
    **kwargs: Any,
) -> tuple[Path, Path, dict[str, Any]]:
    """Run E1.2 and write metrics plus sample-level OOF predictions."""
    return _write_classification_results(
        feature_cache_path,
        splits_path,
        output_dir,
        experiment="E1.2",
        results_filename=RESULTS_FILENAME,
        predictions_filename=PREDICTIONS_FILENAME,
        runner=run_e1a,
        **kwargs,
    )


def write_e1b_results(
    feature_cache_path: Path | str,
    splits_path: Path | str,
    output_dir: Path | str,
    **kwargs: Any,
) -> tuple[Path, Path, dict[str, Any]]:
    """Run E1.3 and write its separate metrics and OOF predictions."""
    return _write_classification_results(
        feature_cache_path,
        splits_path,
        output_dir,
        experiment="E1.3",
        results_filename=E1B_RESULTS_FILENAME,
        predictions_filename=E1B_PREDICTIONS_FILENAME,
        runner=run_e1b,
        **kwargs,
    )
