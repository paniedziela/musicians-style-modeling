"""Leakage-safe nested classification and reporting for the E1a baseline."""

from __future__ import annotations

import json
import platform
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RESULTS_FILENAME = "e1a_results.json"
PREDICTIONS_FILENAME = "e1a_predictions.json"
RESULTS_SCHEMA_VERSION = "e1.2.0"


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


def _metrics(y_true: list[str], y_pred: list[str], labels: list[str]) -> dict[str, Any]:
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "labels": labels,
    }


def _cluster_bootstrap_ci(
    rows: list[dict[str, Any]], *, seed: int, samples: int
) -> list[float]:
    rng = np.random.default_rng(seed)
    clusters: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        clusters[(row["repeat"], row["group_id"])].append(row)
    keys = sorted(clusters)
    scores: list[float] = []
    for _ in range(samples):
        chosen = rng.integers(0, len(keys), size=len(keys))
        selected = [row for index in chosen for row in clusters[keys[int(index)]]]
        scores.append(
            float(balanced_accuracy_score(
                [row["true_composer"] for row in selected],
                [row["predicted_composer"] for row in selected],
            ))
        )
    return [float(value) for value in np.percentile(scores, [2.5, 97.5])]


def _group_permutation_p_value(
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


def run_e1a(
    feature_cache: dict[str, Any],
    splits: dict[str, Any],
    *,
    bootstrap_samples: int = 2000,
    permutations: int = 999,
    model_names: tuple[str, ...] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate both E1a variants with precomputed outer and inner grouped folds."""
    if bootstrap_samples < 1 or permutations < 1:
        raise ValueError("bootstrap_samples and permutations must be positive")
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

    for variant, specification in feature_cache["variants"].items():
        columns = np.asarray(specification["indices"], dtype=int)
        x = matrix[:, columns]
        for repetition in splits["repetitions"]:
            repeat = int(repetition["repeat"])
            seed = int(repetition["seed"])
            available = _models(seed)
            selected_models = model_names or tuple(available)
            unknown = set(selected_models) - set(available)
            if unknown:
                raise ValueError(f"unknown model names: {sorted(unknown)}")
            for fold in repetition["folds"]:
                train_idx = _index(fold["train"]["sample_ids"], positions)
                test_idx = _index(fold["test"]["sample_ids"], positions)
                train_positions = {sample_id: index for index, sample_id in enumerate(fold["train"]["sample_ids"])}
                inner_cv = [
                    (
                        _index(inner["train"]["sample_ids"], train_positions),
                        _index(inner["validation"]["sample_ids"], train_positions),
                    )
                    for inner in fold["inner_folds"]
                ]
                for model_name in selected_models:
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
                        best_params = {key: _json_value(value) for key, value in search.best_params_.items()}
                        inner_score = float(search.best_score_)
                    predicted = fitted.predict(x[test_idx]).tolist()
                    truth = target[test_idx].tolist()
                    fold_results.append(
                        {
                            "variant": variant,
                            "model": model_name,
                            "repeat": repeat,
                            "seed": seed,
                            "fold": int(fold["fold"]),
                            "best_params": best_params,
                            "inner_balanced_accuracy": inner_score,
                            **_metrics(truth, predicted, labels),
                        }
                    )
                    for index, prediction in zip(test_idx, predicted):
                        sample_id = ids[int(index)]
                        predictions.append(
                            {
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
                        )

    summaries: list[dict[str, Any]] = []
    combinations = sorted({(row["variant"], row["model"]) for row in predictions})
    for offset, (variant, model) in enumerate(combinations):
        selected = [row for row in predictions if row["variant"] == variant and row["model"] == model]
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
                "variant": variant,
                "model": model,
                **summary,
                "balanced_accuracy_95_ci_clustered": _cluster_bootstrap_ci(
                    selected, seed=104729 + offset, samples=bootstrap_samples
                ),
                "group_label_permutation_p_value": _group_permutation_p_value(
                    selected, seed=130363 + offset, permutations=permutations
                ),
                "repeat_metrics": repeat_metrics,
            }
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
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "summaries": summaries,
        "fold_results": fold_results,
    }
    return result, predictions


def write_e1a_results(
    feature_cache_path: Path | str,
    splits_path: Path | str,
    output_dir: Path | str,
    **kwargs: Any,
) -> tuple[Path, Path, dict[str, Any]]:
    """Run E1.2 and atomically write its metrics and sample-level OOF predictions."""
    cache = json.loads(Path(feature_cache_path).read_text(encoding="utf-8"))
    splits = json.loads(Path(splits_path).read_text(encoding="utf-8"))
    result, predictions = run_e1a(cache, splits, **kwargs)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    result_path = destination / RESULTS_FILENAME
    predictions_path = destination / PREDICTIONS_FILENAME
    for path, payload in ((result_path, result), (predictions_path, predictions)):
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
    return result_path, predictions_path, result
