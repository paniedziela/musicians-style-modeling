"""Deterministic, leakage-safe nested cross-validation splits for E1.1."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

SPLITS_FILENAME = "splits.json"
SPLITS_SCHEMA_VERSION = "e1.1.0"


def _accepted(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    samples = [
        sample
        for sample in manifest.get("samples", [])
        if sample.get("validation_status") == "accepted"
    ]
    if not samples:
        raise ValueError("manifest contains no accepted samples")
    ids = [sample["sample_id"] for sample in samples]
    if len(ids) != len(set(ids)):
        raise ValueError("accepted sample_id values are not unique")
    return sorted(samples, key=lambda sample: sample["sample_id"])


def _counts(samples: Iterable[dict[str, Any]]) -> dict[str, Any]:
    values = list(samples)
    return {
        "samples_by_composer": dict(sorted(Counter(s["composer"] for s in values).items())),
        "groups_by_composer": {
            composer: len({s["group_id"] for s in values if s["composer"] == composer})
            for composer in sorted({s["composer"] for s in values})
        },
        "sample_count": len(values),
        "group_count": len({s["group_id"] for s in values}),
    }


def _partition(indices: np.ndarray, samples: list[dict[str, Any]]) -> dict[str, Any]:
    selected = [samples[int(index)] for index in indices]
    return {
        "sample_ids": [sample["sample_id"] for sample in selected],
        **_counts(selected),
    }


def _assert_disjoint(
    train_indices: np.ndarray, test_indices: np.ndarray, samples: list[dict[str, Any]]
) -> None:
    train = [samples[int(index)] for index in train_indices]
    test = [samples[int(index)] for index in test_indices]
    for field in ("sample_id", "group_id", "sha256"):
        left = {sample[field] for sample in train if sample.get(field)}
        right = {sample[field] for sample in test if sample.get(field)}
        overlap = left & right
        if overlap:
            raise ValueError(f"{field} leakage between train and test: {sorted(overlap)}")


def build_e1_splits(
    manifest: dict[str, Any],
    *,
    seeds: tuple[int, ...] = (1729, 2718, 3141, 5772, 8119),
    outer_splits: int = 5,
    inner_splits: int = 3,
) -> dict[str, Any]:
    """Create repeated outer folds and the grouped inner folds for every training set."""
    if not seeds:
        raise ValueError("at least one split seed is required")
    if outer_splits < 2 or inner_splits < 2:
        raise ValueError("outer_splits and inner_splits must be at least 2")
    samples = _accepted(manifest)
    y = np.asarray([sample["composer"] for sample in samples])
    groups = np.asarray([sample["group_id"] for sample in samples])
    x = np.zeros((len(samples), 1), dtype=np.uint8)
    repetitions: list[dict[str, Any]] = []

    for repeat, seed in enumerate(seeds):
        outer = StratifiedGroupKFold(
            n_splits=outer_splits, shuffle=True, random_state=int(seed)
        )
        folds: list[dict[str, Any]] = []
        seen_test: list[str] = []
        for outer_fold, (train_idx, test_idx) in enumerate(outer.split(x, y, groups)):
            _assert_disjoint(train_idx, test_idx, samples)
            train_samples = [samples[int(index)] for index in train_idx]
            inner_y = y[train_idx]
            inner_groups = groups[train_idx]
            inner_cv = StratifiedGroupKFold(
                n_splits=inner_splits,
                shuffle=True,
                random_state=int(seed) + outer_fold + 1,
            )
            inner_folds: list[dict[str, Any]] = []
            inner_seen: list[str] = []
            inner_x = np.zeros((len(train_idx), 1), dtype=np.uint8)
            for inner_fold, (inner_train, inner_valid) in enumerate(
                inner_cv.split(inner_x, inner_y, inner_groups)
            ):
                _assert_disjoint(inner_train, inner_valid, train_samples)
                inner_seen.extend(train_samples[int(index)]["sample_id"] for index in inner_valid)
                inner_folds.append(
                    {
                        "fold": inner_fold,
                        "train": _partition(inner_train, train_samples),
                        "validation": _partition(inner_valid, train_samples),
                    }
                )
            expected_inner = sorted(sample["sample_id"] for sample in train_samples)
            if sorted(inner_seen) != expected_inner:
                raise ValueError("inner folds do not cover each outer-training sample exactly once")
            seen_test.extend(samples[int(index)]["sample_id"] for index in test_idx)
            folds.append(
                {
                    "fold": outer_fold,
                    "train": _partition(train_idx, samples),
                    "test": _partition(test_idx, samples),
                    "inner_folds": inner_folds,
                }
            )
        expected = sorted(sample["sample_id"] for sample in samples)
        if sorted(seen_test) != expected:
            raise ValueError("outer folds do not cover each sample exactly once")
        repetitions.append({"repeat": repeat, "seed": int(seed), "folds": folds})

    return {
        "splits_schema_version": SPLITS_SCHEMA_VERSION,
        "manifest_schema_version": manifest.get("manifest_schema_version"),
        "code_commit": manifest.get("code_commit"),
        "outer_splits": outer_splits,
        "inner_splits": inner_splits,
        "seeds": list(seeds),
        "dataset": _counts(samples),
        "repetitions": repetitions,
    }


def write_e1_splits(
    manifest_path: Path | str,
    output_path: Path | str | None = None,
    **kwargs: Any,
) -> Path:
    """Build E1.1 splits from a manifest and atomically write them as JSON."""
    source = Path(manifest_path)
    manifest = json.loads(source.read_text(encoding="utf-8"))
    payload = build_e1_splits(manifest, **kwargs)
    destination = Path(output_path) if output_path else source.with_name(SPLITS_FILENAME)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return destination
