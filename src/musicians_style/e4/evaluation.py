"""Validation metrics and frozen GO/NO-GO decision rules for E4."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline

from ..e1.composition_features import FEATURE_SPECS, extract_composition_features
from ..evaluation.content import content_metrics
from ..midi.types import InternalRepr
from .dataset import COMPOSERS, PieceSegments


def binary_f1(target: np.ndarray, prediction: np.ndarray) -> float:
    target, prediction = target.astype(bool), prediction.astype(bool)
    true_positive = int(np.logical_and(target, prediction).sum())
    false_positive = int(np.logical_and(~target, prediction).sum())
    false_negative = int(np.logical_and(target, ~prediction).sum())
    denominator = 2 * true_positive + false_positive + false_negative
    return 2 * true_positive / denominator if denominator else 1.0


def discriminator_classification_metrics(
    discriminator: torch.nn.Module,
    dataset: object,
    device: torch.device,
    *,
    batch_size: int = 64,
) -> dict[str, Any]:
    """Measure the real-segment domain signal that guides the generator."""
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)  # type: ignore[arg-type]
    confusion = np.zeros((len(COMPOSERS), len(COMPOSERS)), dtype=np.int64)
    was_training = discriminator.training
    discriminator.eval()
    with torch.no_grad():
        for raw in loader:
            x = raw["x"].to(device)
            truth = raw["source"].cpu().numpy()
            _, logits = discriminator(x)
            predicted = logits.argmax(dim=1).cpu().numpy()
            for expected, actual in zip(truth, predicted):
                confusion[int(expected), int(actual)] += 1
    discriminator.train(was_training)
    totals = confusion.sum(axis=1)
    recalls = np.divide(
        np.diag(confusion), totals, out=np.zeros(len(COMPOSERS), dtype=float), where=totals > 0,
    )
    return {
        "balanced_accuracy": float(recalls.mean()),
        "per_composer_recall": {name: float(recalls[index]) for index, name in enumerate(COMPOSERS)},
        "confusion_matrix": confusion.tolist(),
    }


def calibrate_thresholds(
    model: torch.nn.Module,
    pieces: Sequence[PieceSegments],
    device: torch.device,
    grid: Sequence[float],
) -> dict[str, Any]:
    """Calibrate onset/frame thresholds on source-domain identity reconstruction."""
    probabilities: list[list[np.ndarray]] = [[], []]
    targets: list[list[np.ndarray]] = [[], []]
    model.eval()
    with torch.no_grad():
        for piece in pieces:
            label = COMPOSERS.index(piece.composer)
            for segment in piece.segment_map.segments:
                if not segment.nonempty:
                    continue
                x = torch.from_numpy(segment.data.astype(np.float32))[None].to(device)
                probability = torch.sigmoid(model(x, torch.tensor([label], device=device)))[0].cpu().numpy()
                valid = segment.mask.astype(bool)
                for channel in range(2):
                    probabilities[channel].append(probability[channel, valid].reshape(-1))
                    targets[channel].append(segment.data[channel, valid].reshape(-1))
    names = ("onset", "frame")
    result: dict[str, Any] = {}
    for channel, name in enumerate(names):
        truth = np.concatenate(targets[channel]) if targets[channel] else np.zeros(0, dtype=np.uint8)
        scores = np.concatenate(probabilities[channel]) if probabilities[channel] else np.zeros(0, dtype=float)
        candidates = [(float(threshold), binary_f1(truth, scores >= threshold)) for threshold in grid]
        threshold, score = max(candidates, key=lambda item: (item[1], -abs(item[0] - .5)))
        result[name] = {"threshold": threshold, "identity_f1": score, "cells": int(truth.size)}
    return result


class FoldStyleEvaluator:
    """Independent E1b proxy and train-only standardized feature profiles."""

    def __init__(
        self,
        composition_payload: Mapping[str, Any],
        manifest_rows: Mapping[str, Mapping[str, Any]],
        fit_ids: Sequence[str],
        *,
        onset_tolerance_beats: float = 1 / 16,
    ) -> None:
        self.rows = {str(row["sample_id"]): row for row in composition_payload["samples"]}
        self.manifest_rows = manifest_rows
        self.onset_tolerance_beats = onset_tolerance_beats
        matrix = np.asarray([self.rows[item]["values"] for item in fit_ids], dtype=float)
        labels = np.asarray([manifest_rows[item]["composer"] for item in fit_ids])
        self.classifier = Pipeline([
            ("variance", VarianceThreshold()),
            ("model", RandomForestClassifier(
                n_estimators=300, class_weight="balanced", max_features=.5,
                min_samples_leaf=2, random_state=1729, n_jobs=1,
            )),
        ]).fit(matrix, labels)
        self.group_indices = {
            group: np.asarray([index for index, spec in enumerate(FEATURE_SPECS) if spec["group"] == group])
            for group in ("pitch", "rhythm", "texture")
        }
        self.profiles: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.target_max_polyphony: dict[str, int] = {}
        poly_index = next(index for index, spec in enumerate(FEATURE_SPECS) if spec["name"] == "max_polyphony")
        for composer in COMPOSERS:
            selected = np.asarray([self.rows[item]["values"] for item in fit_ids if manifest_rows[item]["composer"] == composer], dtype=float)
            if not len(selected):
                raise ValueError(f"style evaluator has no fit samples for {composer}")
            standard = selected.std(axis=0)
            standard[standard < 1e-6] = 1.0
            self.profiles[composer] = selected.mean(axis=0), standard
            self.target_max_polyphony[composer] = int(np.ceil(selected[:, poly_index].max()))

    def evaluate(
        self,
        sample_id: str,
        source: InternalRepr,
        output: InternalRepr,
        target_composer: str,
    ) -> dict[str, Any]:
        before = np.asarray(self.rows[sample_id]["values"], dtype=float)
        after = extract_composition_features(output)
        classes = [str(value) for value in self.classifier.classes_]
        target_index = classes.index(target_composer)
        p_before = float(self.classifier.predict_proba(before.reshape(1, -1))[0, target_index])
        p_after = float(self.classifier.predict_proba(after.reshape(1, -1))[0, target_index])
        mean, standard = self.profiles[target_composer]
        group_gains: dict[str, float] = {}
        for group, indices in self.group_indices.items():
            before_distance = float(np.sqrt(np.mean(np.square((before[indices] - mean[indices]) / standard[indices]))))
            after_distance = float(np.sqrt(np.mean(np.square((after[indices] - mean[indices]) / standard[indices]))))
            group_gains[group] = before_distance - after_distance
        return {
            "p_target_input": p_before,
            "p_target_output": p_after,
            "delta_p_target": p_after - p_before,
            "style_group_gains": group_gains,
            "content": content_metrics(
                source,
                output,
                onset_tolerance_beats=self.onset_tolerance_beats,
            ),
            "target_max_polyphony": self.target_max_polyphony[target_composer],
        }


def summarize_records(
    records: Sequence[Mapping[str, Any]],
    *,
    melody_min: float,
    fallback_max: float,
    target_similarity_max: float = .99,
) -> dict[str, Any]:
    if not records:
        raise ValueError("cannot summarize an empty E4 evaluation")
    directions: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        directions[f"{record['source_composer']}→{record['target_composer']}"].append(record)
    direction_means = {
        direction: float(np.mean([row["delta_p_target"] for row in values]))
        for direction, values in sorted(directions.items())
    }
    group_means = {
        group: float(np.mean([row["style_group_gains"][group] for row in records]))
        for group in ("pitch", "rhythm", "texture")
    }
    nonempty_segments = sum(int(row["diagnostics"]["nonempty_input_segments"]) for row in records)
    fallback_segments = sum(int(row["diagnostics"]["fallback_segments"]) for row in records)
    fallback_rate = fallback_segments / max(1, nonempty_segments)
    melody_values = [float(row["content"]["melody_trigram_jaccard"]) for row in records]
    length_ok = all(int(row["content"]["length_error_ticks"]) <= int(row["allowed_length_error_ticks"]) for row in records)
    style_groups_ok = sum(value >= 0 for value in group_means.values()) >= 2 and any(value > 0 for value in group_means.values())
    target_similarities = [float(row.get("cross_target_event_similarity", 0.0)) for row in records]
    criteria = {
        "mean_delta_p_target_positive": float(np.mean([row["delta_p_target"] for row in records])) > 0,
        "at_least_four_of_six_directions_positive": sum(value > 0 for value in direction_means.values()) >= 4,
        "style_groups_not_jointly_degraded": style_groups_ok,
        "all_roundtrip_and_structure_valid": all(
            row["artifact_parseable"]
            and row["content"]["ticks_per_beat_preserved"]
            and row["content"]["smf_format_preserved"]
            and row["content"]["meta_preserved"]
            and row["content"]["bar_count_preserved"]
            for row in records
        ),
        "length_within_one_local_step": length_ok,
        "median_melody_similarity_at_least_threshold": float(np.median(melody_values)) >= melody_min,
        "no_nonempty_output_became_empty": all(row["content"]["nonempty"] for row in records),
        "identity_fallback_rate_within_threshold": fallback_rate <= fallback_max,
        "polyphony_within_source_or_target_limit": all(
            row["content"]["max_polyphony_output"]
            <= max(row["content"]["max_polyphony_input"], row["target_max_polyphony"])
            for row in records
        ),
        "target_outputs_not_collapsed": float(np.mean(target_similarities)) < target_similarity_max,
    }
    return {
        "record_count": len(records),
        "mean_delta_p_target": float(np.mean([row["delta_p_target"] for row in records])),
        "direction_means": direction_means,
        "style_group_mean_gains": group_means,
        "median_melody_trigram_jaccard": float(np.median(melody_values)),
        "mean_onset_f1": float(np.mean([row["content"]["onset_f1"] for row in records])),
        "mean_chroma_cosine": float(np.mean([row["content"]["chroma_cosine"] for row in records])),
        "mean_cross_target_event_similarity": float(np.mean(target_similarities)),
        "fallback_segments": fallback_segments,
        "nonempty_input_segments": nonempty_segments,
        "fallback_rate": fallback_rate,
        "orphan_frame_starts": sum(int(row["diagnostics"]["orphan_frame_starts"]) for row in records),
        "active_pitches_at_segment_ends": sum(int(row["diagnostics"]["active_pitches_at_segment_ends"]) for row in records),
        "mean_empty_bar_ratio_input": float(np.mean([row["content"]["empty_bar_ratio_input"] for row in records])),
        "mean_empty_bar_ratio_output": float(np.mean([row["content"]["empty_bar_ratio_output"] for row in records])),
        "criteria": criteria,
        "passed": all(criteria.values()),
    }
