from __future__ import annotations

import json

import pytest

from musicians_style.e1.classification import (
    run_e1a,
    validate_oof_predictions,
    write_e1a_results,
)
from musicians_style.e1.features import build_legacy_feature_cache
from musicians_style.e1.splits import build_e1_splits


def _inputs() -> tuple[dict, dict]:
    samples = []
    for composer_index, composer in enumerate(("Bach", "Chopin", "Beethoven")):
        for number in range(10):
            vector = [0.0] * 42
            vector[composer_index + 1] = 1.0
            vector[13 + number] = 1.0
            samples.append(
                {
                    "sample_id": f"{composer.lower()}-{number}",
                    "composer": composer,
                    "group_id": f"{composer.lower()}-group-{number // 2}",
                    "sha256": f"{composer}-{number}",
                    "validation_status": "accepted",
                    "legacy_features": vector,
                }
            )
    manifest = {"manifest_schema_version": "test", "samples": samples}
    return (
        build_legacy_feature_cache(manifest),
        build_e1_splits(manifest, seeds=(7,), outer_splits=5, inner_splits=3),
    )


def test_dummy_produces_exactly_one_oof_prediction_per_variant_and_sample() -> None:
    cache, splits = _inputs()
    progress = []
    result, predictions = run_e1a(
        cache,
        splits,
        bootstrap_samples=20,
        permutations=20,
        model_names=("dummy_most_frequent",),
        progress_callback=progress.append,
    )
    assert len(predictions) == 2 * (30 + 15)
    assert len(result["summaries"]) == 4
    keys = [
        (row["analysis"], row["variant"], row["repeat"], row["sample_id"])
        for row in predictions
    ]
    assert len(keys) == len(set(keys))
    assert all(0.0 <= summary["balanced_accuracy"] <= 1.0 for summary in result["summaries"])
    assert result["oof_validation"]["passed"] is True
    assert len(result["sensitivity_selections"][0]["sample_ids"]) == 15
    assert progress[0]["event"] == "run_started"
    assert progress[-1]["event"] == "run_completed"
    assert len([event for event in progress if event["event"] == "fit_completed"]) == 20


def test_oof_validator_rejects_duplicate_prediction() -> None:
    cache, splits = _inputs()
    result, predictions = run_e1a(
        cache,
        splits,
        bootstrap_samples=5,
        permutations=5,
        model_names=("dummy_most_frequent",),
        include_group_sensitivity=False,
    )
    predictions.append(dict(predictions[0]))
    with pytest.raises(ValueError, match="duplicate OOF prediction"):
        validate_oof_predictions(
            cache,
            splits,
            predictions,
            model_names=("dummy_most_frequent",),
            analysis_modes=("all_samples",),
        )
    assert result["oof_validation"]["prediction_count"] == 60


def test_retrained_group_permutation_control_refits_models() -> None:
    cache, splits = _inputs()
    result, _ = run_e1a(
        cache,
        splits,
        bootstrap_samples=5,
        permutations=5,
        retraining_permutations=2,
        include_group_sensitivity=False,
        model_names=("logistic_regression",),
    )
    controls = result["retrained_group_permutation_tests"]
    assert len(controls) == 2
    assert all(control["permutations"] == 2 for control in controls)
    assert all(0.0 < control["p_value"] <= 1.0 for control in controls)


def test_writer_records_input_hashes_and_completed_run(tmp_path) -> None:
    cache, splits = _inputs()
    cache_path = tmp_path / "features.json"
    splits_path = tmp_path / "splits.json"
    config_path = tmp_path / "config.yaml"
    manifest_path = tmp_path / "manifest.json"
    cache_path.write_text(json.dumps(cache), encoding="utf-8")
    splits_path.write_text(json.dumps(splits), encoding="utf-8")
    config_path.write_text("experiment_name: test\n", encoding="utf-8")
    manifest_path.write_text('{"samples": []}\n', encoding="utf-8")
    output = tmp_path / "run"
    write_e1a_results(
        cache_path,
        splits_path,
        output,
        config_path=config_path,
        manifest_path=manifest_path,
        bootstrap_samples=5,
        permutations=5,
        include_group_sensitivity=False,
        model_names=("dummy_most_frequent",),
    )
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert manifest["oof_validation"]["passed"] is True
    assert len(manifest["inputs"]["feature_cache"]["sha256"]) == 64
    assert (output / "config_used.yaml").is_file()
    assert (output / "inputs" / "manifest.json").is_file()
    assert (output / "inputs" / "splits.json").is_file()
    assert (output / "inputs" / "feature_cache.json").is_file()
