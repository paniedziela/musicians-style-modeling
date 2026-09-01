from __future__ import annotations

from musicians_style.e1.classification import run_e1a
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
                    "group_id": f"{composer.lower()}-group-{number}",
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
    assert len(predictions) == 2 * 30
    assert len(result["summaries"]) == 2
    keys = [(row["variant"], row["repeat"], row["sample_id"]) for row in predictions]
    assert len(keys) == len(set(keys))
    assert all(0.0 <= summary["balanced_accuracy"] <= 1.0 for summary in result["summaries"])
    assert progress[0]["event"] == "run_started"
    assert progress[-1]["event"] == "run_completed"
    assert len([event for event in progress if event["event"] == "fit_completed"]) == 10
