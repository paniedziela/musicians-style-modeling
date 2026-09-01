from __future__ import annotations

from musicians_style.e1.open_set import run_e1_open
from musicians_style.e1.splits import build_e1_splits


def _inputs() -> tuple[dict, dict]:
    known = []
    all_rows = []
    composers = ("Bach", "Beethoven", "Chopin")
    for composer_index, composer in enumerate(composers):
        for number in range(10):
            row = {
                "sample_id": f"{composer.lower()}-{number}",
                "composer": composer,
                "title": f"Sonata_{number}",
                "form": "sonata",
                "group_id": f"{composer.lower()}-group-{number // 2}",
                "sha256": f"{composer}-{number}",
                "validation_status": "accepted",
                "values": [float(composer_index), float(number % 3), float(number)],
            }
            known.append(row)
            all_rows.append(row)
    for composer, offset in (("Haydn", 8.0), ("Liszt", -8.0)):
        for number in range(6):
            all_rows.append(
                {
                    "sample_id": f"{composer.lower()}-{number}",
                    "composer": composer,
                    "title": f"Work_{number}",
                    "form": "other",
                    "group_id": f"{composer.lower()}-{number}",
                    "sha256": f"{composer}-{number}",
                    "validation_status": "accepted",
                    "values": [offset, float(number % 2), float(number)],
                }
            )
    cache = {
        "features_schema_version": "e1.3.0",
        "variants": {
            "composition_full": {
                "indices": [0, 1, 2],
                "feature_names": ["pitch", "rhythm", "structure"],
            }
        },
        "samples": all_rows,
    }
    split_manifest = {"manifest_schema_version": "test", "samples": known}
    splits = build_e1_splits(
        split_manifest, seeds=(7,), outer_splits=5, inner_splits=3
    )
    return cache, splits


def test_open_set_keeps_calibration_and_test_composers_separate() -> None:
    cache, splits = _inputs()
    result, predictions = run_e1_open(
        cache,
        splits,
        known_composers=("Bach", "Beethoven", "Chopin"),
        calibration_composers=("Haydn",),
        test_composers=("Liszt",),
        model_names=("logistic_regression",),
        known_calibration_folds=(0, 1),
        target_known_tpr=0.8,
        include_rotations=True,
    )
    assert len(result["summaries"]) == 4
    external = next(
        row for row in result["summaries"] if row["scenario"] == "external_unknowns"
    )
    assert external["calibration_unknown_composers"] == ["Haydn"]
    assert external["test_unknown_composers"] == ["Liszt"]
    assert 0.0 <= external["test_metrics"]["auroc_known_vs_unknown"] <= 1.0
    assert 0.0 <= external["test_metrics"]["unknown_recall_at_threshold"] <= 1.0
    calibration_rows = [
        row
        for row in predictions
        if row["scenario"] == "external_unknowns"
        and row["source"] == "unknown_calibration"
    ]
    test_rows = [
        row
        for row in predictions
        if row["scenario"] == "external_unknowns"
        and row["source"] == "unknown_test"
    ]
    assert {row["composer"] for row in calibration_rows} == {"Haydn"}
    assert {row["composer"] for row in test_rows} == {"Liszt"}
    assert all("open_prediction" in row for row in predictions)
