from __future__ import annotations

import pytest
import json

from musicians_style.e1.reporting import write_e1_closure_report

pytestmark = pytest.mark.regression


def _write_json(path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_e14_writes_decision_tables_plots_and_provenance(tmp_path) -> None:
    run = tmp_path / "run"
    _write_json(run / "run_manifest.json", {"status": "completed"})
    summary = {
        "analysis": "all_samples",
        "variant": "composition_full",
        "model": "logistic_regression",
        "balanced_accuracy": 0.8,
        "macro_f1": 0.79,
        "balanced_accuracy_95_ci_clustered": [0.7, 0.9],
        "confusion_matrix": [[8, 1, 1], [1, 8, 1], [0, 1, 9]],
        "labels": ["Bach", "Beethoven", "Chopin"],
    }
    results = {
        "results_schema_version": "e1.3.0",
        "feature_schema_version": "e1.3.0",
        "protocol": {"seeds": [7]},
        "provenance": {"code_commit": "abc"},
        "summaries": [summary],
        "fold_results": [
            {
                "analysis": "all_samples",
                "variant": "composition_full",
                "model": "logistic_regression",
                "balanced_accuracy": 0.8,
            }
        ],
        "retrained_group_permutation_tests": [
            {
                "variant": "composition_full",
                "model": "logistic_regression",
                "p_value": 0.01,
            }
        ],
        "permutation_importance": {
            "by_feature": [
                {
                    "model": "logistic_regression",
                    "feature": "pitch_range",
                    "mean_across_folds": 0.1,
                }
            ]
        },
    }
    _write_json(run / "e1b_results.json", results)
    _write_json(
        run / "e1b_predictions.json",
        [{"sample_id": "bach-1", "predicted_composer": "Bach"}],
    )
    _write_json(
        run / "inputs" / "manifest.json",
        {
            "dataset_fingerprints": {"metadata.csv": "123"},
            "samples": [
                {
                    "sample_id": "bad",
                    "composer": "Bach",
                    "title": "Bad",
                    "validation_status": "excluded",
                    "exclusion_reason": "invalid",
                }
            ],
        },
    )
    report_path, closure = write_e1_closure_report(run, tmp_path / "report")
    assert closure["success_criterion"]["passed"] is True
    assert report_path.is_file()
    assert (tmp_path / "report" / "predictions.csv").is_file()
    assert (tmp_path / "report" / "exclusions.csv").is_file()
    assert (tmp_path / "report" / "plots" / "confusion_matrix.png").is_file()
    assert (tmp_path / "report" / "plots" / "ablations.png").is_file()
    assert (tmp_path / "report" / "plots" / "feature_importance.png").is_file()
    assert "GO" in report_path.read_text(encoding="utf-8")
