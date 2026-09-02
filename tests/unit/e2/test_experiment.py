from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from musicians_style.e2.experiment import (
    E2Config,
    E2Experiment,
    _analyse_results,
    _content_metrics,
    _holm_adjust,
    _onset_f1,
    _valid_completed_record,
    _validate_splits,
    _WorkerContext,
)
from musicians_style.ga.types import IDENTITY_GENOME
from musicians_style.midi.types import InternalRepr, NoteEvent
from musicians_style.config import GAConfig


def _repr_(scale: float = 1.0) -> InternalRepr:
    return InternalRepr(
        ticks_per_beat=480,
        notes=(
            NoteEvent(0, 0, 60, 80, int(240 * scale)),
            NoteEvent(int(480 * scale), 0, 64, 80, int(240 * scale)),
            NoteEvent(int(960 * scale), 0, 67, 80, int(240 * scale)),
        ),
        smf_format=1,
    )


def _e2_config(tmp_path: Path) -> E2Config:
    return E2Config(
        manifest_path=tmp_path / "manifest.json",
        splits_path=tmp_path / "splits.json",
        legacy_features_path=tmp_path / "legacy.json",
        composition_features_path=tmp_path / "composition.json",
        dataset_root=tmp_path,
        output_dir=tmp_path / "run",
        summary_path=tmp_path / "E2.md",
        ga=GAConfig(
            population_size=2,
            generations=1,
            tournament_size=1,
            crossover="uniform",
            mutation_sigma={
                "transpose_semitones": 0.1,
                "rhythm_density_factor": 0.01,
                "note_duration_factor": 0.01,
                "velocity_offset": 0.1,
            },
            elitism_k=1,
            fitness_metric="mahalanobis",
            stagnation_generations=2,
        ),
        bootstrap_samples=20,
        sign_permutations=20,
    )


def test_cli_runtime_values_override_e2_yaml_values(tmp_path: Path) -> None:
    config = _e2_config(tmp_path)
    override_dir = tmp_path / "override"
    experiment = E2Experiment(config, workers=4, run_dir=override_dir)
    assert experiment.workers == 4
    assert experiment.run_dir == override_dir.resolve()
    assert experiment.runtime_overrides == {"workers": True, "run_dir": True}


def test_content_metrics_identity_and_time_scaling() -> None:
    source = _repr_()
    identity = _content_metrics(source, source, source, 1 / 16)
    assert identity["length_error"] == 0.0
    assert identity["onset_f1"] == 1.0
    assert identity["melody_trigram_jaccard"] == 1.0
    assert identity["roundtrip_valid"] is True

    stretched = _repr_(2.0)
    assert _onset_f1(source, stretched, 1 / 16) < 1.0
    assert _content_metrics(source, stretched, stretched, 1 / 16)["length_ratio"] == 2.0


def test_holm_adjust_is_monotone_and_bounded() -> None:
    adjusted = _holm_adjust({"a": 0.001, "b": 0.02, "c": 0.5})
    assert adjusted["a"] <= adjusted["b"] <= adjusted["c"] <= 1.0


def test_completed_record_is_not_resumable_when_midi_is_missing(tmp_path: Path) -> None:
    from musicians_style.midi.printer import MidiPrettyPrinter

    output = tmp_path / "task" / "output.mid"
    MidiPrettyPrinter().write(_repr_(), output)
    genotype = {
        "transpose_semitones": 0.0,
        "rhythm_density_factor": 1.0,
        "note_duration_factor": 1.0,
        "velocity_offset": 0.0,
    }
    history = {
        "best_fitness": [0.0],
        "mean_fitness": [0.0],
        "worst_fitness": [0.0],
        "best_genome": [genotype],
        "stop_reason": "stagnation",
    }
    (tmp_path / "task" / "genotype.json").write_text(json.dumps(genotype), encoding="utf-8")
    (tmp_path / "task" / "history.json").write_text(json.dumps(history), encoding="utf-8")
    (tmp_path / "task" / "ga.jsonl").write_text("{}\n", encoding="utf-8")
    import hashlib

    task = {"task_id": "task", "task_signature": "sig", "phase": "full"}
    result = {
        "status": "completed",
        "task_id": "task",
        "task_signature": "sig",
        "phase": "full",
        "fold": 0,
        "source_id": "source",
        "source_composer": "Bach",
        "target_composer": "Beethoven",
        "seed": 1729,
        "output_path": "task/output.mid",
        "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "content": {
            "roundtrip_valid": True,
            "ticks_per_beat_preserved": True,
            "smf_format_preserved": True,
            "meta_preserved": True,
        },
        "history": {
            "best_fitness": [0.0],
            "mean_fitness": [0.0],
            "worst_fitness": [0.0],
            "best_genome": [{
                "transpose_semitones": 0.0,
                "rhythm_density_factor": 1.0,
                "note_duration_factor": 1.0,
                "velocity_offset": 0.0,
            }],
            "stop_reason": "stagnation",
        },
        "p_target_input": 0.0,
        "p_target_output": 0.0,
        "delta_p_target": 0.0,
        "p_source_input": 0.0,
        "p_source_output": 0.0,
        "identity_fitness": 0.0,
        "generation0_fitness": 0.0,
        "ga_final_fitness": 0.0,
        "fitness_gain_vs_identity": 0.0,
        "best_genome": {
            "transpose_semitones": 0.0,
            "rhythm_density_factor": 1.0,
            "note_duration_factor": 1.0,
            "velocity_offset": 0.0,
        },
        "style_distances_e1b": {},
        "stop_reason": "stagnation",
        "num_generations": 1,
        "output_nonempty": True,
        "genotype_path": "task/genotype.json",
        "history_path": "task/history.json",
    }
    assert _valid_completed_record(tmp_path, task, result)
    output.unlink()
    assert not _valid_completed_record(tmp_path, task, result)


def test_report_rejects_incomplete_run(tmp_path: Path) -> None:
    experiment = E2Experiment(_e2_config(tmp_path))
    try:
        experiment.write_report()
    except ValueError as exc:
        assert "run manifest" in str(exc)
    else:  # pragma: no cover - assertion clarity
        raise AssertionError("incomplete report was accepted")


def test_validate_splits_rejects_group_leakage() -> None:
    manifest = {
        "samples": [
            {"sample_id": "a", "group_id": "g", "validation_status": "accepted"},
            {"sample_id": "b", "group_id": "g", "validation_status": "accepted"},
        ]
    }
    splits = {
        "repetitions": [
            {
                "repeat": 0,
                "folds": [
                    {"fold": 0, "train": {"sample_ids": ["a"]}, "test": {"sample_ids": ["b"]}},
                ],
            }
        ]
    }
    try:
        _validate_splits(manifest, splits, 0)
    except ValueError as exc:
        assert "group leakage" in str(exc)
    else:  # pragma: no cover - assertion clarity
        raise AssertionError("group leakage was not rejected")


def test_prepare_builds_exactly_300_full_and_18_pilot_tasks(tmp_path: Path) -> None:
    rows = []
    for composer in ("Bach", "Beethoven", "Chopin"):
        for index in range(50):
            sample_id = f"{composer.lower()}-{index:03d}"
            rows.append(
                {
                    "sample_id": sample_id,
                    "composer": composer,
                    "title": f"{composer}_{index}",
                    "score_path": f"{sample_id}.mid",
                    "group_id": f"{composer.lower()}-group-{index}",
                    "sha256": f"{index:064x}"[-64:],
                    "note_count": index + 1,
                    "validation_status": "accepted",
                }
            )
    manifest = {"samples": rows, "dataset_root": str(tmp_path)}
    cache_rows = [
        {
            "sample_id": row["sample_id"],
            "composer": row["composer"],
            "group_id": row["group_id"],
            "sha256": row["sha256"],
            "values": [0.0] * 42,
        }
        for row in rows
    ]
    composition_rows = [dict(row, values=[0.0] * 93) for row in cache_rows]
    folds = []
    for fold in range(5):
        test = [row["sample_id"] for row in rows if int(row["sample_id"].rsplit("-", 1)[1]) % 5 == fold]
        train = [row["sample_id"] for row in rows if row["sample_id"] not in test]
        folds.append({"fold": fold, "train": {"sample_ids": train}, "test": {"sample_ids": test}})
    splits = {"repetitions": [{"repeat": 0, "folds": folds}]}
    config = _e2_config(tmp_path)
    config.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    config.splits_path.write_text(json.dumps(splits), encoding="utf-8")
    config.legacy_features_path.write_text(json.dumps({"features_schema_version": "e1.2.0", "samples": cache_rows}), encoding="utf-8")
    config.composition_features_path.write_text(json.dumps({"features_schema_version": "e1.3.0", "samples": composition_rows}), encoding="utf-8")
    experiment = E2Experiment(config)
    experiment._load_inputs = lambda: (manifest, splits, {"features_schema_version": "e1.2.0", "samples": cache_rows}, {"features_schema_version": "e1.3.0", "samples": composition_rows})  # type: ignore[method-assign]
    task_path = experiment.prepare()
    payload = json.loads(task_path.read_text(encoding="utf-8"))
    assert payload["counts"] == {"pilot": 18, "full": 300, "total": 318}
    full = [task for task in payload["tasks"] if task["phase"] == "full"]
    assert len({task["task_id"] for task in full}) == 300
    assert all(task["source_id"] not in task["target_train_ids"] for task in full)
    assert all(task["target_train_ids"] for task in full)
    assert {task["source_composer"] for task in full} == {"Bach", "Beethoven", "Chopin"}


def test_analysis_produces_main_and_directional_statistics(tmp_path: Path) -> None:
    results = []
    for index in range(12):
        source = "Bach" if index % 2 else "Chopin"
        target = "Beethoven" if source == "Bach" else "Bach"
        results.append(
            {
                "task_id": str(index),
                "source_group_id": f"{source}-group-{index}",
                "source_composer": source,
                "target_composer": target,
                "delta_p_target": 0.1 if index % 3 else -0.02,
                "p_target_input": 0.2,
                "p_target_output": 0.3,
                "fitness_gain_vs_identity": 1.0,
                "stop_reason": "stagnation",
                "num_generations": 3,
                "best_genome": {
                    "transpose_semitones": 0.0,
                    "rhythm_density_factor": 1.0,
                    "note_duration_factor": 1.0,
                    "velocity_offset": 0.0,
                },
                "style_distances_e1b": {group: {"gain": 0.1} for group in ("pitch", "melody", "rhythm", "texture", "harmony", "structure")},
                "content": {
                    "length_error": 0.0,
                    "onset_f1": 1.0,
                    "melody_trigram_jaccard": 1.0,
                    "note_count_ratio": 1.0,
                    "roundtrip_valid": True,
                },
            }
        )
    analysis = _analyse_results(results, _e2_config(tmp_path))
    assert analysis["task_count"] == 12
    assert analysis["delta_p_target"]["ci95_clustered"]
    assert set(analysis["directions"]) == {"Bach→Beethoven", "Chopin→Bach"}


def test_worker_executes_one_task_and_writes_roundtrip_artifacts(tmp_path: Path) -> None:
    from musicians_style.midi.printer import MidiPrettyPrinter

    source_repr = _repr_()
    samples = []
    legacy_rows = []
    composition_rows = []
    for composer_index, composer in enumerate(("Bach", "Beethoven", "Chopin")):
        for index in range(3):
            sample_id = f"{composer.lower()}-{index}"
            midi_path = tmp_path / f"{sample_id}.mid"
            MidiPrettyPrinter().write(source_repr, midi_path)
            samples.append(
                {
                    "sample_id": sample_id,
                    "composer": composer,
                    "score_path": midi_path.name,
                    "group_id": f"{composer.lower()}-{index}",
                    "validation_status": "accepted",
                }
            )
            legacy = [0.0] * 42
            legacy[1] = float(composer_index * 4 + index)
            legacy_rows.append({**samples[-1], "sha256": "x", "values": legacy})
            composition = [0.0] * 93
            composition[0] = float(composer_index * 3 + index)
            composition_rows.append({**samples[-1], "sha256": "x", "values": composition})
    manifest_path = tmp_path / "manifest.json"
    splits_path = tmp_path / "splits.json"
    legacy_path = tmp_path / "legacy.json"
    composition_path = tmp_path / "composition.json"
    manifest_path.write_text(json.dumps({"samples": samples}), encoding="utf-8")
    splits_path.write_text(
        json.dumps({"repetitions": [{"repeat": 0, "folds": [{"fold": 0, "train": {"sample_ids": [samples[index]["sample_id"] for index in (0, 1, 3, 4, 6, 7)]}, "test": {"sample_ids": [samples[index]["sample_id"] for index in (2, 5, 8)]}}]}]}),
        encoding="utf-8",
    )
    legacy_path.write_text(json.dumps({"samples": legacy_rows}), encoding="utf-8")
    composition_path.write_text(json.dumps({"samples": composition_rows}), encoding="utf-8")
    config = _e2_config(tmp_path)
    payload = {
        "manifest_path": str(manifest_path),
        "splits_path": str(splits_path),
        "legacy_features_path": str(legacy_path),
        "composition_features_path": str(composition_path),
        "dataset_root": str(tmp_path),
        "run_dir": str(tmp_path / "run"),
        "ga": asdict(config.ga),
        "evaluator_n_estimators": 5,
        "evaluator_max_features": 0.5,
        "evaluator_min_samples_leaf": 1,
        "onset_tolerance_beats": 1 / 16,
        "config_hash": "test",
        "repeat": 0,
    }
    context = _WorkerContext.from_payload(payload)
    task = {
        "task_id": "task-one",
        "task_signature": "signature",
        "phase": "pilot",
        "fold": 0,
            "source_id": "chopin-2",
        "source_composer": "Chopin",
        "target_composer": "Bach",
        "target_train_ids": ["bach-0", "bach-1"],
        "seed": 1729,
    }
    result = context.execute(task)
    assert result["status"] == "completed", result
    assert result["determinism"] == {
        "checked": True,
        "genome_equal": True,
        "history_equal": True,
        "output_equal": True,
    }
    assert (tmp_path / "run" / "tasks" / "task-one" / "output.mid").is_file()
    assert (tmp_path / "run" / "tasks" / "task-one" / "genotype.json").is_file()
    assert (tmp_path / "run" / "tasks" / "task-one" / "history.json").is_file()
    assert result["content"]["roundtrip_valid"] is True
