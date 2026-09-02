"""Runner, metrics and reporting for experiment E2.

E2 intentionally keeps the historical four-gene genetic algorithm unchanged.
This module supplies the experiment protocol around it: fold-local target
profiles, a fold-local E1b evaluator, resumable task execution and a compact
statistical report.  It is deliberately independent from the older
single-manifest inference API, whose manifest schema is not the ASAP E1 schema.
"""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import shutil
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import yaml
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from ..config import GAConfig
from ..e1.composition_features import (
    COMPOSITION_FEATURES_SCHEMA_VERSION,
    FEATURE_GROUPS,
    FEATURE_SPECS,
    extract_composition_features,
)
from ..e1.features import FEATURES_FILENAME, FEATURES_SCHEMA_VERSION
from ..evaluation.distance import prepare_mahalanobis
from ..features.types import AggregatedFeatures, FeatureVector
from ..ga.algorithm import GeneticAlgorithm, WORKING_RANGES
from ..ga.fitness import fitness
from ..ga.transformation import apply_transformation
from ..ga.types import IDENTITY_GENOME, Genome
from ..midi.parser import MidiParser
from ..midi.printer import MidiPrettyPrinter
from ..midi.types import InternalRepr

E2_SCHEMA_VERSION = "e2.0.0"
TASK_MANIFEST_FILENAME = "task_manifest.json"
RUN_MANIFEST_FILENAME = "run_manifest.json"
RESULTS_FILENAME = "results.json"
PILOT_RESULTS_FILENAME = "pilot_results.json"
RESULTS_CSV_FILENAME = "results.csv"
PROGRESS_FILENAME = "progress.jsonl"

_DEFAULT_GA = {
    "population_size": 100,
    "generations": 200,
    "tournament_size": 3,
    "crossover": "uniform",
    "mutation_sigma": {
        "transpose_semitones": 1.5,
        "rhythm_density_factor": 0.1,
        "note_duration_factor": 0.1,
        "velocity_offset": 5.0,
    },
    "elitism_k": 2,
    "fitness_metric": "mahalanobis",
    "stagnation_generations": 30,
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _assert_finite_payload(value: Any, path: str = "payload") -> None:
    """Reject NaN/inf before a result can be accepted or serialized."""
    if isinstance(value, (float, np.floating)) and not np.isfinite(float(value)):
        raise ValueError(f"{path} contains a non-finite number")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _assert_finite_payload(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_finite_payload(item, f"{path}[{index}]")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=_json_default)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    temporary.replace(path)


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _resolve(value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else (Path.cwd() / path).resolve()


def _as_int_tuple(value: Any, default: Sequence[int]) -> tuple[int, ...]:
    raw = default if value is None else value
    if not isinstance(raw, (list, tuple)) or not raw:
        raise ValueError("seed list must be a non-empty sequence")
    result = tuple(int(item) for item in raw)
    if len(set(result)) != len(result):
        raise ValueError("seed list must not contain duplicates")
    return result


@dataclass(frozen=True)
class E2Config:
    """Validated, versioned configuration for E2."""

    manifest_path: Path
    splits_path: Path
    legacy_features_path: Path
    composition_features_path: Path
    dataset_root: Path
    output_dir: Path
    summary_path: Path
    repeat: int = 0
    main_seed: int = 1729
    pilot_seeds: tuple[int, ...] = (1729, 2718, 3141)
    pilot_fold: int = 0
    workers: int = 1
    ga: GAConfig = field(default_factory=lambda: GAConfig.from_dict(_DEFAULT_GA))
    evaluator_n_estimators: int = 300
    evaluator_max_features: float = 0.5
    evaluator_min_samples_leaf: int = 2
    bootstrap_samples: int = 2000
    sign_permutations: int = 999
    alpha: float = 0.05
    onset_tolerance_beats: float = 1.0 / 16.0
    expected_dataset_fingerprints: dict[str, str] = field(default_factory=dict)
    schema_version: str = E2_SCHEMA_VERSION
    raw: dict[str, Any] = field(default_factory=dict, compare=False, repr=False)


def load_e2_config(path: Path | str) -> E2Config:
    """Load and validate the explicit E2 YAML contract."""
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("E2 configuration must be a YAML mapping")
    inputs = raw.get("inputs", {})
    if not isinstance(inputs, Mapping):
        raise ValueError("inputs must be a mapping")
    output = raw.get("output", {})
    if not isinstance(output, Mapping):
        raise ValueError("output must be a mapping")
    ga_raw = raw.get("ga", _DEFAULT_GA)
    if not isinstance(ga_raw, Mapping):
        raise ValueError("ga must be a mapping")
    ga_values = dict(_DEFAULT_GA)
    ga_values.update({key: value for key, value in ga_raw.items()})
    if "mutation_sigma" not in ga_values:
        ga_values["mutation_sigma"] = dict(_DEFAULT_GA["mutation_sigma"])
    else:
        ga_values["mutation_sigma"] = dict(ga_values["mutation_sigma"])
    ga = GAConfig.from_dict(ga_values)
    evaluator = raw.get("evaluator", {})
    if not isinstance(evaluator, Mapping):
        raise ValueError("evaluator must be a mapping")
    pilot = raw.get("pilot", {})
    if not isinstance(pilot, Mapping):
        raise ValueError("pilot must be a mapping")
    manifest_path = _resolve(inputs.get("manifest", "datasets/derived/e1_asap/manifest.json"))
    splits_path = _resolve(inputs.get("splits", "datasets/derived/e1_asap/splits.json"))
    legacy_path = _resolve(inputs.get("legacy_features", manifest_path.with_name(FEATURES_FILENAME)))
    composition_path = _resolve(inputs.get("composition_features", manifest_path.with_name("composition_features.json")))
    dataset_root = _resolve(inputs.get("dataset_root", "datasets/asap-dataset-1.2"))
    output_dir = _resolve(output.get("run_dir", "experiments/e2_asap"))
    summary_path = _resolve(output.get("summary", "docs/results/E2.md"))
    expected = inputs.get("expected_dataset_fingerprints", {})
    if not isinstance(expected, Mapping):
        raise ValueError("expected_dataset_fingerprints must be a mapping")
    workers = int(raw.get("workers", 1))
    if workers < 1:
        raise ValueError("workers must be >= 1")
    repeat = int(raw.get("repeat", 0))
    main_seed = int(raw.get("main_seed", 1729))
    pilot_seeds = _as_int_tuple(pilot.get("seeds"), (1729, 2718, 3141))
    pilot_fold = int(pilot.get("fold", 0))
    n_estimators = int(evaluator.get("n_estimators", 300))
    max_features = float(evaluator.get("max_features", 0.5))
    min_leaf = int(evaluator.get("min_samples_leaf", 2))
    bootstrap_samples = int(evaluator.get("bootstrap_samples", 2000))
    sign_permutations = int(evaluator.get("sign_permutations", 999))
    alpha = float(evaluator.get("alpha", 0.05))
    tolerance = float(evaluator.get("onset_tolerance_beats", 1.0 / 16.0))
    if n_estimators < 1 or min_leaf < 1 or bootstrap_samples < 1 or sign_permutations < 1:
        raise ValueError("evaluator counts must be positive")
    if not 0.0 < max_features <= 1.0 or not 0.0 < alpha < 1.0 or tolerance <= 0.0:
        raise ValueError("invalid evaluator range")
    config = E2Config(
        manifest_path=manifest_path,
        splits_path=splits_path,
        legacy_features_path=legacy_path,
        composition_features_path=composition_path,
        dataset_root=dataset_root,
        output_dir=output_dir,
        summary_path=summary_path,
        repeat=repeat,
        main_seed=main_seed,
        pilot_seeds=pilot_seeds,
        pilot_fold=pilot_fold,
        workers=workers,
        ga=ga,
        evaluator_n_estimators=n_estimators,
        evaluator_max_features=max_features,
        evaluator_min_samples_leaf=min_leaf,
        bootstrap_samples=bootstrap_samples,
        sign_permutations=sign_permutations,
        alpha=alpha,
        onset_tolerance_beats=tolerance,
        expected_dataset_fingerprints={str(k): str(v).lower() for k, v in expected.items()},
        schema_version=str(raw.get("schema_version", E2_SCHEMA_VERSION)),
        raw=dict(raw),
    )
    if config.schema_version != E2_SCHEMA_VERSION:
        raise ValueError(f"unsupported E2 schema version: {config.schema_version}")
    return config


def _genome_dict(genome: Genome) -> dict[str, float]:
    return {
        "transpose_semitones": float(genome.transpose_semitones),
        "rhythm_density_factor": float(genome.rhythm_density_factor),
        "note_duration_factor": float(genome.note_duration_factor),
        "velocity_offset": float(genome.velocity_offset),
    }


def _genome_from_dict(data: Mapping[str, Any]) -> Genome:
    return Genome(
        transpose_semitones=float(data["transpose_semitones"]),
        rhythm_density_factor=float(data["rhythm_density_factor"]),
        note_duration_factor=float(data["note_duration_factor"]),
        velocity_offset=float(data["velocity_offset"]),
    )


def _feature_indices_by_group() -> dict[str, np.ndarray]:
    return {
        group: np.asarray(
            [index for index, specification in enumerate(FEATURE_SPECS) if specification["group"] == group],
            dtype=int,
        )
        for group in FEATURE_GROUPS
    }


def _validate_rows(manifest: Mapping[str, Any], legacy: Mapping[str, Any], composition: Mapping[str, Any]) -> None:
    samples = [row for row in manifest.get("samples", []) if row.get("validation_status") == "accepted"]
    if len(samples) != 150:
        raise ValueError(f"E2 expects 150 accepted E1 samples, found {len(samples)}")
    ids_list = [str(row["sample_id"]) for row in samples]
    if len(set(ids_list)) != len(ids_list):
        raise ValueError("E1 manifest contains duplicate accepted sample IDs")
    ids = set(ids_list)
    for cache_name, cache, expected_schema, value_length in (
        ("legacy", legacy, FEATURES_SCHEMA_VERSION, 42),
        ("composition", composition, COMPOSITION_FEATURES_SCHEMA_VERSION, 93),
    ):
        if cache.get("features_schema_version") != expected_schema:
            raise ValueError(f"{cache_name} cache has an incompatible schema")
        rows = cache.get("samples", [])
        row_ids = [str(row["sample_id"]) for row in rows]
        if len(rows) != len(ids) or len(set(row_ids)) != len(row_ids) or set(row_ids) != ids:
            raise ValueError(f"{cache_name} cache sample IDs do not match manifest")
        for row in rows:
            values = np.asarray(row.get("values"), dtype=float)
            if values.shape != (value_length,) or not np.isfinite(values).all():
                raise ValueError(f"invalid {cache_name} values for {row.get('sample_id')}")
            manifest_row = next(item for item in samples if item["sample_id"] == row["sample_id"])
            if (
                row.get("sha256") != manifest_row.get("sha256")
                or row.get("composer") != manifest_row.get("composer")
                or row.get("group_id") != manifest_row.get("group_id")
            ):
                raise ValueError(f"{cache_name} metadata/SHA mismatch for {row['sample_id']}")


def _validate_splits(manifest: Mapping[str, Any], splits: Mapping[str, Any], repeat: int) -> dict[str, Any]:
    accepted = {row["sample_id"]: row for row in manifest.get("samples", []) if row.get("validation_status") == "accepted"}
    repetitions = splits.get("repetitions", [])
    repetition = next((item for item in repetitions if int(item["repeat"]) == repeat), None)
    if repetition is None:
        raise ValueError(f"split repetition {repeat} is not present")
    seen_test: set[str] = set()
    for fold in repetition.get("folds", []):
        train = set(fold["train"]["sample_ids"])
        test = set(fold["test"]["sample_ids"])
        if train & test:
            raise ValueError(f"fold {fold['fold']} has train/test sample leakage")
        if not train <= set(accepted) or not test <= set(accepted):
            raise ValueError(f"fold {fold['fold']} references an unknown sample")
        if seen_test & test:
            raise ValueError("test samples occur in multiple folds")
        seen_test |= test
        train_groups = {accepted[item]["group_id"] for item in train}
        test_groups = {accepted[item]["group_id"] for item in test}
        if train_groups & test_groups:
            raise ValueError(f"fold {fold['fold']} has group leakage")
    if seen_test != set(accepted):
        raise ValueError("E2 repeat does not assign every accepted sample to one test fold")
    return repetition


def _task_signature(task: Mapping[str, Any], config_hash: str) -> str:
    return hashlib.sha256(_canonical_json({"task": task, "config_hash": config_hash}).encode("utf-8")).hexdigest()


def _task_id(phase: str, repeat: int, fold: int, source_id: str, target: str, seed: int) -> str:
    return f"{phase}__r{repeat:02d}__f{fold:02d}__{source_id}__to__{target}__s{seed}"


def _valid_completed_record(run_dir: Path, task: Mapping[str, Any], result: Mapping[str, Any]) -> bool:
    """Return whether a task record and its output artifact are resumable."""
    try:
        _assert_finite_payload(result, f"task {task.get('task_id')}")
    except ValueError:
        return False
    if (
        result.get("status") != "completed"
        or result.get("task_id") != task.get("task_id")
        or result.get("task_signature") != task.get("task_signature")
    ):
        return False
    for field_name in ("phase", "fold", "source_id", "source_composer", "target_composer", "seed"):
        if field_name in task and result.get(field_name) != task.get(field_name):
            return False
    relative = result.get("output_path")
    if not relative:
        return False
    output = (run_dir / str(relative)).resolve()
    try:
        output.relative_to(run_dir.resolve())
    except ValueError:
        return False
    if not output.is_file() or output.stat().st_size <= 0:
        return False
    ga_log = output.parent / "ga.jsonl"
    if not ga_log.is_file() or ga_log.stat().st_size <= 0:
        return False
    expected_sha = result.get("output_sha256")
    if not expected_sha or expected_sha != _sha256(output):
        return False
    if not isinstance(result.get("content"), Mapping) or not isinstance(result.get("history"), Mapping):
        return False
    content = result["content"]
    if not all(
        bool(content.get(name))
        for name in ("roundtrip_valid", "ticks_per_beat_preserved", "smf_format_preserved", "meta_preserved")
    ) or not bool(result.get("output_nonempty")):
        return False
    required_fields = (
        "p_target_input", "p_target_output", "delta_p_target",
        "p_source_input", "p_source_output", "identity_fitness",
        "generation0_fitness", "ga_final_fitness", "fitness_gain_vs_identity",
        "best_genome", "style_distances_e1b", "stop_reason", "num_generations",
    )
    if any(field_name not in result for field_name in required_fields):
        return False
    genome = result.get("best_genome")
    gene_names = {
        "transpose_semitones",
        "rhythm_density_factor",
        "note_duration_factor",
        "velocity_offset",
    }
    if not isinstance(genome, Mapping) or set(genome) != gene_names:
        return False
    history = result["history"]
    history_fitness = [history.get(name) for name in ("best_fitness", "mean_fitness", "worst_fitness")]
    history_genomes = history.get("best_genome")
    try:
        recorded_generations = int(result.get("num_generations", -1))
    except (TypeError, ValueError):
        return False
    if (
        not all(isinstance(values, list) and values for values in history_fitness)
        or not isinstance(history_genomes, list)
        or not history_genomes
        or len({len(values) for values in history_fitness + [history_genomes]}) != 1
        or history.get("stop_reason") != result.get("stop_reason")
        or recorded_generations != len(history_fitness[0])
    ):
        return False
    for artifact_field in ("genotype_path", "history_path"):
        artifact_value = result.get(artifact_field)
        if not artifact_value:
            return False
        artifact_path = (run_dir / str(artifact_value)).resolve()
        try:
            artifact_path.relative_to(run_dir.resolve())
        except ValueError:
            return False
        if not artifact_path.is_file() or artifact_path.stat().st_size <= 0:
            return False
        try:
            artifact_payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        expected_payload = genome if artifact_field == "genotype_path" else history
        if artifact_payload != expected_payload:
            return False
    try:
        ga_lines = [json.loads(line) for line in ga_log.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError):
        return False
    if not ga_lines:
        return False
    # Pilot records must carry evidence of the same-seed technical repetition;
    # older/incomplete records are deliberately rerun after an interrupted
    # implementation upgrade.
    if str(task.get("phase")) == "pilot":
        deterministic = result.get("determinism")
        if not isinstance(deterministic, Mapping) or not all(
            bool(deterministic.get(name))
            for name in ("checked", "genome_equal", "history_equal", "output_equal")
        ):
            return False
    return True


class E2Experiment:
    """Prepare, execute and report the staged E2 protocol."""

    def __init__(
        self,
        config: E2Config,
        *,
        run_dir: Path | str | None = None,
        workers: int | None = None,
    ) -> None:
        self.config = config
        self.run_dir = Path(run_dir).resolve() if run_dir is not None else config.output_dir
        self.workers = int(config.workers if workers is None else workers)
        if self.workers < 1:
            raise ValueError("workers must be >= 1")
        # YAML is the source of truth for the production configuration.  Unit
        # tests and library callers may construct ``E2Config`` directly,
        # however; in that case hash the resolved dataclass rather than the
        # empty default ``raw`` mapping so two different configurations cannot
        # accidentally share a task manifest.
        hash_payload: Any = config.raw
        if not hash_payload:
            hash_payload = {
                key: value
                for key, value in asdict(config).items()
                if key != "raw"
            }
        self._config_hash = hashlib.sha256(_canonical_json(hash_payload).encode("utf-8")).hexdigest()

    @property
    def task_manifest_path(self) -> Path:
        return self.run_dir / TASK_MANIFEST_FILENAME

    @property
    def run_manifest_path(self) -> Path:
        return self.run_dir / RUN_MANIFEST_FILENAME

    def _load_inputs(self) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
        paths = (self.config.manifest_path, self.config.splits_path, self.config.legacy_features_path, self.config.composition_features_path)
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(f"E2 input does not exist: {path}")
        manifest = json.loads(self.config.manifest_path.read_text(encoding="utf-8"))
        splits = json.loads(self.config.splits_path.read_text(encoding="utf-8"))
        legacy = json.loads(self.config.legacy_features_path.read_text(encoding="utf-8"))
        composition = json.loads(self.config.composition_features_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict) or not isinstance(splits, dict) or not isinstance(legacy, dict) or not isinstance(composition, dict):
            raise ValueError("E2 inputs must all contain JSON objects")
        _validate_rows(manifest, legacy, composition)
        repetition = _validate_splits(manifest, splits, self.config.repeat)
        folds = repetition.get("folds", [])
        if len(folds) != 5 or sum(len(fold.get("test", {}).get("sample_ids", ())) for fold in folds) != 150:
            raise ValueError("E2 repeat must contain five folds covering 150 test samples")
        if self.config.pilot_fold < 0 or self.config.pilot_fold >= len(repetition.get("folds", [])):
            raise ValueError("pilot fold is outside the selected repetition")
        root_from_manifest = _resolve(manifest.get("dataset_root", self.config.dataset_root))
        if root_from_manifest != self.config.dataset_root:
            # A config may explicitly override the location of an otherwise
            # identical E1 snapshot; no silent fallback is allowed.
            if not self.config.dataset_root.is_dir():
                raise ValueError("configured dataset_root does not exist")
        if not self.config.dataset_root.is_dir():
            raise FileNotFoundError(f"dataset_root does not exist: {self.config.dataset_root}")
        expected = self.config.expected_dataset_fingerprints
        actual = manifest.get("dataset_fingerprints", {})
        if not isinstance(actual, Mapping):
            raise ValueError("E1 manifest dataset_fingerprints must be a mapping")
        if expected:
            for name, expected_hash in expected.items():
                if actual.get(name) != expected_hash:
                    raise ValueError(f"E1 dataset fingerprint mismatch in manifest for {name}")
                source = self.config.dataset_root / name
                if not source.is_file() or _sha256(source) != expected_hash:
                    raise ValueError(f"E1 dataset fingerprint mismatch on disk for {name}")
        # Verify every accepted source file before task construction/stage
        # start. This is intentionally done once per stage, never in every
        # worker task.
        for row in manifest.get("samples", []):
            if row.get("validation_status") != "accepted":
                continue
            path = (self.config.dataset_root / str(row["score_path"])).resolve()
            try:
                path.relative_to(self.config.dataset_root.resolve())
            except ValueError as exc:
                raise ValueError(f"accepted E1 file escapes dataset_root: {row['sample_id']}") from exc
            if not path.is_file():
                raise FileNotFoundError(f"accepted E1 file is missing: {path}")
            if _sha256(path) != row.get("sha256"):
                raise ValueError(f"accepted E1 SHA mismatch for {row['sample_id']}")
        return manifest, splits, legacy, composition  # type: ignore[return-value]

    def _existing_task_manifest(self) -> dict[str, Any] | None:
        if not self.task_manifest_path.is_file():
            return None
        value = json.loads(self.task_manifest_path.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or value.get("schema_version") != E2_SCHEMA_VERSION:
            raise ValueError("existing task manifest has an incompatible E2 schema")
        if value.get("config_hash") != self._config_hash:
            raise ValueError("existing task manifest belongs to a different configuration")
        return value

    def _validate_task_manifest_payload(self, payload: Mapping[str, Any]) -> None:
        """Validate counts, IDs and signatures before allowing a resume."""
        snapshots = payload.get("inputs", {}).get("snapshots", {}) if isinstance(payload.get("inputs"), Mapping) else {}
        if snapshots:
            if not isinstance(snapshots, Mapping):
                raise ValueError("E2 task manifest snapshots are malformed")
            for name, record in snapshots.items():
                if not isinstance(record, Mapping) or not record.get("path") or not record.get("sha256"):
                    raise ValueError(f"E2 snapshot record {name!r} is malformed")
                snapshot_path = Path(str(record["path"]))
                if not snapshot_path.is_absolute():
                    snapshot_path = self.run_dir / snapshot_path
                try:
                    snapshot_path.resolve().relative_to(self.run_dir.resolve())
                except ValueError as exc:
                    raise ValueError(f"E2 snapshot {name!r} is outside the run directory") from exc
                if not snapshot_path.is_file() or _sha256(snapshot_path) != str(record["sha256"]):
                    raise ValueError(f"E2 snapshot {name!r} is missing or has a SHA mismatch")
        tasks = payload.get("tasks")
        if not isinstance(tasks, list):
            raise ValueError("E2 task manifest has no task list")
        counts = Counter(str(task.get("phase")) for task in tasks if isinstance(task, Mapping))
        if len(tasks) != 318 or counts.get("pilot", 0) != 18 or counts.get("full", 0) != 300:
            raise ValueError("E2 task manifest must contain exactly 18 pilot and 300 full tasks")
        declared_counts = payload.get("counts")
        if not isinstance(declared_counts, Mapping) or {
            "pilot": int(declared_counts.get("pilot", -1)),
            "full": int(declared_counts.get("full", -1)),
            "total": int(declared_counts.get("total", -1)),
        } != {"pilot": 18, "full": 300, "total": 318}:
            raise ValueError("E2 task manifest has inconsistent declared counts")
        for phase, expected_directions in (("pilot", 6), ("full", 6)):
            directions = {
                (str(task.get("source_composer")), str(task.get("target_composer")))
                for task in tasks
                if isinstance(task, Mapping) and task.get("phase") == phase
            }
            if len(directions) != expected_directions:
                raise ValueError(f"E2 {phase} task manifest does not cover six directions")
        full_source_counts = Counter(
            str(task.get("source_id"))
            for task in tasks
            if isinstance(task, Mapping) and task.get("phase") == "full"
        )
        if not full_source_counts or any(count != 2 for count in full_source_counts.values()):
            raise ValueError("E2 full task manifest must contain two targets per test source")
        pilot_source_counts = Counter(
            str(task.get("source_id"))
            for task in tasks
            if isinstance(task, Mapping) and task.get("phase") == "pilot"
        )
        if not pilot_source_counts or any(count != len(self.config.pilot_seeds) * 2 for count in pilot_source_counts.values()):
            raise ValueError("E2 pilot task manifest must contain both targets for every pilot seed")
        pilot_seeds = {
            int(task.get("seed"))
            for task in tasks
            if isinstance(task, Mapping) and task.get("phase") == "pilot"
        }
        if pilot_seeds != set(self.config.pilot_seeds):
            raise ValueError("E2 pilot task manifest has an unexpected seed set")
        full_seeds = {
            int(task.get("seed"))
            for task in tasks
            if isinstance(task, Mapping) and task.get("phase") == "full"
        }
        if full_seeds != {self.config.main_seed}:
            raise ValueError("E2 full task manifest has an unexpected seed")
        task_ids: set[str] = set()
        for task in tasks:
            if not isinstance(task, Mapping):
                raise ValueError("E2 task manifest contains a malformed task")
            task_id = str(task.get("task_id", ""))
            if not task_id or task_id in task_ids:
                raise ValueError("E2 task manifest contains duplicate/empty task IDs")
            task_ids.add(task_id)
            signature_input = dict(task)
            signature = signature_input.pop("task_signature", None)
            if signature != _task_signature(signature_input, self._config_hash):
                raise ValueError(f"E2 task {task_id} has an invalid signature")
            if task.get("source_id") in task.get("target_train_ids", ()):
                raise ValueError(f"E2 task {task_id} leaks its source into target training")

    def _validate_task_membership(
        self,
        payload: Mapping[str, Any],
        manifest: Mapping[str, Any],
        splits: Mapping[str, Any],
    ) -> None:
        """Cross-check task rows against the immutable E1 fold membership.

        Signatures protect against accidental edits made with the same config,
        while this check protects the more important statistical invariant:
        every source is a test sample and every target profile consists of the
        corresponding fold's training samples, with no work/group overlap.
        """
        repetition = _validate_splits(manifest, splits, self.config.repeat)
        accepted = {
            str(row["sample_id"]): row
            for row in manifest.get("samples", [])
            if row.get("validation_status") == "accepted"
        }
        folds = {int(fold["fold"]): fold for fold in repetition.get("folds", [])}
        for task in payload.get("tasks", []):
            fold_index = int(task["fold"])
            if fold_index not in folds:
                raise ValueError(f"E2 task {task.get('task_id')} references an unknown fold")
            fold = folds[fold_index]
            train_ids = [str(item) for item in fold["train"]["sample_ids"]]
            test_ids = [str(item) for item in fold["test"]["sample_ids"]]
            source_id = str(task.get("source_id"))
            if source_id not in accepted or source_id not in test_ids:
                raise ValueError(f"E2 task {task.get('task_id')} source is not a fold test sample")
            source = accepted[source_id]
            if str(task.get("source_composer")) != str(source["composer"]):
                raise ValueError(f"E2 task {task.get('task_id')} has an incorrect source composer")
            target = str(task.get("target_composer"))
            if target == str(source["composer"]):
                raise ValueError(f"E2 task {task.get('task_id')} transfers to its source composer")
            expected_target_ids = sorted(
                item for item in train_ids if str(accepted[item]["composer"]) == target
            )
            actual_target_ids = sorted(str(item) for item in task.get("target_train_ids", ()))
            if actual_target_ids != expected_target_ids:
                raise ValueError(f"E2 task {task.get('task_id')} has an incorrect target training set")
            source_group = str(source["group_id"])
            if any(str(accepted[item]["group_id"]) == source_group for item in actual_target_ids):
                raise ValueError(f"E2 task {task.get('task_id')} leaks its source group")

    def prepare(self) -> Path:
        """Validate E1 snapshots and build the deterministic pilot/full task list."""
        # Always validate the immutable E1 inputs, even when a task manifest is
        # already present.  This catches a replaced/corrupted cache or source
        # MIDI before a resume can silently reuse it.
        manifest, splits, legacy, composition = self._load_inputs()
        existing = self._existing_task_manifest()
        if existing is not None:
            self._validate_task_manifest_payload(existing)
            self._validate_task_membership(existing, manifest, splits)
            existing_inputs = existing.get("inputs")
            existing_snapshots = (
                existing_inputs.get("snapshots", {})
                if isinstance(existing_inputs, Mapping)
                else {}
            )
            current_sources = {
                "manifest": self.config.manifest_path,
                "splits": self.config.splits_path,
                "legacy_features": self.config.legacy_features_path,
                "composition_features": self.config.composition_features_path,
            }
            if isinstance(existing_snapshots, Mapping) and existing_snapshots:
                for name, source in current_sources.items():
                    snapshot = existing_snapshots.get(name)
                    if isinstance(snapshot, Mapping) and snapshot.get("sha256"):
                        if _sha256(source) != str(snapshot["sha256"]):
                            raise ValueError(
                                f"E1 input {name} changed since E2 preparation; "
                                "use a new --run-dir"
                            )
            # Upgrade manifests created by an earlier E2 patch to include the
            # immutable input snapshots used by workers. Existing snapshots
            # are never overwritten during a resume.
            inputs = existing.get("inputs")
            if not isinstance(inputs, Mapping) or not inputs.get("snapshots"):
                inputs_dir = self.run_dir / "inputs"
                inputs_dir.mkdir(parents=True, exist_ok=True)
                snapshot_records: dict[str, dict[str, str]] = {}
                for name, source in {
                    "manifest": self.config.manifest_path,
                    "splits": self.config.splits_path,
                    "legacy_features": self.config.legacy_features_path,
                    "composition_features": self.config.composition_features_path,
                }.items():
                    destination = inputs_dir / source.name
                    if not destination.is_file():
                        shutil.copy2(source, destination)
                    snapshot_records[name] = {"path": str(destination), "sha256": _sha256(destination)}
                config_snapshot = self.run_dir / "config_used.yaml"
                if config_snapshot.is_file():
                    snapshot_records["config"] = {
                        "path": str(config_snapshot),
                        "sha256": _sha256(config_snapshot),
                    }
                existing_inputs = dict(inputs) if isinstance(inputs, Mapping) else {}
                existing_inputs["snapshots"] = snapshot_records
                existing["inputs"] = existing_inputs
                _atomic_json(self.task_manifest_path, existing)
                if self.run_manifest_path.is_file():
                    run_manifest = json.loads(self.run_manifest_path.read_text(encoding="utf-8"))
                    changed = False
                    if not run_manifest.get("snapshots"):
                        run_manifest["snapshots"] = snapshot_records
                        changed = True
                    if not run_manifest.get("dataset_fingerprints"):
                        run_manifest["dataset_fingerprints"] = {
                            str(name): str(value)
                            for name, value in manifest.get("dataset_fingerprints", {}).items()
                        }
                        changed = True
                    if changed:
                        _atomic_json(self.run_manifest_path, run_manifest)
            if self.run_manifest_path.is_file():
                run_manifest = json.loads(self.run_manifest_path.read_text(encoding="utf-8"))
                changed = False
                existing_inputs = existing.get("inputs")
                snapshot_records = existing_inputs.get("snapshots", {}) if isinstance(existing_inputs, Mapping) else {}
                if isinstance(existing_inputs, Mapping) and isinstance(snapshot_records, Mapping):
                    task_snapshots = dict(snapshot_records)
                    config_snapshot = self.run_dir / "config_used.yaml"
                    if config_snapshot.is_file() and "config" not in task_snapshots:
                        task_snapshots["config"] = {
                            "path": str(config_snapshot),
                            "sha256": _sha256(config_snapshot),
                        }
                        existing_inputs = dict(existing_inputs)
                        existing_inputs["snapshots"] = task_snapshots
                        existing["inputs"] = existing_inputs
                        snapshot_records = task_snapshots
                        _atomic_json(self.task_manifest_path, existing)
                if not run_manifest.get("snapshots") and snapshot_records:
                    run_manifest["snapshots"] = snapshot_records
                    changed = True
                config_snapshot = self.run_dir / "config_used.yaml"
                if config_snapshot.is_file() and isinstance(run_manifest.get("snapshots"), Mapping) and "config" not in run_manifest["snapshots"]:
                    snapshots = dict(run_manifest["snapshots"])
                    snapshots["config"] = {
                        "path": str(config_snapshot),
                        "sha256": _sha256(config_snapshot),
                    }
                    run_manifest["snapshots"] = snapshots
                    changed = True
                if not run_manifest.get("dataset_fingerprints"):
                    run_manifest["dataset_fingerprints"] = {
                        str(name): str(value)
                        for name, value in manifest.get("dataset_fingerprints", {}).items()
                    }
                    changed = True
                if self.task_manifest_path.is_file():
                    task_manifest_sha = _sha256(self.task_manifest_path)
                    task_manifest_record = run_manifest.get("task_manifest")
                    if not isinstance(task_manifest_record, Mapping) or task_manifest_record.get("sha256") != task_manifest_sha:
                        run_manifest["task_manifest"] = {
                            "path": str(self.task_manifest_path),
                            "sha256": task_manifest_sha,
                        }
                        changed = True
                if changed:
                    _atomic_json(self.run_manifest_path, run_manifest)
            else:
                # A missing run manifest is recoverable from the immutable task
                # manifest and snapshots.  Recreate only this bookkeeping file;
                # task/result artifacts are never overwritten here.
                inputs = existing.get("inputs", {})
                snapshots = inputs.get("snapshots", {}) if isinstance(inputs, Mapping) else {}
                _atomic_json(
                    self.run_manifest_path,
                    {
                        "schema_version": E2_SCHEMA_VERSION,
                        "experiment": "E2",
                        "status": "prepared",
                        "created_at_utc": _utc_now(),
                        "config_hash": self._config_hash,
                        "code_commit": _git_commit(),
                        "inputs": {
                            name: {"path": str(path), "sha256": _sha256(path)}
                            for name, path in {
                                "manifest": self.config.manifest_path,
                                "splits": self.config.splits_path,
                                "legacy_features": self.config.legacy_features_path,
                                "composition_features": self.config.composition_features_path,
                            }.items()
                        },
                        "snapshots": snapshots,
                        "task_manifest": {
                            "path": str(self.task_manifest_path),
                            "sha256": _sha256(self.task_manifest_path),
                        },
                        "dataset_fingerprints": {
                            str(name): str(value)
                            for name, value in manifest.get("dataset_fingerprints", {}).items()
                        },
                        "task_counts": existing.get("counts", {}),
                        "environment": {
                            "python": platform.python_version(),
                            "numpy": np.__version__,
                            "platform": platform.platform(),
                        },
                    },
                )
            return self.task_manifest_path
        repetition = _validate_splits(manifest, splits, self.config.repeat)
        samples = {
            row["sample_id"]: row
            for row in manifest.get("samples", [])
            if row.get("validation_status") == "accepted"
        }
        folds = sorted(repetition["folds"], key=lambda item: int(item["fold"]))
        full_tasks: list[dict[str, Any]] = []
        for fold in folds:
            fold_index = int(fold["fold"])
            train_ids = list(fold["train"]["sample_ids"])
            for source_id in sorted(fold["test"]["sample_ids"]):
                source_composer = str(samples[source_id]["composer"])
                for target_composer in sorted({str(samples[item]["composer"]) for item in train_ids} - {source_composer}):
                    task = {
                        "phase": "full",
                        "repeat": self.config.repeat,
                        "fold": fold_index,
                        "source_id": source_id,
                        "source_composer": source_composer,
                        "target_composer": target_composer,
                        "target_train_ids": sorted(
                            item for item in train_ids if str(samples[item]["composer"]) == target_composer
                        ),
                        "seed": self.config.main_seed,
                    }
                    task["task_id"] = _task_id(
                        "full", self.config.repeat, fold_index, source_id, target_composer, self.config.main_seed
                    )
                    task["task_signature"] = _task_signature(task, self._config_hash)
                    full_tasks.append(task)
        if len(full_tasks) != 300 or len({task["task_id"] for task in full_tasks}) != 300:
            raise ValueError(f"E2 full task matrix must contain exactly 300 unique tasks, found {len(full_tasks)}")
        full_directions = {
            (str(task["source_composer"]), str(task["target_composer"]))
            for task in full_tasks
        }
        if len(full_directions) != 6:
            raise ValueError(f"E2 full matrix must contain six transfer directions, found {len(full_directions)}")

        pilot_fold = next(fold for fold in folds if int(fold["fold"]) == self.config.pilot_fold)
        pilot_test = list(pilot_fold["test"]["sample_ids"])
        representatives: dict[str, str] = {}
        for composer in sorted({str(samples[item]["composer"]) for item in pilot_test}):
            candidates = [samples[item] for item in pilot_test if str(samples[item]["composer"]) == composer]
            median_count = float(np.median([int(item["note_count"]) for item in candidates]))
            representatives[composer] = min(
                (str(item["sample_id"]) for item in candidates),
                key=lambda item: (abs(int(samples[item]["note_count"]) - median_count), item),
            )
        pilot_tasks: list[dict[str, Any]] = []
        train_ids = list(pilot_fold["train"]["sample_ids"])
        available_composers = sorted({str(samples[item]["composer"]) for item in train_ids})
        for seed in self.config.pilot_seeds:
            for source_composer in sorted(representatives):
                source_id = representatives[source_composer]
                for target_composer in available_composers:
                    if target_composer == source_composer:
                        continue
                    task = {
                        "phase": "pilot",
                        "repeat": self.config.repeat,
                        "fold": self.config.pilot_fold,
                        "source_id": source_id,
                        "source_composer": source_composer,
                        "target_composer": target_composer,
                        "target_train_ids": sorted(
                            item for item in train_ids if str(samples[item]["composer"]) == target_composer
                        ),
                        "seed": int(seed),
                    }
                    task["task_id"] = _task_id(
                        "pilot", self.config.repeat, self.config.pilot_fold, source_id, target_composer, int(seed)
                    )
                    task["task_signature"] = _task_signature(task, self._config_hash)
                    pilot_tasks.append(task)
        if len(pilot_tasks) != 18 or len({task["task_id"] for task in pilot_tasks}) != 18:
            raise ValueError(f"E2 pilot matrix must contain exactly 18 unique tasks, found {len(pilot_tasks)}")
        if len({(task["source_composer"], task["target_composer"]) for task in pilot_tasks}) != 6:
            raise ValueError("E2 pilot matrix must cover all six transfer directions")
        tasks = sorted(pilot_tasks + full_tasks, key=lambda item: item["task_id"])
        payload = {
            "schema_version": E2_SCHEMA_VERSION,
            "created_at_utc": _utc_now(),
            "config_hash": self._config_hash,
            "repeat": self.config.repeat,
            "main_seed": self.config.main_seed,
            "pilot_seeds": list(self.config.pilot_seeds),
            "pilot_fold": self.config.pilot_fold,
            "counts": {"pilot": len(pilot_tasks), "full": len(full_tasks), "total": len(tasks)},
            "representatives": representatives,
            "tasks": tasks,
            "inputs": {
                "manifest": str(self.config.manifest_path),
                "splits": str(self.config.splits_path),
                "legacy_features": str(self.config.legacy_features_path),
                "composition_features": str(self.config.composition_features_path),
                "dataset_root": str(self.config.dataset_root),
            },
        }
        self.run_dir.mkdir(parents=True, exist_ok=True)
        inputs_dir = self.run_dir / "inputs"
        inputs_dir.mkdir(parents=True, exist_ok=True)
        for source in (self.config.manifest_path, self.config.splits_path, self.config.legacy_features_path, self.config.composition_features_path):
            shutil.copy2(source, inputs_dir / source.name)
        if self.config.raw:
            _atomic_text(self.run_dir / "config_used.yaml", yaml.safe_dump(self.config.raw, sort_keys=False, allow_unicode=True))
        snapshot_paths = {
            name: inputs_dir / source.name
            for name, source in {
                "manifest": self.config.manifest_path,
                "splits": self.config.splits_path,
                "legacy_features": self.config.legacy_features_path,
                "composition_features": self.config.composition_features_path,
            }.items()
        }
        payload["inputs"]["snapshots"] = {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in snapshot_paths.items()
        }
        if self.config.raw:
            config_snapshot = self.run_dir / "config_used.yaml"
            payload["inputs"]["snapshots"]["config"] = {
                "path": str(config_snapshot),
                "sha256": _sha256(config_snapshot),
            }
        _atomic_json(self.task_manifest_path, payload)
        snapshot_records = {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in snapshot_paths.items()
        }
        if self.config.raw:
            config_snapshot = self.run_dir / "config_used.yaml"
            snapshot_records["config"] = {
                "path": str(config_snapshot),
                "sha256": _sha256(config_snapshot),
            }
        _atomic_json(
            self.run_manifest_path,
            {
                "schema_version": E2_SCHEMA_VERSION,
                "experiment": "E2",
                "status": "prepared",
                "created_at_utc": _utc_now(),
                "config_hash": self._config_hash,
                "code_commit": _git_commit(),
                "inputs": {
                    name: {"path": str(path), "sha256": _sha256(path)}
                    for name, path in {
                        "manifest": self.config.manifest_path,
                        "splits": self.config.splits_path,
                        "legacy_features": self.config.legacy_features_path,
                        "composition_features": self.config.composition_features_path,
                    }.items()
                },
                "snapshots": snapshot_records,
                "task_manifest": {
                    "path": str(self.task_manifest_path),
                    "sha256": _sha256(self.task_manifest_path),
                },
                "dataset_fingerprints": {
                    str(name): str(value)
                    for name, value in manifest.get("dataset_fingerprints", {}).items()
                },
                "task_counts": {"pilot": len(pilot_tasks), "full": len(full_tasks)},
                "environment": {
                    "python": platform.python_version(),
                    "numpy": np.__version__,
                    "platform": platform.platform(),
                },
            },
        )
        return self.task_manifest_path

    def ensure_prepared(self) -> None:
        # ``prepare`` is idempotent and performs input/task-manifest
        # validation even on an existing run, which is essential before a
        # resumed worker touches any artifacts.
        self.prepare()

    def _load_task_manifest(self) -> dict[str, Any]:
        manifest = self._existing_task_manifest()
        if manifest is None:
            raise ValueError("E2 task manifest is missing; run prepare first")
        self._validate_task_manifest_payload(manifest)
        return manifest

    def _worker_payload(self) -> dict[str, Any]:
        inputs_dir = self.run_dir / "inputs"

        def snapshot_or_original(path: Path) -> str:
            snapshot = inputs_dir / path.name
            return str(snapshot if snapshot.is_file() else path)

        return {
            "manifest_path": snapshot_or_original(self.config.manifest_path),
            "splits_path": snapshot_or_original(self.config.splits_path),
            "legacy_features_path": snapshot_or_original(self.config.legacy_features_path),
            "composition_features_path": snapshot_or_original(self.config.composition_features_path),
            "dataset_root": str(self.config.dataset_root),
            "run_dir": str(self.run_dir),
            "ga": asdict(self.config.ga),
            "evaluator_n_estimators": self.config.evaluator_n_estimators,
            "evaluator_max_features": self.config.evaluator_max_features,
            "evaluator_min_samples_leaf": self.config.evaluator_min_samples_leaf,
            "onset_tolerance_beats": self.config.onset_tolerance_beats,
            "config_hash": self._config_hash,
            "repeat": self.config.repeat,
        }

    def _task_result_path(self, task: Mapping[str, Any]) -> Path:
        return self.run_dir / "tasks" / str(task["task_id"]) / "result.json"

    def _write_parent_failure(self, task: Mapping[str, Any], exc: BaseException) -> dict[str, Any]:
        """Persist a failure raised outside the worker's task try/except.

        A process can die before :func:`_execute_task` gets a chance to write
        its own record (for example, an initializer or native-library failure).
        The parent still writes a normal, retryable task record so a partial
        stage can never be mistaken for a complete 300-task matrix.
        """
        result = {
            "schema_version": E2_SCHEMA_VERSION,
            "task_id": str(task["task_id"]),
            "task_signature": task.get("task_signature"),
            "phase": task.get("phase"),
            "fold": int(task["fold"]),
            "source_id": task.get("source_id"),
            "source_composer": task.get("source_composer"),
            "target_composer": task.get("target_composer"),
            "seed": int(task["seed"]),
            "status": "failed",
            "elapsed_seconds": None,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        _atomic_json(self._task_result_path(task), result)
        return result

    def _completed_tasks(self, tasks: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        pending: list[dict[str, Any]] = []
        completed: list[dict[str, Any]] = []
        for task in tasks:
            path = self._task_result_path(task)
            try:
                result = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                pending.append(task)
                continue
            if not isinstance(result, Mapping):
                pending.append(task)
                continue
            if _valid_completed_record(self.run_dir, task, result):
                completed.append(result)
            else:
                pending.append(task)
        return pending, completed

    def _validated_aggregate(
        self,
        path: Path,
        tasks: Sequence[dict[str, Any]],
        phase: str,
    ) -> list[dict[str, Any]]:
        """Load an aggregate and prove that it is the current task matrix.

        A stage aggregate is written only after all task records have been
        accepted, but it is still an independent resumable artifact.  Checking
        its config hash, count, IDs, signatures and task records prevents a
        stale/corrupt ``pilot_results.json`` from satisfying the full-stage
        prerequisite.
        """
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"{phase} aggregate is missing or unreadable: {path}") from exc
        if not isinstance(payload, Mapping):
            raise ValueError(f"{phase} aggregate must be a JSON object")
        if payload.get("schema_version") != E2_SCHEMA_VERSION or payload.get("phase") != phase:
            raise ValueError(f"{phase} aggregate has an incompatible schema or phase")
        if payload.get("config_hash") != self._config_hash:
            raise ValueError(f"{phase} aggregate belongs to a different configuration")
        raw_results = payload.get("results")
        if not isinstance(raw_results, list) or len(raw_results) != len(tasks):
            raise ValueError(f"{phase} aggregate does not contain all task results")
        expected = {str(task["task_id"]): task for task in tasks}
        seen: set[str] = set()
        validated: list[dict[str, Any]] = []
        for raw in raw_results:
            if not isinstance(raw, Mapping):
                raise ValueError(f"{phase} aggregate contains a malformed result")
            task_id = str(raw.get("task_id", ""))
            if task_id in seen or task_id not in expected:
                raise ValueError(f"{phase} aggregate contains a duplicate or unknown task")
            task = expected[task_id]
            if not _valid_completed_record(self.run_dir, task, raw):
                raise ValueError(f"{phase} aggregate contains an invalid task artifact: {task_id}")
            seen.add(task_id)
            validated.append(dict(raw))
        if seen != set(expected):
            raise ValueError(f"{phase} aggregate is missing task results")
        try:
            task_count = int(payload.get("task_count", -1))
            completed_count = int(payload.get("completed_count", -1))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{phase} aggregate has invalid completion counts") from exc
        if task_count != len(tasks) or completed_count != len(tasks):
            raise ValueError(f"{phase} aggregate has inconsistent completion counts")
        return validated

    def run_stage(self, phase: str) -> Path:
        """Run ``pilot`` or ``full`` tasks, resuming completed task records."""
        if phase not in {"pilot", "full"}:
            raise ValueError("phase must be 'pilot' or 'full'")
        # Protect the library API as well as the CLI (which calls
        # ``ensure_prepared``): a direct stage invocation must not run against
        # a changed E1 source/cache after preparation.
        manifest, splits, _, _ = self._load_inputs()
        task_manifest = self._load_task_manifest()
        self._validate_task_membership(task_manifest, manifest, splits)
        all_tasks = [task for task in task_manifest.get("tasks", []) if task.get("phase") == phase]
        if not all_tasks:
            raise ValueError(f"task manifest contains no {phase} tasks")
        if phase == "full":
            run_manifest = json.loads(self.run_manifest_path.read_text(encoding="utf-8"))
            if run_manifest.get("config_hash") != self._config_hash:
                raise ValueError("E2 run manifest belongs to a different configuration")
            pilot_tasks = [task for task in task_manifest.get("tasks", []) if task.get("phase") == "pilot"]
            pilot_results_path = self.run_dir / PILOT_RESULTS_FILENAME
            if run_manifest.get("status") not in {
                "pilot_running", "pilot_completed", "full_running", "full_failed",
                "full_completed", "reported", "interrupted",
            }:
                raise ValueError("full E2 run requires a completed pilot stage")
            pilot_pending, pilot_completed = self._completed_tasks(pilot_tasks)
            if pilot_pending or len(pilot_completed) != len(pilot_tasks) or not pilot_results_path.is_file():
                raise ValueError("full E2 run requires all pilot task records")
            self._validated_aggregate(pilot_results_path, pilot_tasks, "pilot")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        run_manifest = json.loads(self.run_manifest_path.read_text(encoding="utf-8"))
        if run_manifest.get("config_hash") != self._config_hash:
            raise ValueError("E2 run manifest belongs to a different configuration")
        run_manifest["status"] = "pilot_running" if phase == "pilot" else "full_running"
        run_manifest["started_at_utc"] = _utc_now()
        run_manifest["workers"] = self.workers
        _atomic_json(self.run_manifest_path, run_manifest)
        pending, completed = self._completed_tasks(all_tasks)
        tasks_dir = self.run_dir / "tasks"
        tasks_dir.mkdir(parents=True, exist_ok=True)
        progress_path = self.run_dir / PROGRESS_FILENAME
        started = time.perf_counter()
        failures: list[dict[str, Any]] = []
        processed = len(completed)

        def record_progress(event: Mapping[str, Any]) -> None:
            with progress_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"timestamp_utc": _utc_now(), **event}, ensure_ascii=False, default=_json_default) + "\n")

        record_progress({"event": "stage_started", "phase": phase, "total": len(all_tasks), "pending": len(pending), "completed": len(completed)})
        if pending:
            payload = self._worker_payload()
            if self.workers == 1:
                try:
                    _init_worker(payload)
                except KeyboardInterrupt:
                    run_manifest["status"] = "interrupted"
                    run_manifest["finished_at_utc"] = _utc_now()
                    run_manifest["elapsed_seconds"] = time.perf_counter() - started
                    _atomic_json(self.run_manifest_path, run_manifest)
                    raise
                except Exception as exc:  # noqa: BLE001 - initializer failure is retryable
                    for task in pending:
                        result = self._write_parent_failure(task, exc)
                        failures.append(result)
                        processed += 1
                        record_progress({"event": "task_completed", "phase": phase, "task_id": task["task_id"], "status": "failed", "elapsed_seconds": None})
                else:
                    iterator = ((_execute_task(task), task) for task in pending)
                    try:
                        for result, task in iterator:
                            if not isinstance(result, Mapping):
                                result = self._write_parent_failure(task, RuntimeError("worker returned a non-mapping result"))
                            if result.get("status") != "completed":
                                failures.append(result)
                            processed += 1
                            record_progress({"event": "task_completed", "phase": phase, "task_id": task["task_id"], "status": result.get("status"), "elapsed_seconds": result.get("elapsed_seconds")})
                            print(f"E2 {phase}: {processed}/{len(all_tasks)} {task['task_id']} [{result.get('status')}]", flush=True)
                    except KeyboardInterrupt:
                        run_manifest["status"] = "interrupted"
                        run_manifest["finished_at_utc"] = _utc_now()
                        run_manifest["elapsed_seconds"] = time.perf_counter() - started
                        _atomic_json(self.run_manifest_path, run_manifest)
                        raise
            else:
                with ProcessPoolExecutor(max_workers=self.workers, initializer=_init_worker, initargs=(payload,)) as executor:
                    futures = {executor.submit(_execute_task, task): task for task in pending}
                    try:
                        for future in as_completed(futures):
                            task = futures[future]
                            try:
                                result = future.result()
                            except Exception as exc:  # noqa: BLE001 - isolate one worker failure
                                result = self._write_parent_failure(task, exc)
                            if not isinstance(result, Mapping):
                                result = self._write_parent_failure(task, RuntimeError("worker returned a non-mapping result"))
                            if result.get("status") != "completed":
                                failures.append(result)
                            processed += 1
                            record_progress({"event": "task_completed", "phase": phase, "task_id": task["task_id"], "status": result.get("status"), "elapsed_seconds": result.get("elapsed_seconds")})
                            print(f"E2 {phase}: task {task['task_id']} [{result.get('status')}]", flush=True)
                    except KeyboardInterrupt:
                        for future in futures:
                            future.cancel()
                        run_manifest["status"] = "interrupted"
                        run_manifest["finished_at_utc"] = _utc_now()
                        run_manifest["elapsed_seconds"] = time.perf_counter() - started
                        _atomic_json(self.run_manifest_path, run_manifest)
                        raise
        all_results: list[dict[str, Any]] = []
        for task in all_tasks:
            task_result_path = self._task_result_path(task)
            try:
                result = json.loads(task_result_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                result = self._write_parent_failure(
                    task,
                    RuntimeError(f"missing or unreadable task result: {exc}"),
                )
            if not isinstance(result, Mapping):
                result = self._write_parent_failure(
                    task,
                    RuntimeError("task result is not a JSON object"),
                )
            if result.get("task_signature") != task.get("task_signature"):
                result = self._write_parent_failure(
                    task,
                    RuntimeError("task result has an incompatible task signature"),
                )
            elif result.get("status") == "completed" and not _valid_completed_record(self.run_dir, task, result):
                result = self._write_parent_failure(
                    task,
                    RuntimeError("completed task result is missing or has a corrupt MIDI artifact"),
                )
            all_results.append(result)
        failed_results = [result for result in all_results if result.get("status") != "completed"]
        pilot_gate_error: str | None = None
        if phase == "pilot" and not failed_results:
            try:
                self._validate_pilot_gate(all_results)
            except Exception as exc:  # noqa: BLE001 - gate failure is recorded in the run manifest
                pilot_gate_error = f"{type(exc).__name__}: {exc}"
                failed_results = [{"status": "failed", "error": pilot_gate_error}]
        result_path = self.run_dir / (PILOT_RESULTS_FILENAME if phase == "pilot" else RESULTS_FILENAME)
        if phase == "pilot" and not failed_results:
            _atomic_json(result_path, self._aggregate_payload(all_results, phase))
        if phase == "full" and not failed_results:
            _atomic_json(result_path, self._aggregate_payload(all_results, phase))
            self._write_csv(all_results, self.run_dir / RESULTS_CSV_FILENAME)
        run_manifest["status"] = (
            "pilot_completed" if phase == "pilot" and not failed_results else
            "full_completed" if phase == "full" and not failed_results else
            "pilot_failed" if phase == "pilot" else "full_failed"
        )
        run_manifest["finished_at_utc"] = _utc_now()
        run_manifest["elapsed_seconds"] = time.perf_counter() - started
        run_manifest["task_status"] = {
            "expected": len(all_tasks),
            "completed": len([result for result in all_results if result.get("status") == "completed"]),
            "failed": len(failed_results),
            "resumed": len(completed),
        }
        if pilot_gate_error is not None:
            run_manifest["pilot_gate_error"] = pilot_gate_error
        _atomic_json(self.run_manifest_path, run_manifest)
        record_progress({"event": "stage_completed", "phase": phase, "status": run_manifest["status"], "elapsed_seconds": run_manifest["elapsed_seconds"]})
        if failed_results:
            raise RuntimeError(f"E2 {phase} stage has {len(failed_results)} failed tasks; rerun to retry them")
        return result_path

    def _validate_pilot_gate(self, results: Sequence[Mapping[str, Any]]) -> None:
        """Validate the pilot's technical (not quality) acceptance gate."""
        if len(results) != 18:
            raise ValueError(f"pilot requires 18 task records, found {len(results)}")
        finite_fields = (
            "p_target_input", "p_target_output", "delta_p_target",
            "p_source_input", "p_source_output", "identity_fitness",
            "generation0_fitness", "ga_final_fitness", "elapsed_seconds",
        )
        for result in results:
            if result.get("status") != "completed":
                raise ValueError(f"pilot task {result.get('task_id')} is not completed")
            _assert_finite_payload(result, f"pilot task {result.get('task_id')}")
            output_path = self.run_dir / str(result.get("output_path", ""))
            if not output_path.is_file() or output_path.stat().st_size <= 0:
                raise ValueError(f"pilot task {result.get('task_id')} has an empty/missing MIDI")
            if result.get("output_sha256") != _sha256(output_path):
                raise ValueError(f"pilot task {result.get('task_id')} has an invalid MIDI SHA")
            for field_name in finite_fields:
                value = result.get(field_name)
                if value is None or not np.isfinite(float(value)):
                    raise ValueError(f"pilot task {result.get('task_id')} has non-finite {field_name}")
            content = result.get("content", {})
            if not content.get("roundtrip_valid") or not result.get("output_nonempty", False):
                raise ValueError(f"pilot task {result.get('task_id')} failed MIDI round-trip/non-empty checks")
            if not all(
                bool(content.get(name))
                for name in ("ticks_per_beat_preserved", "smf_format_preserved", "meta_preserved")
            ):
                raise ValueError(f"pilot task {result.get('task_id')} did not preserve MIDI metadata")
            deterministic = result.get("determinism", {})
            if not all(
                bool(deterministic.get(name))
                for name in ("checked", "genome_equal", "history_equal", "output_equal")
            ):
                raise ValueError(f"pilot task {result.get('task_id')} failed deterministic repetition")

    def _aggregate_payload(self, results: list[dict[str, Any]], phase: str) -> dict[str, Any]:
        return {
            "schema_version": E2_SCHEMA_VERSION,
            "experiment": "E2",
            "phase": phase,
            "created_at_utc": _utc_now(),
            "config_hash": self._config_hash,
            "protocol": {
                "repeat": self.config.repeat,
                "main_seed": self.config.main_seed,
                "ga": asdict(self.config.ga),
                "evaluator": {
                    "model": "random_forest",
                    "n_estimators": self.config.evaluator_n_estimators,
                    "class_weight": "balanced",
                    "max_features": self.config.evaluator_max_features,
                    "min_samples_leaf": self.config.evaluator_min_samples_leaf,
                    "random_state": 1729,
                    "feature_variant": "composition_full",
                },
                "onset_tolerance_beats": self.config.onset_tolerance_beats,
                "statistics": {
                    "bootstrap_replicates": self.config.bootstrap_samples,
                    "sign_permutations": self.config.sign_permutations,
                    "bootstrap_cluster": "source_group_id",
                    "sign_permutation_alternative": "two-sided",
                    "directional_correction": "Holm",
                },
            },
            "task_count": len(results),
            "completed_count": sum(result.get("status") == "completed" for result in results),
            "results": sorted(results, key=lambda result: str(result.get("task_id"))),
        }

    @staticmethod
    def _write_csv(results: Sequence[Mapping[str, Any]], path: Path) -> None:
        columns = [
            "task_id", "fold", "source_id", "source_group_id", "source_composer", "target_composer", "seed",
            "p_target_input", "p_target_output", "delta_p_target", "p_source_input", "p_source_output",
            "target_class_input", "target_class_output", "target_hit_input", "target_hit_output",
            "source_probability_drop", "identity_fitness", "generation0_fitness", "ga_final_fitness",
            "fitness_gain_vs_identity", "fitness_gain_vs_generation0", "identity_worse",
            "stop_reason", "num_generations", "length_ratio", "length_error", "onset_f1",
            "melody_trigram_jaccard", "note_count_input", "note_count_output", "max_polyphony_output",
            "mean_polyphony_input", "mean_polyphony_output", "zero_duration_ratio_output",
            "roundtrip_valid", "roundtrip_semantic_equal", "elapsed_seconds",
        ]
        # Keep the tabular artifact resumable as well: a killed report stage
        # must never leave a truncated CSV that looks complete to a later
        # consumer.  Task-level records have already been validated before
        # this method is called, so flattening the content block is lossless
        # for the scalar columns below.
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.parent.mkdir(parents=True, exist_ok=True)
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            for result in sorted(results, key=lambda item: str(item.get("task_id"))):
                row = dict(result)
                content = result.get("content")
                if isinstance(content, Mapping):
                    row.update(content)
                writer.writerow({key: row.get(key) for key in columns})
        temporary.replace(path)

    def write_report(self) -> Path:
        """Create JSON/CSV/plots and the tracked Markdown E2 summary."""
        if not self.run_manifest_path.is_file():
            raise ValueError("E2 run manifest is missing; execute prepare and run first")
        run_manifest = json.loads(self.run_manifest_path.read_text(encoding="utf-8"))
        if run_manifest.get("config_hash") != self._config_hash:
            raise ValueError("E2 run manifest belongs to a different configuration")
        if run_manifest.get("status") not in {"full_completed", "reported"}:
            raise ValueError("E2 report requires a completed full stage")
        manifest, splits, _, _ = self._load_inputs()
        task_manifest = self._load_task_manifest()
        self._validate_task_membership(task_manifest, manifest, splits)
        full_tasks = [task for task in task_manifest.get("tasks", []) if task.get("phase") == "full"]
        pending, completed_records = self._completed_tasks(full_tasks)
        if pending or len(completed_records) != 300:
            raise ValueError("E2 report requires 300 intact completed task artifacts")
        result_path = self.run_dir / RESULTS_FILENAME
        if not result_path.is_file():
            raise ValueError("E2 full results are missing")
        results = self._validated_aggregate(result_path, full_tasks, "full")
        if len(results) != 300:
            raise ValueError(f"E2 report requires 300 completed tasks, found {len(results)}")
        expected_signatures = {str(task["task_id"]): task["task_signature"] for task in full_tasks}
        actual_signatures = {str(result.get("task_id")): result.get("task_signature") for result in results}
        if actual_signatures != expected_signatures:
            raise ValueError("E2 results do not match the prepared full task manifest")
        analysis = _analyse_results(results, self.config)
        _atomic_json(self.run_dir / "analysis.json", analysis)
        plots_dir = self.run_dir / "plots"
        _write_plots(results, analysis, plots_dir)
        report_path = self.run_dir / "E2.md"
        report = _render_markdown_report(analysis, run_manifest, self.run_dir)
        _atomic_text(report_path, report)
        summary_path = self.config.summary_path
        _atomic_text(summary_path, report)
        run_manifest["report"] = {
            "path": str(report_path),
            "summary_path": str(summary_path),
            "analysis": str(self.run_dir / "analysis.json"),
            "plots": sorted(str(path.relative_to(self.run_dir)) for path in plots_dir.glob("*.png")),
        }
        run_manifest["status"] = "reported"
        _atomic_json(self.run_manifest_path, run_manifest)
        return report_path


def _git_commit() -> str:
    try:
        import subprocess

        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
        )
    except OSError:
        return "unknown"
    return completed.stdout.strip() if completed.returncode == 0 and completed.stdout.strip() else "unknown"


# The worker context is built once per process.  It contains only read-only
# snapshots and caches; task-specific outputs are written into disjoint dirs.
_WORKER: "_WorkerContext | None" = None


def _init_worker(payload: Mapping[str, Any]) -> None:
    global _WORKER
    _WORKER = _WorkerContext.from_payload(payload)


def _execute_task(task: Mapping[str, Any]) -> dict[str, Any]:
    if _WORKER is None:
        raise RuntimeError("E2 worker was not initialized")
    return _WORKER.execute(task)


def _aggregate_legacy(values: np.ndarray) -> AggregatedFeatures:
    if values.ndim != 2 or values.shape[1] != 42 or values.shape[0] < 1:
        raise ValueError(f"legacy profile has invalid shape {values.shape}")
    covariance = np.cov(values, rowvar=False) if values.shape[0] > 1 else np.zeros((42, 42), dtype=float)
    covariance = np.asarray(covariance, dtype=np.float64)
    if covariance.shape != (42, 42) or not np.isfinite(covariance).all():
        raise ValueError("legacy profile covariance is invalid")
    return AggregatedFeatures(
        mean=FeatureVector.from_array(values.mean(axis=0)),
        median=FeatureVector.from_array(np.median(values, axis=0)),
        std=FeatureVector.from_array(values.std(axis=0, ddof=0)),
        covariance=covariance,
    )


class _WorkerContext:
    def __init__(self, payload: Mapping[str, Any]) -> None:
        self.run_dir = Path(str(payload["run_dir"]))
        self.dataset_root = Path(str(payload["dataset_root"]))
        self.ga_config = GAConfig.from_dict(dict(payload["ga"]))
        self.n_estimators = int(payload["evaluator_n_estimators"])
        self.max_features = float(payload["evaluator_max_features"])
        self.min_samples_leaf = int(payload["evaluator_min_samples_leaf"])
        self.onset_tolerance_beats = float(payload["onset_tolerance_beats"])
        self.config_hash = str(payload["config_hash"])
        self.parser = MidiParser()
        self.printer = MidiPrettyPrinter()
        self.extractor = None
        manifest = json.loads(Path(str(payload["manifest_path"])).read_text(encoding="utf-8"))
        splits = json.loads(Path(str(payload["splits_path"])).read_text(encoding="utf-8"))
        legacy = json.loads(Path(str(payload["legacy_features_path"])).read_text(encoding="utf-8"))
        composition = json.loads(Path(str(payload["composition_features_path"])).read_text(encoding="utf-8"))
        self.samples = {
            str(row["sample_id"]): row
            for row in manifest["samples"]
            if row.get("validation_status") == "accepted"
        }
        self.legacy = {
            str(row["sample_id"]): np.asarray(row["values"], dtype=np.float64)
            for row in legacy["samples"]
        }
        self.composition = {
            str(row["sample_id"]): np.asarray(row["values"], dtype=np.float64)
            for row in composition["samples"]
        }
        repetitions = {int(item["repeat"]): item for item in splits["repetitions"]}
        self.repetition = repetitions[int(payload.get("repeat", 0))] if int(payload.get("repeat", 0)) in repetitions else next(iter(repetitions.values()))
        self.fold_data: dict[int, dict[str, Any]] = {}
        self.source_cache: dict[str, InternalRepr] = {}
        self.target_cache: dict[tuple[int, str], AggregatedFeatures] = {}
        self.target_inverse_cache: dict[tuple[int, str], np.ndarray] = {}
        self.evaluator_cache: dict[int, Any] = {}
        self.group_indices = _feature_indices_by_group()
        self._build_fold_caches()

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "_WorkerContext":
        return cls(payload)

    def _build_fold_caches(self) -> None:
        for fold in self.repetition["folds"]:
            fold_index = int(fold["fold"])
            train_ids = list(fold["train"]["sample_ids"])
            self.fold_data[fold_index] = {"train_ids": train_ids, "test_ids": list(fold["test"]["sample_ids"])}
            for composer in sorted({str(self.samples[item]["composer"]) for item in train_ids}):
                ids = [item for item in train_ids if str(self.samples[item]["composer"]) == composer]
                target_profile = _aggregate_legacy(
                    np.asarray([self.legacy[item] for item in ids], dtype=np.float64)
                )
                cache_key = (fold_index, composer)
                self.target_cache[cache_key] = target_profile
                self.target_inverse_cache[cache_key] = prepare_mahalanobis(
                    target_profile.covariance,
                    dimension=42,
                )
            train_matrix = np.asarray([self.composition[item] for item in train_ids], dtype=np.float64)
            train_labels = np.asarray([str(self.samples[item]["composer"]) for item in train_ids])
            classifier = Pipeline(
                [
                    ("variance", VarianceThreshold()),
                    (
                        "model",
                        RandomForestClassifier(
                            n_estimators=self.n_estimators,
                            class_weight="balanced",
                            max_features=self.max_features,
                            min_samples_leaf=self.min_samples_leaf,
                            random_state=1729,
                            n_jobs=1,
                        ),
                    ),
                ]
            )
            self.evaluator_cache[fold_index] = classifier.fit(train_matrix, train_labels)

    def _source_repr(self, sample_id: str) -> InternalRepr:
        if sample_id not in self.source_cache:
            self.source_cache[sample_id] = self.parser.parse(
                self.dataset_root / str(self.samples[sample_id]["score_path"])
            )
        return self.source_cache[sample_id]

    def execute(self, task: Mapping[str, Any]) -> dict[str, Any]:
        task_id = str(task["task_id"])
        task_dir = self.run_dir / "tasks" / task_id
        result_path = task_dir / "result.json"
        if result_path.is_file():
            try:
                old = json.loads(result_path.read_text(encoding="utf-8"))
                if _valid_completed_record(self.run_dir, task, old):
                    return old
            except (OSError, json.JSONDecodeError):
                pass
        started = time.perf_counter()
        source_id = str(task.get("source_id", ""))
        source_sample = self.samples.get(source_id)
        base = {
            "schema_version": E2_SCHEMA_VERSION,
            "task_id": task_id,
            "task_signature": task.get("task_signature"),
            "phase": task.get("phase"),
            "fold": int(task["fold"]),
            "source_id": source_id,
            "source_composer": task.get("source_composer"),
            "source_group_id": source_sample.get("group_id") if source_sample else None,
            "target_composer": task.get("target_composer"),
            "seed": int(task["seed"]),
        }
        try:
            source = self._source_repr(str(task["source_id"]))
            cache_key = (int(task["fold"]), str(task["target_composer"]))
            target = self.target_cache[cache_key]
            prepared_inverse = self.target_inverse_cache[cache_key]
            fold_train_ids = self.fold_data[cache_key[0]]["train_ids"]
            expected_target_ids = sorted(
                item for item in fold_train_ids
                if str(self.samples[item]["composer"]) == str(task["target_composer"])
            )
            if sorted(str(item) for item in task.get("target_train_ids", ())) != expected_target_ids:
                raise ValueError("task target_train_ids do not match the fold-local target profile")
            if str(task["source_id"]) in fold_train_ids:
                raise ValueError("source sample occurs in the fold training set")
            ga = GeneticAlgorithm(extractor=self._legacy_extractor())
            ga_log = task_dir / "ga.jsonl"
            best_genome, history = ga.run(
                source,
                target,
                self.ga_config,
                int(task["seed"]),
                log_path=ga_log,
                prepared_inverse_covariance=prepared_inverse,
            )
            output_repr = apply_transformation(source, best_genome)
            if not output_repr.notes:
                raise ValueError("GA output contains no notes after transformation")
            determinism = {"checked": False}
            if str(task.get("phase")) == "pilot":
                # The pilot's technical gate includes an actual same-seed
                # repetition. Keep the repeat out of the primary log, but
                # compare the complete GA history and transformed MIDI.
                repeated_genome, repeated_history = ga.run(
                    source,
                    target,
                    self.ga_config,
                    int(task["seed"]),
                    log_path=None,
                    prepared_inverse_covariance=prepared_inverse,
                )
                repeated_output = apply_transformation(source, repeated_genome)
                determinism = {
                    "checked": True,
                    "genome_equal": bool(repeated_genome == best_genome),
                    "history_equal": bool(repeated_history == history),
                    "output_equal": bool(repeated_output == output_repr),
                }
                if not all(bool(value) for key, value in determinism.items() if key != "checked"):
                    raise RuntimeError("GA same-seed pilot repetition is not deterministic")
            task_dir.mkdir(parents=True, exist_ok=True)
            output_path = task_dir / "output.mid"
            temporary_output = task_dir / "output.mid.tmp"
            self.printer.write(output_repr, temporary_output)
            temporary_output.replace(output_path)
            roundtrip = self.parser.parse(output_path)
            input_values = self.composition[str(task["source_id"])]
            # Evaluate the bytes that were actually written and read back.
            # Dense transformed passages can create same-pitch overlaps; the
            # legacy parser's FIFO pairing may then assign a different duration
            # to an otherwise valid note.  The technical round-trip criterion
            # is therefore parseability plus preserved SMF metadata; a separate
            # semantic-equality flag below keeps this corner case visible.
            output_values = extract_composition_features(roundtrip)
            classifier = self.evaluator_cache[int(task["fold"])]
            classes = [str(value) for value in classifier.classes_]
            input_probability = classifier.predict_proba(input_values.reshape(1, -1))[0]
            output_probability = classifier.predict_proba(output_values.reshape(1, -1))[0]
            p_input = float(input_probability[classes.index(str(task["target_composer"]))])
            p_output = float(output_probability[classes.index(str(task["target_composer"]))])
            source_index = classes.index(str(task["source_composer"]))
            input_class = str(classifier.predict(input_values.reshape(1, -1))[0])
            output_class = str(classifier.predict(output_values.reshape(1, -1))[0])
            evaluator_feature_count = int(
                classifier.named_steps["variance"].get_support().sum()
            )
            identity_fitness = fitness(
                IDENTITY_GENOME,
                source,
                target,
                self.ga_config.fitness_metric,  # type: ignore[arg-type]
                extractor=self._legacy_extractor(),
                prepared_inverse_covariance=prepared_inverse,
            )
            generation0 = float(history.best_fitness[0])
            final_fitness = float(history.best_fitness[-1])
            generation0_genome = history.best_genome[0]
            content = _content_metrics(source, output_repr, roundtrip, self.onset_tolerance_beats)
            if not content["roundtrip_valid"] or not content["ticks_per_beat_preserved"] or not content["smf_format_preserved"] or not content["meta_preserved"]:
                raise ValueError("GA output failed MIDI round-trip or metadata preservation")
            result = {
                **base,
                "status": "completed",
                "elapsed_seconds": float(time.perf_counter() - started),
                "output_path": str(output_path.relative_to(self.run_dir)),
                "output_sha256": _sha256(output_path),
                "p_target_input": p_input,
                "p_target_output": p_output,
                "delta_p_target": p_output - p_input,
                "target_class_input": input_class,
                "target_class_output": output_class,
                "target_hit_input": bool(input_class == str(task["target_composer"])),
                "target_hit_output": bool(output_class == str(task["target_composer"])),
                "evaluator_classes": classes,
                "evaluator_feature_count": evaluator_feature_count,
                "p_source_input": float(input_probability[source_index]),
                "p_source_output": float(output_probability[source_index]),
                "source_probability_drop": float(input_probability[source_index] - output_probability[source_index]),
                "identity_fitness": float(identity_fitness),
                "generation0_fitness": generation0,
                "ga_final_fitness": final_fitness,
                "fitness_gain_vs_identity": final_fitness - float(identity_fitness),
                "fitness_gain_vs_generation0": final_fitness - generation0,
                "identity_worse": bool(final_fitness < float(identity_fitness)),
                "generation0_genome": _genome_dict(generation0_genome),
                "determinism": determinism,
                "output_nonempty": bool(output_repr.notes),
                "target_train_count": len(task.get("target_train_ids", ())),
                "stop_reason": history.stop_reason,
                "num_generations": history.num_generations,
                "best_genome": _genome_dict(best_genome),
                "history": {
                    "best_fitness": [float(value) for value in history.best_fitness],
                    "mean_fitness": [float(value) for value in history.mean_fitness],
                    "worst_fitness": [float(value) for value in history.worst_fitness],
                    "best_genome": [_genome_dict(value) for value in history.best_genome],
                    "stop_reason": history.stop_reason,
                },
                "style_distances_e1b": _style_distances(
                    input_values,
                    output_values,
                    self.composition,
                    self.fold_data[int(task["fold"])] ["train_ids"],
                    str(task["target_composer"]),
                    self.samples,
                    self.group_indices,
                ),
                "content": content,
            }
            genotype_path = task_dir / "genotype.json"
            history_path = task_dir / "history.json"
            _atomic_json(genotype_path, result["best_genome"])
            _atomic_json(history_path, result["history"])
            result["genotype_path"] = str(genotype_path.relative_to(self.run_dir))
            result["history_path"] = str(history_path.relative_to(self.run_dir))
            _assert_finite_payload(result, "task_result")
            _atomic_json(result_path, result)
            return result
        except Exception as exc:  # noqa: BLE001 - one task must not abort a resumable matrix
            result = {
                **base,
                "status": "failed",
                "elapsed_seconds": float(time.perf_counter() - started),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            _atomic_json(result_path, result)
            return result

    def _legacy_extractor(self):
        # Import lazily to keep process start-up and configuration validation
        # independent of the parser's logger setup.
        if self.extractor is None:
            from ..features.extractor import FeatureExtractor

            self.extractor = FeatureExtractor(parser=self.parser)
        return self.extractor


def _onsets(repr_: InternalRepr) -> list[int]:
    return sorted({int(note.tick) for note in repr_.notes})


def _melody(repr_: InternalRepr) -> list[int]:
    by_tick: dict[int, list[int]] = defaultdict(list)
    for note in repr_.notes:
        by_tick[int(note.tick)].append(int(note.pitch))
    return [max(by_tick[tick]) for tick in sorted(by_tick)]


def _contour_tokens(repr_: InternalRepr) -> tuple[int, ...]:
    melody = _melody(repr_)
    return tuple(int(np.sign(right - left)) for left, right in zip(melody, melody[1:]))


def _multiset_jaccard(left: Sequence[Any], right: Sequence[Any], n: int = 3) -> float:
    def grams(values: Sequence[Any]) -> Counter[tuple[Any, ...]]:
        return Counter(tuple(values[index : index + n]) for index in range(max(0, len(values) - n + 1)))

    first, second = grams(left), grams(right)
    if not first and not second:
        return 1.0
    union = sum((first | second).values())
    return float(sum((first & second).values()) / union) if union else 1.0


def _onset_f1(before: InternalRepr, after: InternalRepr, tolerance_beats: float) -> float:
    before_ticks = [tick / before.ticks_per_beat for tick in _onsets(before)]
    after_ticks = [tick / after.ticks_per_beat for tick in _onsets(after)]
    used: set[int] = set()
    matches = 0
    for tick in before_ticks:
        candidates = [
            (abs(tick - other), index)
            for index, other in enumerate(after_ticks)
            if index not in used and abs(tick - other) <= tolerance_beats
        ]
        if candidates:
            _, index = min(candidates)
            used.add(index)
            matches += 1
    if not before_ticks and not after_ticks:
        return 1.0
    precision = matches / len(after_ticks) if after_ticks else 0.0
    recall = matches / len(before_ticks) if before_ticks else 0.0
    return float(2.0 * precision * recall / (precision + recall)) if precision + recall else 0.0


def _max_polyphony(repr_: InternalRepr) -> int:
    boundaries: list[tuple[int, int]] = []
    for note in repr_.notes:
        if note.duration_ticks > 0:
            boundaries.extend(((note.tick, 1), (note.tick + note.duration_ticks, -1)))
    active = maximum = 0
    for _, change in sorted(boundaries, key=lambda item: (item[0], item[1])):
        active += change
        maximum = max(maximum, active)
    return maximum


def _mean_polyphony(repr_: InternalRepr) -> float:
    """Time-weighted mean number of simultaneously sounding positive notes."""
    boundaries: list[tuple[int, int]] = []
    for note in repr_.notes:
        if note.duration_ticks > 0:
            boundaries.extend(((note.tick, 1), (note.tick + note.duration_ticks, -1)))
    if not boundaries:
        return 0.0
    active = 0
    area = 0
    previous = min(tick for tick, _ in boundaries)
    for tick, change in sorted(boundaries, key=lambda item: (item[0], item[1])):
        area += active * (tick - previous)
        active += change
        previous = tick
    span = max(tick for tick, _ in boundaries) - min(tick for tick, _ in boundaries)
    return float(area / span) if span else float(active)


def _end_tick(repr_: InternalRepr) -> int:
    return max((note.tick + note.duration_ticks for note in repr_.notes), default=0)


def _content_metrics(
    source: InternalRepr,
    output: InternalRepr,
    roundtrip: InternalRepr,
    onset_tolerance_beats: float,
) -> dict[str, Any]:
    # Content is measured on the parsed artifact (``roundtrip``), i.e. on what
    # a downstream MIDI consumer will receive.  ``output`` is retained only for
    # a diagnostic semantic-equality flag because overlapping same-pitch notes
    # can be paired differently by a valid MIDI parser without making the file
    # unreadable.
    measured = roundtrip
    source_end = _end_tick(source)
    output_end = _end_tick(measured)
    length_ratio = output_end / source_end if source_end else 1.0
    source_notes = len(source.notes)
    output_notes = len(measured.notes)
    serialized_metadata_ok = bool(
        roundtrip.ticks_per_beat == output.ticks_per_beat
        and roundtrip.smf_format == output.smf_format
        and roundtrip.meta == output.meta
    )
    semantic_equal = bool(
        roundtrip.ticks_per_beat == output.ticks_per_beat
        and roundtrip.smf_format == output.smf_format
        and roundtrip.notes == output.notes
        and roundtrip.meta == output.meta
    )
    return {
        "ticks_per_beat_input": int(source.ticks_per_beat),
        "ticks_per_beat_output": int(output.ticks_per_beat),
        "smf_format_input": int(source.smf_format),
        "smf_format_output": int(output.smf_format),
        "meta_event_count_input": int(len(source.meta)),
        "meta_event_count_output": int(len(output.meta)),
        "length_beats_input": float(source_end / source.ticks_per_beat),
        "length_beats_output": float(output_end / output.ticks_per_beat),
        "length_ratio": float(length_ratio),
        "length_error": float(abs(length_ratio - 1.0)),
        "onset_f1": _onset_f1(source, measured, onset_tolerance_beats),
        "melody_trigram_jaccard": _multiset_jaccard(_contour_tokens(source), _contour_tokens(measured)),
        "note_count_input": source_notes,
        "note_count_output": output_notes,
        "note_count_ratio": float(output_notes / source_notes) if source_notes else 1.0,
        "max_polyphony_input": _max_polyphony(source),
        "max_polyphony_output": _max_polyphony(measured),
        "mean_polyphony_input": _mean_polyphony(source),
        "mean_polyphony_output": _mean_polyphony(measured),
        "zero_duration_ratio_input": float(sum(note.duration_ticks == 0 for note in source.notes) / source_notes) if source_notes else 0.0,
        "zero_duration_ratio_output": float(sum(note.duration_ticks == 0 for note in measured.notes) / output_notes) if output_notes else 0.0,
        "roundtrip_valid": bool(roundtrip.notes) and serialized_metadata_ok,
        "roundtrip_semantic_equal": semantic_equal,
        "ticks_per_beat_preserved": bool(source.ticks_per_beat == roundtrip.ticks_per_beat),
        "smf_format_preserved": bool(source.smf_format == roundtrip.smf_format),
        "meta_preserved": bool(source.meta == roundtrip.meta),
    }


def _style_distances(
    input_values: np.ndarray,
    output_values: np.ndarray,
    composition: Mapping[str, np.ndarray],
    train_ids: Sequence[str],
    target_composer: str,
    samples: Mapping[str, Mapping[str, Any]],
    group_indices: Mapping[str, np.ndarray],
) -> dict[str, dict[str, float]]:
    target_ids = [item for item in train_ids if str(samples[item]["composer"]) == target_composer]
    matrix = np.asarray([composition[item] for item in target_ids], dtype=np.float64)
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0, ddof=0)
    safe_std = np.where(std > 1e-12, std, 1.0)
    result: dict[str, dict[str, float]] = {}
    for group, indices in group_indices.items():
        before = float(np.linalg.norm((input_values[indices] - mean[indices]) / safe_std[indices]))
        after = float(np.linalg.norm((output_values[indices] - mean[indices]) / safe_std[indices]))
        result[group] = {"before": before, "after": after, "gain": before - after}
    return result


def _direction_key(source: Mapping[str, Any]) -> str:
    return f"{source['source_composer']}→{source['target_composer']}"


def _cluster_values(results: Sequence[Mapping[str, Any]], value_key: str) -> dict[str, list[float]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in results:
        grouped[str(row["source_group_id"])].append(float(row[value_key]))
    return grouped


def _bootstrap_ci(
    results: Sequence[Mapping[str, Any]],
    value_key: str,
    samples: int,
    seed: int,
) -> list[float]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in results:
        grouped[str(row["source_group_id"])].append(row)
    strata: dict[str, list[str]] = defaultdict(list)
    for group, rows in grouped.items():
        strata[str(rows[0]["source_composer"])].append(group)
    rng = np.random.default_rng(seed)
    scores: list[float] = []
    for _ in range(samples):
        values: list[float] = []
        for groups in strata.values():
            if not groups:
                continue
            selected = rng.integers(0, len(groups), size=len(groups))
            for index in selected:
                values.extend(float(row[value_key]) for row in grouped[groups[int(index)]])
        scores.append(float(np.mean(values)) if values else float("nan"))
    finite = np.asarray([value for value in scores if np.isfinite(value)], dtype=float)
    return [float(value) for value in np.percentile(finite, [2.5, 97.5])] if finite.size else [float("nan"), float("nan")]


def _sign_permutation_p(
    results: Sequence[Mapping[str, Any]], value_key: str, permutations: int, seed: int
) -> float:
    grouped = _cluster_values(results, value_key)
    values = np.asarray([np.mean(items) for items in grouped.values()], dtype=float)
    if values.size == 0:
        return float("nan")
    observed = float(values.mean())
    observed_abs = abs(observed)
    rng = np.random.default_rng(seed)
    exceedances = 0
    for _ in range(permutations):
        null = float(np.mean(values * rng.choice(np.asarray([-1.0, 1.0]), size=values.size)))
        # No GO/NO-GO direction is predeclared for E2, therefore the sign test
        # is two-sided: either a positive or a negative systematic shift is
        # evidence against the zero-effect null.
        exceedances += int(abs(null) >= observed_abs)
    return float((exceedances + 1) / (permutations + 1))


def _holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    valid = sorted(((key, value) for key, value in p_values.items() if np.isfinite(value)), key=lambda item: item[1])
    adjusted: dict[str, float] = {key: float("nan") for key in p_values}
    previous = 0.0
    for index, (key, value) in enumerate(valid):
        corrected = min(1.0, (len(valid) - index) * value)
        corrected = max(corrected, previous)
        adjusted[key] = corrected
        previous = corrected
    return adjusted


def _finite_stats(values: Iterable[float]) -> dict[str, float | int | None]:
    array = np.asarray([float(value) for value in values if np.isfinite(value)], dtype=float)
    if not array.size:
        return {"count": 0, "mean": None, "median": None, "std": None, "min": None, "max": None}
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "std": float(array.std(ddof=0)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _source_probability_drop(row: Mapping[str, Any]) -> float:
    if "source_probability_drop" in row:
        return float(row["source_probability_drop"])
    return float(row.get("p_source_input", float("nan"))) - float(row.get("p_source_output", float("nan")))


def _generation0_gain(row: Mapping[str, Any]) -> float:
    if "fitness_gain_vs_generation0" in row:
        return float(row["fitness_gain_vs_generation0"])
    return float(row.get("ga_final_fitness", float("nan"))) - float(row.get("generation0_fitness", float("nan")))


def _correlation(left: Iterable[float], right: Iterable[float]) -> float | None:
    first = np.asarray(list(left), dtype=float)
    second = np.asarray(list(right), dtype=float)
    mask = np.isfinite(first) & np.isfinite(second)
    if int(mask.sum()) < 2:
        return None
    first, second = first[mask], second[mask]
    if float(first.std()) == 0.0 or float(second.std()) == 0.0:
        return None
    return float(np.corrcoef(first, second)[0, 1])


def _analyse_results(results: Sequence[Mapping[str, Any]], config: E2Config) -> dict[str, Any]:
    if not results:
        raise ValueError("cannot analyse an empty E2 result set")
    directions: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in results:
        directions[_direction_key(row)].append(row)
    direction_p = {
        direction: _sign_permutation_p(rows, "delta_p_target", config.sign_permutations, 910001 + index)
        for index, (direction, rows) in enumerate(sorted(directions.items()))
    }
    direction_analysis: dict[str, Any] = {}
    for direction, rows in sorted(directions.items()):
        direction_analysis[direction] = {
            "task_count": len(rows),
            "delta_p_target": _finite_stats(float(row["delta_p_target"]) for row in rows),
            "delta_p_target_ci95_clustered": _bootstrap_ci(rows, "delta_p_target", config.bootstrap_samples, 920001 + len(direction_analysis)),
            "sign_permutation_p_value": direction_p[direction],
            "source_probability_drop": _finite_stats(_source_probability_drop(row) for row in rows),
            "target_classification": {
                "input_hit_rate": float(np.mean([bool(row.get("target_hit_input", False)) for row in rows])),
                "output_hit_rate": float(np.mean([bool(row.get("target_hit_output", False)) for row in rows])),
            },
            "content": {
                "length_error": _finite_stats(row["content"]["length_error"] for row in rows),
                "onset_f1": _finite_stats(row["content"]["onset_f1"] for row in rows),
                "melody_trigram_jaccard": _finite_stats(row["content"]["melody_trigram_jaccard"] for row in rows),
                "note_count_ratio": _finite_stats(row["content"]["note_count_ratio"] for row in rows),
                "max_polyphony_input": _finite_stats(row["content"].get("max_polyphony_input", float("nan")) for row in rows),
                "max_polyphony_output": _finite_stats(row["content"].get("max_polyphony_output", float("nan")) for row in rows),
                "mean_polyphony_input": _finite_stats(row["content"].get("mean_polyphony_input", float("nan")) for row in rows),
                "mean_polyphony_output": _finite_stats(row["content"].get("mean_polyphony_output", float("nan")) for row in rows),
                "zero_duration_ratio_output": _finite_stats(row["content"].get("zero_duration_ratio_output", float("nan")) for row in rows),
                "length_within_5_percent": float(np.mean([row["content"]["length_error"] <= 0.05 for row in rows])),
                "roundtrip_valid_rate": float(np.mean([bool(row["content"]["roundtrip_valid"]) for row in rows])),
            },
            "fitness_gain_vs_identity": _finite_stats(row["fitness_gain_vs_identity"] for row in rows),
            "fitness_gain_vs_generation0": _finite_stats(_generation0_gain(row) for row in rows),
            "identity_worse_rate": float(np.mean([row["fitness_gain_vs_identity"] < 0.0 for row in rows])),
            "elapsed_seconds": _finite_stats(row.get("elapsed_seconds", float("nan")) for row in rows),
        }
    holm = _holm_adjust(direction_p)
    for direction in direction_analysis:
        direction_analysis[direction]["sign_permutation_p_value_holm"] = holm[direction]
    group_distances: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in results:
        for group, values in row["style_distances_e1b"].items():
            for metric_name in ("before", "after", "gain"):
                if metric_name in values:
                    group_distances[group][metric_name].append(float(values[metric_name]))
    style_groups = {
        group: {
            metric_name: _finite_stats(metric_values)
            for metric_name, metric_values in sorted(metric_map.items())
        }
        for group, metric_map in sorted(group_distances.items())
    }
    final_genomes = [row["best_genome"] for row in results]
    identity_worse_ids = sorted(
        str(row.get("task_id"))
        for row in results
        if float(row.get("fitness_gain_vs_identity", float("nan"))) < 0.0
    )
    generation0_worse_ids = sorted(
        str(row.get("task_id"))
        for row in results
        if _generation0_gain(row) < 0.0
    )
    genes: dict[str, Any] = {}
    for gene in ("transpose_semitones", "rhythm_density_factor", "note_duration_factor", "velocity_offset"):
        values = np.asarray([float(genome[gene]) for genome in final_genomes], dtype=float)
        low, high = WORKING_RANGES[gene]
        genes[gene] = {
            **_finite_stats(values),
            "at_lower_bound_rate": float(np.mean(np.isclose(values, low))),
            "at_upper_bound_rate": float(np.mean(np.isclose(values, high))),
        }
    boundary_genes = [
        gene for gene, values in genes.items()
        if values["at_lower_bound_rate"] > 0.05 or values["at_upper_bound_rate"] > 0.05
    ]
    return {
        "schema_version": E2_SCHEMA_VERSION,
        "experiment": "E2",
        "decision": "baseline_completed_no_go_no_go",
        "task_count": len(results),
        "statistics": {
            "bootstrap_replicates": config.bootstrap_samples,
            "sign_permutations": config.sign_permutations,
            "bootstrap_cluster": "source_group_id",
            "sign_permutation_alternative": "two-sided",
            "directional_correction": "Holm",
        },
        "delta_p_target": {
            **_finite_stats(float(row["delta_p_target"]) for row in results),
            "ci95_clustered": _bootstrap_ci(results, "delta_p_target", config.bootstrap_samples, 930001),
            "sign_permutation_p_value": _sign_permutation_p(results, "delta_p_target", config.sign_permutations, 930002),
        },
        "p_target": {
            "input": _finite_stats(row["p_target_input"] for row in results),
            "output": _finite_stats(row["p_target_output"] for row in results),
            "classification": {
                "input_hit_rate": float(np.mean([bool(row.get("target_hit_input", False)) for row in results])),
                "output_hit_rate": float(np.mean([bool(row.get("target_hit_output", False)) for row in results])),
            },
        },
        "source_probability_drop": _finite_stats(_source_probability_drop(row) for row in results),
        "content": {
            "length_error": _finite_stats(row["content"]["length_error"] for row in results),
            "onset_f1": _finite_stats(row["content"]["onset_f1"] for row in results),
            "melody_trigram_jaccard": _finite_stats(row["content"]["melody_trigram_jaccard"] for row in results),
            "note_count_ratio": _finite_stats(row["content"]["note_count_ratio"] for row in results),
            "max_polyphony_input": _finite_stats(row["content"].get("max_polyphony_input", float("nan")) for row in results),
            "max_polyphony_output": _finite_stats(row["content"].get("max_polyphony_output", float("nan")) for row in results),
            "mean_polyphony_input": _finite_stats(row["content"].get("mean_polyphony_input", float("nan")) for row in results),
            "mean_polyphony_output": _finite_stats(row["content"].get("mean_polyphony_output", float("nan")) for row in results),
            "zero_duration_ratio_output": _finite_stats(row["content"].get("zero_duration_ratio_output", float("nan")) for row in results),
            "length_within_5_percent_rate": float(np.mean([row["content"]["length_error"] <= 0.05 for row in results])),
            "roundtrip_valid_rate": float(np.mean([bool(row["content"]["roundtrip_valid"]) for row in results])),
        },
        "content_tradeoff": {
            "delta_vs_length_error": _correlation(
                (float(row["delta_p_target"]) for row in results),
                (float(row["content"]["length_error"]) for row in results),
            ),
            "delta_vs_onset_loss": _correlation(
                (float(row["delta_p_target"]) for row in results),
                (1.0 - float(row["content"]["onset_f1"]) for row in results),
            ),
            "delta_vs_contour_loss": _correlation(
                (float(row["delta_p_target"]) for row in results),
                (1.0 - float(row["content"]["melody_trigram_jaccard"]) for row in results),
            ),
        },
        "fitness": {
            "gain_vs_identity": _finite_stats(row["fitness_gain_vs_identity"] for row in results),
            "gain_vs_generation0": _finite_stats(_generation0_gain(row) for row in results),
            "identity_worse_rate": float(np.mean([row["fitness_gain_vs_identity"] < 0.0 for row in results])),
            "identity_worse_task_ids": identity_worse_ids,
            "generation0_worse_rate": float(np.mean([
                _generation0_gain(row) < 0.0
                for row in results
            ])),
            "generation0_worse_task_ids": generation0_worse_ids,
            "stop_reasons": dict(Counter(str(row["stop_reason"]) for row in results)),
            "generations": _finite_stats(row["num_generations"] for row in results),
        },
        "runtime": {
            "elapsed_seconds": _finite_stats(row.get("elapsed_seconds", float("nan")) for row in results),
        },
        "evaluator": {
            "feature_count": _finite_stats(row.get("evaluator_feature_count", float("nan")) for row in results),
            "expected_feature_count": 93,
        },
        "genes": genes,
        "style_distance_by_group": style_groups,
        # Backwards-compatible concise view used by early E2 notebooks.
        "style_distance_gain_by_group": {
            group: values.get("gain", {"count": 0, "mean": None, "median": None, "std": None, "min": None, "max": None})
            for group, values in style_groups.items()
        },
        "e3_diagnostics": {
            "objective": [
                "Revisit the 42-feature Mahalanobis objective together with explicit content constraints; E1b is evaluation-only.",
                "Keep velocity/content terms explicit if they are intended to influence optimization, because velocity is absent from the historical fitness vector.",
            ],
            "genotype": [
                "Inspect the four global genes and their saturation rates before extending the genotype.",
                *([f"Boundary use is elevated for: {', '.join(boundary_genes)}."] if boundary_genes else ["No gene exceeds the 5% boundary-use diagnostic threshold."]),
            ],
            "operators": [
                "Use stagnation and generation-cost diagnostics to decide whether E3 needs new search operators or a richer representation.",
            ],
        },
        "directions": direction_analysis,
        "limitations": [
            "The E1b classifier is an independent style proxy, not proof of authorship attribution.",
            "The historical 42-feature fitness contains tempo-derived features; velocity_offset is not represented in that vector.",
            "Global onset and duration scaling can change length and rhythm substantially.",
            "Round-trip validity means successful parsing with preserved SMF metadata; overlapping same-pitch notes may not be FIFO-semantically identical (reported separately).",
            "Only repeat 0 of the predeclared E1 grouped split is used for the E2 transfer matrix.",
        ],
    }


def _write_plots(results: Sequence[Mapping[str, Any]], analysis: Mapping[str, Any], directory: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory.mkdir(parents=True, exist_ok=True)
    directions = sorted({_direction_key(row) for row in results})
    values = [[float(row["delta_p_target"]) for row in results if _direction_key(row) == direction] for direction in directions]
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.boxplot(values, labels=directions, showmeans=True)
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_ylabel("Δp target")
    axis.set_title("E2: zmiana prawdopodobieństwa kompozytora docelowego")
    figure.tight_layout()
    figure.savefig(directory / "style_gain_by_direction.png", dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 5))
    x = [float(row["content"]["length_error"]) for row in results]
    y = [float(row["delta_p_target"]) for row in results]
    axis.scatter(x, y, s=12, alpha=0.45)
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xlabel("Błąd długości bezwzględny")
    axis.set_ylabel("Δp target")
    axis.set_title("Zysk stylu a utrata długości")
    figure.tight_layout()
    figure.savefig(directory / "style_gain_vs_length_error.png", dpi=150)
    plt.close(figure)

    genes = ["transpose_semitones", "rhythm_density_factor", "note_duration_factor", "velocity_offset"]
    figure, axes = plt.subplots(1, 4, figsize=(12, 3.5))
    for axis, gene in zip(axes, genes):
        axis.hist([float(row["best_genome"][gene]) for row in results], bins=15)
        axis.set_title(gene.replace("_", " "))
    figure.suptitle("Końcowe genomy GA")
    figure.tight_layout()
    figure.savefig(directory / "genome_distributions.png", dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 5))
    axis.hist([float(row["content"]["length_error"]) for row in results], bins=20)
    axis.axvline(0.05, color="red", linestyle="--", label="5%")
    axis.set_xlabel("Błąd długości bezwzględny")
    axis.set_ylabel("Liczba zadań")
    axis.legend()
    axis.set_title("Rozkład błędu długości")
    figure.tight_layout()
    figure.savefig(directory / "length_error.png", dpi=150)
    plt.close(figure)

    # Boundary use and runtime/stagnation are protocol diagnostics rather than
    # additional objectives.  Keeping them in separate, deterministic plots
    # makes the report useful for deciding what E3 must redesign.
    genes = ["transpose_semitones", "rhythm_density_factor", "note_duration_factor", "velocity_offset"]
    lower = [float(analysis["genes"][gene]["at_lower_bound_rate"]) for gene in genes]
    upper = [float(analysis["genes"][gene]["at_upper_bound_rate"]) for gene in genes]
    x_positions = np.arange(len(genes))
    figure, axis = plt.subplots(figsize=(10, 4))
    axis.bar(x_positions - 0.18, lower, width=0.36, label="dolna granica")
    axis.bar(x_positions + 0.18, upper, width=0.36, label="górna granica")
    axis.set_xticks(x_positions, [gene.replace("_", " ") for gene in genes], rotation=20, ha="right")
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Udział zadań")
    axis.set_title("E2: częstość trafiania genów w granice")
    axis.legend()
    figure.tight_layout()
    figure.savefig(directory / "gene_boundary_rates.png", dpi=150)
    plt.close(figure)

    stop_reasons = analysis["fitness"]["stop_reasons"]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(list(stop_reasons), list(stop_reasons.values()))
    axes[0].set_title("Powody zatrzymania")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].set_ylabel("Liczba zadań")
    axes[1].scatter(
        [float(row["num_generations"]) for row in results],
        [float(row.get("elapsed_seconds", float("nan"))) for row in results],
        s=12,
        alpha=0.5,
    )
    axes[1].set_xlabel("Liczba pokoleń")
    axes[1].set_ylabel("Czas [s]")
    axes[1].set_title("Koszt czasowy a długość przebiegu")
    figure.tight_layout()
    figure.savefig(directory / "runtime_and_stagnation.png", dpi=150)
    plt.close(figure)


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float) and not np.isfinite(value):
        return "n/a"
    return f"{float(value):.{digits}f}" if isinstance(value, (float, int)) else str(value)


def _render_markdown_report(analysis: Mapping[str, Any], run_manifest: Mapping[str, Any], run_dir: Path) -> str:
    main = analysis["delta_p_target"]
    content = analysis["content"]
    lines = [
        "# Wyniki eksperymentu E2",
        "",
        f"Wygenerowano: {_utc_now()}",
        "",
        "## Decyzja",
        "",
        "E2 zakończony jako baseline; eksperyment nie ma bramki GO/NO-GO.",
        "",
        "## Wynik główny",
        "",
        f"Średnia Δp_target: **{_fmt(main['mean'])}**, mediana: **{_fmt(main['median'])}**, "
        f"95% CI klastrowane: **[{_fmt(main['ci95_clustered'][0])}, {_fmt(main['ci95_clustered'][1])}]**.",
        f"Dwustronny test permutacyjny znaku: p = **{_fmt(main['sign_permutation_p_value'], 4)}**.",
        f"Spadek prawdopodobieństwa źródła (średnia): {_fmt(analysis['source_probability_drop']['mean'])}; "
        f"trafienie klasy celu input/output: {_fmt(100 * analysis['p_target']['classification']['input_hit_rate'], 1)}% / "
        f"{_fmt(100 * analysis['p_target']['classification']['output_hit_rate'], 1)}%.",
        "",
        "## Zachowanie treści",
        "",
        f"Błąd długości (średnia): {_fmt(content['length_error']['mean'])}; "
        f"zadania w tolerancji 5%: {_fmt(100 * content['length_within_5_percent_rate'], 1)}%.",
        f"Onset-F1: {_fmt(content['onset_f1']['mean'])}; "
        f"podobieństwo trigramów konturu: {_fmt(content['melody_trigram_jaccard']['mean'])}.",
        f"Zmiana liczby nut (ratio): {_fmt(content['note_count_ratio']['mean'])}; "
        f"polifonia średnia/max output: {_fmt(content['mean_polyphony_output']['mean'])} / "
        f"{_fmt(content['max_polyphony_output']['mean'])}; "
        f"udział nut o zerowej długości: {_fmt(content['zero_duration_ratio_output']['mean'])}.",
        f"Korelacja Δp z błędem długości: {_fmt(analysis['content_tradeoff']['delta_vs_length_error'])}; "
        f"z utratą onset-F1: {_fmt(analysis['content_tradeoff']['delta_vs_onset_loss'])}.",
        f"Poprawny round-trip MIDI: {_fmt(100 * content['roundtrip_valid_rate'], 1)}%.",
        f"Ewaluator użył pełnego kontraktu 93 cech (średnio zachowanych po filtrze wariancji: "
        f"{_fmt(analysis['evaluator']['feature_count']['mean'], 1)}).",
        "",
        "## Kierunki transferu",
        "",
        "| Kierunek | N | Śr. Δp | CI 95% | p Holma |",
        "|---|---:|---:|---:|---:|",
    ]
    for direction, values in analysis["directions"].items():
        ci = values["delta_p_target_ci95_clustered"]
        lines.append(
            f"| {direction} | {values['task_count']} | {_fmt(values['delta_p_target']['mean'])} | "
            f"[{_fmt(ci[0])}, {_fmt(ci[1])}] | {_fmt(values['sign_permutation_p_value_holm'], 4)} |"
        )
    lines.extend([
        "",
        "## Standaryzowane odległości E1b",
        "",
        "| Grupa cech | Śr. przed | Śr. po | Śr. zysk (przed − po) |",
        "|---|---:|---:|---:|",
    ])
    for group, values in analysis.get("style_distance_by_group", {}).items():
        lines.append(
            f"| {group} | {_fmt(values.get('before', {}).get('mean'))} | "
            f"{_fmt(values.get('after', {}).get('mean'))} | "
            f"{_fmt(values.get('gain', {}).get('mean'))} |"
        )
    lines.extend([
        "",
        "## Diagnostyka GA",
        "",
        f"Zysk fitness względem identity: {_fmt(analysis['fitness']['gain_vs_identity']['mean'])}; "
        f"odsetek przypadków gorszych od identity: {_fmt(100 * analysis['fitness']['identity_worse_rate'], 1)}%.",
        f"Identyfikatory tych przypadków ({len(analysis['fitness']['identity_worse_task_ids'])}): "
        f"`{analysis['fitness']['identity_worse_task_ids']}`.",
        f"Powody stopu: `{analysis['fitness']['stop_reasons']}`.",
        f"Zysk względem najlepszego osobnika generacji 0: {_fmt(analysis['fitness']['gain_vs_generation0']['mean'])}; "
        f"przypadki pogorszenia: {_fmt(100 * analysis['fitness']['generation0_worse_rate'], 1)}%.",
        f"Czas zadania (średnia/mediana): {_fmt(analysis['runtime']['elapsed_seconds']['mean'])} s / "
        f"{_fmt(analysis['runtime']['elapsed_seconds']['median'])} s.",
        "Częstość trafienia genów w granice jest zapisana w `analysis.json` "
        "i na wykresie `plots/gene_boundary_rates.png`.",
        "",
        "## Ograniczenia",
        "",
    ])
    lines.extend(f"- {item}" for item in analysis["limitations"])
    diagnostics = analysis.get("e3_diagnostics", {})
    lines.extend([
        "",
        "## Wskazania dla E3",
        "",
    ])
    for category, items in diagnostics.items():
        lines.append(f"**{category}:**")
        lines.extend(f"- {item}" for item in items)
        lines.append("")
    lines.extend([
        "## Artefakty",
        "",
        f"- katalog przebiegu: `{run_dir}`",
        "- `results.json`, `results.csv`, `analysis.json` oraz `tasks/*/result.json`",
        "- `plots/style_gain_by_direction.png`, `plots/style_gain_vs_length_error.png`, `plots/genome_distributions.png`, `plots/length_error.png`, `plots/gene_boundary_rates.png`, `plots/runtime_and_stagnation.png`",
        "",
        f"Status run manifestu: `{run_manifest.get('status')}`.",
    ])
    return "\n".join(lines) + "\n"
