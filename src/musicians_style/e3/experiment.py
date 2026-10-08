"""Staged, resumable experiment harness for E3."""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline

from ..features.composition import extract_composition_features
from ..midi.parser import MidiParser
from ..midi.printer import MidiPrettyPrinter
from .algorithm import E3GeneticAlgorithm, SearchConfig
from .profile import TargetProfile, build_target_profile

SCHEMA_VERSION = "e3.0.0"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"timestamp_utc": _now(), **value}, ensure_ascii=False) + "\n")


def _resolve(value: Any) -> Path:
    path = Path(str(value))
    return path.resolve() if path.is_absolute() else (Path.cwd() / path).resolve()


def _hash_value(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class E3Config:
    manifest_path: Path
    splits_path: Path
    composition_features_path: Path
    dataset_root: Path
    run_dir: Path
    summary_path: Path
    search: SearchConfig = field(default_factory=SearchConfig)
    repeat: int = 0
    main_seed: int = 1729
    pilot_fold: int = 0
    pilot_seed: int = 1729
    workers: int = 1
    n_estimators: int = 300
    max_features: float = 0.5
    min_samples_leaf: int = 2
    random_state: int = 1729
    raw: dict[str, Any] = field(default_factory=dict, compare=False)


def load_e3_config(path: Path | str) -> E3Config:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping) or raw.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported or malformed E3 configuration")
    inputs, output = raw.get("inputs", {}), raw.get("output", {})
    ga, pilot, evaluator = raw.get("ga", {}), raw.get("pilot", {}), raw.get("evaluator", {})
    search = SearchConfig(
        population_size=int(ga.get("population_size", 32)), generations=int(ga.get("generations", 60)),
        elitism_k=int(ga.get("elitism_k", 2)), tournament_size=int(ga.get("tournament_size", 3)),
        stagnation_generations=int(ga.get("stagnation_generations", 12)),
        mutation_sigma=tuple(float(value) for value in ga.get("mutation_sigma", [1, .12, .12, .12, .12])),
    )
    if len(search.mutation_sigma) != 5:
        raise ValueError("E3 mutation_sigma must contain five values")
    config = E3Config(
        manifest_path=_resolve(inputs.get("manifest", "datasets/derived/e1_asap/manifest.json")),
        splits_path=_resolve(inputs.get("splits", "datasets/derived/e1_asap/splits.json")),
        composition_features_path=_resolve(inputs.get("composition_features", "datasets/derived/e1_asap/composition_features.json")),
        dataset_root=_resolve(inputs.get("dataset_root", "datasets/asap-dataset-1.2")),
        run_dir=_resolve(output.get("run_dir", "experiments/e3_asap")),
        summary_path=_resolve(output.get("summary", "docs/results/E3.md")), search=search,
        repeat=int(raw.get("repeat", 0)), main_seed=int(raw.get("main_seed", 1729)),
        pilot_fold=int(pilot.get("fold", 0)), pilot_seed=int(pilot.get("seed", 1729)),
        workers=int(raw.get("workers", 1)), n_estimators=int(evaluator.get("n_estimators", 300)),
        max_features=float(evaluator.get("max_features", .5)), min_samples_leaf=int(evaluator.get("min_samples_leaf", 2)),
        random_state=int(evaluator.get("random_state", 1729)), raw=dict(raw),
    )
    if config.workers < 1 or config.n_estimators < 1:
        raise ValueError("workers and n_estimators must be positive")
    return config


def _task_id(phase: str, repeat: int, fold: int, source: str, target: str, seed: int) -> str:
    return f"{phase}__r{repeat:02d}__f{fold:02d}__{source}__to__{target}__s{seed}"


class E3Experiment:
    def __init__(self, config: E3Config, *, run_dir: Path | str | None = None, workers: int | None = None) -> None:
        self.config = config
        self.run_dir = Path(run_dir).resolve() if run_dir else config.run_dir
        self.workers = config.workers if workers is None else int(workers)
        self.runtime_overrides = {
            "workers": workers is not None,
            "run_dir": run_dir is not None,
        }
        if self.workers < 1:
            raise ValueError("workers must be positive")
        hash_payload = config.raw or {
            key: value for key, value in asdict(config).items() if key != "raw"
        }
        self.config_hash = _hash_value(hash_payload)

    def _effective_runtime(self) -> dict[str, Any]:
        return {
            "workers": self.workers,
            "workers_source": "cli" if self.runtime_overrides["workers"] else "config",
            "run_dir": str(self.run_dir),
            "run_dir_source": "cli" if self.runtime_overrides["run_dir"] else "config",
        }

    @property
    def task_manifest_path(self) -> Path:
        return self.run_dir / "task_manifest.json"

    def _inputs(self) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        manifest = json.loads(self.config.manifest_path.read_text(encoding="utf-8"))
        splits = json.loads(self.config.splits_path.read_text(encoding="utf-8"))
        features = json.loads(self.config.composition_features_path.read_text(encoding="utf-8"))
        accepted = [row for row in manifest.get("samples", []) if row.get("validation_status") == "accepted"]
        ids = {str(row["sample_id"]) for row in accepted}
        if len(accepted) != 150 or {str(row["sample_id"]) for row in features.get("samples", [])} != ids:
            raise ValueError("E3 requires the canonical 150-sample E1 input contract")
        return manifest, splits, features

    def prepare(self) -> Path:
        manifest, splits, _ = self._inputs()
        samples = {str(row["sample_id"]): row for row in manifest["samples"] if row.get("validation_status") == "accepted"}
        repetition = next((item for item in splits["repetitions"] if int(item["repeat"]) == self.config.repeat), None)
        if repetition is None:
            raise ValueError(f"split repeat {self.config.repeat} is missing")
        tasks: list[dict[str, Any]] = []
        for fold in repetition["folds"]:
            train_ids, test_ids = list(fold["train"]["sample_ids"]), list(fold["test"]["sample_ids"])
            if set(train_ids) & set(test_ids):
                raise ValueError("train/test leakage")
            targets = sorted({str(samples[item]["composer"]) for item in train_ids})
            for source_id in sorted(test_ids):
                source_composer = str(samples[source_id]["composer"])
                for target in targets:
                    if target == source_composer:
                        continue
                    task = {"phase": "full", "repeat": self.config.repeat, "fold": int(fold["fold"]),
                            "source_id": source_id, "source_composer": source_composer,
                            "target_composer": target, "target_train_ids": sorted(item for item in train_ids if samples[item]["composer"] == target),
                            "seed": self.config.main_seed}
                    task["task_id"] = _task_id("full", self.config.repeat, int(fold["fold"]), source_id, target, self.config.main_seed)
                    tasks.append(task)
        full = [task for task in tasks if task["phase"] == "full"]
        if len(full) != 300:
            raise ValueError(f"E3 full matrix must contain 300 tasks, found {len(full)}")
        pilot_fold = next(item for item in repetition["folds"] if int(item["fold"]) == self.config.pilot_fold)
        pilot_test = list(pilot_fold["test"]["sample_ids"])
        representatives = {}
        for composer in sorted({samples[item]["composer"] for item in pilot_test}):
            candidates = [item for item in pilot_test if samples[item]["composer"] == composer]
            median = float(np.median([samples[item]["note_count"] for item in candidates]))
            representatives[composer] = min(candidates, key=lambda item: (abs(samples[item]["note_count"] - median), item))
        pilot_train = list(pilot_fold["train"]["sample_ids"])
        pilot_tasks = []
        for source_composer, source_id in sorted(representatives.items()):
            for target in sorted({samples[item]["composer"] for item in pilot_train} - {source_composer}):
                task = {"phase": "pilot", "repeat": self.config.repeat, "fold": self.config.pilot_fold,
                        "source_id": source_id, "source_composer": source_composer, "target_composer": target,
                        "target_train_ids": sorted(item for item in pilot_train if samples[item]["composer"] == target),
                        "seed": self.config.pilot_seed}
                task["task_id"] = _task_id("pilot", self.config.repeat, self.config.pilot_fold, source_id, target, self.config.pilot_seed)
                pilot_tasks.append(task)
        if len(pilot_tasks) != 6:
            raise ValueError("E3 pilot must cover six directions")
        pilot_tasks[0]["determinism_check"] = True
        all_tasks = pilot_tasks + full
        for task in all_tasks:
            task["task_signature"] = _hash_value({"config": self.config_hash, "task": task})
        payload = {"schema_version": SCHEMA_VERSION, "created_at_utc": _now(), "config_hash": self.config_hash,
                   "counts": {"pilot": 6, "full": 300}, "representatives": representatives,
                   "tasks": sorted(all_tasks, key=lambda item: item["task_id"])}
        self.run_dir.mkdir(parents=True, exist_ok=True)
        snapshots = self.run_dir / "inputs"
        snapshots.mkdir(exist_ok=True)
        for path in (self.config.manifest_path, self.config.splits_path, self.config.composition_features_path):
            destination = snapshots / path.name
            if destination.exists() and _sha256(destination) != _sha256(path):
                raise ValueError(f"input snapshot changed: {path.name}; use a new run directory")
            if not destination.exists():
                shutil.copy2(path, destination)
        config_used = self.run_dir / "config_used.yaml"
        if not config_used.exists():
            _atomic_text(config_used, yaml.safe_dump(self.config.raw, sort_keys=False, allow_unicode=True))
        if self.task_manifest_path.exists():
            old = json.loads(self.task_manifest_path.read_text(encoding="utf-8"))
            if old.get("config_hash") != self.config_hash:
                raise ValueError("prepared E3 configuration differs; use a new run directory")
        else:
            _atomic_json(self.task_manifest_path, payload)
        run_manifest = self.run_dir / "run_manifest.json"
        if not run_manifest.exists():
            _atomic_json(run_manifest, {"schema_version": SCHEMA_VERSION, "status": "prepared", "created_at_utc": _now(),
                                        "config_hash": self.config_hash, "task_counts": payload["counts"],
                                        "effective_runtime": self._effective_runtime()})
        else:
            run_payload = json.loads(run_manifest.read_text(encoding="utf-8"))
            run_payload["effective_runtime"] = self._effective_runtime()
            _atomic_json(run_manifest, run_payload)
        return self.task_manifest_path

    def ensure_prepared(self) -> None:
        self.prepare()

    def _worker_payload(self) -> dict[str, Any]:
        return {"run_dir": str(self.run_dir), "dataset_root": str(self.config.dataset_root),
                "manifest_path": str(self.run_dir / "inputs" / self.config.manifest_path.name),
                "splits_path": str(self.run_dir / "inputs" / self.config.splits_path.name),
                "composition_features_path": str(self.run_dir / "inputs" / self.config.composition_features_path.name),
                "repeat": self.config.repeat, "search": asdict(self.config.search),
                "n_estimators": self.config.n_estimators, "max_features": self.config.max_features,
                "min_samples_leaf": self.config.min_samples_leaf, "random_state": self.config.random_state}

    def run_stage(self, phase: str) -> Path:
        if phase not in {"pilot", "full"}:
            raise ValueError("phase must be pilot or full")
        self.ensure_prepared()
        if phase == "full":
            pilot_path = self.run_dir / "pilot_results.json"
            if not pilot_path.exists():
                raise ValueError("successful E3 pilot is required before the full run")
            pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
            checked = [row for row in pilot.get("results", []) if row.get("determinism", {}).get("checked")]
            if pilot.get("completed") != 6 or pilot.get("failed") or len(checked) != 1 or not checked[0]["determinism"].get("equal"):
                raise ValueError("E3 pilot gate is not satisfied")
        manifest = json.loads(self.task_manifest_path.read_text(encoding="utf-8"))
        tasks = [task for task in manifest["tasks"] if task["phase"] == phase]
        pending = [task for task in tasks if not _valid_result(self.run_dir, task)]
        results = [json.loads((self.run_dir / "tasks" / task["task_id"] / "result.json").read_text(encoding="utf-8")) for task in tasks if task not in pending]
        started = time.perf_counter()
        progress_path = self.run_dir / "progress.jsonl"
        _append_jsonl(progress_path, {"event": "stage_started", "phase": phase, "pending": len(pending), "resumed": len(results), "total": len(tasks), **self._effective_runtime()})
        if pending and self.workers == 1:
            worker = _Worker(self._worker_payload())
            for task in pending:
                row = worker.execute(task)
                results.append(row)
                _append_jsonl(progress_path, {"event": "task_completed", "phase": phase, "task_id": task["task_id"], "status": row.get("status")})
        elif pending:
            with ProcessPoolExecutor(max_workers=self.workers, initializer=_init_worker, initargs=(self._worker_payload(),)) as pool:
                futures = {pool.submit(_execute_task, task): task for task in pending}
                for future in as_completed(futures):
                    row = future.result()
                    results.append(row)
                    _append_jsonl(progress_path, {"event": "task_completed", "phase": phase, "task_id": row.get("task_id"), "status": row.get("status")})
        failures = [row for row in results if row.get("status") != "completed"]
        aggregate = {"schema_version": SCHEMA_VERSION, "phase": phase, "completed_at_utc": _now(),
                     "elapsed_seconds_this_invocation": time.perf_counter() - started,
                     "completed": len(results) - len(failures), "failed": len(failures), "results": sorted(results, key=lambda row: row["task_id"])}
        output = self.run_dir / ("pilot_results.json" if phase == "pilot" else "results.json")
        _atomic_json(output, aggregate)
        _append_jsonl(progress_path, {"event": "stage_completed", "phase": phase, "completed": aggregate["completed"], "failed": aggregate["failed"]})
        if failures:
            raise RuntimeError(f"{len(failures)} E3 {phase} tasks failed; see {output}")
        if phase == "pilot":
            checked = [row for row in results if row.get("determinism", {}).get("checked")]
            if len(results) != 6 or len(checked) != 1 or not checked[0]["determinism"].get("equal"):
                raise RuntimeError("E3 pilot gate failed")
            run_manifest_path = self.run_dir / "run_manifest.json"
            run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
            seconds = sum(float(row["elapsed_seconds"]) for row in results) / len(results)
            run_manifest.update({"status": "pilot_completed", "pilot_completed_at_utc": _now(),
                                 "estimated_full_seconds_single_worker": seconds * 300,
                                 "effective_runtime": self._effective_runtime()})
            _atomic_json(run_manifest_path, run_manifest)
        else:
            run_manifest_path = self.run_dir / "run_manifest.json"
            run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
            run_manifest.update({"status": "completed", "full_completed_at_utc": _now(), "full_results_sha256": _sha256(output),
                                 "effective_runtime": self._effective_runtime()})
            _atomic_json(run_manifest_path, run_manifest)
        return output

    def write_report(self, e2_run_dir: Path | str | None = None) -> Path:
        """Close E3 from frozen outputs; no GA task is executed here."""
        from .reporting import build_closure_report

        results_path = self.run_dir / "results.json"
        if not results_path.exists():
            raise ValueError("full E3 results are required before report")
        rows = json.loads(results_path.read_text(encoding="utf-8"))["results"]
        text, analysis = build_closure_report(self.config, self.run_dir, rows, e2_run_dir)
        _atomic_json(self.run_dir / "closure_analysis.json", analysis)
        _atomic_text(self.config.summary_path, text)
        return self.config.summary_path


def _group_means(rows: Sequence[Mapping[str, Any]], value) -> np.ndarray:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        grouped.setdefault(str(row["source_group_id"]), []).append(float(value(row)))
    return np.asarray([np.mean(grouped[key]) for key in sorted(grouped)], dtype=float)


def _bootstrap_group_ci(values: np.ndarray, *, seed: int, samples: int = 2000) -> tuple[float, float]:
    if not values.size:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    estimates = [float(rng.choice(values, size=len(values), replace=True).mean()) for _ in range(samples)]
    return float(np.quantile(estimates, .025)), float(np.quantile(estimates, .975))


def _sign_permutation_p(values: np.ndarray, *, seed: int, permutations: int = 9999) -> float:
    if not values.size:
        return 1.0
    observed = abs(float(values.mean()))
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(permutations):
        signs = rng.choice((-1.0, 1.0), size=len(values))
        exceed += abs(float(np.mean(values * signs))) >= observed
    return float((exceed + 1) / (permutations + 1))


def _holm(values: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted(values, key=values.get)
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for rank, key in enumerate(ordered):
        running = max(running, min(1.0, (count - rank) * float(values[key])))
        adjusted[key] = running
    return adjusted


def _valid_result(run_dir: Path, task: Mapping[str, Any]) -> bool:
    path = run_dir / "tasks" / str(task["task_id"]) / "result.json"
    if not path.exists():
        return False
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
        output = run_dir / str(row["output_path"])
        return row.get("status") == "completed" and row.get("task_signature") == task.get("task_signature") and output.is_file() and _sha256(output) == row.get("output_sha256")
    except (OSError, KeyError, json.JSONDecodeError):
        return False


_WORKER: "_Worker | None" = None


def _init_worker(payload: Mapping[str, Any]) -> None:
    global _WORKER
    _WORKER = _Worker(payload)


def _execute_task(task: Mapping[str, Any]) -> dict[str, Any]:
    if _WORKER is None:
        raise RuntimeError("E3 worker not initialized")
    return _WORKER.execute(task)


class _Worker:
    def __init__(self, payload: Mapping[str, Any]) -> None:
        self.run_dir, self.dataset_root = Path(payload["run_dir"]), Path(payload["dataset_root"])
        manifest = json.loads(Path(payload["manifest_path"]).read_text(encoding="utf-8"))
        splits = json.loads(Path(payload["splits_path"]).read_text(encoding="utf-8"))
        features = json.loads(Path(payload["composition_features_path"]).read_text(encoding="utf-8"))
        self.samples = {str(row["sample_id"]): row for row in manifest["samples"] if row.get("validation_status") == "accepted"}
        self.feature_rows = {str(row["sample_id"]): row for row in features["samples"]}
        self.repeat = next(item for item in splits["repetitions"] if int(item["repeat"]) == int(payload["repeat"]))
        self.search = SearchConfig(**payload["search"])
        self.parser, self.printer = MidiParser(), MidiPrettyPrinter()
        self.reprs: dict[str, Any] = {}
        self.profiles: dict[tuple[int, str], TargetProfile] = {}
        self.classifiers: dict[int, Any] = {}
        for fold in self.repeat["folds"]:
            fold_index, train_ids = int(fold["fold"]), list(fold["train"]["sample_ids"])
            matrix = np.asarray([self.feature_rows[item]["values"] for item in train_ids])
            labels = np.asarray([self.samples[item]["composer"] for item in train_ids])
            self.classifiers[fold_index] = Pipeline([("variance", VarianceThreshold()), ("model", RandomForestClassifier(
                n_estimators=int(payload["n_estimators"]), class_weight="balanced", max_features=float(payload["max_features"]),
                min_samples_leaf=int(payload["min_samples_leaf"]), random_state=int(payload["random_state"]), n_jobs=1))]).fit(matrix, labels)

    def _repr(self, sample_id: str):
        if sample_id not in self.reprs:
            self.reprs[sample_id] = self.parser.parse(self.dataset_root / self.samples[sample_id]["score_path"])
        return self.reprs[sample_id]

    def _profile(self, task: Mapping[str, Any]) -> TargetProfile:
        key = (int(task["fold"]), str(task["target_composer"]))
        if key not in self.profiles:
            ids = [str(value) for value in task["target_train_ids"]]
            train_rows = [self.samples[value] for value in ids]
            fold = next(item for item in self.repeat["folds"] if int(item["fold"]) == key[0])
            forbidden = [self.samples[value] for value in fold["test"]["sample_ids"]]
            self.profiles[key] = build_target_profile(key[1], train_rows, {value: self._repr(value) for value in ids}, forbidden_rows=forbidden)
        return self.profiles[key]

    def execute(self, task: Mapping[str, Any]) -> dict[str, Any]:
        task_dir = self.run_dir / "tasks" / str(task["task_id"])
        started = time.perf_counter()
        try:
            source, profile = self._repr(str(task["source_id"])), self._profile(task)
            history_lines = []
            result = E3GeneticAlgorithm(self.search).run(source, profile, seed=int(task["seed"]), progress=history_lines.append)
            determinism = {"checked": False}
            if task.get("determinism_check"):
                repeated = E3GeneticAlgorithm(self.search).run(source, profile, seed=int(task["seed"]))
                determinism = {"checked": True, "equal": repeated.genome == result.genome and repeated.output == result.output and repeated.history == result.history}
                if not determinism["equal"]:
                    raise RuntimeError("same-seed result is not deterministic")
            task_dir.mkdir(parents=True, exist_ok=True)
            output_path = task_dir / "output.mid"
            temporary = task_dir / "output.mid.tmp"
            self.printer.write(result.output, temporary)
            temporary.replace(output_path)
            roundtrip = self.parser.parse(output_path)
            input_values = np.asarray(self.feature_rows[str(task["source_id"])]["values"], dtype=float)
            output_values = extract_composition_features(roundtrip)
            classifier = self.classifiers[int(task["fold"])]
            classes = [str(value) for value in classifier.classes_]
            target_index = classes.index(str(task["target_composer"]))
            p_input = float(classifier.predict_proba(input_values.reshape(1, -1))[0][target_index])
            p_output = float(classifier.predict_proba(output_values.reshape(1, -1))[0][target_index])
            row = {"schema_version": SCHEMA_VERSION, "status": "completed", "task_id": task["task_id"],
                   "task_signature": task["task_signature"], "phase": task["phase"], "fold": task["fold"],
                   "source_id": task["source_id"], "source_composer": task["source_composer"], "target_composer": task["target_composer"],
                   "source_group_id": self.samples[str(task["source_id"])]["group_id"], "seed": task["seed"],
                   "elapsed_seconds": time.perf_counter() - started, "profile_fingerprint": profile.fingerprint,
                   "profile_train_ids": list(profile.train_sample_ids), "best_genome": asdict(result.genome),
                   "style_gain": result.evaluation.style_gain, "group_gains": result.evaluation.group_gains,
                   "constraints": asdict(result.evaluation.constraints), "stop_reason": result.stop_reason,
                   "unique_candidates": result.unique_candidates, "cache_hits": result.cache_hits,
                   "history": list(result.history), "determinism": determinism,
                   "p_target_input": p_input, "p_target_output": p_output, "delta_p_target": p_output - p_input,
                   "output_path": str(output_path.relative_to(self.run_dir)), "output_sha256": _sha256(output_path)}
            _atomic_json(task_dir / "result.json", row)
            _atomic_text(task_dir / "ga.jsonl", "".join(json.dumps(value, ensure_ascii=False) + "\n" for value in history_lines))
            return row
        except Exception as exc:
            row = {"schema_version": SCHEMA_VERSION, "status": "failed", "task_id": task["task_id"],
                   "task_signature": task.get("task_signature"), "phase": task.get("phase"),
                   "elapsed_seconds": time.perf_counter() - started, "error_type": type(exc).__name__, "error": str(exc)}
            _atomic_json(task_dir / "result.json", row)
            return row
