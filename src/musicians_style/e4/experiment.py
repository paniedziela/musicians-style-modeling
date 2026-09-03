"""Reproducible E4 harness from data audit to a guarded outer test.

E4 is deliberately a decision experiment, not a hyper-parameter search.  The
validation split selects the checkpoint and decoding thresholds.  The outer
test is exposed as a separate, one-way stage and refuses to run before the
validation gate has passed.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import subprocess
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

from ..midi.parser import MidiParser
from ..midi.printer import MidiPrettyPrinter
from .dataset import COMPOSERS, BalancedComposerSampler, PieceSegments, SegmentDataset, assert_split_isolation
from .evaluation import FoldStyleEvaluator, calibrate_thresholds, summarize_records
from .model import ConditionalGenerator, PatchDiscriminator
from .segmentation import PITCH_HIGH, PITCH_LOW, STEPS_PER_BAR, Segment, encode_piece, quantization_errors, stitch_segments
from .training import gan_step

SCHEMA_VERSION = "e4.5.0"


@dataclass(frozen=True)
class E4Config:
    manifest_path: Path | None
    splits_path: Path | None
    dataset_root: Path | None
    run_dir: Path | None
    composition_features_path: Path | None = None
    summary_path: Path | None = None
    repeat: int = 0
    outer_fold: int = 0
    inner_fold: int = 0
    seed: int = 1729
    batch_size: int = 16
    max_epochs: int = 30
    patience: int = 5
    learning_rate: float = .0002
    beta1: float = .5
    beta2: float = .999
    lambda_cls: float = 1.0
    lambda_cyc: float = 10.0
    lambda_id: float = 10.0
    smoke_epochs: int = 2
    smoke_per_composer: int = 4
    smoke_files: int = 6
    conv_dim: int = 32
    residual_blocks: int = 3
    device: str = "auto"
    threshold_grid: tuple[float, ...] = tuple(value / 10 for value in range(1, 10))
    melody_similarity_min: float = .95
    fallback_rate_max: float = .01
    onset_tolerance_beats: float = 1 / 16
    min_bar_quarter_notes: float = .5
    max_bar_quarter_notes: float = 12.0
    allow_terminal_partial_bar: bool = True


def _resolve(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (Path.cwd() / path).resolve()


def load_e4_config(path: Path | str) -> E4Config:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping) or raw.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported E4 config; expected {SCHEMA_VERSION}")
    inputs, output = raw.get("inputs", {}), raw.get("output", {})
    training, smoke = raw.get("training", {}), raw.get("smoke", {})
    model, decoding, gates, audit = (raw.get(name, {}) for name in ("model", "decoding", "gates", "audit"))
    grid = tuple(float(value) for value in decoding.get("threshold_grid", [value / 10 for value in range(1, 10)]))
    if not grid or any(not 0 < value < 1 for value in grid):
        raise ValueError("decoding.threshold_grid must contain probabilities between zero and one")
    return E4Config(
        manifest_path=_resolve(str(inputs["manifest"])),
        splits_path=_resolve(str(inputs["splits"])),
        dataset_root=_resolve(str(inputs["dataset_root"])),
        run_dir=_resolve(str(output["run_dir"])),
        composition_features_path=_resolve(str(inputs["composition_features"])),
        summary_path=_resolve(str(output.get("summary", "docs/results/E4.md"))),
        repeat=int(raw.get("repeat", 0)), outer_fold=int(raw.get("outer_fold", 0)),
        inner_fold=int(raw.get("inner_fold", 0)), seed=int(raw.get("seed", 1729)),
        batch_size=int(training.get("batch_size", 16)), max_epochs=int(training.get("max_epochs", 30)),
        patience=int(training.get("patience", 5)), learning_rate=float(training.get("learning_rate", .0002)),
        beta1=float(training.get("beta1", .5)), beta2=float(training.get("beta2", .999)),
        lambda_cls=float(training.get("lambda_cls", 1)), lambda_cyc=float(training.get("lambda_cyc", 10)),
        lambda_id=float(training.get("lambda_id", 10)), smoke_epochs=int(smoke.get("epochs", 2)),
        smoke_per_composer=int(smoke.get("per_composer", 4)), smoke_files=int(smoke.get("files", 6)),
        conv_dim=int(model.get("conv_dim", 32)), residual_blocks=int(model.get("residual_blocks", 3)),
        device=str(training.get("device", "auto")), threshold_grid=grid,
        melody_similarity_min=float(gates.get("melody_similarity_min", .95)),
        fallback_rate_max=float(gates.get("fallback_rate_max", .01)),
        onset_tolerance_beats=float(gates.get("onset_tolerance_beats", 1 / 16)),
        min_bar_quarter_notes=float(audit.get("min_bar_quarter_notes", .5)),
        max_bar_quarter_notes=float(audit.get("max_bar_quarter_notes", 12)),
        allow_terminal_partial_bar=bool(audit.get("allow_terminal_partial_bar", True)),
    )


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _protocol(splits: Mapping[str, Any], config: E4Config) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    repeat = next((item for item in splits["repetitions"] if int(item["repeat"]) == config.repeat), None)
    outer = next((item for item in (repeat or {}).get("folds", []) if int(item["fold"]) == config.outer_fold), None)
    inner = next((item for item in (outer or {}).get("inner_folds", []) if int(item["fold"]) == config.inner_fold), None)
    if not repeat or not outer or not inner:
        raise ValueError("requested E4 repeat/fold is missing")
    return outer, inner


def _split_ids(splits: Mapping[str, Any], config: E4Config) -> dict[str, set[str]]:
    outer, inner = _protocol(splits, config)
    result = {
        "train": set(inner["train"]["sample_ids"]),
        "validation": set(inner["validation"]["sample_ids"]),
        "test": set(outer["test"]["sample_ids"]),
    }
    if any(result[left] & result[right] for left in result for right in result if left != right):
        raise ValueError("sample_id leakage between E4 splits")
    return result


def _inputs(config: E4Config) -> tuple[dict[str, Any], dict[str, set[str]], dict[str, dict[str, Any]], dict[str, Any]]:
    if not config.manifest_path or not config.splits_path or not config.composition_features_path:
        raise ValueError("E4 input paths are required")
    manifest = json.loads(config.manifest_path.read_text(encoding="utf-8"))
    splits = json.loads(config.splits_path.read_text(encoding="utf-8"))
    composition = json.loads(config.composition_features_path.read_text(encoding="utf-8"))
    partitions = _split_ids(splits, config)
    rows = {str(item["sample_id"]): item for item in manifest.get("samples", []) if item.get("validation_status") == "accepted"}
    if set().union(*partitions.values()) != set(rows):
        raise ValueError("E4 splits do not partition accepted manifest samples")
    groups = {name: {str(rows[item]["group_id"]) for item in ids} for name, ids in partitions.items()}
    if any(groups[left] & groups[right] for left in groups for right in groups if left != right):
        raise ValueError("group_id leakage between E4 splits")
    if {str(item["sample_id"]) for item in composition.get("samples", [])} != set(rows):
        raise ValueError("E4 composition feature cache does not match the manifest")
    return manifest, partitions, rows, composition


def _geometry(config: E4Config) -> dict[str, Any]:
    return {
        "bars_per_segment": 4, "steps_per_bar": STEPS_PER_BAR, "steps_per_segment": 64,
        "pitch_low": PITCH_LOW, "pitch_high": PITCH_HIGH, "channels": ["onset", "frame"],
        "composers": list(COMPOSERS), "conv_dim": config.conv_dim, "residual_blocks": config.residual_blocks,
    }


def audit_e4(config: E4Config) -> Path:
    _, partitions, rows, _ = _inputs(config)
    if not config.run_dir or not config.dataset_root:
        raise ValueError("E4 output and dataset paths are required")
    parser = MidiParser()
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION, "geometry": _geometry(config), "splits": {},
        "rejections": [], "exclusions": [], "unusual_bars": [], "resolved_terminal_partial_bars": [],
    }
    pitch_values: list[int] = []
    quantization_passed = True
    for split, ids in partitions.items():
        counters: dict[str, Counter[str]] = defaultdict(Counter)
        errors: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
        for sample_id in sorted(ids):
            row, exclusion = rows[sample_id], None
            try:
                source = parser.parse(config.dataset_root / str(row["score_path"]))
                mapped = encode_piece(source)
            except Exception as exc:
                report["rejections"].append({"sample_id": sample_id, "split": split, "reason": f"parser_or_segmentation: {exc}"})
                continue
            for bar in mapped.bars:
                length = bar.length_ticks / source.ticks_per_beat
                if config.min_bar_quarter_notes <= length <= config.max_bar_quarter_notes:
                    continue
                record = {"sample_id": sample_id, "split": split, "bar": bar.index, "length_quarter_notes": length, "meter": f"{bar.numerator}/{bar.denominator}"}
                report["unusual_bars"].append(record)
                terminal_partial = bar.index == len(mapped.bars) - 1 and length < config.min_bar_quarter_notes
                if terminal_partial and config.allow_terminal_partial_bar:
                    report["resolved_terminal_partial_bars"].append(record)
                else:
                    exclusion = "bar_length_outside_frozen_range"
            if exclusion:
                report["exclusions"].append({"sample_id": sample_id, "split": split, "reason": exclusion})
                continue
            composer = str(row["composer"])
            values = counters[composer]
            values["pieces"] += 1
            values["bars"] += len(mapped.bars)
            values["segments"] += len(mapped.segments)
            values["empty_segments"] += sum(not segment.nonempty for segment in mapped.segments)
            values["nonempty_segments"] += sum(segment.nonempty for segment in mapped.segments)
            values["active_cells"] += sum(int(segment.data[1].sum()) for segment in mapped.segments)
            values["onset_cells"] += sum(int(segment.data[0].sum()) for segment in mapped.segments)
            for bar in mapped.bars:
                values[f"meter:{bar.numerator}/{bar.denominator}"] += 1
            for note in source.notes:
                pitch_values.append(note.pitch)
                values["outside_pitch"] += int(not PITCH_LOW <= note.pitch < PITCH_HIGH)
            for meter, onset_error, end_error, onset_step, end_step in quantization_errors(mapped):
                errors[meter].append((
                    abs(onset_error), abs(end_error), onset_step, end_step,
                ))
        per_composer = {composer: dict(values) for composer, values in sorted(counters.items())}
        for values in per_composer.values():
            steps = max(1, values["segments"] * 64)
            values["occupancy"] = values["active_cells"] / (steps * 84)
            values["onset_density"] = values["onset_cells"] / steps
        report["splits"][split] = per_composer
        quantization: dict[str, Any] = {}
        for meter, values in sorted(errors.items()):
            onset = sorted(item[0] for item in values)
            onset_ratios = sorted(item[0] / item[2] for item in values if item[2] > 0)
            end_ratios = [item[1] / item[3] for item in values if item[3] > 0]
            onset_p95 = onset[int(.95 * (len(onset) - 1))]
            end_max = max(item[1] for item in values)
            onset_ratio_p95 = onset_ratios[int(.95 * (len(onset_ratios) - 1))]
            end_ratio_max = max(end_ratios)
            passed = onset_ratio_p95 <= .5 + 1e-9 and end_ratio_max <= 1 + 1e-9
            quantization_passed &= passed
            quantization[meter] = {
                "count": len(values), "onset_abs_p95_ticks": onset_p95, "end_abs_max_ticks": end_max,
                "onset_error_local_steps_p95": onset_ratio_p95,
                "end_error_local_steps_max": end_ratio_max,
                "local_step_ticks_min": min(min(item[2], item[3]) for item in values),
                "local_step_ticks_max": max(max(item[2], item[3]) for item in values),
                "passed": passed,
            }
        report.setdefault("quantization", {})[split] = quantization
    enough = all(report["splits"]["train"].get(composer, {}).get("nonempty_segments", 0) >= 100 for composer in COMPOSERS)
    report["pitch"] = {"min": min(pitch_values, default=None), "max": max(pitch_values, default=None)}
    report["gate"] = {
        "group_isolation": True, "no_parser_errors": not report["rejections"],
        "at_least_100_train_nonempty_per_composer": enough, "quantization_within_limits": quantization_passed,
    }
    report["gate"]["passed"] = all(report["gate"].values())
    target = config.run_dir / "audit.json"
    _atomic_json(target, report)
    return target


def _eligible_partitions(config: E4Config, partitions: Mapping[str, set[str]]) -> dict[str, set[str]]:
    if not config.run_dir:
        raise ValueError("run_dir is required")
    audit_path = config.run_dir / "audit.json"
    if not audit_path.is_file():
        audit_e4(config)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    excluded = {str(row["sample_id"]) for row in audit.get("exclusions", []) + audit.get("rejections", [])}
    return {name: set(ids) - excluded for name, ids in partitions.items()}


def _pieces(config: E4Config, partitions: Mapping[str, set[str]], rows: Mapping[str, Mapping[str, Any]]) -> dict[str, list[PieceSegments]]:
    if not config.dataset_root:
        raise ValueError("dataset_root is required")
    parser, result = MidiParser(), {name: [] for name in partitions}
    membership = {sample: split for split, samples in partitions.items() for sample in samples}
    for sample_id in sorted(membership):
        row, split = rows[sample_id], membership[sample_id]
        source = parser.parse(config.dataset_root / str(row["score_path"]))
        result[split].append(PieceSegments(sample_id, str(row["group_id"]), str(row["composer"]), split, encode_piece(source)))
    return result


def prepare_e4(config: E4Config) -> Path:
    if not config.run_dir or not config.manifest_path or not config.splits_path or not config.composition_features_path:
        raise ValueError("E4 paths are required")
    # The full-corpus audit is intentionally reusable inside one versioned run
    # directory.  Remove audit.json explicitly when changing its frozen policy.
    audit_path = config.run_dir / "audit.json"
    if not audit_path.is_file():
        audit_path = audit_e4(config)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if not audit["gate"]["passed"]:
        raise ValueError("E4 audit gate failed; refusing to prepare training")
    _, partitions, rows, _ = _inputs(config)
    partitions = _eligible_partitions(config, partitions)
    pieces = _pieces(config, partitions, rows)
    config.run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(config.manifest_path, config.run_dir / "manifest.snapshot.json")
    shutil.copyfile(config.splits_path, config.run_dir / "splits.snapshot.json")
    shutil.copyfile(config.composition_features_path, config.run_dir / "composition_features.snapshot.json")
    snapshot = {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()}
    _atomic_json(config.run_dir / "config.snapshot.json", snapshot)
    segment_path = config.run_dir / "segments.jsonl"
    segment_path.unlink(missing_ok=True)
    for split, collection in pieces.items():
        for item in collection:
            _append_jsonl(segment_path, {
                "sample_id": item.sample_id, "group_id": item.group_id, "composer": item.composer, "split": split,
                "segments": len(item.segment_map.segments),
                "nonempty_segments": sum(segment.nonempty for segment in item.segment_map.segments),
                "geometry": _geometry(config),
            })
    _atomic_json(config.run_dir / "status.json", {
        "schema_version": SCHEMA_VERSION, "status": "prepared", "prepared_at_utc": _utc(),
        "seed": config.seed, "commit": _commit(), "geometry": _geometry(config),
    })
    _append_jsonl(config.run_dir / "progress.jsonl", {"event": "prepared", "at_utc": _utc()})
    return segment_path


def _device(config: E4Config) -> torch.device:
    if config.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(config.device)


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _atomic_checkpoint(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def _checkpoint(
    config: E4Config,
    epoch: int,
    generator: ConditionalGenerator,
    discriminator: PatchDiscriminator,
    generator_optimizer: torch.optim.Optimizer,
    discriminator_optimizer: torch.optim.Optimizer,
    best: float | Sequence[float],
    *,
    thresholds: Mapping[str, Any] | None = None,
    validation: Mapping[str, Any] | None = None,
    epochs_without_improvement: int = 0,
) -> dict[str, Any]:
    best_key = [float(best)] if isinstance(best, (int, float)) else [float(value) for value in best]
    return {
        "schema_version": SCHEMA_VERSION, "epoch": epoch, "best_key": best_key,
        "geometry": _geometry(config), "composer_to_index": {name: index for index, name in enumerate(COMPOSERS)},
        "generator": generator.state_dict(), "discriminator": discriminator.state_dict(),
        "generator_optimizer": generator_optimizer.state_dict(), "discriminator_optimizer": discriminator_optimizer.state_dict(),
        "seed": config.seed, "thresholds": dict(thresholds or {}), "validation": dict(validation or {}),
        "epochs_without_improvement": epochs_without_improvement,
    }


def _load_models(config: E4Config, checkpoint: Path, device: torch.device) -> ConditionalGenerator:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    expected_mapping = {name: index for index, name in enumerate(COMPOSERS)}
    if payload.get("schema_version") != SCHEMA_VERSION or payload.get("geometry") != _geometry(config) or payload.get("composer_to_index") != expected_mapping:
        raise ValueError("checkpoint schema, geometry, or composer mapping does not match configuration")
    model = ConditionalGenerator(conv_dim=config.conv_dim, residual_blocks=config.residual_blocks).to(device)
    model.load_state_dict(payload["generator"])
    model.eval()
    return model


def _threshold_values(thresholds: Mapping[str, Any] | None) -> tuple[float, float]:
    thresholds = thresholds or {}
    onset = thresholds.get("onset", .5)
    frame = thresholds.get("frame", .5)
    if isinstance(onset, Mapping):
        onset = onset.get("threshold", .5)
    if isinstance(frame, Mapping):
        frame = frame.get("threshold", .5)
    return float(onset), float(frame)


def _clean_binary_segment(
    data: np.ndarray,
    active_at_start: np.ndarray | None = None,
) -> tuple[np.ndarray, int, np.ndarray]:
    """Enforce onset=>frame while carrying sounding pitches across windows."""
    cleaned = data.copy()
    cleaned[1] = np.logical_or(cleaned[1], cleaned[0]).astype(np.uint8)
    orphaned = 0
    active = (
        np.zeros(cleaned.shape[2], dtype=bool)
        if active_at_start is None
        else np.asarray(active_at_start, dtype=bool).copy()
    )
    if active.shape != (cleaned.shape[2],):
        raise ValueError("active pitch state does not match segment geometry")
    for step in range(cleaned.shape[1]):
        onset, frame = cleaned[0, step].astype(bool), cleaned[1, step].astype(bool)
        orphan = frame & ~active & ~onset
        orphaned += int(orphan.sum())
        cleaned[1, step, orphan] = 0
        frame = cleaned[1, step].astype(bool)
        active = (active & frame) | onset
        active &= frame
    return cleaned, orphaned, active


def _infer_piece(
    model: ConditionalGenerator,
    piece: PieceSegments,
    target: int,
    device: torch.device,
    thresholds: Mapping[str, Any] | None = None,
) -> tuple[Any, dict[str, int]]:
    onset_threshold, frame_threshold = _threshold_values(thresholds)
    decoded: list[Segment] = []
    fallback = orphaned = stuck = 0
    active = np.zeros(piece.segment_map.pitch_high - piece.segment_map.pitch_low, dtype=bool)
    with torch.no_grad():
        for segment in piece.segment_map.segments:
            x = torch.from_numpy(segment.data.astype(np.float32))[None].to(device)
            probabilities = torch.sigmoid(model(x, torch.tensor([target], device=device)))[0].cpu().numpy()
            binary = np.stack((probabilities[0] >= onset_threshold, probabilities[1] >= frame_threshold)).astype(np.uint8)
            binary[:, ~segment.mask.astype(bool), :] = 0
            binary, segment_orphans, active = _clean_binary_segment(binary, active)
            orphaned += segment_orphans
            stuck += int(active.sum())
            if segment.nonempty and not binary[:, segment.mask.astype(bool), :].any():
                # Re-evaluate the carry state after the identity fallback.  A
                # source sustain at a boundary is valid only if the preceding
                # decoded window also left that pitch sounding.
                binary, _, active = _clean_binary_segment(segment.data, active)
                fallback += 1
            decoded.append(Segment(segment.index, segment.start_bar, segment.bars, binary, segment.mask.copy()))
    output = stitch_segments(piece.segment_map, decoded)
    output_end = max((note.tick + note.duration_ticks for note in output.notes), default=0)
    if output_end != piece.segment_map.end_tick:
        terminal = len(decoded) - 1
        if terminal >= 0 and decoded[terminal].data is not piece.segment_map.segments[terminal].data:
            decoded[terminal] = piece.segment_map.segments[terminal]
            fallback += int(piece.segment_map.segments[terminal].nonempty)
        output = stitch_segments(piece.segment_map, decoded)
        output_end = max((note.tick + note.duration_ticks for note in output.notes), default=0)
    if output_end != piece.segment_map.end_tick:
        output = piece.segment_map.source
        fallback = sum(segment.nonempty for segment in piece.segment_map.segments)
    return output, {
        "fallback_segments": fallback,
        "nonempty_input_segments": sum(segment.nonempty for segment in piece.segment_map.segments),
        "orphan_frame_starts": orphaned,
        "active_pitches_at_segment_ends": stuck,
    }


def _selection_pieces(collection: Sequence[PieceSegments]) -> list[PieceSegments]:
    selected: list[PieceSegments] = []
    for composer in COMPOSERS:
        candidates = [piece for piece in collection if piece.composer == composer]
        if not candidates:
            raise ValueError(f"validation split has no {composer} piece")
        median = float(np.median([piece.segment_map.segment_count for piece in candidates]))
        selected.append(min(candidates, key=lambda piece: (abs(piece.segment_map.segment_count - median), piece.sample_id)))
    return selected


def _allowed_length_error(piece: PieceSegments) -> int:
    return max(1, max(round(bar.length_ticks / STEPS_PER_BAR) for bar in piece.segment_map.bars))


def _fit_ids(config: E4Config, partitions: Mapping[str, set[str]], *, scope: str) -> list[str]:
    if scope == "validation":
        return sorted(partitions["train"])
    if scope == "test":
        return sorted(partitions["train"] | partitions["validation"])
    raise ValueError("scope must be validation or test")


def _evaluate_collection(
    config: E4Config,
    model: ConditionalGenerator,
    pieces: Sequence[PieceSegments],
    evaluator: FoldStyleEvaluator,
    device: torch.device,
    thresholds: Mapping[str, Any],
    *,
    output_dir: Path | None = None,
) -> list[dict[str, Any]]:
    printer, parser = MidiPrettyPrinter(), MidiParser()
    records: list[dict[str, Any]] = []
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
    for piece in pieces:
        source_index = COMPOSERS.index(piece.composer)
        for target_index, target_composer in enumerate(COMPOSERS):
            if target_index == source_index:
                continue
            generated, diagnostics = _infer_piece(model, piece, target_index, device, thresholds)
            parse_error: str | None = None
            output_path: Path | None = None
            try:
                if output_dir:
                    output_path = output_dir / f"{piece.sample_id}_to_{target_composer}.mid"
                    printer.write(generated, output_path)
                    measured = parser.parse(output_path)
                else:
                    measured = parser.parse_bytes(printer.to_bytes(generated))
            except Exception as exc:
                measured = generated
                parse_error = f"{type(exc).__name__}: {exc}"
            record = {
                "sample_id": piece.sample_id, "group_id": piece.group_id,
                "source_composer": piece.composer, "target_composer": target_composer,
                "output_path": str(output_path) if output_path else None,
                "artifact_parseable": parse_error is None, "parse_error": parse_error,
                "allowed_length_error_ticks": _allowed_length_error(piece), "diagnostics": diagnostics,
                **evaluator.evaluate(piece.sample_id, piece.segment_map.source, measured, target_composer),
            }
            records.append(record)
    return records


def _selection_key(records: Sequence[Mapping[str, Any]], config: E4Config) -> tuple[float, float, float, float]:
    valid = np.mean([
        row["artifact_parseable"]
        and row["content"]["ticks_per_beat_preserved"]
        and row["content"]["smf_format_preserved"]
        and row["content"]["meta_preserved"]
        and row["content"]["bar_count_preserved"]
        and row["content"]["length_error_ticks"] <= row["allowed_length_error_ticks"]
        and row["content"]["nonempty"]
        for row in records
    ])
    content = np.mean([
        row["content"]["melody_trigram_jaccard"] >= config.melody_similarity_min
        and row["content"]["max_polyphony_output"]
        <= max(row["content"]["max_polyphony_input"], row["target_max_polyphony"])
        for row in records
    ])
    nonempty_segments = sum(int(row["diagnostics"]["nonempty_input_segments"]) for row in records)
    fallback_segments = sum(int(row["diagnostics"]["fallback_segments"]) for row in records)
    negative_fallback_rate = -fallback_segments / max(1, nonempty_segments)
    style = np.mean([row["delta_p_target"] for row in records])
    return float(valid), float(content), float(negative_fallback_rate), float(style)


def _model_and_optimizers(config: E4Config, device: torch.device):
    generator = ConditionalGenerator(conv_dim=config.conv_dim, residual_blocks=config.residual_blocks).to(device)
    discriminator = PatchDiscriminator(conv_dim=config.conv_dim).to(device)
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=config.learning_rate, betas=(config.beta1, config.beta2))
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=config.learning_rate, betas=(config.beta1, config.beta2))
    return generator, discriminator, generator_optimizer, discriminator_optimizer


def smoke_e4(config: E4Config) -> Path:
    if not config.run_dir:
        raise ValueError("run_dir is required")
    _, partitions, rows, _ = _inputs(config)
    if not (config.run_dir / "segments.jsonl").is_file():
        prepare_e4(config)
    eligible = _eligible_partitions(config, partitions)
    smoke_partitions = {"train": set(), "validation": set(), "test": set()}
    for composer in COMPOSERS:
        train = sorted(item for item in eligible["train"] if rows[item]["composer"] == composer)
        validation = sorted(item for item in eligible["validation"] if rows[item]["composer"] == composer)
        smoke_partitions["train"].update(train[:config.smoke_per_composer])
        smoke_partitions["validation"].update(validation[:max(1, config.smoke_files // len(COMPOSERS))])
    collections = _pieces(config, smoke_partitions, rows)
    datasets = {name: SegmentDataset(values, split=name) for name, values in collections.items()}
    assert_split_isolation(datasets)
    selected_indices: list[int] = []
    for label in range(len(COMPOSERS)):
        selected_indices.extend([
            index for index, entry in enumerate(datasets["train"].index)
            if datasets["train"].composer_to_index[entry.composer] == label
        ][:config.smoke_per_composer])
    train = Subset(datasets["train"], selected_indices)
    _seed(config.seed)
    device = _device(config)
    generator, discriminator, generator_optimizer, discriminator_optimizer = _model_and_optimizers(config, device)
    loader = DataLoader(train, batch_size=config.batch_size, shuffle=True)
    losses: list[float] = []
    for _ in range(config.smoke_epochs):
        for raw in loader:
            batch = {key: value.to(device) for key, value in raw.items() if key in {"x", "mask", "source"}}
            metrics = gan_step(
                generator, discriminator, generator_optimizer, discriminator_optimizer, batch,
                lambda_cls=config.lambda_cls, lambda_cyc=config.lambda_cyc, lambda_id=config.lambda_id,
            )
            losses.append(metrics.generator_loss)
    checkpoint = config.run_dir / "smoke.pt"
    _atomic_checkpoint(checkpoint, _checkpoint(config, config.smoke_epochs, generator, discriminator, generator_optimizer, discriminator_optimizer, [-float(np.mean(losses))]))
    selected = collections["validation"][:config.smoke_files]
    output_dir = config.run_dir / "smoke_midi"
    results = []
    for piece in selected:
        target = (COMPOSERS.index(piece.composer) + 1) % len(COMPOSERS)
        rendered, diagnostics = _infer_piece(generator, piece, target, device)
        path = output_dir / f"{piece.sample_id}_to_{COMPOSERS[target]}.mid"
        MidiPrettyPrinter().write(rendered, path)
        parsed = MidiParser().parse(path)
        results.append({
            "sample_id": piece.sample_id, "path": str(path), "parsed": True,
            "end_tick": piece.segment_map.end_tick,
            "result_end_tick": max((note.tick + note.duration_ticks for note in parsed.notes), default=0),
            "diagnostics": diagnostics,
        })
    passed = len(results) == config.smoke_files and all(row["result_end_tick"] == row["end_tick"] for row in results)
    path = config.run_dir / "smoke.json"
    _atomic_json(path, {"schema_version": SCHEMA_VERSION, "files": results, "passed": passed, "epochs": config.smoke_epochs})
    _atomic_json(config.run_dir / "status.json", {"schema_version": SCHEMA_VERSION, "status": "smoke_passed" if passed else "smoke_failed", "finished_at_utc": _utc()})
    return path


def train_e4(config: E4Config) -> Path:
    if not config.run_dir:
        raise ValueError("run_dir is required")
    smoke_path = config.run_dir / "smoke.json"
    if not smoke_path.is_file() or not json.loads(smoke_path.read_text(encoding="utf-8")).get("passed"):
        raise ValueError("a passing E4 smoke gate is required before training")
    _, partitions, rows, composition = _inputs(config)
    eligible = _eligible_partitions(config, partitions)
    collections = _pieces(config, eligible, rows)
    datasets = {name: SegmentDataset(values, split=name) for name, values in collections.items()}
    assert_split_isolation(datasets)
    sampler = BalancedComposerSampler(datasets["train"], seed=config.seed)
    loader = DataLoader(datasets["train"], batch_size=config.batch_size, sampler=sampler)
    selection = _selection_pieces(collections["validation"])
    evaluator = FoldStyleEvaluator(
        composition,
        rows,
        _fit_ids(config, eligible, scope="validation"),
        onset_tolerance_beats=config.onset_tolerance_beats,
    )
    _seed(config.seed)
    device = _device(config)
    generator, discriminator, generator_optimizer, discriminator_optimizer = _model_and_optimizers(config, device)
    start, best_key, stale = 0, (-1.0, -1.0, -1.0, float("-inf")), 0
    last_path = config.run_dir / "last.pt"
    if last_path.is_file():
        state = torch.load(last_path, map_location=device, weights_only=False)
        if state.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("last.pt belongs to an incompatible E4 schema; use a new run directory")
        generator.load_state_dict(state["generator"])
        discriminator.load_state_dict(state["discriminator"])
        generator_optimizer.load_state_dict(state["generator_optimizer"])
        discriminator_optimizer.load_state_dict(state["discriminator_optimizer"])
        start = int(state["epoch"])
        best_key = tuple(float(value) for value in state.get("best_key", best_key))  # type: ignore[assignment]
        stale = int(state.get("epochs_without_improvement", 0))
    _atomic_json(config.run_dir / "status.json", {"schema_version": SCHEMA_VERSION, "status": "training", "started_at_utc": _utc()})
    for epoch in range(start, config.max_epochs):
        sampler.set_epoch(epoch)
        generator.train()
        losses = []
        for raw in loader:
            batch = {key: value.to(device) for key, value in raw.items() if key in {"x", "mask", "source"}}
            losses.append(gan_step(
                generator, discriminator, generator_optimizer, discriminator_optimizer, batch,
                lambda_cls=config.lambda_cls, lambda_cyc=config.lambda_cyc, lambda_id=config.lambda_id,
            ).generator_loss)
        generator.eval()
        thresholds = calibrate_thresholds(generator, selection, device, config.threshold_grid)
        preview = _evaluate_collection(config, generator, selection, evaluator, device, thresholds)
        key = _selection_key(preview, config)
        improved = key > best_key
        if improved:
            best_key, stale = key, 0
            _atomic_checkpoint(config.run_dir / "best.pt", _checkpoint(
                config, epoch + 1, generator, discriminator, generator_optimizer, discriminator_optimizer,
                best_key, thresholds=thresholds,
                validation={"selection_key": list(key), "records": preview},
            ))
        else:
            stale += 1
        _atomic_checkpoint(last_path, _checkpoint(
            config, epoch + 1, generator, discriminator, generator_optimizer, discriminator_optimizer,
            best_key, thresholds=thresholds, validation={"selection_key": list(key)}, epochs_without_improvement=stale,
        ))
        _append_jsonl(config.run_dir / "progress.jsonl", {
            "event": "epoch_completed", "stage": "train", "epoch": epoch + 1,
            "generator_loss": float(np.mean(losses)), "selection_key": list(key), "improved": improved, "at_utc": _utc(),
        })
        if stale >= config.patience:
            break
    _atomic_json(config.run_dir / "status.json", {
        "schema_version": SCHEMA_VERSION, "status": "training_completed", "finished_at_utc": _utc(),
        "best_key": list(best_key), "best_checkpoint_sha256": _sha256(config.run_dir / "best.pt"),
    })
    return config.run_dir / "best.pt"


def _evaluate_scope(config: E4Config, scope: str) -> Path:
    if not config.run_dir:
        raise ValueError("run_dir is required")
    checkpoint_path = config.run_dir / "best.pt"
    if not checkpoint_path.is_file():
        raise ValueError("best.pt is missing; complete E4 training first")
    validation_path = config.run_dir / "validation.json"
    if scope == "test":
        if not validation_path.is_file() or not json.loads(validation_path.read_text(encoding="utf-8")).get("summary", {}).get("passed"):
            raise ValueError("outer test is locked until the validation GO gate passes")
        output_path, split = config.run_dir / "outer_test.json", "test"
    else:
        output_path, split = validation_path, "validation"
    _, partitions, rows, composition = _inputs(config)
    eligible = _eligible_partitions(config, partitions)
    collections = _pieces(config, {name: (ids if name == split else set()) for name, ids in eligible.items()}, rows)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    thresholds = checkpoint.get("thresholds")
    if not thresholds:
        raise ValueError("best.pt has no frozen validation thresholds")
    device = _device(config)
    model = _load_models(config, checkpoint_path, device)
    evaluator = FoldStyleEvaluator(
        composition,
        rows,
        _fit_ids(config, eligible, scope=scope),
        onset_tolerance_beats=config.onset_tolerance_beats,
    )
    records = _evaluate_collection(
        config, model, collections[split], evaluator, device, thresholds,
        output_dir=config.run_dir / ("validation_midi" if scope == "validation" else "outer_test_midi"),
    )
    summary = summarize_records(records, melody_min=config.melody_similarity_min, fallback_max=config.fallback_rate_max)
    payload = {
        "schema_version": SCHEMA_VERSION, "scope": "inner_validation" if scope == "validation" else "outer_test",
        "created_at_utc": _utc(), "checkpoint_sha256": _sha256(checkpoint_path), "thresholds": thresholds,
        "records": records, "summary": summary,
    }
    _atomic_json(output_path, payload)
    status = ("validation_go" if summary["passed"] else "validation_no_go") if scope == "validation" else ("completed_go" if summary["passed"] else "completed_no_go")
    _atomic_json(config.run_dir / "status.json", {"schema_version": SCHEMA_VERSION, "status": status, "finished_at_utc": _utc(), "checkpoint_sha256": payload["checkpoint_sha256"]})
    return output_path


def evaluate_e4(config: E4Config) -> Path:
    return _evaluate_scope(config, "validation")


def test_e4(config: E4Config) -> Path:
    return _evaluate_scope(config, "test")


def report_e4(config: E4Config) -> Path:
    if not config.run_dir:
        raise ValueError("run_dir is required")
    validation_path, test_path = config.run_dir / "validation.json", config.run_dir / "outer_test.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8")) if validation_path.is_file() else None
    outer = json.loads(test_path.read_text(encoding="utf-8")) if test_path.is_file() else None
    audit_path, smoke_path = config.run_dir / "audit.json", config.run_dir / "smoke.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.is_file() else None
    smoke = json.loads(smoke_path.read_text(encoding="utf-8")) if smoke_path.is_file() else None
    best_exists = (config.run_dir / "best.pt").is_file()
    if outer:
        decision = "GO" if outer["summary"]["passed"] else "NO-GO"
        explanation = "Outer test przeszedł wszystkie zamrożone bramki." if decision == "GO" else "Outer test nie przeszedł wszystkich zamrożonych bramek; E5 nie powinno być uruchamiane."
    elif validation:
        decision = "VALIDATION GO — OUTER TEST OCZEKUJE" if validation["summary"]["passed"] else "NO-GO"
        explanation = "Uruchom osobny etap `test` dokładnie raz." if validation["summary"]["passed"] else "Validation nie przeszło bramki; outer test pozostaje zablokowany."
    elif best_exists:
        decision, explanation = "GOTOWE DO WALIDACJI", "Trening zakończony; należy uruchomić etap `evaluate`."
    elif smoke and smoke.get("passed"):
        decision, explanation = "GOTOWE DO TRENINGU", "Implementacja i smoke są poprawne; wynik badawczy E4 jeszcze nie istnieje."
    elif audit and not audit.get("gate", {}).get("passed"):
        decision, explanation = "BLOKADA TECHNICZNA", "Audyt danych nie przeszedł bramki; trening pozostaje zablokowany."
    else:
        decision, explanation = "NIEKOMPLETNE", "Brak właściwej ewaluacji validation."
    lines = ["# E4 — warunkowany GAN inspirowany StarGAN", "", f"Wygenerowano: {_utc()}", "", "## Decyzja", "", f"**{decision}.** {explanation}"]
    lines.extend([
        "", "## Stan potoku", "", "| Etap | Stan |", "|---|---|",
        f"| Audit | {'PASS' if audit and audit.get('gate', {}).get('passed') else 'brak/FAIL'} |",
        f"| Prepare | {'gotowe' if (config.run_dir / 'segments.jsonl').is_file() else 'brak'} |",
        f"| Smoke | {'PASS' if smoke and smoke.get('passed') else 'brak/FAIL'} |",
        f"| Trening / `best.pt` | {'gotowe' if best_exists else 'nieuruchomiony'} |",
        f"| Validation | {'GO' if validation and validation['summary']['passed'] else ('NO-GO' if validation else 'brak')} |",
        f"| Outer test | {'GO' if outer and outer['summary']['passed'] else ('NO-GO' if outer else 'zablokowany/brak')} |",
    ])
    if audit:
        lines.extend([
            "",
            f"Audyt: **{len(audit.get('rejections', []))}** błędów parsera, "
            f"**{len(audit.get('exclusions', []))}** jawnych wykluczeń i "
            f"**{len(audit.get('resolved_terminal_partial_bars', []))}** zaakceptowanych końcowych taktów częściowych.",
        ])
    for title, payload in (("Validation", validation), ("Outer test", outer)):
        if not payload:
            continue
        summary = payload["summary"]
        lines.extend([
            "", f"## {title}", "",
            f"- Liczba kierunkowych wyników: **{summary['record_count']}**.",
            f"- Średnie Δp_target: **{summary['mean_delta_p_target']:.4f}**.",
            f"- Dodatnie kierunki: **{sum(value > 0 for value in summary['direction_means'].values())}/6**.",
            f"- Mediana podobieństwa melodii: **{summary['median_melody_trigram_jaccard']:.4f}**.",
            f"- Średni onset-F1: **{summary['mean_onset_f1']:.4f}**; chroma cosine: **{summary['mean_chroma_cosine']:.4f}**.",
            f"- Fallback identity: **{summary['fallback_segments']}/{summary['nonempty_input_segments']} ({100 * summary['fallback_rate']:.2f}%)**.",
            f"- Osierocone początki frame: **{summary['orphan_frame_starts']}**; aktywne wysokości na końcach okien: **{summary['active_pitches_at_segment_ends']}**.",
            f"- Średni udział pustych taktów wejście/wynik: **{summary['mean_empty_bar_ratio_input']:.4f} / {summary['mean_empty_bar_ratio_output']:.4f}**.",
            "", "| Kryterium | Wynik |", "|---|---|",
            *[f"| `{name}` | {'PASS' if value else 'FAIL'} |" for name, value in summary["criteria"].items()],
            "", "| Kierunek | Średnie Δp_target |", "|---|---:|",
            *[f"| {direction} | {value:.4f} |" for direction, value in summary["direction_means"].items()],
        ])
    lines.extend(["", "## Ograniczenia", "", "- E4 używa jednego seeda i jednego outer foldu; jest eksperymentem decyzyjnym, nie pełnym porównaniem modeli.", "- E1b pozostaje proxy skorelowanym z repertuarem i formą; wynik musi być czytany razem z metrykami treści.", "- Outer test nie służy do ponownego strojenia modelu ani progów."])
    text = "\n".join(lines) + "\n"
    run_report = config.run_dir / "report.md"
    _atomic_text(run_report, text)
    if config.summary_path:
        _atomic_text(config.summary_path, text)
    return run_report
