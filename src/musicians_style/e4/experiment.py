"""E4.0 audit: manifest/split verification and frozen representation facts."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from ..midi.parser import MidiParser
from .segmentation import PITCH_HIGH, PITCH_LOW, STEPS_PER_BAR, encode_piece, quantization_errors

SCHEMA_VERSION = "e4.0.0"

@dataclass(frozen=True)
class E4Config:
    manifest_path: Path
    splits_path: Path
    dataset_root: Path
    run_dir: Path
    repeat: int = 0
    outer_fold: int = 0
    inner_fold: int = 0

def _resolve(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (Path.cwd() / path).resolve()

def load_e4_config(path: Path | str) -> E4Config:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping) or raw.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported or malformed E4 configuration")
    inputs, output = raw.get("inputs", {}), raw.get("output", {})
    return E4Config(_resolve(str(inputs["manifest"])), _resolve(str(inputs["splits"])), _resolve(str(inputs["dataset_root"])), _resolve(str(output["run_dir"])), int(raw.get("repeat", 0)), int(raw.get("outer_fold", 0)), int(raw.get("inner_fold", 0)))

def _split_ids(splits: Mapping[str, Any], config: E4Config) -> dict[str, set[str]]:
    repeat = next((x for x in splits["repetitions"] if int(x["repeat"]) == config.repeat), None)
    if repeat is None: raise ValueError("requested repeat is missing")
    outer = next((x for x in repeat["folds"] if int(x["fold"]) == config.outer_fold), None)
    if outer is None: raise ValueError("requested outer fold is missing")
    inner = next((x for x in outer["inner_folds"] if int(x["fold"]) == config.inner_fold), None)
    if inner is None: raise ValueError("requested inner fold is missing")
    result = {"train": set(inner["train"]["sample_ids"]), "validation": set(inner["validation"]["sample_ids"]), "test": set(outer["test"]["sample_ids"])}
    if any(result[a] & result[b] for a in result for b in result if a != b): raise ValueError("sample_id leakage between E4 splits")
    return result

def audit_e4(config: E4Config) -> Path:
    manifest = json.loads(config.manifest_path.read_text(encoding="utf-8"))
    splits = json.loads(config.splits_path.read_text(encoding="utf-8"))
    rows = {str(x["sample_id"]): x for x in manifest.get("samples", []) if x.get("validation_status") == "accepted"}
    partitions = _split_ids(splits, config)
    if set().union(*partitions.values()) != set(rows): raise ValueError("E4 split does not partition accepted manifest samples")
    groups = {name: {str(rows[x]["group_id"]) for x in ids} for name, ids in partitions.items()}
    if any(groups[a] & groups[b] for a in groups for b in groups if a != b): raise ValueError("group_id leakage between E4 splits")
    report: dict[str, Any] = {"schema_version": SCHEMA_VERSION, "geometry": {"bars_per_segment": 4, "steps_per_bar": STEPS_PER_BAR, "pitch_range": [PITCH_LOW, PITCH_HIGH]}, "splits": {}, "rejections": []}
    parser = MidiParser()
    global_pitch_min: int | None = None
    global_pitch_max: int | None = None
    report["unusual_bar_lengths_quarter_notes"] = []
    for split, ids in partitions.items():
        counters: dict[str, Counter[str]] = defaultdict(Counter)
        meter_quantization: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for sample_id in sorted(ids):
            row = rows[sample_id]; composer = str(row["composer"])
            try:
                source = parser.parse(config.dataset_root / row["score_path"])
                mapped = encode_piece(source)
            except Exception as exc:  # audit must document, not hide a malformed score
                report["rejections"].append({"sample_id": sample_id, "reason": f"parser_or_segmentation: {exc}"}); continue
            counters[composer]["pieces"] += 1; counters[composer]["bars"] += len(mapped.bars); counters[composer]["segments"] += len(mapped.segments); counters[composer]["empty_segments"] += sum(not s.nonempty for s in mapped.segments)
            for segment in mapped.segments:
                counters[composer]["nonempty_segments"] += int(segment.nonempty)
                counters[composer]["active_cells"] += int(segment.data[1].sum())
                counters[composer]["onset_cells"] += int(segment.data[0].sum())
            for bar in mapped.bars:
                counters[composer][f"meter:{bar.numerator}/{bar.denominator}"] += 1
                length_quarters = bar.length_ticks / source.ticks_per_beat
                if not .5 <= length_quarters <= 12:
                    report["unusual_bar_lengths_quarter_notes"].append({"sample_id": sample_id, "bar": bar.index, "length": length_quarters, "meter": f"{bar.numerator}/{bar.denominator}"})
            for note in source.notes:
                global_pitch_min = note.pitch if global_pitch_min is None else min(global_pitch_min, note.pitch); global_pitch_max = note.pitch if global_pitch_max is None else max(global_pitch_max, note.pitch)
                counters[composer]["outside_pitch"] += int(not PITCH_LOW <= note.pitch < PITCH_HIGH)
            for meter, onset_error, end_error in quantization_errors(mapped):
                meter_quantization[meter].append((abs(onset_error), abs(end_error)))
        per_composer = {composer: dict(sorted(values.items())) for composer, values in sorted(counters.items())}
        for composer, values in per_composer.items():
            steps = max(1, values.get("segments", 0) * 64)
            values["occupancy"] = values.get("active_cells", 0) / (steps * (PITCH_HIGH - PITCH_LOW))
            values["onset_density"] = values.get("onset_cells", 0) / steps
        report["splits"][split] = per_composer
        report.setdefault("quantization", {})[split] = {meter: {"count": len(values), "onset_abs_p95": float(sorted(x[0] for x in values)[int(.95 * (len(values) - 1))]), "end_abs_max": max(x[1] for x in values)} for meter, values in sorted(meter_quantization.items())}
    train = report["splits"]["train"]
    sufficient = all(train.get(composer, {}).get("nonempty_segments", 0) >= 100 for composer in ("Bach", "Beethoven", "Chopin"))
    report["pitch"] = {"min": global_pitch_min, "max": global_pitch_max}
    report["gate"] = {"group_isolation": True, "parser_errors": not bool(report["rejections"]), "at_least_100_train_nonempty_per_composer": sufficient, "passed": not report["rejections"] and sufficient}
    destination = config.run_dir / "audit.json"; destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".json.tmp"); temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"); temporary.replace(destination)
    return destination
