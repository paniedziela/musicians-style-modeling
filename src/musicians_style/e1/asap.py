"""ASAP adapter and quality audit for stage E1.0.

The adapter deliberately understands only ASAP's ``metadata.csv`` contract.
It never copies or modifies source files and treats one ``(composer, title)``
pair as one sample.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import subprocess
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml
from mido import MidiFile

from musicians_style.features.extractor import FeatureExtractor
from musicians_style.midi.parser import MidiParser

MANIFEST_FILENAME = "manifest.json"
QUALITY_REPORT_FILENAME = "quality_report.json"


@dataclass(frozen=True)
class E1Config:
    """Frozen configuration of the E1.0 data audit."""

    dataset_root: Path
    output_dir: Path
    composers: tuple[str, ...]
    minimum_samples_per_class: int
    schema_version: str
    expected_fingerprints: dict[str, str]
    experiment_name: str = "e1_asap"
    split_seeds: tuple[int, ...] = (1729, 2718, 3141, 5772, 8119)
    outer_splits: int = 5
    inner_splits: int = 3


def load_e1_config(path: Path | str) -> E1Config:
    """Load and validate the small, experiment-specific YAML configuration."""
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("E1 configuration must be a YAML mapping")
    composers = tuple(str(value) for value in raw.get("composers", ()))
    if not composers:
        raise ValueError("E1 configuration requires at least one composer")
    expected = raw.get("expected_fingerprints", {})
    if not isinstance(expected, dict):
        raise ValueError("expected_fingerprints must be a mapping")
    split_seeds = tuple(int(value) for value in raw.get("split_seeds", (1729, 2718, 3141, 5772, 8119)))
    if not split_seeds:
        raise ValueError("split_seeds must contain at least one seed")
    outer_splits = int(raw.get("outer_splits", 5))
    inner_splits = int(raw.get("inner_splits", 3))
    if outer_splits < 2 or inner_splits < 2:
        raise ValueError("outer_splits and inner_splits must be at least 2")
    return E1Config(
        dataset_root=Path(raw["dataset_root"]),
        output_dir=Path(raw["output_dir"]),
        composers=composers,
        minimum_samples_per_class=int(raw.get("minimum_samples_per_class", 30)),
        schema_version=str(raw.get("schema_version", "e1.0.0")),
        expected_fingerprints={str(k): str(v).lower() for k, v in expected.items()},
        experiment_name=str(raw.get("experiment_name", "e1_asap")),
        split_seeds=split_seeds,
        outer_splits=outer_splits,
        inner_splits=inner_splits,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _slug(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", normalized.lower()).strip("-")


def group_id_for(composer: str, title: str) -> str:
    """Return the conservative work-level group prescribed by E1_PLAN.md."""
    if composer == "Bach":
        bwv = re.search(r"(?:Prelude|Fugue)_bwv_(\d+)", title, re.IGNORECASE)
        work = f"bwv-{bwv.group(1)}" if bwv else _slug(title)
    elif composer == "Beethoven":
        sonata = re.match(r"Piano_Sonatas_(\d+)(?:-|$)", title, re.IGNORECASE)
        work = f"piano-sonata-{sonata.group(1)}" if sonata else _slug(title)
    elif composer == "Chopin":
        sonata = re.match(r"Sonata_(\d+)(?:_|$)", title, re.IGNORECASE)
        work = f"sonata-{sonata.group(1)}" if sonata else _slug(title)
    else:
        work = _slug(title)
    return f"{_slug(composer)}--{work}"


def _canonical_score(paths: Iterable[str]) -> tuple[str, list[str]]:
    unique = sorted({path.replace("\\", "/") for path in paths})
    if not unique:
        return "", []

    def rank(path: str) -> tuple[int, str]:
        lowered = path.lower()
        is_alternate = "_no_repeat" in lowered or "_no_2_repeat" in lowered
        return (int(is_alternate), lowered)

    canonical = min(unique, key=rank)
    return canonical, [path for path in unique if path != canonical]


def _read_candidates(config: E1Config) -> list[dict[str, Any]]:
    metadata_path = config.dataset_root / "metadata.csv"
    grouped: dict[tuple[str, str], list[str]] = defaultdict(list)
    with metadata_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"composer", "title", "midi_score"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"metadata.csv lacks columns: {sorted(required)}")
        for row in reader:
            composer = row["composer"].strip()
            if composer in config.composers:
                grouped[(composer, row["title"].strip())].append(row["midi_score"].strip())

    candidates = []
    for (composer, title), score_paths in sorted(grouped.items()):
        score_path, alternates = _canonical_score(score_paths)
        candidates.append(
            {
                "sample_id": f"{_slug(composer)}--{_slug(title)}",
                "composer": composer,
                "title": title,
                "score_path": score_path,
                "alternate_score_paths": alternates,
                "group_id": group_id_for(composer, title),
            }
        )
    return candidates


def _max_polyphony(notes: Iterable[Any]) -> int:
    boundaries: list[tuple[int, int]] = []
    for note in notes:
        if note.duration_ticks == 0:
            continue
        boundaries.append((note.tick, 1))
        boundaries.append((note.tick + note.duration_ticks, -1))
    active = maximum = 0
    # Notes ending at a tick do not overlap notes starting at that same tick.
    for _, change in sorted(boundaries, key=lambda item: (item[0], item[1])):
        active += change
        maximum = max(maximum, active)
    return maximum


def _length_bars(repr_: Any, end_tick: int) -> float:
    signatures = [(0, 4.0)]
    for event in repr_.meta:
        if event.kind == "time_signature":
            numerator = int(event.payload["numerator"])
            denominator = int(event.payload["denominator"])
            if numerator > 0 and denominator > 0:
                signatures.append((event.tick, numerator * 4.0 / denominator))
    # At a given tick the last parsed signature wins; sorted dict is deterministic.
    changes = sorted(dict(signatures).items())
    bars = 0.0
    for index, (start, beats_per_bar) in enumerate(changes):
        if start >= end_tick:
            break
        stop = min(end_tick, changes[index + 1][0] if index + 1 < len(changes) else end_tick)
        if stop > start:
            bars += ((stop - start) / repr_.ticks_per_beat) / beats_per_bar
    return bars


def _audit_candidate(candidate: dict[str, Any], config: E1Config, commit: str) -> dict[str, Any]:
    record = {
        **candidate,
        "sha256": None,
        "smf_format": None,
        "ticks_per_beat": None,
        "track_count": None,
        "note_count": None,
        "zero_duration_note_count": None,
        "meta_event_count": None,
        "pitch_min": None,
        "pitch_max": None,
        "max_polyphony": None,
        "length_beats": None,
        "length_bars": None,
        "legacy_features": None,
        "validation_status": "excluded",
        "exclusion_reason": None,
        "manifest_schema_version": config.schema_version,
        "code_commit": commit,
    }
    try:
        if not candidate["score_path"]:
            raise ValueError("metadata row has no midi_score path")
        relative_path = Path(candidate["score_path"])
        if relative_path.suffix.lower() not in {".mid", ".midi"}:
            raise ValueError("score path does not have a MIDI extension")
        score_path = config.dataset_root / relative_path
        if not score_path.is_file():
            raise FileNotFoundError(f"score MIDI does not exist: {candidate['score_path']}")

        record["sha256"] = _sha256(score_path)
        raw_midi = MidiFile(score_path)
        if raw_midi.ticks_per_beat <= 0:
            raise ValueError("ticks_per_beat must be positive (SMPTE timing is unsupported)")
        parser = MidiParser()
        parser.validate(score_path)
        repr_ = parser.parse(score_path)
        if repr_.ticks_per_beat <= 0:
            raise ValueError("ticks_per_beat must be positive")
        if not repr_.notes:
            raise ValueError("score contains no complete notes")
        invalid_notes = [
            note
            for note in repr_.notes
            if not (note.tick >= 0 and 0 <= note.pitch <= 127 and 1 <= note.velocity <= 127 and note.duration_ticks >= 0)
        ]
        if invalid_notes:
            raise ValueError(f"score contains {len(invalid_notes)} invalid note events")

        features = FeatureExtractor().extract(repr_, source=score_path).as_array()
        if not np.isfinite(features).all():
            raise ValueError("legacy feature vector contains NaN or infinity")

        end_tick = max(note.tick + note.duration_ticks for note in repr_.notes)
        pitches = [note.pitch for note in repr_.notes]
        record.update(
            {
                "smf_format": int(raw_midi.type),
                "ticks_per_beat": int(raw_midi.ticks_per_beat),
                "track_count": len(raw_midi.tracks),
                "note_count": len(repr_.notes),
                "zero_duration_note_count": sum(note.duration_ticks == 0 for note in repr_.notes),
                "meta_event_count": sum(message.is_meta for track in raw_midi.tracks for message in track),
                "pitch_min": min(pitches),
                "pitch_max": max(pitches),
                "max_polyphony": _max_polyphony(repr_.notes),
                "length_beats": end_tick / repr_.ticks_per_beat,
                "length_bars": _length_bars(repr_, end_tick),
                "legacy_features": [float(value) for value in features],
                "validation_status": "accepted",
            }
        )
    except Exception as exc:  # one bad sample must remain documented in the manifest
        record["exclusion_reason"] = f"{type(exc).__name__}: {exc}"
    return record


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _outliers(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    accepted = [record for record in records if record["validation_status"] == "accepted"]
    result: list[dict[str, Any]] = []
    for field in ("note_count", "length_beats", "max_polyphony"):
        values = np.asarray([record[field] for record in accepted], dtype=float)
        if values.size < 4:
            continue
        q1, q3 = np.percentile(values, [25, 75])
        lower, upper = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
        for record in accepted:
            value = float(record[field])
            if value < lower or value > upper:
                result.append({"sample_id": record["sample_id"], "field": field, "value": value, "lower_fence": float(lower), "upper_fence": float(upper)})
    return result


def build_e1_manifest(config: E1Config) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the complete manifest and E1.0 quality-gate report in memory."""
    root = config.dataset_root
    fingerprints = {
        name: _sha256(root / name) if (root / name).is_file() else None
        for name in config.expected_fingerprints
    }
    mismatches = {
        name: {"expected": expected, "actual": fingerprints.get(name)}
        for name, expected in config.expected_fingerprints.items()
        if fingerprints.get(name) != expected
    }
    commit = _git_commit()
    records = [_audit_candidate(candidate, config, commit) for candidate in _read_candidates(config)]
    sha_groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for record in records:
        if record["sha256"]:
            sha_groups[record["sha256"]].append({"sample_id": record["sample_id"], "composer": record["composer"]})
    duplicates = [
        {"sha256": sha, "samples": samples, "cross_class": len({s["composer"] for s in samples}) > 1}
        for sha, samples in sorted(sha_groups.items())
        if len(samples) > 1
    ]
    accepted_counts = Counter(record["composer"] for record in records if record["validation_status"] == "accepted")
    excluded_counts = Counter(record["composer"] for record in records if record["validation_status"] == "excluded")
    group_counts = {
        composer: len({record["group_id"] for record in records if record["composer"] == composer and record["validation_status"] == "accepted"})
        for composer in config.composers
    }
    failures: list[str] = []
    if mismatches:
        failures.append("dataset fingerprint mismatch")
    if any(record["validation_status"] not in {"accepted", "excluded"} for record in records):
        failures.append("unaccounted validation status")
    if any(item["cross_class"] for item in duplicates):
        failures.append("identical SHA-256 appears in more than one class")
    for composer in config.composers:
        if accepted_counts[composer] < config.minimum_samples_per_class:
            failures.append(f"{composer} has fewer than {config.minimum_samples_per_class} accepted samples")

    created_at = datetime.now(timezone.utc).isoformat()
    manifest = {
        "manifest_schema_version": config.schema_version,
        "created_at_utc": created_at,
        "code_commit": commit,
        "dataset_root": root.as_posix(),
        "dataset_fingerprints": fingerprints,
        "selection": {"composers": list(config.composers), "unit": "unique (composer, title)", "source": "score MIDI only"},
        "samples": records,
    }
    report = {
        "manifest_schema_version": config.schema_version,
        "created_at_utc": created_at,
        "candidate_count": len(records),
        "accepted_count": sum(accepted_counts.values()),
        "excluded_count": sum(excluded_counts.values()),
        "accepted_by_composer": dict(accepted_counts),
        "excluded_by_composer": dict(excluded_counts),
        "groups_by_composer": group_counts,
        "fingerprint_mismatches": mismatches,
        "duplicate_sha_groups": duplicates,
        "outliers_iqr": _outliers(records),
        "exclusions": [{"sample_id": record["sample_id"], "reason": record["exclusion_reason"]} for record in records if record["validation_status"] == "excluded"],
        "quality_gate": {"passed": not failures, "failures": failures},
    }
    return manifest, report


def write_e1_artifacts(config: E1Config) -> tuple[Path, Path, dict[str, Any]]:
    """Run E1.0 and atomically replace its two derived JSON artifacts."""
    manifest, report = build_e1_manifest(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = config.output_dir / MANIFEST_FILENAME
    report_path = config.output_dir / QUALITY_REPORT_FILENAME
    for path, payload in ((manifest_path, manifest), (report_path, report)):
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
    return manifest_path, report_path, report
