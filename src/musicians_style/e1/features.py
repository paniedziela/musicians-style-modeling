"""Versioned legacy feature cache and E1a ablations for stage E1.2."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

FEATURES_FILENAME = "legacy_features.json"
FEATURES_SCHEMA_VERSION = "e1.2.0"

LEGACY_FULL_NAMES = (
    ["tempo_bpm"]
    + [f"pitch_class_{index}" for index in range(12)]
    + [f"melodic_interval_{index:+d}" for index in range(-12, 13)]
    + ["note_density_per_s", "mean_note_duration_s", "std_note_duration_s", "rest_ratio"]
)
SCORE_ONLY_INDICES = tuple(index for index in range(42) if index not in {0, 38, 39, 40})


def build_legacy_feature_cache(manifest: dict[str, Any]) -> dict[str, Any]:
    """Extract the audited vectors without reparsing MIDI and describe both E1a variants."""
    rows: list[dict[str, Any]] = []
    for sample in sorted(manifest.get("samples", []), key=lambda item: item["sample_id"]):
        if sample.get("validation_status") != "accepted":
            continue
        vector = np.asarray(sample.get("legacy_features"), dtype=np.float64)
        if vector.shape != (42,) or not np.isfinite(vector).all():
            raise ValueError(f"invalid legacy feature vector for {sample['sample_id']}")
        rows.append(
            {
                "sample_id": sample["sample_id"],
                "composer": sample["composer"],
                "group_id": sample["group_id"],
                "sha256": sample["sha256"],
                "values": vector.tolist(),
            }
        )
    if not rows:
        raise ValueError("manifest contains no accepted feature vectors")
    return {
        "features_schema_version": FEATURES_SCHEMA_VERSION,
        "manifest_schema_version": manifest.get("manifest_schema_version"),
        "code_commit": manifest.get("code_commit"),
        "variants": {
            "legacy_full": {"indices": list(range(42)), "feature_names": LEGACY_FULL_NAMES},
            "legacy_score_only": {
                "indices": list(SCORE_ONLY_INDICES),
                "feature_names": [LEGACY_FULL_NAMES[index] for index in SCORE_ONLY_INDICES],
            },
        },
        "samples": rows,
    }


def write_legacy_feature_cache(
    manifest_path: Path | str, output_path: Path | str | None = None
) -> Path:
    source = Path(manifest_path)
    payload = build_legacy_feature_cache(json.loads(source.read_text(encoding="utf-8")))
    destination = Path(output_path) if output_path else source.with_name(FEATURES_FILENAME)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return destination
