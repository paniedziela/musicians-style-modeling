"""ASAP cache construction and ablations for the historical E1b experiment."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from ..features.composition import (
    COMPOSITION_FEATURES_SCHEMA_VERSION,
    FEATURE_GROUPS,
    FEATURE_SPECS,
    extract_composition_features,
)
from ..midi.parser import MidiParser

COMPOSITION_FEATURES_FILENAME = "composition_features.json"


def infer_musical_form(title: str) -> str:
    """Map ASAP title conventions to coarse, predeclared form families."""
    normalized = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
    rules = (
        ("prelude_fugue", ("prelude", "fugue")),
        ("sonata", ("sonata", "piano_sonatas")),
        ("etude", ("etude", "etudes")),
        ("ballade", ("ballade", "ballades")),
        ("scherzo", ("scherzo", "scherzi")),
        ("concerto", ("concerto",)),
        ("mazurka", ("mazurka", "mazurkas")),
        ("nocturne", ("nocturne", "nocturnes")),
        ("polonaise", ("polonaise", "polonaises")),
        ("waltz", ("waltz", "waltzes", "valse")),
    )
    for form, tokens in rules:
        if any(token in normalized for token in tokens):
            return form
    return "other"


def _variants() -> dict[str, dict[str, Any]]:
    indices_by_group = {
        group: [index for index, feature in enumerate(FEATURE_SPECS) if feature["group"] == group]
        for group in FEATURE_GROUPS
    }
    all_indices = list(range(len(FEATURE_SPECS)))
    variants: dict[str, dict[str, Any]] = {
        "composition_full": {"indices": all_indices, "kind": "full", "groups": list(FEATURE_GROUPS)}
    }
    for group in FEATURE_GROUPS:
        variants[f"only_{group}"] = {
            "indices": indices_by_group[group], "kind": "group_only", "groups": [group]
        }
    for group in FEATURE_GROUPS:
        variants[f"without_{group}"] = {
            "indices": [index for index in all_indices if index not in indices_by_group[group]],
            "kind": "leave_one_group_out",
            "groups": [item for item in FEATURE_GROUPS if item != group],
        }
    for specification in variants.values():
        specification["feature_names"] = [
            FEATURE_SPECS[index]["name"] for index in specification["indices"]
        ]
    return variants


def build_composition_feature_cache(
    manifest: dict[str, Any], *, dataset_root: Path | str | None = None
) -> dict[str, Any]:
    """Parse accepted score MIDIs and build all predeclared E1b ablations."""
    root_value = dataset_root if dataset_root is not None else manifest.get("dataset_root")
    if root_value is None:
        raise ValueError("dataset_root is required for composition feature extraction")
    root = Path(root_value)
    parser = MidiParser()
    rows = []
    for sample in sorted(manifest.get("samples", []), key=lambda item: item["sample_id"]):
        if sample.get("validation_status") != "accepted":
            continue
        score_path = root / sample["score_path"]
        vector = extract_composition_features(parser.parse(score_path))
        rows.append({
            "sample_id": sample["sample_id"], "composer": sample["composer"],
            "title": sample["title"], "form": infer_musical_form(sample["title"]),
            "group_id": sample["group_id"], "sha256": sample["sha256"],
            "values": vector.tolist(),
        })
    if not rows:
        raise ValueError("manifest contains no accepted samples")
    return {
        "features_schema_version": COMPOSITION_FEATURES_SCHEMA_VERSION,
        "manifest_schema_version": manifest.get("manifest_schema_version"),
        "code_commit": manifest.get("code_commit"),
        "feature_contract": list(FEATURE_SPECS),
        "importance_variant": "composition_full",
        "variants": _variants(),
        "samples": rows,
    }


def write_composition_feature_cache(
    manifest_path: Path | str,
    output_path: Path | str | None = None,
    *,
    dataset_root: Path | str | None = None,
) -> Path:
    source = Path(manifest_path)
    manifest = json.loads(source.read_text(encoding="utf-8"))
    payload = build_composition_feature_cache(manifest, dataset_root=dataset_root)
    destination = (
        Path(output_path)
        if output_path
        else source.with_name(COMPOSITION_FEATURES_FILENAME)
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return destination
