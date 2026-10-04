"""Thin wrapper around the frozen E1b contract and implementation."""

from pathlib import Path

from ..e1.composition_features import (
    COMPOSITION_FEATURES_SCHEMA_VERSION, FEATURE_SPECS, extract_composition_features,
)
from ..midi.parser import MidiParser


def extract(path: Path) -> dict:
    values = extract_composition_features(MidiParser().parse(path)).tolist()
    return {"schema": [dict(spec) for spec in FEATURE_SPECS], "values": values,
            "diagnostics": {"missing": [], "nonfinite": [], "excluded": []}}
