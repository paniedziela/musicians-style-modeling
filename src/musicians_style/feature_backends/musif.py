"""MIDI-only musif adapter, loaded exclusively in the isolated worker environment."""

from __future__ import annotations

import importlib.metadata
import math
import numbers
import re
from pathlib import Path

FEATURES = ("core", "ambitus", "melody", "tempo", "density", "texture", "scale", "key", "dynamics", "rhythm")
CONFIG = {"features": list(FEATURES), "basic_modules": ["scoring"], "parallel": 1,
          "ignore_errors": False, "cache_dir": None, "expand_repeats": False,
          "window_size": None, "parts_filter": [], "remove_unpitched_objects": True}

# Reviewed score-level musical fields only. Never select by numeric dtype alone:
# Id/WindowId, filename-derived encodings, names and part/instrument headers are excluded.
GLOBAL_FIELDS = {
    "core": ("Score_Notes", "Score_NotesMean", "Score_SoundingMeasures", "Score_SoundingMeasuresMean", "Measures"),
    "density": ("Score_Density", "Score_SoundingDensity"),
    "rhythm": ("Score_AverageDuration", "Score_RhythmInt", "Score_DottedRhythm", "Score_DoubleDottedRhythm"),
    "tempo": ("NumericTempo", "NumberOfBeats"),
    "dynamics": ("Score_DynMean", "Score_DynMean_weighted", "Score_DynAbruptness", "Score_DynGrad"),
}


def musical_family(name: str, melody_names: set[str]) -> str | None:
    for family, fields in GLOBAL_FIELDS.items():
        if name in fields:
            return family
    if name in melody_names or re.fullmatch(r"Score_Interval[PMAmd]-?\d+_(Count|Per)", name):
        return "melody"
    if re.fullmatch(r"Score_Degree(?:bbb|bb|b|#x|#|x)?[1-7]_(Count|Per)", name):
        return "scale"
    return None


def project_row(row: dict, melody_names: set[str]) -> dict:
    """Leakage-safe, ordered nullable numeric result; no inferred/learned encodings."""
    schema, values = [], []
    diagnostics = {"missing": [], "nonfinite": [], "excluded": [], "invalid_numeric": []}
    for name in sorted(row):
        family = musical_family(name, melody_names)
        if family is None:
            diagnostics["excluded"].append(name)
            continue
        value = row[name]
        schema.append({"name": name, "group": family, "unit": "musif_1.2.4_defined", "nullable": True})
        if value is None or isinstance(value, str) and value in {"NA", "NaN", ""} or isinstance(value, numbers.Real) and math.isnan(float(value)):
            values.append(None)
            diagnostics["missing"].append(name)
        elif not isinstance(value, numbers.Real) or isinstance(value, bool):
            values.append(None)
            diagnostics["invalid_numeric"].append(name)
        elif not math.isfinite(float(value)):
            values.append(None)
            diagnostics["nonfinite"].append(name)
        else:
            values.append(float(value))
    if not schema or not any(value is not None for value in values):
        raise ValueError("no finite allowlisted musical features")
    if diagnostics["invalid_numeric"]:
        raise ValueError(f"nonnumeric values in numeric musical fields: {diagnostics['invalid_numeric']}")
    return {"schema": schema, "values": values, "diagnostics": diagnostics}


def extract(path: Path) -> dict:
    if path.suffix.lower() != ".mid":
        raise ValueError("musif feasibility accepts MIDI only; no notation substitution")
    if importlib.metadata.version("musif") != "1.2.4":
        raise ValueError("musif adapter requires the pinned 1.2.4 backend")
    from musif.extract.extract import FeaturesExtractor
    from musif.extract.features.melody.constants import SCORE_FEATURES

    # MeanInterval, Largest/SmallestInterval and KeySignature are musical category
    # strings in 1.2.4. They remain excluded rather than becoming label encodings.
    melody_names = {"Score_" + name for name in SCORE_FEATURES
                    if name != "MeanInterval" and not re.fullmatch(r"(?:Largest|Smallest)Interval(?:Asc|Desc|All)", name)}
    frame = FeaturesExtractor(None, data_dir=str(path), output_dir=str(path.parent), **CONFIG).extract()
    if len(frame) != 1 or not frame.columns.is_unique:
        raise ValueError(f"expected exactly one row with unique columns, got {frame.shape}")
    row = frame.astype(object).where(frame.notna(), None).iloc[0].to_dict()
    result = project_row(row, melody_names)
    result["raw_feature_count"] = len(frame.columns)
    result["configuration"] = CONFIG
    return result
