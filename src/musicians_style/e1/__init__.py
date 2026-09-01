"""Experiment E1: composer discrimination on ASAP score MIDI files."""

from .asap import E1Config, build_e1_manifest, load_e1_config, write_e1_artifacts
from .classification import run_e1a, write_e1a_results
from .features import build_legacy_feature_cache, write_legacy_feature_cache
from .splits import build_e1_splits, write_e1_splits

__all__ = [
    "E1Config",
    "build_e1_manifest",
    "load_e1_config",
    "write_e1_artifacts",
    "build_e1_splits",
    "write_e1_splits",
    "build_legacy_feature_cache",
    "write_legacy_feature_cache",
    "run_e1a",
    "write_e1a_results",
]
