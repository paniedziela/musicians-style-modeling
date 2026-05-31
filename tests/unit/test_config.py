"""Testy jednostkowe loadera konfiguracji (``musicians_style.config``).

Zakres (zadanie 1.2):
* rozpoznawanie formatu po rozszerzeniu (YAML / JSON),
* normalizacja struktury do zestawu ``@dataclass``,
* walidacja pól wymaganych (``seed``, ``model.mode``, ``model.artists`` dla
  ``conditional``, ``model.reference_artist`` dla ``per_artist``),
* zgłaszanie :class:`ConfigValidationError` dla niespójnych konfiguracji.

Testy YAML są pomijane (``pytest.importorskip``), gdy pakiet PyYAML nie jest
zainstalowany w środowisku - ścieżka JSON pozostaje w pełni weryfikowalna.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from musicians_style.config import (
    Config,
    ConfigValidationError,
    DatasetConfig,
    EvaluationConfig,
    GAConfig,
    ModelConfig,
    TrainingConfig,
    load_config,
    parse_config_text,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = REPO_ROOT / "configs"


# --------------------------------------------------------------------------- #
# Pomocnicze dane
# --------------------------------------------------------------------------- #
def _minimal_conditional_dict() -> dict:
    return {
        "seed": 7,
        "model": {"mode": "conditional", "artists": ["beatles", "queen"]},
    }


def _minimal_per_artist_dict() -> dict:
    return {
        "seed": 7,
        "model": {
            "mode": "per_artist",
            "artists": ["beatles"],
            "reference_artist": "control",
        },
    }


# --------------------------------------------------------------------------- #
# Rozpoznawanie formatu + ładowanie z pliku
# --------------------------------------------------------------------------- #
def test_load_json_by_extension(tmp_path: Path) -> None:
    cfg_path = tmp_path / "exp.json"
    cfg_path.write_text(json.dumps(_minimal_conditional_dict()), encoding="utf-8")

    config = load_config(cfg_path)

    assert isinstance(config, Config)
    assert config.seed == 7
    assert config.model.mode == "conditional"
    assert config.model.artists == ["beatles", "queen"]


def test_load_yaml_by_extension(tmp_path: Path) -> None:
    yaml = pytest.importorskip("yaml")
    cfg_path = tmp_path / "exp.yaml"
    cfg_path.write_text(yaml.safe_dump(_minimal_conditional_dict()), encoding="utf-8")

    config = load_config(cfg_path)

    assert config.seed == 7
    assert config.model.mode == "conditional"


def test_unsupported_extension_raises(tmp_path: Path) -> None:
    cfg_path = tmp_path / "exp.txt"
    cfg_path.write_text("seed: 1", encoding="utf-8")

    with pytest.raises(ConfigValidationError, match="rozszerzenie"):
        load_config(cfg_path)


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigValidationError, match="nie istnieje"):
        load_config(tmp_path / "brak.json")


# --------------------------------------------------------------------------- #
# Walidacja pól wymaganych
# --------------------------------------------------------------------------- #
def test_missing_seed_raises() -> None:
    data = {"model": {"mode": "conditional", "artists": ["a"]}}
    with pytest.raises(ConfigValidationError, match="seed"):
        Config.from_dict(data)


def test_seed_must_be_int() -> None:
    data = {"seed": "42", "model": {"mode": "conditional", "artists": ["a"]}}
    with pytest.raises(ConfigValidationError, match="seed"):
        Config.from_dict(data)


def test_missing_model_mode_raises() -> None:
    data = {"seed": 1, "model": {"artists": ["a"]}}
    with pytest.raises(ConfigValidationError, match="model.mode"):
        Config.from_dict(data)


def test_invalid_mode_raises() -> None:
    data = {"seed": 1, "model": {"mode": "bogus", "artists": ["a"]}}
    with pytest.raises(ConfigValidationError, match="model.mode"):
        Config.from_dict(data)


def test_conditional_requires_artists() -> None:
    data = {"seed": 1, "model": {"mode": "conditional", "artists": []}}
    with pytest.raises(ConfigValidationError, match="model.artists"):
        Config.from_dict(data)


def test_per_artist_requires_reference_artist() -> None:
    data = {"seed": 1, "model": {"mode": "per_artist", "artists": ["beatles"]}}
    with pytest.raises(ConfigValidationError, match="reference_artist"):
        Config.from_dict(data)


def test_per_artist_valid() -> None:
    config = Config.from_dict(_minimal_per_artist_dict())
    assert config.model.mode == "per_artist"
    assert config.model.reference_artist == "control"


# --------------------------------------------------------------------------- #
# Normalizacja i wartości domyślne
# --------------------------------------------------------------------------- #
def test_defaults_applied_for_optional_sections() -> None:
    config = Config.from_dict(_minimal_conditional_dict())

    assert config.experiment_name == "experiment"
    assert isinstance(config.dataset, DatasetConfig)
    assert isinstance(config.training, TrainingConfig)
    assert isinstance(config.ga, GAConfig)
    assert isinstance(config.evaluation, EvaluationConfig)
    # wartości domyślne ze schematu Data Models
    assert config.training.epochs == 200
    assert config.ga.population_size == 100
    assert config.evaluation.plot_format == "pdf"
    assert config.features.pitch_range == (24, 108)


def test_pitch_range_normalized_to_tuple() -> None:
    data = _minimal_conditional_dict()
    data["features"] = {"pitch_range": [12, 96]}
    config = Config.from_dict(data)
    assert config.model  # sanity
    assert config.features.pitch_range == (12, 96)


def test_unknown_keys_are_ignored() -> None:
    data = _minimal_conditional_dict()
    data["model"]["nieznane_pole"] = 123
    data["zupelnie_obca_sekcja"] = {"x": 1}
    # nie powinno rzucać - nieznane klucze są filtrowane
    config = Config.from_dict(data)
    assert config.seed == 7


def test_ga_mutation_sigma_merged_with_defaults() -> None:
    data = _minimal_conditional_dict()
    data["ga"] = {"mutation_sigma": {"transpose_semitones": 3.0}}
    config = Config.from_dict(data)
    # nadpisana wartość
    assert config.ga.mutation_sigma["transpose_semitones"] == 3.0
    # pozostałe domyślne nadal obecne
    assert config.ga.mutation_sigma["velocity_offset"] == 5.0


# --------------------------------------------------------------------------- #
# Równoważność formatów (zalążek Property 17)
# --------------------------------------------------------------------------- #
def test_json_and_yaml_produce_equal_config() -> None:
    yaml = pytest.importorskip("yaml")
    data = _minimal_conditional_dict()

    from_json = parse_config_text(json.dumps(data), "json")
    from_yaml = parse_config_text(yaml.safe_dump(data), "yaml")

    assert from_json == from_yaml


# --------------------------------------------------------------------------- #
# Dostarczone pliki domyślne
# --------------------------------------------------------------------------- #
def test_default_conditional_file_loads() -> None:
    config = load_config(CONFIGS_DIR / "default_conditional.yaml")
    assert config.model.mode == "conditional"
    assert config.model.artists  # niepusta lista


def test_default_per_artist_file_loads() -> None:
    config = load_config(CONFIGS_DIR / "default_per_artist.yaml")
    assert config.model.mode == "per_artist"
    assert config.model.reference_artist


def test_default_ga_file_loads() -> None:
    config = load_config(CONFIGS_DIR / "default_ga.yaml")
    assert config.ga.fitness_metric in ("euclidean", "mahalanobis")
    assert config.seed == 42
