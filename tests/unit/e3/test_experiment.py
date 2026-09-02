from pathlib import Path

import numpy as np

from musicians_style.e3.experiment import (
    _bootstrap_group_ci,
    _holm,
    _sign_permutation_p,
    load_e3_config,
)


def test_repository_config_loads_with_frozen_budget():
    root = Path(__file__).resolve().parents[3]
    config = load_e3_config(root / "configs" / "e3_asap.yaml")
    assert config.search.population_size == 32
    assert config.search.generations == 60
    assert config.search.stagnation_generations == 12


def test_report_statistics_are_deterministic_and_holm_is_monotone():
    values = np.asarray([0.1, 0.2, 0.3])
    assert _bootstrap_group_ci(values, seed=7, samples=50) == _bootstrap_group_ci(values, seed=7, samples=50)
    assert _sign_permutation_p(values, seed=7, permutations=99) <= 1.0
    adjusted = _holm({"a": 0.01, "b": 0.04, "c": 0.2})
    assert adjusted["a"] <= adjusted["b"] <= adjusted["c"]
