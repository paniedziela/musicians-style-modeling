import pytest

from musicians_style.e3 import algorithm
from musicians_style.e3.algorithm import E3GeneticAlgorithm, SearchConfig
from musicians_style.e3.profile import build_target_profile
from musicians_style.e3.structure import analyse_structure
from musicians_style.e3.types import IDENTITY_GENOME


@pytest.fixture
def target_profile(source_piece):
    row = {"sample_id": "a", "group_id": "train", "sha256": "1", "composer": "target"}
    return build_target_profile("target", [row], {"a": source_piece})


def test_small_search_is_deterministic_and_never_worse_than_identity(source_piece):
    row = {"sample_id": "a", "group_id": "train", "sha256": "1", "composer": "target"}
    profile = build_target_profile("target", [row], {"a": source_piece})
    config = SearchConfig(population_size=4, generations=2, elitism_k=1, stagnation_generations=2)
    first = E3GeneticAlgorithm(config).run(source_piece, profile, seed=11)
    second = E3GeneticAlgorithm(config).run(source_piece, profile, seed=11)
    assert first.genome == second.genome
    assert first.output == second.output
    assert first.history == second.history
    assert first.evaluation.style_gain >= 0.0
    assert first.unique_candidates > 0


def test_explicit_defaults_match_implicit_defaults(source_piece, target_profile):
    search = E3GeneticAlgorithm(SearchConfig(
        population_size=4, generations=3, elitism_k=1, stagnation_generations=4,
    ))
    default = search.run(source_piece, target_profile, seed=11)
    progress = []
    explicit = search.run(
        source_piece, target_profile, seed=11, progress=progress.append,
        canonicalize=algorithm._clamp, transform=algorithm.apply_transformation,
    )

    assert explicit == default
    assert tuple(progress) == default.history


def test_callbacks_precede_cache_lookup_and_are_local_to_run(source_piece, target_profile):
    config = SearchConfig(
        population_size=4, generations=3, elitism_k=1, stagnation_generations=4,
    )
    search = E3GeneticAlgorithm(config)
    default = search.run(source_piece, target_profile, seed=11)
    canonicalized = []
    transformed = []

    def canonicalize(genome):
        canonicalized.append(genome)
        return IDENTITY_GENOME

    def transform(source, genome, profile, *, seed, structure):
        assert source is source_piece
        assert profile is target_profile
        assert seed == 11
        assert structure == analyse_structure(source_piece)
        transformed.append(genome)
        return source

    result = search.run(
        source_piece, target_profile, seed=11,
        canonicalize=canonicalize, transform=transform,
    )

    # Identity plus four evaluations per generation; three offspring per step.
    assert len(canonicalized) == 17 + 9
    assert transformed == [IDENTITY_GENOME]
    assert result.genome == IDENTITY_GENOME
    assert result.output == source_piece
    assert result.unique_candidates == 1
    assert result.cache_hits == 16
    assert search.run(source_piece, target_profile, seed=11) == default


@pytest.mark.parametrize("callback", ["canonicalize", "transform"])
def test_callback_errors_propagate(source_piece, target_profile, callback):
    def fail(*args, **kwargs):
        raise RuntimeError("callback failed")

    search = E3GeneticAlgorithm(SearchConfig(population_size=2, generations=0))
    with pytest.raises(RuntimeError, match="callback failed"):
        search.run(source_piece, target_profile, seed=11, **{callback: fail})
