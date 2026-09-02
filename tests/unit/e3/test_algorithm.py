from musicians_style.e3.algorithm import E3GeneticAlgorithm, SearchConfig
from musicians_style.e3.profile import build_target_profile


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
