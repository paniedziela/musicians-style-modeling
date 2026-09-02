from musicians_style.e3.objective import validate_constraints
from musicians_style.e3.profile import build_target_profile
from musicians_style.e3.transformation import apply_transformation
from musicians_style.e3.types import E3Genome, IDENTITY_GENOME


def _profile(piece):
    row = {"sample_id": "a", "group_id": "train", "sha256": "1", "composer": "target"}
    return build_target_profile("target", [row], {"a": piece})


def test_identity_is_exact(source_piece):
    assert apply_transformation(source_piece, IDENTITY_GENOME, _profile(source_piece), seed=7) is source_piece


def test_operators_are_deterministic_and_preserve_protected_melody(source_piece):
    profile = _profile(source_piece)
    genome = E3Genome(2, 0.7, 0.6, 0.0, 0.4)
    first = apply_transformation(source_piece, genome, profile, seed=7)
    second = apply_transformation(source_piece, genome, profile, seed=7)
    assert first == second
    report = validate_constraints(source_piece, first, profile, genome)
    assert "melody" not in report.violations
    assert first.meta == source_piece.meta
