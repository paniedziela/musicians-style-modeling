import pytest

from musicians_style.e3.profile import build_target_profile


def _row(sample_id, group="g", sha="sha"):
    return {"sample_id": sample_id, "group_id": group, "sha256": sha, "composer": "target"}


def test_profile_is_train_only_and_has_fingerprint(source_piece):
    profile = build_target_profile("target", [_row("a")], {"a": source_piece})
    assert profile.train_sample_ids == ("a",)
    assert len(profile.fingerprint) == 64
    with pytest.raises(ValueError, match="group_id leakage"):
        build_target_profile("target", [_row("a")], {"a": source_piece}, forbidden_rows=[_row("x")])
