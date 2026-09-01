from __future__ import annotations

from musicians_style.e1.splits import build_e1_splits


def _manifest() -> dict:
    samples = []
    for composer in ("Bach", "Chopin", "Beethoven"):
        for number in range(10):
            samples.append(
                {
                    "sample_id": f"{composer.lower()}-{number}",
                    "composer": composer,
                    "group_id": f"{composer.lower()}-group-{number}",
                    "sha256": f"{composer}-{number}",
                    "validation_status": "accepted",
                }
            )
    return {"manifest_schema_version": "test", "code_commit": "abc", "samples": samples}


def test_splits_are_deterministic_complete_and_group_disjoint() -> None:
    first = build_e1_splits(_manifest(), seeds=(11, 22), outer_splits=5, inner_splits=3)
    second = build_e1_splits(_manifest(), seeds=(11, 22), outer_splits=5, inner_splits=3)
    assert first == second
    expected = {sample["sample_id"] for sample in _manifest()["samples"]}
    for repetition in first["repetitions"]:
        test_ids = []
        for fold in repetition["folds"]:
            train = set(fold["train"]["sample_ids"])
            test = set(fold["test"]["sample_ids"])
            assert train.isdisjoint(test)
            assert train | test == expected
            test_ids.extend(test)
            inner_validation = [
                sample_id
                for inner in fold["inner_folds"]
                for sample_id in inner["validation"]["sample_ids"]
            ]
            assert sorted(inner_validation) == sorted(train)
        assert sorted(test_ids) == sorted(expected)
