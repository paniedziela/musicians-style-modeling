from __future__ import annotations

from musicians_style.e1.features import SCORE_ONLY_INDICES, build_legacy_feature_cache


def test_feature_cache_defines_score_only_ablation() -> None:
    manifest = {
        "manifest_schema_version": "test",
        "samples": [
            {
                "sample_id": "bach-one",
                "composer": "Bach",
                "group_id": "bach-work",
                "sha256": "abc",
                "validation_status": "accepted",
                "legacy_features": list(range(42)),
            }
        ],
    }
    cache = build_legacy_feature_cache(manifest)
    assert len(cache["variants"]["legacy_full"]["indices"]) == 42
    assert cache["variants"]["legacy_score_only"]["indices"] == list(SCORE_ONLY_INDICES)
    assert {0, 38, 39, 40}.isdisjoint(SCORE_ONLY_INDICES)
    assert len(cache["variants"]["legacy_score_only"]["indices"]) == 38
