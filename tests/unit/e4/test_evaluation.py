from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from musicians_style.e4.evaluation import binary_f1, summarize_records
from musicians_style.e4.experiment import E4Config, _clean_binary_segment, test_e4 as run_outer_test


def test_binary_cleanup_carries_sustains_between_segments() -> None:
    first = np.zeros((2, 2, 3), dtype=np.uint8)
    first[:, 1, 1] = 1
    cleaned_first, orphaned_first, active = _clean_binary_segment(first)

    second = np.zeros((2, 2, 3), dtype=np.uint8)
    second[1, 0, 1] = 1
    cleaned_second, orphaned_second, active = _clean_binary_segment(second, active)

    assert cleaned_first[1, 1, 1] == 1
    assert cleaned_second[1, 0, 1] == 1
    assert orphaned_first == orphaned_second == 0
    assert not active.any()

    cleaned_without_carry, orphaned, _ = _clean_binary_segment(second)
    assert cleaned_without_carry[1, 0, 1] == 0
    assert orphaned == 1


def _record(source: str, target: str, *, delta: float = .1) -> dict:
    return {
        "source_composer": source,
        "target_composer": target,
        "delta_p_target": delta,
        "style_group_gains": {"pitch": .1, "rhythm": .1, "texture": 0.0},
        "artifact_parseable": True,
        "allowed_length_error_ticks": 30,
        "target_max_polyphony": 8,
        "diagnostics": {
            "fallback_segments": 0,
            "nonempty_input_segments": 4,
            "orphan_frame_starts": 0,
            "active_pitches_at_segment_ends": 0,
        },
        "content": {
            "ticks_per_beat_preserved": True,
            "smf_format_preserved": True,
            "meta_preserved": True,
            "bar_count_preserved": True,
            "length_error_ticks": 0,
            "melody_trigram_jaccard": .98,
            "onset_f1": .97,
            "chroma_cosine": .96,
            "nonempty": True,
            "max_polyphony_input": 6,
            "max_polyphony_output": 7,
            "empty_bar_ratio_input": 0.0,
            "empty_bar_ratio_output": 0.0,
        },
    }


def test_frozen_gate_requires_all_content_and_style_contracts() -> None:
    composers = ("Bach", "Beethoven", "Chopin")
    records = [_record(source, target) for source in composers for target in composers if source != target]
    summary = summarize_records(records, melody_min=.95, fallback_max=.01)
    assert set(summary["direction_means"]) == {
        "Bach→Beethoven", "Bach→Chopin", "Beethoven→Bach",
        "Beethoven→Chopin", "Chopin→Bach", "Chopin→Beethoven",
    }
    assert summary["passed"]

    records[0]["content"]["meta_preserved"] = False
    assert not summarize_records(records, melody_min=.95, fallback_max=.01)["passed"]


def test_outer_test_is_locked_before_validation_go(tmp_path: Path) -> None:
    (tmp_path / "best.pt").write_bytes(b"not loaded before the lock")
    with pytest.raises(ValueError, match="locked"):
        run_outer_test(E4Config(None, None, None, tmp_path))


def test_binary_f1_empty_contract() -> None:
    empty = np.zeros(0, dtype=np.uint8)
    assert binary_f1(empty, empty) == 1.0
