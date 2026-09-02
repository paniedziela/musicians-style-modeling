from __future__ import annotations

from collections import Counter

import pytest

from musicians_style.e4.dataset import BalancedComposerSampler, PieceSegments, SegmentDataset, assert_split_isolation
from musicians_style.e4.segmentation import encode_piece
from musicians_style.midi.types import InternalRepr, NoteEvent


def _piece(tick: int, pitch: int = 60) -> InternalRepr:
    return InternalRepr(480, (NoteEvent(tick, 0, pitch, 90, 480),), (), 1)


def test_segment_dataset_is_indexed_balanced_and_split_isolated() -> None:
    train_pieces = [
        PieceSegments("bach", "g-bach", "Bach", "train", encode_piece(_piece(0))),
        PieceSegments("beethoven", "g-beethoven", "Beethoven", "train", encode_piece(_piece(1920))),
        PieceSegments("chopin", "g-chopin", "Chopin", "train", encode_piece(_piece(2 * 1920))),
    ]
    train = SegmentDataset(train_pieces, split="train")
    validation = SegmentDataset([PieceSegments("valid", "g-valid", "Bach", "validation", encode_piece(_piece(0)))], split="validation")
    test = SegmentDataset([PieceSegments("test", "g-test", "Chopin", "test", encode_piece(_piece(0)))], split="test")
    assert len(train) == 3
    assert train[0]["x"].shape == (2, 64, 84)
    assert_split_isolation({"train": train, "validation": validation, "test": test})
    sampled = list(BalancedComposerSampler(train, seed=7))
    assert Counter(int(train[index]["source"]) for index in sampled) == Counter({0: 1, 1: 1, 2: 1})
    with pytest.raises(ValueError, match="belongs to train"):
        SegmentDataset(train_pieces, split="validation")
    leaked = SegmentDataset([PieceSegments("again", "g-bach", "Bach", "test", encode_piece(_piece(0)))], split="test")
    with pytest.raises(ValueError, match="group_id leakage"):
        assert_split_isolation({"train": train, "validation": validation, "test": leaked})
