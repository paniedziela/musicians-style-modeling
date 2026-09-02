"""Segment-indexed, split-safe data access for the E4 pilot."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from .segmentation import SegmentMap

COMPOSERS = ("Bach", "Beethoven", "Chopin")


@dataclass(frozen=True)
class PieceSegments:
    """All E4 segments of one score, with its frozen split membership."""

    sample_id: str
    group_id: str
    composer: str
    split: str
    segment_map: SegmentMap


@dataclass(frozen=True)
class SegmentIndexEntry:
    sample_id: str
    group_id: str
    composer: str
    split: str
    segment_index: int


class SegmentDataset(Dataset[dict[str, object]]):
    """A dataset whose rows are nonempty windows, never entire pieces.

    ``pieces`` must contain exactly one requested split.  Split and group checks
    are deliberately performed here too, so a training loader cannot silently
    receive validation/test fragments.
    """

    def __init__(self, pieces: Iterable[PieceSegments], *, split: str, composers: Sequence[str] = COMPOSERS) -> None:
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"unknown split: {split}")
        self.split = split
        self.composers = tuple(composers)
        self.composer_to_index = {name: index for index, name in enumerate(self.composers)}
        self._maps: dict[str, SegmentMap] = {}
        self.index: list[SegmentIndexEntry] = []
        group_splits: dict[str, str] = {}
        for piece in pieces:
            if piece.split != split:
                raise ValueError(f"piece {piece.sample_id} belongs to {piece.split}, not {split}")
            if piece.composer not in self.composer_to_index:
                raise ValueError(f"unknown composer: {piece.composer}")
            previous = group_splits.setdefault(piece.group_id, piece.split)
            if previous != piece.split:
                raise ValueError(f"group_id {piece.group_id} crosses splits")
            if piece.sample_id in self._maps:
                raise ValueError(f"duplicate sample_id: {piece.sample_id}")
            self._maps[piece.sample_id] = piece.segment_map
            self.index.extend(
                SegmentIndexEntry(piece.sample_id, piece.group_id, piece.composer, piece.split, segment.index)
                for segment in piece.segment_map.segments
                if segment.nonempty
            )
        if split == "train" and not self.index:
            raise ValueError("training dataset has no nonempty segments")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int) -> dict[str, object]:
        item = self.index[index]
        segment = self._maps[item.sample_id].segments[item.segment_index]
        return {
            "x": torch.from_numpy(segment.data.astype(np.float32, copy=False)),
            "mask": torch.from_numpy(segment.mask.astype(np.float32, copy=False)),
            "source": torch.tensor(self.composer_to_index[item.composer], dtype=torch.long),
            "sample_id": item.sample_id,
            "group_id": item.group_id,
            "split": item.split,
        }


def assert_split_isolation(datasets: Mapping[str, SegmentDataset]) -> None:
    """Reject sample or group leakage between any E4 data split."""
    expected = {"train", "validation", "test"}
    if set(datasets) != expected:
        raise ValueError("datasets must contain train, validation, and test")
    sample_sets = {name: {entry.sample_id for entry in data.index} for name, data in datasets.items()}
    group_sets = {name: {entry.group_id for entry in data.index} for name, data in datasets.items()}
    for left in expected:
        for right in expected - {left}:
            if sample_sets[left] & sample_sets[right]:
                raise ValueError("sample_id leakage between E4 datasets")
            if group_sets[left] & group_sets[right]:
                raise ValueError("group_id leakage between E4 datasets")


class BalancedComposerSampler(Sampler[int]):
    """Deterministic balanced stream, oversampling minority composers only in train."""

    def __init__(self, dataset: SegmentDataset, *, seed: int = 1729) -> None:
        if dataset.split != "train":
            raise ValueError("balanced sampling is permitted only for the training dataset")
        by_composer: dict[int, list[int]] = defaultdict(list)
        for position, entry in enumerate(dataset.index):
            by_composer[dataset.composer_to_index[entry.composer]].append(position)
        missing = set(range(len(dataset.composers))) - set(by_composer)
        if missing:
            raise ValueError(f"training data misses composer labels: {sorted(missing)}")
        self._by_composer = dict(by_composer)
        self.seed = seed
        self.epoch = 0
        self._per_composer = max(map(len, self._by_composer.values()))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self._per_composer * len(self._by_composer)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        draws: dict[int, list[int]] = {}
        for label, positions in self._by_composer.items():
            source = torch.tensor(positions, dtype=torch.long)
            choice = torch.randint(len(source), (self._per_composer,), generator=generator)
            draws[label] = source[choice].tolist()
        for offset in range(self._per_composer):
            for label in sorted(draws):
                yield draws[label][offset]
