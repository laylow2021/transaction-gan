"""Utilities for splitting transaction records into train/holdout subsets."""

from __future__ import annotations

import random
from typing import List, Sequence, Tuple

from .data_loader import TransactionRecord


def split_records(
    records: Sequence[TransactionRecord],
    *,
    train_fraction: float = 0.8,
    seed: int = 42,
) -> Tuple[List[TransactionRecord], List[TransactionRecord]]:
    """Split records into train and holdout partitions.

    Args:
        records: Sequence of transaction dicts.
        train_fraction: Fraction of rows to keep for training (0 < fraction <= 1).
        seed: Seed used for deterministic shuffling.
    """

    if not records:
        return [], []
    if not (0.0 < train_fraction <= 1.0):
        raise ValueError("train_fraction must be within (0, 1].")

    indices = list(range(len(records)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    train_count = max(1, min(len(records), int(round(len(records) * train_fraction))))
    train_indices = set(indices[:train_count])

    train_split = [records[idx] for idx in range(len(records)) if idx in train_indices]
    holdout_split = [records[idx] for idx in range(len(records)) if idx not in train_indices]

    return train_split, holdout_split


__all__ = ["split_records"]
