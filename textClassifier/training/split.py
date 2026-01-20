"""Train/val/test split helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class Split:
    train_idx: List[int]
    val_idx: List[int]
    test_idx: List[int]


def stratified_split(
    labels: Sequence[int],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Split:
    """Create stratified indices for train/val/test."""
    labels_np = np.asarray(labels)
    rng = np.random.default_rng(seed)

    train_idx: List[int] = []
    val_idx: List[int] = []
    test_idx: List[int] = []

    for cls in np.unique(labels_np):
        cls_idx = np.where(labels_np == cls)[0]
        rng.shuffle(cls_idx)

        n = len(cls_idx)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        n_train = min(max(n_train, 1), n - 2) if n >= 3 else max(n_train, 1)
        n_val = min(max(n_val, 1), n - n_train - 1) if n - n_train >= 2 else 0

        train_idx.extend(cls_idx[:n_train].tolist())
        val_idx.extend(cls_idx[n_train : n_train + n_val].tolist())
        test_idx.extend(cls_idx[n_train + n_val :].tolist())

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    return Split(train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)
