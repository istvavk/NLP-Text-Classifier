"""PyTorch Dataset for text classification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch
from torch.utils.data import Dataset

from utils.preprocessing import tokenize
from utils.vocab import Vocabulary


@dataclass(frozen=True)
class EncodedExample:
    x: torch.Tensor
    y: torch.Tensor


class TextDataset(Dataset[EncodedExample]):
    texts: Sequence[str]
    labels: Sequence[int]
    vocab: Vocabulary
    max_len: int

    def __init__(
        self,
        texts: Sequence[str],
        labels: Sequence[int],
        vocab: Vocabulary,
        max_len: int = 200,
    ) -> None:
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens: List[str] = tokenize(self.texts[idx])
        encoded: List[int] = self.vocab.encode(tokens)[: self.max_len]
        padded: List[int] = encoded + [0] * (self.max_len - len(encoded))
        return torch.tensor(padded, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)
