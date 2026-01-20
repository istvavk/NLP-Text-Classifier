"""A tiny vocabulary helper

Includes type hints (PEP484) and variable annotations (PEP526).

Run doctests:
    python -m doctest -v utils/vocab.py
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List, Sequence


class Vocabulary:
    stoi: Dict[str, int]
    itos: List[str]
    min_freq: int

    def __init__(self, min_freq: int = 2) -> None:
        self.stoi = {"<PAD>": 0, "<UNK>": 1}
        self.itos = ["<PAD>", "<UNK>"]
        self.min_freq = min_freq

    def build(self, tokenized_texts: Iterable[Sequence[str]]) -> None:
        counter: Counter[str] = Counter()
        for tokens in tokenized_texts:
            counter.update(tokens)

        for word, freq in counter.items():
            if freq >= self.min_freq and word not in self.stoi:
                self.stoi[word] = len(self.itos)
                self.itos.append(word)

    def encode(self, tokens: Sequence[str]) -> List[int]:
        unk_idx: int = self.stoi["<UNK>"]
        return [self.stoi.get(t, unk_idx) for t in tokens]
