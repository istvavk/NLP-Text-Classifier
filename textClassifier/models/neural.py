"""Neural model: a compact LSTM classifier."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    """LSTM text classifier with masked mean pooling."""

    embedding: nn.Embedding
    lstm: nn.LSTM
    fc: nn.Linear
    pad_idx: int

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 100,
        hidden_dim: int = 128,
        num_classes: int = 2,
        pad_idx: int = 0,
    ) -> None:
        super().__init__()
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb: torch.Tensor = self.embedding(x)  
        output, _ = self.lstm(emb)  

        # Mask padding to avoid skewing pooling
        mask: torch.Tensor = (x != self.pad_idx).unsqueeze(-1)  
        masked: torch.Tensor = output * mask

        lengths: torch.Tensor = mask.sum(dim=1).clamp(min=1)  
        pooled: torch.Tensor = masked.sum(dim=1) / lengths  
        return self.fc(pooled)
