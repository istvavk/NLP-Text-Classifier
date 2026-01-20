"""Training script for the LSTM classifier"""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from models.neural import LSTMClassifier
from training.dataset import TextDataset
from training.split import stratified_split
from utils.concurrency import parallel_map
from utils.decorators import measure_time
from utils.preprocessing import tokenize
from utils.vocab import Vocabulary

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
DATA_PATH: Path = PROJECT_ROOT / "data" / "matches.csv"
MODEL_DIR: Path = PROJECT_ROOT / "saved_models"
MODEL_PATH: Path = MODEL_DIR / "lstm.pt"


def _select(items: Sequence[str], idx: Sequence[int]) -> List[str]:
    return [items[i] for i in idx]


def _select_labels(items: Sequence[int], idx: Sequence[int]) -> List[int]:
    return [int(items[i]) for i in idx]


@measure_time
def train_model(
    epochs: int = 10,
    batch_size: int = 16,
    lr: float = 1e-3,
    seed: int = 42,
) -> None:
    torch.manual_seed(seed)

    df = pd.read_csv(DATA_PATH)
    texts: List[str] = df["text"].astype(str).tolist()
    labels: List[int] = df["label"].astype(int).tolist()

    split = stratified_split(labels, seed=seed)
    train_texts = _select(texts, split.train_idx)
    val_texts = _select(texts, split.val_idx)

    train_labels = _select_labels(labels, split.train_idx)
    val_labels = _select_labels(labels, split.val_idx)

    # Build vocab from training data only (prevents leakage)
    tokenized_train: List[List[str]] = parallel_map(tokenize, train_texts, max_workers=4)
    vocab = Vocabulary(min_freq=2)
    vocab.build(tokenized_train)

    train_ds = TextDataset(train_texts, train_labels, vocab=vocab, max_len=200)
    val_ds = TextDataset(val_texts, val_labels, vocab=vocab, max_len=200)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = LSTMClassifier(vocab_size=len(vocab.itos), pad_idx=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    best_val_f1: float = -1.0
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss: float = 0.0

        for x, y in train_loader:
            optimizer.zero_grad()
            outputs = model(x)
            loss = loss_fn(outputs, y)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())

        avg_loss: float = total_loss / max(len(train_loader), 1)

        # Validation
        model.eval()
        val_preds: List[int] = []
        with torch.no_grad():
            for x, _y in val_loader:
                logits = model(x)
                preds = torch.argmax(logits, dim=1)
                val_preds.extend([int(p) for p in preds.tolist()])

        val_f1: float = float(f1_score(val_labels, val_preds, average="macro"))
        print(f"Epoch {epoch}/{epochs} - loss: {avg_loss:.4f} - val_f1(macro): {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save({"model": model.state_dict(), "vocab": vocab}, MODEL_PATH)
            print(f"[SAVE] New best model saved to {MODEL_PATH} (val_f1={best_val_f1:.4f})")

    print(f"Training complete. Best val_f1(macro)={best_val_f1:.4f}")


if __name__ == "__main__":
    train_model()
