"""Evaluation script"""

from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd
import torch
import torch.serialization
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader

from models.neural import LSTMClassifier
from training.dataset import TextDataset
from training.split import stratified_split
from utils.vocab import Vocabulary

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
DATA_PATH: Path = PROJECT_ROOT / "data" / "matches.csv"
MODEL_PATH: Path = PROJECT_ROOT / "saved_models" / "lstm.pt"

torch.serialization.add_safe_globals([Vocabulary])


def evaluate() -> None:
    checkpoint = torch.load(MODEL_PATH, weights_only=False)
    vocab: Vocabulary = checkpoint["vocab"]

    model = LSTMClassifier(vocab_size=len(vocab.itos), pad_idx=0)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    df = pd.read_csv(DATA_PATH)
    texts: List[str] = df["text"].astype(str).tolist()
    labels: List[int] = df["label"].astype(int).tolist()

    split = stratified_split(labels, seed=42)
    test_texts = [texts[i] for i in split.test_idx]
    test_labels = [labels[i] for i in split.test_idx]

    ds = TextDataset(test_texts, test_labels, vocab=vocab, max_len=200)
    loader = DataLoader(ds, batch_size=16, shuffle=False)

    all_preds: List[int] = []
    all_labels: List[int] = []

    with torch.no_grad():
        for x, y in loader:
            outputs = model(x)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend([int(p) for p in preds.tolist()])
            all_labels.extend([int(v) for v in y.tolist()])

    acc = float(accuracy_score(all_labels, all_preds))
    f1 = float(f1_score(all_labels, all_preds, average="macro"))
    print(f"Test accuracy: {acc:.4f}")
    print(f"Test F1 (macro): {f1:.4f}")
    print("\nClassification report:")
    print(classification_report(all_labels, all_preds, target_names=["PREVIEW", "REPORT"]))


if __name__ == "__main__":
    evaluate()
