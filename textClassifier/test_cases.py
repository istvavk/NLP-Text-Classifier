"""Quick manual test cases for the trained model.

This is *not* part of the automated test suite, but useful for demoing.

Run:
    python test_cases.py
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F
import torch.serialization

from models.neural import LSTMClassifier
from utils.preprocessing import tokenize
from utils.vocab import Vocabulary

PROJECT_ROOT: Path = Path(__file__).resolve().parent
MODEL_PATH: Path = PROJECT_ROOT / "saved_models" / "lstm.pt"

LABELS: List[str] = ["PREVIEW", "REPORT"]

torch.serialization.add_safe_globals([Vocabulary])


def load_model() -> tuple[LSTMClassifier, Vocabulary]:
    checkpoint = torch.load(MODEL_PATH, weights_only=False)
    vocab: Vocabulary = checkpoint["vocab"]

    model = LSTMClassifier(vocab_size=len(vocab.itos), pad_idx=0)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, vocab


def predict(model: LSTMClassifier, vocab: Vocabulary, text: str, max_len: int = 200) -> tuple[int, List[float]]:
    tokens = tokenize(text)
    encoded = vocab.encode(tokens)[:max_len]
    padded = encoded + [0] * (max_len - len(encoded))
    x = torch.tensor([padded], dtype=torch.long)

    with torch.no_grad():
        output = model(x)
        probs = F.softmax(output, dim=1).squeeze()
        pred_idx = int(torch.argmax(probs).item())
    return pred_idx, [float(p) for p in probs.tolist()]


def main() -> None:
    model, vocab = load_model()

    test_texts: List[str] = [
        (
            "Wales could follow England's lead by training with a rugby league club "
            "to boost tackling technique. Coaches are expected to trial new sessions."
        ),
        (
            "Manchester United will face Liverpool in the Premier League this weekend. "
            "The teams are expected to fight for top spot."
        ),
        (
            "Tottenham played well but couldn't score, finishing 0-0. Spurs fans were "
            "disappointed by the goalless draw."
        ),
    ]

    for text in test_texts:
        pred_idx, probs = predict(model, vocab, text)
        print("=" * 80)
        print(f"TEXT: {text}\n")
        print(f"TOKENS: {tokenize(text)}")
        print(f"PREDICTION: {LABELS[pred_idx]}")
        print(f"PROBS: {dict(zip(LABELS, probs))}")
        print("=" * 80, end="\n\n")


if __name__ == "__main__":
    main()
