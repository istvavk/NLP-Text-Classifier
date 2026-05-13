from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

# Ensure the textClassifier directory is on the path so bare imports resolve
sys.path.insert(0, str(Path(__file__).resolve().parent))

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


def predict_text(
    model: LSTMClassifier, vocab: Vocabulary, text: str, max_len: int = 200
) -> tuple[int, List[float]]:
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
    parser = argparse.ArgumentParser(
        description="Match text classifier (LSTM) — classifies sports text as PREVIEW or REPORT."
    )
    parser.add_argument(
        "text",
        nargs="?",
        help="Text to classify. If omitted, text is read from stdin.",
    )
    args = parser.parse_args()

    if args.text:
        text = args.text
    else:
        print("Enter text to classify (Ctrl+D / Ctrl+Z to submit):", file=sys.stderr)
        text = sys.stdin.read().strip()

    if not text:
        print("Error: no text provided.", file=sys.stderr)
        sys.exit(1)

    model, vocab = load_model()
    pred_idx, probs = predict_text(model, vocab, text)

    print(f"Prediction: {LABELS[pred_idx]}")
    for i, label in enumerate(LABELS):
        print(f"  {label}: {probs[i] * 100:.1f}%")


if __name__ == "__main__":
    main()
