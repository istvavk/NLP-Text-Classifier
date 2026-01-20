from __future__ import annotations

from pathlib import Path
from typing import List

import tkinter as tk

import torch
import torch.nn.functional as F
import torch.serialization

from models.neural import LSTMClassifier
from utils.preprocessing import tokenize
from utils.vocab import Vocabulary

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
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


def predict_text(model: LSTMClassifier, vocab: Vocabulary, text: str, max_len: int = 200) -> tuple[int, List[float]]:
    tokens = tokenize(text)
    encoded = vocab.encode(tokens)[:max_len]
    padded = encoded + [0] * (max_len - len(encoded))

    x = torch.tensor([padded], dtype=torch.long)
    with torch.no_grad():
        output = model(x)
        probs = F.softmax(output, dim=1).squeeze()
        pred_idx = int(torch.argmax(probs).item())
    return pred_idx, [float(p) for p in probs.tolist()]


def run_app() -> None:
    model, vocab = load_model()

    root = tk.Tk()
    root.title("Match text classifier (LSTM)")

    tk.Label(root, text="Enter match-related text:").pack(padx=10, pady=(10, 0))

    text_box = tk.Text(root, height=10, width=70)
    text_box.pack(padx=10, pady=10)

    output_label = tk.Label(root, text="", justify="left")
    output_label.pack(padx=10, pady=(0, 10))

    def on_predict() -> None:
        text = text_box.get("1.0", tk.END).strip()
        if not text:
            output_label.config(text="Please enter some text.")
            return

        pred_idx, probs = predict_text(model, vocab, text)
        prob_text = "\n".join([f"{LABELS[i]}: {probs[i] * 100:.1f}%" for i in range(len(probs))])
        output_label.config(text=f"Prediction: {LABELS[pred_idx]}\n\n{prob_text}")

    tk.Button(root, text="Predict", command=on_predict).pack(pady=(0, 10))
    root.mainloop()


if __name__ == "__main__":
    run_app()
