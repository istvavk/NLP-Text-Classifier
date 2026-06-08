from __future__ import annotations

import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
import torch.serialization
import gradio as gr

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


def run_app() -> None:
    model, vocab = load_model()

    def predict(text: str) -> str:
        if not text.strip():
            return "Please enter some text."
        tokens = tokenize(text)
        encoded = vocab.encode(tokens)[:200]
        padded = encoded + [0] * (200 - len(encoded))
        x = torch.tensor([padded], dtype=torch.long)
        with torch.no_grad():
            output = model(x)
            probs = F.softmax(output, dim=1).squeeze()
            pred_idx = int(torch.argmax(probs).item())
        probs_list = [float(p) for p in probs.tolist()]
        lines = [f"Prediction: {LABELS[pred_idx]}", ""]
        for i, label in enumerate(LABELS):
            lines.append(f"  {label}: {probs_list[i] * 100:.1f}%")
        return "\n".join(lines)

    demo = gr.Interface(
        fn=predict,
        inputs=gr.Textbox(
            lines=10,
            label="Enter match-related text",
            placeholder="Type or paste a sports article here...",
        ),
        outputs=gr.Textbox(label="Result", lines=5),
        title="Match Text Classifier (LSTM)",
        description="Classifies sports match text as PREVIEW (upcoming match) or REPORT (completed match).",
    )
    demo.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    run_app()
