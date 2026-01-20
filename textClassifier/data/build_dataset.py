"""Build a small dataset for the project.

NOTE:
This script uses a simple pseudo-labeling heuristic to split sport news
into PREVIEW vs REPORT. In the seminar report, explicitly discuss that this
introduces label noise.

Run:
    python data/build_dataset.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List

import pandas as pd
from datasets import load_dataset

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
OUT_PATH: Path = PROJECT_ROOT / "data" / "matches.csv"


def make_labeler(preview_keywords: List[str]) -> Callable[[str], int]:
    """Return a labeling function (closure) based on keyword list.

    The returned function captures `keywords` from outer scope.
    """
    keywords = [k.lower() for k in preview_keywords]

    def _label(text: str) -> int:
        t = text.lower()
        return 0 if any(k in t for k in keywords) else 1  

    return _label


def main() -> None:
    print("Loading BBC News dataset...")
    dataset = load_dataset("SetFit/bbc-news", split="train")

    labeler = make_labeler(preview_keywords=["will", "expected", "face", "ahead of", "set to"])

    rows: List[Dict[str, object]] = []
    for item in dataset:
        if item.get("label_text") == "sport":
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            label = labeler(text)
            rows.append({"text": text, "label": label})

    df = pd.DataFrame(rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"Saved dataset to {OUT_PATH} with {len(df)} samples.")


if __name__ == "__main__":
    main()
