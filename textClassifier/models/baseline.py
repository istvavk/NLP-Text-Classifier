"""Baseline model: TF-IDF + Logistic Regression."""

from __future__ import annotations

from typing import Sequence

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from models.interfaces import TrainableTextClassifier


class BaselineClassifier(TrainableTextClassifier):

    vectorizer: TfidfVectorizer
    model: LogisticRegression

    def __init__(self, max_features: int = 5000) -> None:
        self.vectorizer = TfidfVectorizer(max_features=max_features)
        self.model = LogisticRegression(max_iter=1000)

    def fit(self, texts: Sequence[str], labels: Sequence[int]) -> None:
        x = self.vectorizer.fit_transform(texts)
        self.model.fit(x, labels)

    def predict(self, text: str) -> int:
        x = self.vectorizer.transform([text])
        return int(self.model.predict(x)[0])
