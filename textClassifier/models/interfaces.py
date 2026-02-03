"""Interfaces, protocols and ABCs"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol, Sequence


class TextClassifier(ABC):
    """Abstract base class for text classifiers."""

    @abstractmethod
    def predict(self, text: str) -> int:
        raise NotImplementedError


class TrainableTextClassifier(TextClassifier, ABC):
    """ABC for models that can be trained."""

    @abstractmethod
    def fit(self, texts: Sequence[str], labels: Sequence[int]) -> None:
        raise NotImplementedError


class SupportsPredict(Protocol):

    def predict(self, text: str) -> int:  
        ...
