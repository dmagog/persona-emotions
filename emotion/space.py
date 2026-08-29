"""Emotion representation spaces.

An ``EmotionSpace`` defines how an emotion label is turned into a numeric vector
and back. We start with a discrete one-hot space over the 7 ISEAR emotions
(matching the project dataset); VAD / circumplex spaces can be added later behind
the same interface.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

# Canonical ISEAR emotion set used across the project dataset.
ISEAR_EMOTIONS: tuple[str, ...] = (
    "anger",
    "disgust",
    "fear",
    "guilt",
    "joy",
    "sadness",
    "shame",
)

# Все 42 упорядоченные пары X-Y для стадии композиции. Упорядоченные:
# anger-sadness и sadness-anger — разные операции, обе считаются. Единый
# источник для конфигов, цепочки и сборщика — чтобы «полная матрица пар»
# нигде не расходилась.
ALL_PAIRS: list[str] = [
    f"{a}-{b}" for a in ISEAR_EMOTIONS for b in ISEAR_EMOTIONS if a != b
]


class EmotionSpace(ABC):
    """Maps emotion labels to vectors and interprets vectors back to labels."""

    @property
    @abstractmethod
    def labels(self) -> list[str]:
        """Ordered emotion labels spanning the space."""

    @property
    def dim(self) -> int:
        return len(self.labels)

    @abstractmethod
    def vectorize(self, label: str) -> np.ndarray:
        """Turn an emotion label into a vector in this space."""

    @abstractmethod
    def interpret(self, vector: np.ndarray) -> list[tuple[str, float]]:
        """Turn a vector into (label, score) pairs sorted by descending score."""


class DiscreteEmotionSpace(EmotionSpace):
    """One-hot space over a fixed set of emotion labels (default: ISEAR 7)."""

    def __init__(self, labels: list[str] | None = None) -> None:
        self._labels = list(labels) if labels is not None else list(ISEAR_EMOTIONS)
        if len(set(self._labels)) != len(self._labels):
            raise ValueError("emotion labels must be unique")
        self._index = {label: i for i, label in enumerate(self._labels)}

    @property
    def labels(self) -> list[str]:
        return list(self._labels)

    def vectorize(self, label: str) -> np.ndarray:
        if label not in self._index:
            raise KeyError(f"unknown emotion label: {label!r}")
        vec = np.zeros(self.dim, dtype=np.float32)
        vec[self._index[label]] = 1.0
        return vec

    def interpret(self, vector: np.ndarray) -> list[tuple[str, float]]:
        vector = np.asarray(vector, dtype=np.float32).reshape(-1)
        if vector.shape[0] != self.dim:
            raise ValueError(f"expected vector of dim {self.dim}, got {vector.shape[0]}")
        pairs = list(zip(self._labels, (float(x) for x in vector)))
        pairs.sort(key=lambda kv: kv[1], reverse=True)
        return pairs
