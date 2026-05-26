"""Text -> emotion vector encoders.

An ``EmotionEncoder`` maps arbitrary text to a vector in a given ``EmotionSpace``.
Concrete encoders (classifier-based, VAD-lexicon, SAE-based) implement
``encode``; the convenience methods here build on it via the space's
``interpret``.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from emotion.space import EmotionSpace


class EmotionEncoder(ABC):
    def __init__(self, space: EmotionSpace) -> None:
        self.space = space

    @abstractmethod
    def encode(self, text: str) -> np.ndarray:
        """Map text to a vector of length ``space.dim``."""

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.space.dim), dtype=np.float32)
        return np.stack([self.encode(t) for t in texts])

    def predict(self, text: str) -> list[tuple[str, float]]:
        """Encode then interpret to (label, score) pairs, sorted descending."""
        return self.space.interpret(self.encode(text))

    def top_emotion(self, text: str) -> str:
        return self.predict(text)[0][0]
