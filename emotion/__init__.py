"""Emotion measurement: representation spaces and text encoders."""
from emotion import metrics
from emotion.encoder import EmotionEncoder
from emotion.space import ISEAR_EMOTIONS, DiscreteEmotionSpace, EmotionSpace

__all__ = [
    "EmotionSpace",
    "DiscreteEmotionSpace",
    "ISEAR_EMOTIONS",
    "EmotionEncoder",
    "metrics",
]
