"""Emotion measurement: representation spaces and text encoders."""
from emotion import metrics
from emotion.classifier_encoder import (
    GO_EMOTIONS_TO_ISEAR,
    ClassifierBasedEncoder,
    aggregate_scores,
)
from emotion.encoder import EmotionEncoder
from emotion.space import ISEAR_EMOTIONS, DiscreteEmotionSpace, EmotionSpace

__all__ = [
    "EmotionSpace",
    "DiscreteEmotionSpace",
    "ISEAR_EMOTIONS",
    "EmotionEncoder",
    "ClassifierBasedEncoder",
    "GO_EMOTIONS_TO_ISEAR",
    "aggregate_scores",
    "metrics",
]
