"""Classifier-based emotion encoder.

Wraps a pretrained HuggingFace text-classification model that scores text over
its own emotion labels, then aggregates those scores into a target
``EmotionSpace`` via a label map. Default: ``SamLowe/roberta-base-go_emotions``
(GoEmotions, 28 labels) aggregated to the 7 ISEAR emotions.

GoEmotions does not have ISEAR's ``guilt`` / ``shame`` directly, so we map the
closest fine-grained labels (guilt<-remorse, shame<-embarrassment). This mapping
is a modelling choice — override ``label_map`` to change it.
"""
from __future__ import annotations

import numpy as np

from emotion.encoder import EmotionEncoder
from emotion.space import ISEAR_EMOTIONS, DiscreteEmotionSpace, EmotionSpace

DEFAULT_MODEL = "SamLowe/roberta-base-go_emotions"

# target ISEAR label -> GoEmotions source labels whose scores are summed into it.
# Unmapped GoEmotions labels (confusion, curiosity, realization, surprise,
# disapproval, neutral) are intentionally dropped.
GO_EMOTIONS_TO_ISEAR: dict[str, list[str]] = {
    "anger": ["anger", "annoyance"],
    "disgust": ["disgust"],
    "fear": ["fear", "nervousness"],
    "guilt": ["remorse"],
    "joy": [
        "joy", "amusement", "excitement", "gratitude", "love", "optimism",
        "pride", "relief", "admiration", "approval", "caring", "desire",
    ],
    "sadness": ["sadness", "grief", "disappointment"],
    "shame": ["embarrassment"],
}


def aggregate_scores(
    label_scores: dict[str, float],
    label_map: dict[str, list[str]],
    space: EmotionSpace,
    normalize: bool = True,
) -> np.ndarray:
    """Sum mapped source-label scores into a vector in ``space`` order."""
    vec = np.zeros(space.dim, dtype=np.float32)
    for i, target in enumerate(space.labels):
        vec[i] = sum(label_scores.get(src, 0.0) for src in label_map.get(target, []))
    if normalize:
        total = float(vec.sum())
        if total > 0.0:
            vec /= total
    return vec


class ClassifierBasedEncoder(EmotionEncoder):
    def __init__(
        self,
        space: EmotionSpace | None = None,
        model_name: str = DEFAULT_MODEL,
        label_map: dict[str, list[str]] | None = None,
        device: int | str | None = None,
        normalize: bool = True,
    ) -> None:
        super().__init__(space or DiscreteEmotionSpace(list(ISEAR_EMOTIONS)))
        self.model_name = model_name
        self.label_map = label_map or GO_EMOTIONS_TO_ISEAR
        self.device = device
        self.normalize = normalize
        self._pipeline = None

    def _load(self):
        if self._pipeline is None:
            from transformers import pipeline

            self._pipeline = pipeline(
                "text-classification",
                model=self.model_name,
                top_k=None,
                device=self.device,
            )
        return self._pipeline

    def encode(self, text: str) -> np.ndarray:
        clf = self._load()
        preds = clf(text, truncation=True)
        if preds and isinstance(preds[0], list):  # pipeline returns [[...]] for one input
            preds = preds[0]
        label_scores = {p["label"]: float(p["score"]) for p in preds}
        return aggregate_scores(label_scores, self.label_map, self.space, self.normalize)
