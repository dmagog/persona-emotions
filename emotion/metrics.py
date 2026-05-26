"""Metrics over emotion vectors for evaluating encoders and steering.

All functions operate on plain vectors (length ``space.dim``) so they work for
any ``EmotionEncoder`` output, independent of the model that produced it.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from emotion.space import EmotionSpace


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def target_score(vector: np.ndarray, label: str, space: EmotionSpace) -> float:
    """Score the vector assigns to one emotion label."""
    idx = space.labels.index(label)
    return float(np.asarray(vector, dtype=np.float32).reshape(-1)[idx])


def top1_accuracy(
    vectors: Sequence[np.ndarray], true_labels: Sequence[str], space: EmotionSpace
) -> float:
    """Fraction of vectors whose argmax label matches the true label."""
    n = len(true_labels)
    if n == 0:
        return 0.0
    correct = sum(space.interpret(v)[0][0] == lbl for v, lbl in zip(vectors, true_labels))
    return correct / n


def separability_auc(pos_scores: Sequence[float], neg_scores: Sequence[float]) -> float:
    """AUC = P(random pos score > random neg score), ties counted as 0.5.

    Threshold-free, scale-robust measure of how well the target-emotion score
    separates positive (emotion-eliciting) from negative (neutral) responses.
    0.5 = no separation, 1.0 = perfect. Computed via average-rank Mann-Whitney U.
    """
    pos = np.asarray(pos_scores, dtype=np.float64).reshape(-1)
    neg = np.asarray(neg_scores, dtype=np.float64).reshape(-1)
    n_pos, n_neg = len(pos), len(neg)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="mergesort")
    sorted_v = allv[order]
    ranks = np.empty(len(allv), dtype=np.float64)
    i = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and sorted_v[j + 1] == sorted_v[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    rank_sum_pos = float(ranks[:n_pos].sum())
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def mean_target_shift(
    pos_vectors: Sequence[np.ndarray],
    neg_vectors: Sequence[np.ndarray],
    label: str,
    space: EmotionSpace,
) -> float:
    """mean(target score over pos) - mean(target score over neg).

    The core "did steering work" signal: how much more the target emotion is
    expressed in positive responses than in neutral (negative) ones.
    """
    pos = [target_score(v, label, space) for v in pos_vectors]
    neg = [target_score(v, label, space) for v in neg_vectors]
    mp = float(np.mean(pos)) if pos else 0.0
    mn = float(np.mean(neg)) if neg else 0.0
    return mp - mn
