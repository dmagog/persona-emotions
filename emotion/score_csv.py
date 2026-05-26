"""Score generated emotion responses with a ClassifierBasedEncoder.

Reads ``{emotion}_pos.csv`` / ``{emotion}_neg.csv`` (columns include ``answer``)
produced by the emotion inference step, encodes each answer, and reports whether
positive (emotion-eliciting) responses express the target emotion more than
negative (neutral) ones.

Usage:
    python -m emotion.score_csv --data-dir eval_emotion/Qwen2.5-3B-Instruct \
        --emotion joy --limit 50
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

from emotion import ClassifierBasedEncoder, metrics
from emotion.space import ISEAR_EMOTIONS

csv.field_size_limit(sys.maxsize)


def load_answers(path: Path, limit: int | None = None) -> list[str]:
    answers: list[str] = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            answers.append(row.get("answer", ""))
            if limit and len(answers) >= limit:
                break
    return answers


def score_emotion(
    data_dir: Path, emotion: str, encoder: ClassifierBasedEncoder, limit: int | None = None
) -> dict:
    space = encoder.space
    pos = encoder.encode_batch(load_answers(data_dir / f"{emotion}_pos.csv", limit))
    neg = encoder.encode_batch(load_answers(data_dir / f"{emotion}_neg.csv", limit))
    pos_mean = float(np.mean([metrics.target_score(v, emotion, space) for v in pos])) if len(pos) else 0.0
    neg_mean = float(np.mean([metrics.target_score(v, emotion, space) for v in neg])) if len(neg) else 0.0
    return {
        "emotion": emotion,
        "n_pos": len(pos),
        "n_neg": len(neg),
        "pos_target": pos_mean,
        "neg_target": neg_mean,
        "shift": pos_mean - neg_mean,
        "top1_pos": metrics.top1_accuracy(pos, [emotion] * len(pos), space),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Score emotion responses with a classifier encoder.")
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--emotion", default="all", help="one ISEAR emotion or 'all'")
    ap.add_argument("--limit", type=int, default=None, help="cap rows per file (quick runs)")
    ap.add_argument("--model", default=None, help="override HF classifier model")
    args = ap.parse_args()

    encoder = ClassifierBasedEncoder(model_name=args.model) if args.model else ClassifierBasedEncoder()
    emotions = list(ISEAR_EMOTIONS) if args.emotion == "all" else [args.emotion]

    print(f"{'emotion':<9} {'n_pos':>5} {'n_neg':>5} {'pos':>6} {'neg':>6} {'shift':>7} {'top1_pos':>8}")
    rows = []
    for emo in emotions:
        r = score_emotion(args.data_dir, emo, encoder, args.limit)
        rows.append(r)
        print(
            f"{r['emotion']:<9} {r['n_pos']:>5} {r['n_neg']:>5} "
            f"{r['pos_target']:>6.3f} {r['neg_target']:>6.3f} {r['shift']:>7.3f} {r['top1_pos']:>8.3f}"
        )
    if len(rows) > 1:
        print(f"\nmean shift: {np.mean([r['shift'] for r in rows]):.3f}")


if __name__ == "__main__":
    main()
