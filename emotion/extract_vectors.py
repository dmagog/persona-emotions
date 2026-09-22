"""Extract emotion steering vectors from generated positive and negative responses.

Rows are joined by their question and instruction key, then optionally filtered
by pairwise-judge scores before hidden-state pooling.

Saves ``{emotion}_response_avg_diff.pt`` (+ prompt_avg / prompt_last) =
mean(activations_pos) - mean(activations_neg) per layer.

Usage (GPU):
    python -m emotion.extract_vectors --model_name Qwen/Qwen2.5-3B-Instruct \
        --data-dir eval_emotion/Qwen2.5-3B-Instruct \
        --judge-scores results/judge_scores_qwen2.5-3b.csv \
        --save-dir emotion_vectors/Qwen2.5-3B-Instruct --emotion all
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from emotion.loader import LoadSpec, load_model_and_tokenizer

from emotion.pairwise_judge import _join_key
from emotion.space import ISEAR_EMOTIONS
from emotion.hidden_states import pooled_hidden_states

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))  # 2**31-1: C long is 32-bit on Windows


def _load_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_kept_keys(judge_csv: Path, emotion: str, threshold: float) -> set[str]:
    kept: set[str] = set()
    for row in _load_csv(judge_csv):
        if row.get("emotion") != emotion:
            continue
        try:
            score = float(row["score"])
        except (ValueError, TypeError):
            continue
        if score >= threshold:
            kept.add(row["key"])
    return kept


def build_filtered_pairs(data_dir: Path, emotion: str, kept_keys: set[str]):
    pos = {_join_key(r["question_id"]): r for r in _load_csv(data_dir / f"{emotion}_pos.csv")}
    neg = {_join_key(r["question_id"]): r for r in _load_csv(data_dir / f"{emotion}_neg.csv")}
    keys = [k for k in pos if k in neg and (not kept_keys or k in kept_keys)]
    pos_prompts = [pos[k]["prompt"] for k in keys]
    pos_responses = [pos[k]["answer"] for k in keys]
    neg_prompts = [neg[k]["prompt"] for k in keys]
    neg_responses = [neg[k]["answer"] for k in keys]
    return keys, pos_prompts, pos_responses, neg_prompts, neg_responses


def save_emotion_vector(model, tokenizer, data_dir, emotion, kept_keys, save_dir):
    keys, pp, pr, np_, nr = build_filtered_pairs(data_dir, emotion, kept_keys)
    if not keys:
        raise SystemExit(
            f"[{emotion}] no pairs survived filtering. This emotion used to be "
            "skipped in silence, and the missing vector then surfaced further down "
            "the chain as an unrelated error. Check the pairs and the judge threshold."
        )
    empty = [k for k, r in zip(keys, pr + nr) if not str(r).strip()]
    if empty:
        raise SystemExit(
            f"[{emotion}] empty answers: {len(empty)}. An empty answer gives an empty "
            "activation slice, which gives NaN, and NaN consumes the whole vector "
            f"during averaging. First few: {empty[:3]}"
        )
    pos_pa, pos_pl, pos_ra = pooled_hidden_states(model, tokenizer, pp, pr)
    neg_pa, neg_pl, neg_ra = pooled_hidden_states(model, tokenizer, np_, nr)
    n_layers = len(pos_ra)

    def diff(pos_layers, neg_layers):
        return torch.stack(
            [pos_layers[l].mean(0).float() - neg_layers[l].mean(0).float() for l in range(n_layers)],
            dim=0,
        )

    variants = {
        "prompt_avg": diff(pos_pa, neg_pa),
        "response_avg": diff(pos_ra, neg_ra),
        "prompt_last": diff(pos_pl, neg_pl),
    }
    # Check BEFORE writing: a NaN vector on disk looks ordinary and passes the
    # rest of the chain, yielding 448 generations of garbage without one error.
    for name, vec in variants.items():
        if not torch.isfinite(vec).all():
            bad = int((~torch.isfinite(vec)).sum())
            raise SystemExit(
                f"[{emotion}/{name}] the vector holds {bad} non-finite values out of "
                f"{vec.numel()}. The usual cause is empty answers, or fp16 overflow on a "
                "model trained in bf16. The vector was not saved."
            )
        norms = vec.norm(dim=-1)
        if float(norms.max()) <= 1e-6:
            raise SystemExit(
                f"[{emotion}/{name}] the vector is zero on every layer. It was not saved."
            )

    os.makedirs(save_dir, exist_ok=True)
    for name, vec in variants.items():
        torch.save(vec, f"{save_dir}/{emotion}_{name}_diff.pt")
    print(f"[{emotion}] saved vectors from {len(keys)} pairs -> {save_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract emotion steering vectors from pos/neg responses.")
    ap.add_argument("--model_name", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--dtype", default="auto",
                    help="auto decides from the hardware and the training dtype")
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--emotion", default="all", help="one ISEAR emotion or 'all'")
    ap.add_argument("--judge-scores", type=Path, default=None, help="judge CSV for filtering; omit to use all matched pairs")
    ap.add_argument("--threshold", type=float, default=60.0)
    ap.add_argument("--save-dir", required=True)
    args = ap.parse_args()

    model, tokenizer, load_info = load_model_and_tokenizer(
        LoadSpec(hf_id=args.model_name, dtype=args.dtype), for_generation=False)
    emotions = list(ISEAR_EMOTIONS) if args.emotion == "all" else [args.emotion]
    for emo in emotions:
        kept = load_kept_keys(args.judge_scores, emo, args.threshold) if args.judge_scores else set()
        save_emotion_vector(model, tokenizer, args.data_dir, emo, kept, args.save_dir)


if __name__ == "__main__":
    main()
