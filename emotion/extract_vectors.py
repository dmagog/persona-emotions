"""Extract emotion steering vectors from generated pos/neg responses.

Reuses ``generate_vec.get_hidden_p_and_r`` (same residual-stream pooling as
PERSONA) so the saved ``.pt`` files are consumable by ``activation_steer.py``.
Differs from ``generate_vec`` only in row selection: pos/neg rows are joined by
their (question, instruction) key and filtered by the pairwise-judge scores
(``results/judge_scores_*.csv``, score >= threshold), instead of the inline
0-100 columns ``generate_vec`` expects.

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

from emotion.pairwise_judge import _join_key
from emotion.space import ISEAR_EMOTIONS
from generate_vec import get_hidden_p_and_r

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
            f"[{emotion}] после фильтрации не осталось пар. Раньше эмоция молча "
            "пропускалась, и дальше по цепочке отсутствие вектора всплывало как "
            "непонятная ошибка. Проверьте пары и порог судьи."
        )
    empty = [k for k, r in zip(keys, pr + nr) if not str(r).strip()]
    if empty:
        raise SystemExit(
            f"[{emotion}] пустых ответов: {len(empty)}. Пустой ответ даёт пустой срез "
            "активаций, тот даёт NaN, и NaN съедает весь вектор при усреднении. "
            f"Первые: {empty[:3]}"
        )
    pos_pa, pos_pl, pos_ra = get_hidden_p_and_r(model, tokenizer, pp, pr)
    neg_pa, neg_pl, neg_ra = get_hidden_p_and_r(model, tokenizer, np_, nr)
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
    # Проверять ДО записи: NaN-вектор на диске выглядит как обычный и проходит
    # всю дальнейшую цепочку, давая 448 генераций мусора без единой ошибки.
    for name, vec in variants.items():
        if not torch.isfinite(vec).all():
            bad = int((~torch.isfinite(vec)).sum())
            raise SystemExit(
                f"[{emotion}/{name}] в векторе {bad} нечисловых значений из {vec.numel()}. "
                "Обычная причина — пустые ответы или переполнение fp16 на модели, "
                "обученной в bf16. Вектор не сохранён."
            )
        norms = vec.norm(dim=-1)
        if float(norms.max()) <= 1e-6:
            raise SystemExit(
                f"[{emotion}/{name}] вектор нулевой на всех слоях. Вектор не сохранён."
            )

    os.makedirs(save_dir, exist_ok=True)
    for name, vec in variants.items():
        torch.save(vec, f"{save_dir}/{emotion}_{name}_diff.pt")
    print(f"[{emotion}] saved vectors from {len(keys)} pairs -> {save_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract emotion steering vectors from pos/neg responses.")
    ap.add_argument("--model_name", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--emotion", default="all", help="one ISEAR emotion or 'all'")
    ap.add_argument("--judge-scores", type=Path, default=None, help="judge CSV for filtering; omit to use all matched pairs")
    ap.add_argument("--threshold", type=float, default=60.0)
    ap.add_argument("--save-dir", required=True)
    args = ap.parse_args()

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, device_map="auto", torch_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    emotions = list(ISEAR_EMOTIONS) if args.emotion == "all" else [args.emotion]
    for emo in emotions:
        kept = load_kept_keys(args.judge_scores, emo, args.threshold) if args.judge_scores else set()
        save_emotion_vector(model, tokenizer, args.data_dir, emo, kept, args.save_dir)


if __name__ == "__main__":
    main()
