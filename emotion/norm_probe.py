"""Замер масштаба активаций модели на слое и пересчёт coeff ↔ безразмерная сила.

Нужен, чтобы наведение было сопоставимо между моделями: S = coeff*||vec|| / mean||h_L||.

Usage:
    python -m emotion.norm_probe --model google/gemma-2-2b-it \
        --vector-dir emotion_vectors/gemma-2-2b-it --layer 12 --coeff 8
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from emotion.loader import LoadSpec, load_model_and_tokenizer

from emotion.space import ISEAR_EMOTIONS
from emotion.steer_eval import build_prompt, load_eval_prompts


def main() -> None:
    ap = argparse.ArgumentParser(description="Масштаб активаций и калибровка силы наведения.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--vector-dir", required=True, type=Path)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--coeff", type=float, default=None,
                    help="если задан — посчитать соответствующую ему безразмерную силу S")
    ap.add_argument("--strength", type=float, default=None,
                    help="если задана — посчитать coeff на каждую эмоцию")
    ap.add_argument("--n-prompts", type=int, default=8)
    args = ap.parse_args()

    model, tokenizer, _ = load_model_and_tokenizer(LoadSpec(hf_id=args.model))
    questions = load_eval_prompts("anger", args.n_prompts)
    prompts = [build_prompt(tokenizer, q) for q in questions]

    norms = []
    with torch.no_grad():
        for prompt in prompts:
            enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
            h = model(**enc, output_hidden_states=True).hidden_states[args.layer + 1][0]
            norms.append(h.norm(dim=-1).mean().item())
    mean_h = sum(norms) / len(norms)

    print(f"модель: {args.model}")
    print(f"слой {args.layer}: mean||h|| = {mean_h:.2f}  (по {len(norms)} промптам)")

    vec_norms = {}
    for emo in ISEAR_EMOTIONS:
        f = args.vector_dir / f"{emo}_response_avg_diff.pt"
        if f.is_file():
            vec_norms[emo] = torch.load(f, map_location="cpu")[args.layer + 1].norm().item()
    if not vec_norms:
        raise SystemExit(f"нет векторов в {args.vector_dir}")
    mean_v = sum(vec_norms.values()) / len(vec_norms)
    print(f"||vec|| среднее {mean_v:.2f} "
          f"(мин {min(vec_norms.values()):.2f}, макс {max(vec_norms.values()):.2f})")

    if args.coeff is not None:
        print(f"\ncoeff {args.coeff} → S = {args.coeff * mean_v / mean_h:.4f} (по средней норме)")
        for emo, vn in vec_norms.items():
            print(f"   {emo:<8} S = {args.coeff * vn / mean_h:.4f}")
    if args.strength is not None:
        print(f"\nS {args.strength} → coeff на эмоцию:")
        for emo, vn in vec_norms.items():
            print(f"   {emo:<8} coeff = {args.strength * mean_h / vn:.2f}")


if __name__ == "__main__":
    main()
