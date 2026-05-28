"""Compositional steering: steer with linear combinations of emotion vectors.

This is where activation steering could beat prompting (which can't easily dose a
blend or subtract an emotion): addition (vec[A]+vec[B] -> blend) and subtraction
(vec[A]-vec[B] -> induce A while suppressing B). Generates baseline + the single
emotions involved + the compositions on neutral prompts, saving answers for the
judge. Vectors from a {emotion}_response_avg_diff.pt dir (e.g. gemma-sae-feat).

Compositions: comma-separated specs like 'joy+sadness,joy-sadness,anger-sadness'.

Usage (GPU):
    python -m emotion.steer_compose --vector-dir emotion_vectors/gemma-sae-feat \
        --model_name google/gemma-2-2b-it --layer 12 --coeff 8 --per-emotion 8 \
        --specs joy+sadness,joy-sadness,anger-sadness --out results/compose_gemma.csv
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from activation_steer import ActivationSteerer
from emotion.classifier_encoder import ClassifierBasedEncoder
from emotion.space import ISEAR_EMOTIONS
from emotion.steer_eval import build_prompt, generate
from emotion.steer_specificity import common_prompts, encode_all

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))


def parse_spec(spec: str) -> list[tuple[float, str]]:
    """'joy+sadness' -> [(+1,joy),(+1,sadness)]; 'joy-sadness' -> [(+1,joy),(-1,sadness)]."""
    return [(-1.0 if sign == "-" else 1.0, emo) for sign, emo in re.findall(r"([+-]?)([a-z]+)", spec)]


def main() -> None:
    ap = argparse.ArgumentParser(description="Compositional (additive/subtractive) emotion steering.")
    ap.add_argument("--model_name", default="google/gemma-2-2b-it")
    ap.add_argument("--vector-dir", required=True, type=Path)
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--coeff", type=float, default=8.0)
    ap.add_argument("--per-emotion", type=int, default=8)
    ap.add_argument("--specs", required=True, help="comma-separated, e.g. joy+sadness,joy-sadness")
    ap.add_argument("--max-new-tokens", type=int, default=120)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map="auto", torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    encoder = ClassifierBasedEncoder()

    questions = common_prompts(args.per_emotion)
    prompts = [build_prompt(tokenizer, q) for q in questions]
    vecs = {
        emo: torch.load(args.vector_dir / f"{emo}_response_avg_diff.pt", map_location="cpu")[args.layer + 1]
        for emo in ISEAR_EMOTIONS
    }
    rows = []

    def run(label, vec):
        for pid, prompt in enumerate(prompts):
            if vec is None:
                ans = generate(model, tokenizer, prompt, args.max_new_tokens)
            else:
                with ActivationSteerer(model, vec, coeff=args.coeff, layer_idx=args.layer, positions="all"):
                    ans = generate(model, tokenizer, prompt, args.max_new_tokens)
            rows.append({"steer": label, "prompt_id": pid, "answer": ans, **encode_all(encoder, ans)})
        print(f"  done {label}")

    specs = [s.strip() for s in args.specs.split(",") if s.strip()]
    singles = sorted({emo for spec in specs for _, emo in parse_spec(spec)})

    run("baseline", None)
    for emo in singles:
        run(emo, vecs[emo])  # single-emotion references on the same prompts
    for spec in specs:
        combined = sum((sign * vecs[emo] for sign, emo in parse_spec(spec)), torch.zeros_like(vecs[singles[0]]))
        run(spec, combined)

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["steer", "prompt_id", *ISEAR_EMOTIONS, "answer"], extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
