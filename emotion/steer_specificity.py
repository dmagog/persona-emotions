"""Steering specificity: steer toward each emotion, measure ALL 7 emotion scores.

On a fixed common set of neutral prompts, we generate a baseline (no steering)
and then, for each emotion's vector, a steered generation; every response is
encoded into all 7 ISEAR scores. From the CSV we build the steering specificity
matrix (steer-toward X vs measured Y): the diagonal should rise most if steering
is emotion-specific; off-diagonal entries are leakage (expected, given the high
cross-emotion vector cosines).

Usage (GPU):
    python -m emotion.steer_specificity --vector-dir emotion_vectors/Qwen2.5-3B-Instruct \
        --layer 14 --coeff 8 --per-emotion 2 --out results/steer_specificity_qwen2.5-3b.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from activation_steer import ActivationSteerer
from emotion.classifier_encoder import ClassifierBasedEncoder
from emotion.space import ISEAR_EMOTIONS
from emotion.steer_eval import build_prompt, generate


def common_prompts(per_emotion: int) -> list[str]:
    """Pool a few held-out questions from each emotion into one neutral set."""
    prompts: list[str] = []
    for emo in ISEAR_EMOTIONS:
        qs = json.loads(Path(f"data_generation/emotion_data_eval/{emo}.json").read_text())["questions"]
        prompts.extend(qs[:per_emotion])
    return prompts


def encode_all(encoder: ClassifierBasedEncoder, text: str) -> dict[str, float]:
    vec = encoder.encode(text)
    return {label: round(float(vec[i]), 4) for i, label in enumerate(encoder.space.labels)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Steering specificity matrix across all emotions.")
    ap.add_argument("--model_name", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--vector-dir", required=True, type=Path)
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--coeff", type=float, default=8.0)
    ap.add_argument("--per-emotion", type=int, default=2, help="held-out prompts pooled per emotion")
    ap.add_argument("--max-new-tokens", type=int, default=120)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, device_map="auto", torch_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    encoder = ClassifierBasedEncoder()

    questions = common_prompts(args.per_emotion)
    prompts = [build_prompt(tokenizer, q) for q in questions]
    fields = ["steer", "layer", "coeff", "prompt_id", *ISEAR_EMOTIONS]
    rows = []

    def run_block(steer_label, vec, layer, coeff):
        for pid, prompt in enumerate(prompts):
            if vec is None:
                ans = generate(model, tokenizer, prompt, args.max_new_tokens)
            else:
                with ActivationSteerer(model, vec, coeff=coeff, layer_idx=layer, positions="all"):
                    ans = generate(model, tokenizer, prompt, args.max_new_tokens)
            scores = encode_all(encoder, ans)
            rows.append({"steer": steer_label, "layer": layer, "coeff": coeff, "prompt_id": pid, **scores})

    print(f"baseline + {len(ISEAR_EMOTIONS)} emotions x {len(prompts)} prompts @ layer {args.layer} coeff {args.coeff}")
    run_block("baseline", None, "", 0.0)
    for emo in ISEAR_EMOTIONS:
        vec = torch.load(args.vector_dir / f"{emo}_response_avg_diff.pt", map_location="cpu")[args.layer + 1]
        run_block(emo, vec, args.layer, args.coeff)
        print(f"  done steer={emo}")

    # specificity matrix: mean measured score per (steer, emotion)
    from collections import defaultdict
    agg = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for e in ISEAR_EMOTIONS:
            agg[r["steer"]][e].append(r[e])
    header = "steer\\meas " + " ".join(f"{e[:4]:>5}" for e in ISEAR_EMOTIONS)
    print(header)
    for steer in ["baseline", *ISEAR_EMOTIONS]:
        means = [sum(agg[steer][e]) / len(agg[steer][e]) for e in ISEAR_EMOTIONS]
        print(f"{steer:>9} " + " ".join(f"{m:>5.2f}" for m in means))

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
