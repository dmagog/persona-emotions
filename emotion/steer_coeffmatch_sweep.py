"""Coeff-matched raw vs SAE-feature steering (Phase 2.3, W2).

The earlier "SAE leaks less" claim doesn't control magnitude: the SAE vector's
diagonal effect was weaker, so "less off-target" might just mean "weaker steering".
To compare fairly we sweep the steering coeff for BOTH vector types (raw and
SAE-feature, both renormed to the same length) and judge the result. Plotting
off-target (sadness leakage) vs diagonal effect gives a "specificity frontier":
if the SAE curve sits below the raw curve (less leakage at the same diagonal),
SAE is genuinely more specific — not just weaker.

One process loads the model once and loops over (vector type, coeff). Per-condition
CSV checkpoint so a crash keeps finished conditions. Judge afterwards.

Usage (GPU):
    python -m emotion.steer_coeffmatch_sweep --model_name google/gemma-2-2b-it \
        --raw-vector-dir emotion_vectors/gemma-2-2b-it \
        --sae-vector-dir emotion_vectors/gemma-sae-feat \
        --layer 12 --coeffs 4,8,16 --per-emotion 4 \
        --out results/steer_coeffmatch_sweep.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from activation_steer import ActivationSteerer
from emotion.classifier_encoder import ClassifierBasedEncoder
from emotion.space import ISEAR_EMOTIONS
from emotion.steer_eval import build_prompt, generate
from emotion.steer_specificity import common_prompts, encode_all


def load_vecs(vdir: Path, layer: int) -> dict:
    return {e: torch.load(vdir / f"{e}_response_avg_diff.pt", map_location="cpu")[layer + 1].float()
            for e in ISEAR_EMOTIONS}


def main() -> None:
    ap = argparse.ArgumentParser(description="Coeff-matched raw vs SAE steering sweep (W2).")
    ap.add_argument("--model_name", default="google/gemma-2-2b-it")
    ap.add_argument("--raw-vector-dir", required=True, type=Path)
    ap.add_argument("--sae-vector-dir", required=True, type=Path)
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--coeffs", default="4,8,16", help="comma-separated steering coeffs to sweep")
    ap.add_argument("--per-emotion", type=int, default=4, help="held-out prompts pooled per emotion")
    ap.add_argument("--max-new-tokens", type=int, default=120)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    coeffs = [float(x) for x in args.coeffs.split(",")]
    vecsets = {"raw": load_vecs(args.raw_vector_dir, args.layer),
               "sae": load_vecs(args.sae_vector_dir, args.layer)}

    model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map="auto", torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    encoder = ClassifierBasedEncoder()

    questions = common_prompts(args.per_emotion)
    prompts = [build_prompt(tokenizer, q) for q in questions]
    fields = ["vtype", "coeff", "steer", "layer", "prompt_id", *ISEAR_EMOTIONS, "answer"]
    rows = []

    def run_block(vtype, coeff, steer_label, vec):
        for pid, prompt in enumerate(prompts):
            if vec is None:
                ans = generate(model, tokenizer, prompt, args.max_new_tokens)
            else:
                with ActivationSteerer(model, vec, coeff=coeff, layer_idx=args.layer, positions="all"):
                    ans = generate(model, tokenizer, prompt, args.max_new_tokens)
            scores = encode_all(encoder, ans)
            rows.append({"vtype": vtype, "coeff": coeff, "steer": steer_label, "layer": args.layer,
                         "prompt_id": pid, "answer": ans, **scores})

    def checkpoint():
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"  checkpoint -> {args.out} ({len(rows)} rows)", flush=True)

    print(f"baseline: {len(prompts)} prompts", flush=True)
    run_block("baseline", 0.0, "baseline", None)
    checkpoint()

    for vtype, vecs in vecsets.items():
        for coeff in coeffs:
            print(f"=== {vtype} coeff={coeff} ===", flush=True)
            for emo in ISEAR_EMOTIONS:
                run_block(vtype, coeff, emo, vecs[emo])
                print(f"  {vtype} c={coeff} steer={emo} done", flush=True)
            checkpoint()

    print(f"DONE wrote {args.out} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
