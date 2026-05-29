"""Off-distribution coeff sweep: does a bigger coeff override a factual prompt? (Phase 5.4b)

Phase 5.4 found that at coeff 8 the SAE-feature vector barely injects emotion into rigid
factual prompts (anger/fear/guilt/sadness => 0). Open question: does a larger coeff
override the instruction (and at what cost to coherence)? Here we sweep the coeff on the
same neutral factual prompts. Output schema matches judge_coeffmatch_sweep (vtype/coeff),
so we reuse that judge + coeffmatch_summary to get diagonal-effect-vs-coeff; run
coherence_check on the same CSV to track text breakage.

Usage (GPU):
    python -m emotion.steer_random_coeff_sweep --model_name google/gemma-2-2b-it \
        --vector-dir emotion_vectors/gemma-sae-feat --layer 12 --coeffs 8,16,24 \
        --out results/steer_random_coeff_sweep.csv
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
from emotion.steer_eval import generate
from emotion.steer_specificity import encode_all
from emotion.steer_random_prompts import RANDOM_PROMPTS, build_neutral


def main() -> None:
    ap = argparse.ArgumentParser(description="Off-distribution coeff sweep on neutral prompts.")
    ap.add_argument("--model_name", default="google/gemma-2-2b-it")
    ap.add_argument("--vector-dir", required=True, type=Path)
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--coeffs", default="8,16,24", help="comma-separated coeffs to sweep")
    ap.add_argument("--max-new-tokens", type=int, default=120)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    coeffs = [float(x) for x in args.coeffs.split(",")]
    model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map="auto", torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    encoder = ClassifierBasedEncoder()
    vecs = {e: torch.load(args.vector_dir / f"{e}_response_avg_diff.pt", map_location="cpu")[args.layer + 1]
            for e in ISEAR_EMOTIONS}

    prompts = [build_neutral(tokenizer, q) for q in RANDOM_PROMPTS]
    fields = ["vtype", "coeff", "steer", "layer", "prompt_id", *ISEAR_EMOTIONS, "answer"]
    rows = []

    def run_block(vtype, coeff, steer_label, vec):
        for pid, prompt in enumerate(prompts):
            if vec is None:
                ans = generate(model, tokenizer, prompt, args.max_new_tokens)
            else:
                with ActivationSteerer(model, vec, coeff=coeff, layer_idx=args.layer, positions="all"):
                    ans = generate(model, tokenizer, prompt, args.max_new_tokens)
            rows.append({"vtype": vtype, "coeff": coeff, "steer": steer_label, "layer": args.layer,
                         "prompt_id": pid, "answer": ans, **encode_all(encoder, ans)})

    def checkpoint():
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"  checkpoint -> {args.out} ({len(rows)} rows)", flush=True)

    print(f"baseline: {len(prompts)} neutral prompts", flush=True)
    run_block("baseline", 0.0, "baseline", None)
    checkpoint()
    for coeff in coeffs:
        print(f"=== coeff={coeff} ===", flush=True)
        for emo in ISEAR_EMOTIONS:
            run_block("sae", coeff, emo, vecs[emo])
            print(f"  c={coeff} steer={emo} done", flush=True)
        checkpoint()
    print(f"DONE wrote {args.out} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
