"""Top-k sweep: how few SAE features are enough to steer? (Phase 3.2, W6 "minimal").

For each k in --ks, build each emotion's steering vector from only its top-k
contrastive SAE features (decode `sum_f delta_f * W_dec[f]`, renorm to the raw
Gemma vector's length so the coeff is comparable across k), steer toward each
emotion on a fixed held-out prompt set, and save the generated answers. One
process loads the model + SAE once and loops over k (cheap vs re-running
steer_specificity per k). Judge the answers afterwards (judge_specificity) to get
a k-vs-quality curve that earns the word "minimal".

Usage (GPU):
    python -m emotion.steer_topk_sweep --model_name google/gemma-2-2b-it \
        --raw-vector-dir emotion_vectors/gemma-2-2b-it \
        --contrastive-json results/sae_contrastive_layer12.json \
        --layer 12 --coeff 8 --per-emotion 4 --ks 1,2,4,8,20 \
        --out results/steer_topk_sweep.csv
Needs HF_TOKEN (GemmaScope gated).
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import json
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from activation_steer import ActivationSteerer
from emotion.classifier_encoder import ClassifierBasedEncoder
from emotion.gemma_sae import load_sae, pick_sae_file
from emotion.space import ISEAR_EMOTIONS
from emotion.steer_eval import build_prompt, generate
from emotion.steer_specificity import common_prompts, encode_all


def build_topk_vec(feats, W_dec, raw, k):
    """Decode top-k contrastive features into a residual-stream vector, renormed
    to the raw vector's length (comparable coeff across k)."""
    vec = torch.zeros(W_dec.shape[1], dtype=torch.float32)
    for idx, delta in feats[:k]:
        vec = vec + float(delta) * W_dec[int(idx)]
    return vec * (raw.norm() / (vec.norm() + 1e-8))


def main() -> None:
    ap = argparse.ArgumentParser(description="Top-k SAE-feature steering sweep (minimal vector).")
    ap.add_argument("--model_name", default="google/gemma-2-2b-it")
    ap.add_argument("--raw-vector-dir", required=True, type=Path, help="dir with raw {emotion}_response_avg_diff.pt (shape + norm match)")
    ap.add_argument("--contrastive-json", required=True, type=Path)
    ap.add_argument("--repo", default="google/gemma-scope-2b-pt-res")
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--width", default="16k")
    ap.add_argument("--l0-target", type=int, default=None)
    ap.add_argument("--coeff", type=float, default=8.0)
    ap.add_argument("--per-emotion", type=int, default=4, help="held-out prompts pooled per emotion")
    ap.add_argument("--ks", default="1,2,4,8,20", help="comma-separated k values to sweep")
    ap.add_argument("--max-new-tokens", type=int, default=120)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    ks = [int(x) for x in args.ks.split(",")]
    sae = load_sae(args.repo, pick_sae_file(args.repo, args.layer, args.width, args.l0_target))
    W_dec = torch.tensor(np.asarray(sae["W_dec"]), dtype=torch.float32)  # [d_sae, d_model]
    cj = json.loads(args.contrastive_json.read_text())
    raw = {e: torch.load(args.raw_vector_dir / f"{e}_response_avg_diff.pt", map_location="cpu")[args.layer + 1].float()
           for e in ISEAR_EMOTIONS}

    model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map="auto", torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    encoder = ClassifierBasedEncoder()

    questions = common_prompts(args.per_emotion)
    prompts = [build_prompt(tokenizer, q) for q in questions]
    fields = ["k", "steer", "layer", "coeff", "prompt_id", *ISEAR_EMOTIONS, "answer"]
    rows = []

    def run_block(k, steer_label, vec):
        for pid, prompt in enumerate(prompts):
            if vec is None:
                ans = generate(model, tokenizer, prompt, args.max_new_tokens)
            else:
                with ActivationSteerer(model, vec, coeff=args.coeff, layer_idx=args.layer, positions="all"):
                    ans = generate(model, tokenizer, prompt, args.max_new_tokens)
            scores = encode_all(encoder, ans)
            rows.append({"k": k, "steer": steer_label, "layer": args.layer, "coeff": args.coeff,
                         "prompt_id": pid, "answer": ans, **scores})

    # baseline is k-independent: generate once, label k=0
    print(f"baseline: {len(prompts)} prompts", flush=True)
    run_block(0, "baseline", None)

    for k in ks:
        print(f"=== k={k} ===", flush=True)
        for emo in ISEAR_EMOTIONS:
            vec = build_topk_vec(cj["emotions"][emo]["top"], W_dec, raw[emo], k)
            run_block(k, emo, vec)
            print(f"  k={k} steer={emo} done", flush=True)
        # checkpoint after each k so a crash keeps finished k's
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"  checkpoint -> {args.out} ({len(rows)} rows)", flush=True)

    print(f"DONE wrote {args.out} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
