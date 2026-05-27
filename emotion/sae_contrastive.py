"""Contrastive SAE feature activations — the proper "minimal emotion vector".

Run the actual judge-filtered pos/neg responses through Gemma + a GemmaScope SAE,
average each SAE feature's (JumpReLU) activation over the response tokens, and
rank features per emotion by mean(pos) - mean(neg). Unlike projecting the
difference *direction* (which stayed entangled, Jaccard 0.84-0.91), this measures
real activations where the pos-neg contrast cancels the shared baseline. Reports
top features per emotion and their cross-emotion Jaccard overlap.

Needs HF_TOKEN (gemma + GemmaScope are gated). Heavy: a Gemma forward per response.

Usage (2070):
    python -m emotion.sae_contrastive --data-dir eval_emotion/Qwen2.5-3B-Instruct \
        --judge-scores results/judge_scores_qwen2.5-3b.csv --layer 12 --width 16k \
        --top-k 20 --out results/sae_contrastive_layer12.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from emotion.extract_vectors import build_filtered_pairs, load_kept_keys
from emotion.gemma_sae import load_sae, pick_sae_file
from emotion.space import ISEAR_EMOTIONS


@torch.no_grad()
def response_feature_means(model, tokenizer, prompts, responses, W_enc, b_enc, thr, layer):
    """Mean SAE feature activation over response tokens, averaged over responses."""
    acc = torch.zeros(W_enc.shape[1], dtype=torch.float64)
    n = 0
    for prompt, answer in zip(prompts, responses):
        inputs = tokenizer(prompt + answer, return_tensors="pt", add_special_tokens=False).to(model.device)
        prompt_len = len(tokenizer.encode(prompt, add_special_tokens=False))
        out = model(**inputs, output_hidden_states=True)
        h = out.hidden_states[layer + 1][0, prompt_len:, :].float()  # [n_resp_tok, d_model]
        del out
        if h.shape[0] == 0:
            continue
        pre = h @ W_enc + b_enc
        acts = torch.where(pre > thr, pre, torch.zeros_like(pre))  # JumpReLU [n_resp_tok, d_sae]
        acc += acts.mean(0).double().cpu()
        n += 1
    return acc / max(n, 1)


def main() -> None:
    ap = argparse.ArgumentParser(description="Contrastive SAE feature activations per emotion.")
    ap.add_argument("--model_name", default="google/gemma-2-2b-it")
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--judge-scores", type=Path, default=None)
    ap.add_argument("--threshold", type=float, default=60.0)
    ap.add_argument("--repo", default="google/gemma-scope-2b-pt-res")
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--width", default="16k")
    ap.add_argument("--l0-target", type=int, default=None)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map="auto", torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    device = next(model.parameters()).device

    sae_file = pick_sae_file(args.repo, args.layer, args.width, args.l0_target)
    print("SAE file:", sae_file)
    sae = load_sae(args.repo, sae_file)
    W_enc = torch.tensor(np.asarray(sae["W_enc"]), dtype=torch.float32, device=device)
    b_enc = torch.tensor(np.asarray(sae["b_enc"]), dtype=torch.float32, device=device)
    thr = torch.tensor(np.asarray(sae["threshold"]), dtype=torch.float32, device=device)
    print("W_enc", tuple(W_enc.shape))

    top_features = {}
    result = {"sae_file": sae_file, "layer": args.layer, "method": "contrastive_activation", "emotions": {}}
    for emo in ISEAR_EMOTIONS:
        kept = load_kept_keys(args.judge_scores, emo, args.threshold) if args.judge_scores else set()
        keys, pp, pr, np_, nr = build_filtered_pairs(args.data_dir, emo, kept)
        pos = response_feature_means(model, tokenizer, pp, pr, W_enc, b_enc, thr, args.layer)
        neg = response_feature_means(model, tokenizer, np_, nr, W_enc, b_enc, thr, args.layer)
        diff = (pos - neg).numpy()
        order = np.argsort(diff)[::-1][: args.top_k]
        feats = [(int(i), round(float(diff[i]), 4)) for i in order]
        top_features[emo] = {i for i, _ in feats}
        result["emotions"][emo] = {"n_pairs": len(keys), "top": feats}
        print(f"{emo:>8}: n={len(keys):>3} top5={[i for i,_ in feats[:5]]} (max d={feats[0][1]:.3f})")

    print("\nJaccard overlap of contrastive top-k features:")
    jacc = []
    for a, b in itertools.combinations(ISEAR_EMOTIONS, 2):
        sa, sb = top_features[a], top_features[b]
        j = len(sa & sb) / len(sa | sb) if (sa | sb) else 0.0
        jacc.append(j)
    result["mean_offdiag_jaccard"] = sum(jacc) / len(jacc)
    print(f"mean off-diagonal Jaccard: {result['mean_offdiag_jaccard']:.3f} (cf. direction-projection 0.84-0.91)")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
