"""Build per-emotion steering vectors from contrastive SAE features.

Decode each emotion's top contrastive SAE features back into the residual stream:
``vec = sum_f delta_f * W_dec[f]`` (delta_f = mean(pos)-mean(neg) activation from
sae_contrastive). This is the sparse "minimal emotion vector" in residual space.
Renorm to the raw Gemma emotion vector's length (so a steering coeff is comparable)
and save in the same ``{emotion}_response_avg_diff.pt`` [n_layers+1, d_model]
layout (only the SAE layer's row set), so `steer_specificity.py` can steer with it
unchanged.

Needs HF_TOKEN (GemmaScope gated).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from emotion.gemma_sae import load_sae, pick_sae_file
from emotion.space import ISEAR_EMOTIONS


def main() -> None:
    ap = argparse.ArgumentParser(description="Decode contrastive SAE features into steering vectors.")
    ap.add_argument("--contrastive-json", required=True, type=Path)
    ap.add_argument("--raw-vector-dir", required=True, type=Path, help="dir with raw {emotion}_response_avg_diff.pt (for shape + norm match)")
    ap.add_argument("--repo", default="google/gemma-scope-2b-pt-res")
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--width", default="16k")
    ap.add_argument("--l0-target", type=int, default=None)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()

    sae = load_sae(args.repo, pick_sae_file(args.repo, args.layer, args.width, args.l0_target))
    W_dec = torch.tensor(np.asarray(sae["W_dec"]), dtype=torch.float32)  # [d_sae, d_model]
    cj = json.loads(args.contrastive_json.read_text())
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for emo in ISEAR_EMOTIONS:
        raw_all = torch.load(args.raw_vector_dir / f"{emo}_response_avg_diff.pt", map_location="cpu")
        raw = raw_all[args.layer + 1].float()
        vec = torch.zeros(W_dec.shape[1], dtype=torch.float32)
        for idx, delta in cj["emotions"][emo]["top"]:
            vec = vec + float(delta) * W_dec[int(idx)]
        vec = vec * (raw.norm() / (vec.norm() + 1e-8))  # match raw norm for comparable coeff
        full = torch.zeros_like(raw_all, dtype=torch.float32)
        full[args.layer + 1] = vec
        torch.save(full, args.out_dir / f"{emo}_response_avg_diff.pt")
        print(f"{emo:>8}: built from {len(cj['emotions'][emo]['top'])} features, |vec|={vec.norm():.2f}")
    print(f"wrote vectors to {args.out_dir}")


if __name__ == "__main__":
    main()
