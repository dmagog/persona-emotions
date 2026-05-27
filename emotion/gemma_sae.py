"""Decompose Gemma emotion vectors with a GemmaScope SAE (minimal emotion vector).

For each emotion's Gemma vector (response_avg_diff at the SAE's layer), run it
through a GemmaScope JumpReLU SAE to get sparse feature activations. Report the
top-k features per emotion (a sparse, interpretable "minimal emotion vector") and
the cross-emotion feature overlap (Jaccard) — testing whether emotions occupy
*distinct* SAE features (disentangled) or *shared* ones (entangled, as the dense
cosines suggested).

GemmaScope residual SAEs: repo ``google/gemma-scope-2b-pt-res``, files
``layer_{L}/width_{W}/average_l0_{X}/params.npz`` with keys
W_enc[d_model,d_sae], W_dec[d_sae,d_model], b_enc[d_sae], b_dec[d_model],
threshold[d_sae] (JumpReLU: acts = pre * (pre > threshold), pre = x@W_enc+b_enc).

Usage (on the box with HF token):
    python -m emotion.gemma_sae --vector-dir emotion_vectors/gemma-2-2b-it \
        --layer 12 --width 16k --top-k 20 --out results/gemma_sae_layer12.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch


def pick_sae_file(repo: str, layer: int, width: str, l0_target: int | None) -> str:
    from huggingface_hub import list_repo_files

    files = list_repo_files(repo)
    pat = re.compile(rf"layer_{layer}/width_{width}/average_l0_(\d+)/params\.npz$")
    cands = [(int(m.group(1)), f) for f in files for m in [pat.search(f)] if m]
    if not cands:
        raise SystemExit(f"no SAE params for layer {layer} width {width} in {repo}")
    if l0_target is None:
        # pick the median-l0 SAE (a balanced sparsity)
        cands.sort()
        return cands[len(cands) // 2][1]
    return min(cands, key=lambda c: abs(c[0] - l0_target))[1]


def load_sae(repo: str, filename: str) -> dict:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo, filename)
    npz = np.load(path)
    return {k: npz[k] for k in npz.files}


def encode(vec: np.ndarray, sae: dict) -> np.ndarray:
    pre = vec @ sae["W_enc"] + sae["b_enc"]
    return np.where(pre > sae["threshold"], pre, 0.0)


def main() -> None:
    ap = argparse.ArgumentParser(description="GemmaScope SAE decomposition of emotion vectors.")
    ap.add_argument("--vector-dir", required=True, type=Path)
    ap.add_argument("--repo", default="google/gemma-scope-2b-pt-res")
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--width", default="16k")
    ap.add_argument("--l0-target", type=int, default=None)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--center", action="store_true",
                    help="subtract the mean emotion vector (remove shared affect), renorm, before SAE projection")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    from emotion.space import ISEAR_EMOTIONS

    sae_file = pick_sae_file(args.repo, args.layer, args.width, args.l0_target)
    print(f"SAE file: {sae_file}")
    sae = load_sae(args.repo, sae_file)
    print("SAE shapes:", {k: tuple(v.shape) for k, v in sae.items()})

    # load each emotion's direction at the SAE layer (residual after block L)
    vecs = {
        emo: torch.load(args.vector_dir / f"{emo}_response_avg_diff.pt", map_location="cpu")[args.layer + 1].float()
        for emo in ISEAR_EMOTIONS
    }
    if args.center:
        mean_vec = torch.stack([vecs[e] for e in ISEAR_EMOTIONS]).mean(0)
        for e in ISEAR_EMOTIONS:
            centered = vecs[e] - mean_vec
            vecs[e] = centered * (vecs[e].norm() / (centered.norm() + 1e-8))  # renorm to original length
        print("centered: subtracted mean emotion vector before SAE projection")

    top_features = {}
    result = {"sae_file": sae_file, "layer": args.layer, "centered": args.center, "emotions": {}}
    for emo in ISEAR_EMOTIONS:
        v = vecs[emo].numpy()
        acts = encode(v, sae)
        nz = int((acts > 0).sum())
        order = np.argsort(acts)[::-1][: args.top_k]
        feats = [(int(i), round(float(acts[i]), 4)) for i in order if acts[i] > 0]
        top_features[emo] = {i for i, _ in feats}
        result["emotions"][emo] = {"n_active": nz, "top": feats}
        print(f"{emo:>8}: active={nz:>4}  top5={[i for i,_ in feats[:5]]}")

    # cross-emotion overlap (Jaccard of top-k feature sets)
    print("\nJaccard overlap of top-k features:")
    print("        " + " ".join(f"{e[:4]:>5}" for e in ISEAR_EMOTIONS))
    overlap = {}
    for a in ISEAR_EMOTIONS:
        row = []
        for b in ISEAR_EMOTIONS:
            sa, sb = top_features[a], top_features[b]
            j = len(sa & sb) / len(sa | sb) if (sa | sb) else 0.0
            row.append(j)
        overlap[a] = dict(zip(ISEAR_EMOTIONS, row))
        print(f"{a:>7} " + " ".join(f"{x:>5.2f}" for x in row))
    result["jaccard"] = overlap

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
