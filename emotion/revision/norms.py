"""Plan p6: relative perturbation strength S (R1 M3, Eq. 12 of the paper).

S = coeff * ||v|| / mean ||h||, where h is the residual-stream output of the
steered block (``hidden_states[layer + 1]``) on the 56 held-out prompts,
averaged over all prompt tokens without steering. Reported for the seven single
directions, all 42 ordered differences, and the p1 controls.
Only forward passes, no generation.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from emotion.revision import vectors as V
from emotion.space import ALL_PAIRS, ISEAR_EMOTIONS
from emotion.steer_eval import build_prompt


@torch.no_grad()
def run_p6(model, tok, questions, vecs, layer, coeff, slug, out_path: Path) -> None:
    norms = []
    for q in questions:
        enc = tok(build_prompt(tok, q), return_tensors="pt", add_special_tokens=False).to(model.device)
        hs = model(**enc, output_hidden_states=True).hidden_states[layer + 1][0].float()
        norms.append(hs.norm(dim=-1).mean().item())
    h = sum(norms) / len(norms)

    def s(v):
        return coeff * V.norm(v) / h

    single = {e: {"norm": V.norm(vecs[e]), "S": s(vecs[e])} for e in ISEAR_EMOTIONS}
    pairs = {}
    for spec in ALL_PAIRS:
        a, b = spec.split("-")
        va, vb = vecs[a], vecs[b]
        geo = V.geo_random(va, vb, V.pair_seed(slug, a, b, 1, "geo"))
        iso = V.iso_random(vb, V.pair_seed(slug, a, b, 1, "iso"))
        pairs[spec] = {
            "cos_ab": V.cos(va, vb),
            "S_ab": s(va - vb), "S_a": s(va), "ratio_ab_a": V.norm(va - vb) / V.norm(va),
            "S_geo": s(va - geo), "S_iso": s(va - iso), "S_lam": s(V.lam_a(va, vb)),
            "S_proj": s(V.proj_remove(va, vb)),
        }
    out = {"slug": slug, "layer": layer, "coeff": coeff, "mean_residual_norm": h,
           "n_prompts": len(questions), "single": single, "pairs": pairs}
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path}: mean ||h|| = {h:.2f}")
