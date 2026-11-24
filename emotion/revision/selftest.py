"""Offline checks for the revision code: no GPU, no network, no model download.

    python -m emotion.revision.selftest
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from emotion.revision import stats as S
from emotion.revision import vectors as V
from emotion.space import ALL_PAIRS, ISEAR_EMOTIONS


def check_vectors() -> None:
    g = torch.Generator().manual_seed(0)
    va = torch.randn(512, generator=g)
    vb = 0.6 * va + 0.8 * torch.randn(512, generator=g)
    vecs = {e: torch.randn(512, generator=g) for e in ISEAR_EMOTIONS}
    r = V.geo_random(va, vb, 7)
    assert abs(V.norm(r) - V.norm(vb)) < 1e-3, "geo: norm"
    ua = va / va.norm()
    assert abs(float(r @ ua) - float(vb @ ua)) < 1e-3, "geo: projection on va"
    assert abs(V.norm(va - r) - V.norm(va - vb)) < 1e-3, "geo: ||va - r|| == ||va - vb||"
    assert abs(V.cos(r, vb)) < 0.999, "geo: r must differ from vb"
    assert V.geo_random(va, vb, 7).equal(r) and not V.geo_random(va, vb, 8).equal(r), "geo: seeding"
    iso = V.iso_random(vb, 3)
    assert abs(V.norm(iso) - V.norm(vb)) < 1e-3, "iso: norm"
    lam = V.lam_a(va, vb)
    assert abs(float((va - vb) @ ua) - float(lam @ ua)) < 1e-3, "lambda*a: same component along va"
    assert V.cos(lam, va) > 0.999 or V.cos(lam, va) < -0.999, "lambda*a: parallel to va"
    p = V.proj_remove(va, vb)
    assert abs(float(p @ vb)) < 1e-3, "proj: orthogonal to vb"
    assert abs(V.norm(V.rescale(va - vb, V.norm(va))) - V.norm(va)) < 1e-3, "rescale"
    c = V.centered(vecs, "anger")
    mu = torch.stack(list(vecs.values())).mean(0)
    assert torch.allclose(c, vecs["anger"] - mu, atol=1e-5), "centered"
    assert V.pair_seed("m", "a", "b", 1, "geo") != V.pair_seed("m", "a", "b", 2, "geo"), "seed rule"
    print("vectors: ok")


def check_stats() -> None:
    """Synthetic matrix where a-b lowers b by exactly 0.1 and nothing else moves."""
    rng = np.random.default_rng(0)
    rows = []
    labels = ["baseline", *ISEAR_EMOTIONS, *ALL_PAIRS] + [f"{p}|{c}" for p in ALL_PAIRS for c in S.P1_CONTROLS]
    base = rng.uniform(0.2, 0.4, (56, 7))
    for lab in labels:
        m = base.copy()
        spec = lab.split("|")[0]
        if lab in ISEAR_EMOTIONS:
            m[:, ISEAR_EMOTIONS.index(lab)] += 0.2
        elif "-" in spec:
            a, b = spec.split("-")
            m[:, ISEAR_EMOTIONS.index(a)] += 0.2
            if "|" not in lab:
                m[:, ISEAR_EMOTIONS.index(b)] -= 0.1
        for pid in range(56):
            rows.append({"steer": lab, "prompt_id": pid, **dict(zip(ISEAR_EMOTIONS, m[pid]))})
    df = pd.DataFrame(rows)
    cells, boots = S.analyse_model(df, "toy", "judge", S.P1_CONTROLS, set(), with_wrong=False)
    assert np.allclose(cells["T_ab"], 0.2) and np.allclose(cells["U_ab"], -0.1), "T/U definitions"
    assert np.allclose(cells["dU_geo1"], -0.1), "dU against control"
    assert (cells["p_iut"] < 0.01).all(), "IUT p-values"
    p = np.array([0.001, 0.01, 0.02, 0.9])
    assert S.bh(p).tolist() == [True, True, True, False] and S.bh(p, by=True).sum() <= 3, "BH/BY"
    df2 = df[df["prompt_id"] >= 8]
    cells2, _ = S.analyse_model(df2, "toy", "judge", [], set(), with_wrong=True)
    assert (cells2["n_ids"] == 48).all(), "exclusion / missing IDs"
    print("stats: ok")


if __name__ == "__main__":
    check_vectors()
    check_stats()
    print("revision selftest passed")
