"""Additive prediction check (paper Table 5) and its p2 variant (rule 6).

Published prediction for ordered difference a-b and component k:
    pred_pub = D(+a)_k - D(+b)_k            (D = condition mean - baseline mean)
p2 variant, with the -b effect measured directly:
    pred_p2  = D(+a)_k + D(-b)_k            (D(-b) from runs_revision/<slug>/p2.csv,
                                             relative to the p2 baseline)
Observed: D(a-b)_k from the published run. Units: 11 models x 42 pairs x 7
components; encoder / B_k, judge / 100. Means use the prompt IDs available for
every condition involved (intersection), as in the paper.

    python -m emotion.revision.additivity --out runs_revision/additivity.md
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from emotion.revision.stats import B_K, EMO, ALL_PAIRS

REPO = Path(__file__).resolve().parents[2]


def _load(path: Path, instrument: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[["steer", "prompt_id", *EMO]].copy()
    df["prompt_id"] = df["prompt_id"].astype(int)
    df = df.groupby(["steer", "prompt_id"], as_index=False)[EMO].mean()
    if instrument == "encoder":
        for e in EMO:
            df[e] = df[e] / B_K[e]
    else:
        df[EMO] = df[EMO] / 100.0
    return df.set_index(["steer", "prompt_id"])


def _mean(df, label, ids):
    return df.loc[label].reindex(ids)[EMO].mean().to_numpy(float)


def _ids(df, labels):
    s = None
    for lab in labels:
        cur = set(df.loc[lab].index) if lab in df.index.get_level_values(0) else set()
        s = cur if s is None else s & cur
    return sorted(s)


def metrics(pred: np.ndarray, obs: np.ndarray) -> dict:
    ok = np.isfinite(pred) & np.isfinite(obs)
    x, y = pred[ok], obs[ok]
    r = np.corrcoef(x, y)[0, 1]
    rho = pd.Series(x).rank().corr(pd.Series(y).rank())
    ccc = 2 * np.cov(x, y, bias=True)[0, 1] / (x.var() + y.var() + (x.mean() - y.mean()) ** 2)
    slope = np.polyfit(x, y, 1)[0]
    return {"n": int(ok.sum()), "r": r, "rho": rho, "CCC": ccc, "slope": slope, "MAE": np.abs(x - y).mean()}


def run(instrument: str, models: list[str] | None):
    rows = []
    for d in sorted((REPO / "runs").iterdir()):
        pub = d / ("compose_allpairs.csv" if instrument == "encoder" else "compose_allpairs_judge_wide.csv")
        p2 = REPO / "runs_revision" / d.name / ("p2.csv" if instrument == "encoder" else "p2_judge_wide.csv")
        if not pub.is_file() or (models and d.name not in models):
            continue
        P = _load(pub, instrument)
        N = _load(p2, instrument) if p2.is_file() else None
        for spec in ALL_PAIRS:
            a, b = spec.split("-")
            ids = _ids(P, ["baseline", a, b, spec])
            base = _mean(P, "baseline", ids)
            Da, Db, Dab = (_mean(P, x, ids) - base for x in (a, b, spec))
            if N is not None:
                nids = _ids(N, ["baseline", f"neg_{b}"])
                Dnb = _mean(N, f"neg_{b}", nids) - _mean(N, "baseline", nids)
            else:
                Dnb = np.full(7, np.nan)
            for k, e in enumerate(EMO):
                rows.append({"model": d.name, "pair": spec, "k": e, "obs": Dab[k],
                             "pred_pub": Da[k] - Db[k], "pred_p2": Da[k] + Dnb[k],
                             "D_plus_b": Db[k], "D_neg_b": Dnb[k]})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--instruments", default="encoder,judge")
    ap.add_argument("--models", default=None)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    models = args.models.split(",") if args.models else None
    L = ["# Additive prediction: published (D(+a) - D(+b)) vs measured -b (D(+a) + D(-b))", ""]
    frames = []
    for inst in args.instruments.split(","):
        df = run(inst, models)
        df["instrument"] = inst
        frames.append(df)
        both = df.dropna(subset=["pred_p2"])
        L += [f"## {inst}", "",
              f"Models with p2: {both['model'].nunique()} of {df['model'].nunique()}.", "",
              "| Prediction | units | r | rho | CCC | slope | MAE |", "|---|---:|---:|---:|---:|---:|---:|"]
        for name, sub, col in (("published, all models", df, "pred_pub"),
                               ("published, models with p2", both, "pred_pub"),
                               ("measured -b (p2)", both, "pred_p2")):
            if not len(sub):
                L.append(f"| {name} | 0 | | | | | |")
                continue
            m = metrics(sub[col].to_numpy(float), sub["obs"].to_numpy(float))
            L.append(f"| {name} | {m['n']} | {m['r']:.3f} | {m['rho']:.3f} | {m['CCC']:.3f} | "
                     f"{m['slope']:.2f} | {m['MAE']:.3f} |")
        if len(both):
            s = both.groupby("model").apply(lambda g: pd.Series({
                "r_pub": np.corrcoef(g.pred_pub, g.obs)[0, 1], "r_p2": np.corrcoef(g.pred_p2, g.obs)[0, 1]}))
            L += ["", "Per-model r (published / measured -b): " +
                  ", ".join(f"{m} {r.r_pub:.2f}/{r.r_p2:.2f}" for m, r in s.iterrows())]
            # symmetry of -v_b vs +v_b on the measured component b itself
            own = both[both["k"] == both["pair"].str.split("-").str[1]]
            L += ["", f"On component b: mean D(+b) {own.D_plus_b.mean():+.3f}, mean D(-b) {own.D_neg_b.mean():+.3f}, "
                      f"corr(D(+b), -D(-b)) over pairs {np.corrcoef(own.D_plus_b, -own.D_neg_b)[0, 1]:.2f}"]
            m_pub = metrics(both.pred_pub.to_numpy(float), both.obs.to_numpy(float))
            m_p2 = metrics(both.pred_p2.to_numpy(float), both.obs.to_numpy(float))
            ok = m_p2["CCC"] >= m_pub["CCC"] and abs(1 - m_p2["slope"]) <= abs(1 - m_pub["slope"])
            L += ["", f"**Rule 6 ({inst}):** measured -b CCC {m_p2['CCC']:.3f} vs {m_pub['CCC']:.3f}, slope "
                      f"{m_p2['slope']:.2f} vs {m_pub['slope']:.2f} -> "
                      f"{'no worse: keep partial composition' if ok else 'worse: narrow the claim'}."]
        L.append("")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(L) + "\n")
    pd.concat(frames).to_csv(args.out.with_suffix(".csv"), index=False)
    print("\n".join(L))


if __name__ == "__main__":
    main()
