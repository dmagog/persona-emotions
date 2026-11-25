"""Additive prediction check (paper Table 5) and its p2 variant (rule 6).

Published prediction for ordered difference a-b and component k:
    pred_pub   = D(+a)_k - D(+b)_k          (D = condition mean - baseline mean)
p2 variant, with the -b effect measured directly:
    pred_p2    = D(+a)_k + D(-b)_k          (D(-b) from runs_revision/<slug>/p2.csv,
                                             relative to the p2 baseline)
Reference predictor that ignores b (asked for in the response to reviewers):
    pred_aonly = D(+a)_k
Floor-aware proportional prediction, component b only (-v_b scales b down
multiplicatively instead of shifting it, so it cannot push b below zero):
    level_b(a-b) = level_b(+a) * level_b(-b) / level_b(base_p2)
    pred_prop    = level_b(a-b) - level_b(base_pub)      (k = b; pred_p2 elsewhere)
Observed: D(a-b)_k from the published run. Units: 11 models x 42 pairs x 7
components; encoder / B_k, judge / 100. Means use the prompt IDs available for
every condition involved, separately within the published and the p2 run.

Uncertainty: two-level bootstrap (models with replacement, then pairs with
replacement within each drawn model; all 7 components of a pair move together),
2000 resamples, seed 1234. Reported as 95% percentile intervals.

--exclude-degenerate drops every (model, b) whose p2 neg_b cell has more than
10% degenerate answers, by the published pipeline's rule
(emotion.run_model_chain._rep_ratio > 0.15: share of repeated word 4-grams; the
same 10% ceiling pick_operating_point uses). The report then shows the analysis
with and without the exclusion.

    python -m emotion.revision.additivity --exclude-degenerate --out runs_revision/additivity.md
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from emotion.revision.stats import B_K, EMO, ALL_PAIRS
from emotion.run_model_chain import _rep_ratio

warnings.filterwarnings("ignore", category=RuntimeWarning)
REPO = Path(__file__).resolve().parents[2]
N_BOOT = 2000
SEED = 1234
DEGEN_RATIO, DEGEN_MAX = 0.15, 0.10
PREDS = [("published", "pred_pub"), ("+a only", "pred_aonly"), ("measured -b (p2)", "pred_p2"),
         ("proportional on b (p2)", "pred_prop")]


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


def degeneration(slug: str) -> dict[str, float]:
    """Share of degenerate answers per p2 condition (neg_<b>), published rule."""
    p = REPO / "runs_revision" / slug / "p2.csv"
    if not p.is_file():
        return {}
    d = pd.read_csv(p, usecols=["steer", "answer"])
    d["degen"] = d["answer"].map(lambda t: _rep_ratio(t) > DEGEN_RATIO)
    return d.groupby("steer")["degen"].mean().to_dict()


def metrics(pred: np.ndarray, obs: np.ndarray) -> dict:
    ok = np.isfinite(pred) & np.isfinite(obs)
    x, y = pred[ok], obs[ok]
    r = np.corrcoef(x, y)[0, 1]
    rho = pd.Series(x).rank().corr(pd.Series(y).rank())
    ccc = 2 * np.cov(x, y, bias=True)[0, 1] / (x.var() + y.var() + (x.mean() - y.mean()) ** 2)
    slope = np.polyfit(x, y, 1)[0]
    return {"n": int(ok.sum()), "r": r, "rho": rho, "CCC": ccc, "slope": slope, "MAE": np.abs(x - y).mean()}


def _take(v: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """v[idx] with index -1 meaning padding (NaN)."""
    return np.append(np.asarray(v, float), np.nan)[idx]


def _fast(x: np.ndarray, y: np.ndarray):
    """r, CCC, slope over the last axis (rows = bootstrap replicates); NaN = padding."""
    mx, my = np.nanmean(x, -1, keepdims=True), np.nanmean(y, -1, keepdims=True)
    dx, dy = x - mx, y - my
    cov = np.nanmean(dx * dy, -1)
    vx, vy = np.nanmean(dx ** 2, -1), np.nanmean(dy ** 2, -1)
    r = cov / np.sqrt(vx * vy)
    ccc = 2 * cov / (vx + vy + (mx[..., 0] - my[..., 0]) ** 2)
    return r, ccc, cov / vx


def boot_rows(df: pd.DataFrame, rng) -> np.ndarray:
    """(N_BOOT, n) row positions: models, then pairs within model, components together."""
    pos = (df.reset_index().groupby(["model", "pair"])["index"].apply(list))
    n_m = df["model"].nunique()
    per = pos.groupby(level=0).size()
    width = pos.map(len)
    if per.nunique() != 1 or width.nunique() != 1:     # e.g. after excluding cells: slow path
        groups = {m: list(g) for m, g in pos.groupby(level=0)}
        models = list(groups)
        out = []
        for _ in range(N_BOOT):
            rows = [r for m in rng.choice(len(models), len(models))
                    for p in rng.integers(0, len(groups[models[m]]), len(groups[models[m]]))
                    for r in groups[models[m]][p]]
            out.append(np.array(rows))
        n = max(len(o) for o in out)        # replicates differ in size: pad with -1 (-> NaN, see _take)
        return np.stack([np.pad(o, (0, n - len(o)), constant_values=-1) for o in out])
    P, C = int(per.iloc[0]), int(width.iloc[0])
    arr = np.array(pos.tolist()).reshape(n_m, P, C)        # (models, pairs, components)
    mi = rng.integers(0, n_m, (N_BOOT, n_m))
    pi = rng.integers(0, P, (N_BOOT, n_m, P))
    return arr[mi[..., None], pi].reshape(N_BOOT, -1)


def run(instrument: str, models: list[str] | None):
    rows = []
    for d in sorted((REPO / "runs").iterdir()):
        pub = d / ("compose_allpairs.csv" if instrument == "encoder" else "compose_allpairs_judge_wide.csv")
        p2 = REPO / "runs_revision" / d.name / ("p2.csv" if instrument == "encoder" else "p2_judge_wide.csv")
        if not pub.is_file() or (models and d.name not in models):
            continue
        P = _load(pub, instrument)
        N = _load(p2, instrument) if p2.is_file() else None
        degen = degeneration(d.name)
        for spec in ALL_PAIRS:
            a, b = spec.split("-")
            ib = EMO.index(b)
            ids = _ids(P, ["baseline", a, b, spec])
            base = _mean(P, "baseline", ids)
            La = _mean(P, a, ids)
            Da, Db, Dab = (_mean(P, x, ids) - base for x in (a, b, spec))
            if N is not None:
                nids = _ids(N, ["baseline", f"neg_{b}"])
                Nb, Nneg = _mean(N, "baseline", nids), _mean(N, f"neg_{b}", nids)
                Dnb = Nneg - Nb
                ratio = Nneg[ib] / Nb[ib] if Nb[ib] > 1e-9 else np.nan
                prop_b = La[ib] * ratio - base[ib]
            else:
                Dnb, prop_b = np.full(7, np.nan), np.nan
            for k, e in enumerate(EMO):
                comp = "a" if e == a else "b" if e == b else "other"
                rows.append({"model": d.name, "pair": spec, "a": a, "b": b, "k": e, "component": comp,
                             "obs": Dab[k], "pred_pub": Da[k] - Db[k], "pred_aonly": Da[k],
                             "pred_p2": Da[k] + Dnb[k],
                             "pred_prop": prop_b if comp == "b" else Da[k] + Dnb[k],
                             "D_plus_b": Db[k], "D_neg_b": Dnb[k],
                             "degen_neg_b": degen.get(f"neg_{b}", np.nan)})
    return pd.DataFrame(rows)


def section(df: pd.DataFrame, inst: str, rng_seed: int) -> list[str]:
    both = df.dropna(subset=["pred_p2"]).reset_index(drop=True)
    L = [f"Models with p2: {both['model'].nunique()} of {df['model'].nunique()}; "
         f"(model, b) cells used: {both[['model', 'b']].drop_duplicates().shape[0]}; units: {len(both)}.", ""]
    if not len(both):
        return L
    idx = boot_rows(both, np.random.default_rng(rng_seed))
    obs = both["obs"].to_numpy(float)
    ob = _take(obs, idx)
    pub_r, pub_c, pub_s = _fast(_take(both["pred_pub"], idx), ob)

    L += ["| Prediction | units | r [95% CI] | CCC [95% CI] | slope [95% CI] | MAE | dCCC vs published [95% CI] |",
          "|---|---:|---|---|---|---:|---|"]
    rule = {}
    for name, col in PREDS:
        x = both[col].to_numpy(float)
        m = metrics(x, obs)
        br, bc, bs = _fast(_take(x, idx), ob)
        ci = lambda v: f"[{np.nanpercentile(v, 2.5):.3f}, {np.nanpercentile(v, 97.5):.3f}]"
        dc = "" if col == "pred_pub" else f"{m['CCC'] - metrics(both.pred_pub.to_numpy(float), obs)['CCC']:+.3f} {ci(bc - pub_c)}"
        L.append(f"| {name} | {m['n']} | {m['r']:.3f} {ci(br)} | {m['CCC']:.3f} {ci(bc)} | "
                 f"{m['slope']:.2f} {ci(bs)} | {m['MAE']:.3f} | {dc} |")
        rule[col] = (m, bc - pub_c)

    # per component
    L += ["", "Per component (CCC; published / +a only / measured -b / proportional on b; "
              "dCCC measured -b minus published with 95% CI):", "",
          "| Component | units | published | +a only | measured -b | proportional | dCCC (-b vs pub) |",
          "|---|---:|---:|---:|---:|---:|---|"]
    for comp in ("a", "b", "other"):
        sub = both[both["component"] == comp].reset_index(drop=True)
        si = boot_rows(sub, np.random.default_rng(rng_seed + 1))
        so = sub["obs"].to_numpy(float)
        c = {col: metrics(sub[col].to_numpy(float), so)["CCC"] for _, col in PREDS}
        _, bp, _ = _fast(_take(sub["pred_pub"], si), _take(so, si))
        _, b2, _ = _fast(_take(sub["pred_p2"], si), _take(so, si))
        d = b2 - bp
        L.append(f"| k = {comp} | {len(sub)} | {c['pred_pub']:.3f} | {c['pred_aonly']:.3f} | {c['pred_p2']:.3f} | "
                 f"{c['pred_prop']:.3f} | {c['pred_p2'] - c['pred_pub']:+.3f} "
                 f"[{np.percentile(d, 2.5):+.3f}, {np.percentile(d, 97.5):+.3f}] |")
    own = both[both["component"] == "b"]
    L += ["", f"On component b: mean observed D(a-b) {own.obs.mean():+.3f}; mean D(+b) {own.D_plus_b.mean():+.3f}, "
              f"mean D(-b) {own.D_neg_b.mean():+.3f}; corr(D(+b), -D(-b)) over pairs "
              f"{np.corrcoef(own.D_plus_b, -own.D_neg_b)[0, 1]:.2f}; mean prediction: published "
              f"{own.pred_pub.mean():+.3f}, measured -b {own.pred_p2.mean():+.3f}, proportional {own.pred_prop.mean():+.3f}."]

    s = both.groupby("model")[["pred_pub", "pred_p2", "obs"]].apply(lambda g: pd.Series({
        "r_pub": np.corrcoef(g.pred_pub, g.obs)[0, 1], "r_p2": np.corrcoef(g.pred_p2, g.obs)[0, 1]}))
    L += ["", "Per-model r (published / measured -b): " +
          ", ".join(f"{m} {r.r_pub:.2f}/{r.r_p2:.2f}" for m, r in s.iterrows())]

    m_pub, m_p2 = rule["pred_pub"][0], rule["pred_p2"][0]
    d = rule["pred_p2"][1]
    ok = m_p2["CCC"] >= m_pub["CCC"] and abs(1 - m_p2["slope"]) <= abs(1 - m_pub["slope"])
    lo, hi = np.percentile(d, [2.5, 97.5])
    sig = "CI excludes 0" if lo > 0 or hi < 0 else "CI includes 0, so the direction is not established"
    L += ["", f"**Rule 6 ({inst}), as pre-registered (point estimates):** measured -b CCC {m_p2['CCC']:.3f} vs "
              f"{m_pub['CCC']:.3f}, slope {m_p2['slope']:.2f} vs {m_pub['slope']:.2f} -> "
              f"{'no worse: keep partial composition' if ok else 'worse: narrow the claim'}. "
              f"Uncertainty: dCCC {m_p2['CCC'] - m_pub['CCC']:+.3f} [{lo:+.3f}, {hi:+.3f}], {sig}. "
              f"+a only reaches CCC {rule['pred_aonly'][0]['CCC']:.3f}, so measured -b adds "
              f"{m_p2['CCC'] - rule['pred_aonly'][0]['CCC']:+.3f} over ignoring b."]
    return L + [""]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--instruments", default="encoder,judge")
    ap.add_argument("--models", default=None)
    ap.add_argument("--exclude-degenerate", action="store_true",
                    help="also report the analysis without (model, b) cells whose p2 neg_b is >10%% degenerate")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    models = args.models.split(",") if args.models else None
    L = ["# Additive prediction: published (D(+a) - D(+b)) vs measured -b (D(+a) + D(-b))", "",
         f"Bootstrap: models, then pairs within model, {N_BOOT} resamples, seed {SEED}; 95% percentile intervals.",
         f"Degeneration rule: share of answers with repeated word-4-gram ratio > {DEGEN_RATIO} "
         f"(emotion.run_model_chain._rep_ratio, the published pipeline's rule); a p2 cell is degenerate above "
         f"{DEGEN_MAX:.0%}, the ceiling pick_operating_point uses.", ""]
    frames = []
    for inst in args.instruments.split(","):
        df = run(inst, models)
        df["instrument"] = inst
        frames.append(df)
        L += [f"## {inst}: all p2 cells", ""] + section(df, inst, SEED)
        if args.exclude_degenerate:
            bad = df[df["degen_neg_b"] > DEGEN_MAX][["model", "b", "degen_neg_b"]].drop_duplicates()
            L += [f"## {inst}: degenerate p2 cells excluded", "",
                  "Excluded (model, b, degenerate share of neg_b): " +
                  (", ".join(f"{r.model} {r.b} {r.degen_neg_b:.0%}" for r in bad.itertuples()) or "none") + ".", ""]
            keep = ~df.set_index(["model", "b"]).index.isin(bad.set_index(["model", "b"]).index)
            L += section(df[keep].reset_index(drop=True), inst, SEED)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(L) + "\n")
    pd.concat(frames).to_csv(args.out.with_suffix(".csv"), index=False)
    print("\n".join(L))


if __name__ == "__main__":
    main()
