"""Pair-level statistics for the revision (requirements S1-S5, E1, E3-E5, overlap check).

No GPU and no API: works on the published matrices and on the new runs.

For a model, an ordered pair (a, b) and a condition X:
    T(X) = mean_i [ s_a(X, i) - s_a(baseline, i) ]     target retention
    U(X) = mean_i [ s_b(X, i) - s_b(a-only, i) ]       attenuation (negative = b lower)
Scores are range-normalised as in the paper: encoder / B_k (number of GoEmotions
labels mapped to the ISEAR category), judge / 100.

Inference (pre-registered, REVISION_EXPERIMENTS.md section 2):
* one paired bootstrap per model: prompt IDs are resampled jointly for every
  condition, stratified by the 7 prompt categories (8 IDs each, pid // 8);
* one-sided bootstrap p-values, intersection-union test for "T > 0 and U < 0"
  (p = max(p_T, p_U));
* Benjamini-Hochberg over the 462 cells per instrument, Benjamini-Yekutieli as
  sensitivity;
* controls: dU = U(a-b) - U(control) on the same measured b; grid mean with CI.

Usage:
    # published 42-pair matrices, all prompts and without the 8 anger prompts
    # that were also used for calibration:
    python -m emotion.revision.stats --source published --out docs_revision/stats_published.md
    python -m emotion.revision.stats --source published --exclude-prompts 0-7 \
        --out docs_revision/stats_published_no_calib.md
    # new runs (plan p1) once encoder scores and judge wide CSVs exist:
    python -m emotion.revision.stats --source p1 --out docs_revision/stats_p1.md
"""
from __future__ import annotations

import argparse
import warnings
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from emotion.space import ALL_PAIRS, ISEAR_EMOTIONS

warnings.filterwarnings("ignore", category=RuntimeWarning)
REPO = Path(__file__).resolve().parents[2]
EMO = list(ISEAR_EMOTIONS)
B_K = {"anger": 2, "disgust": 1, "fear": 2, "guilt": 1, "joy": 3, "sadness": 3, "shame": 1}
TAUS = (0.0, 0.02, 0.05)
N_BOOT = 10_000
SEED = 1234
Q = 0.05

# control suffixes produced by run.py plan p1 (label = f"{a}-{b}|{suffix}")
P1_CONTROLS = ["geo1", "geo2", "iso1", "lam", "proj", "norm", "geo1|norm", "proj|norm"]


def _parse_ids(spec: str | None) -> set[int]:
    out: set[int] = set()
    for part in (spec or "").split(","):
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-")
            out |= set(range(int(lo), int(hi) + 1))
        else:
            out.add(int(part))
    return out


def load_matrix(path: Path, instrument: str) -> pd.DataFrame:
    """steer, prompt_id -> 7 normalised scores (mean over samples if present)."""
    df = pd.read_csv(path)
    df = df[["steer", "prompt_id", *EMO]].copy()
    df["prompt_id"] = df["prompt_id"].astype(int)
    df = df.groupby(["steer", "prompt_id"], as_index=False)[EMO].mean()
    if instrument == "encoder":
        for e in EMO:
            df[e] = df[e] / B_K[e]
    else:
        df[EMO] = df[EMO] / 100.0
    return df


def _cube(df: pd.DataFrame, labels: list[str], exclude: set[int]):
    """Return ids and {label: array (n_ids, 7)}; NaN where a judge score is missing.

    Missing judge rows are excluded, not imputed: every mean is a nanmean, so the
    point estimates equal the published per-condition means, and the bootstrap
    stays paired because all conditions share the resampled prompt IDs.
    """
    present = df[df["steer"].isin(labels)]
    missing = sorted(set(labels) - set(present["steer"]))
    if missing:
        raise KeyError(f"labels missing from matrix: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    ids = sorted(set(present["prompt_id"]) - exclude)
    piv = {lab: g.set_index("prompt_id").reindex(ids)[EMO].to_numpy(float)
           for lab, g in present.groupby("steer")}
    return np.array(ids), piv


def _boot_index(ids: np.ndarray, rng) -> np.ndarray:
    strata = ids // 8
    cols = []
    for s in np.unique(strata):
        pos = np.flatnonzero(strata == s)
        cols.append(pos[rng.integers(0, len(pos), (N_BOOT, len(pos)))])
    return np.concatenate(cols, axis=1)            # (N_BOOT, n_ids) positions


def _p_greater(boot: np.ndarray) -> float:         # H1: effect > 0
    return (1 + np.sum(boot <= 0)) / (1 + len(boot))


def _p_less(boot: np.ndarray) -> float:            # H1: effect < 0
    return (1 + np.sum(boot >= 0)) / (1 + len(boot))


def bh(p: np.ndarray, q: float = Q, by: bool = False) -> np.ndarray:
    m = len(p)
    order = np.argsort(p)
    c = np.sum(1.0 / np.arange(1, m + 1)) if by else 1.0
    thresh = q * np.arange(1, m + 1) / (m * c)
    passed = p[order] <= thresh
    k = np.max(np.flatnonzero(passed)) + 1 if passed.any() else 0
    out = np.zeros(m, bool)
    out[order[:k]] = True
    return out


def analyse_model(df: pd.DataFrame, slug: str, instrument: str, controls: list[str],
                  exclude: set[int], with_wrong: bool):
    have = set(df["steer"])
    pairs = [p for p in ALL_PAIRS if p in have]           # parity has a subset of pairs
    controls = [c for c in controls if all(f"{p}|{c}" in have for p in pairs)]
    with_wrong = with_wrong and len(pairs) == len(ALL_PAIRS)
    labels = ["baseline", *EMO, *pairs] + [f"{p}|{c}" for p in pairs for c in controls]
    ids, M = _cube(df, labels, exclude)
    rng = np.random.default_rng(SEED)
    idx = _boot_index(ids, rng)
    mean = {lab: np.nanmean(M[lab], 0) for lab in labels}
    bmean = {}

    def boot(lab):                               # (N_BOOT, 7), cached
        if lab not in bmean:
            bmean[lab] = np.nanmean(M[lab][idx], 1)
        return bmean[lab]

    rows, boots = [], {}
    for spec in pairs:
        a, b = spec.split("-")
        ia, ib = EMO.index(a), EMO.index(b)
        r = {"model": slug, "instrument": instrument, "pair": spec, "n_ids": len(ids),
             "n_valid_ab": int(np.isfinite(M[spec][:, 0]).sum())}
        variants = {"ab": spec, **{c: f"{spec}|{c}" for c in controls}}
        for name, lab in variants.items():
            T = mean[lab][ia] - mean["baseline"][ia]
            U = mean[lab][ib] - mean[a][ib]
            bT = boot(lab)[:, ia] - boot("baseline")[:, ia]
            bU = boot(lab)[:, ib] - boot(a)[:, ib]
            r[f"T_{name}"], r[f"U_{name}"] = T, U
            r[f"strict_{name}"] = mean[lab][ib] < mean["baseline"][ib]
            if name == "ab":
                r.update(T_lo=np.percentile(bT, 2.5), T_hi=np.percentile(bT, 97.5),
                         U_lo=np.percentile(bU, 2.5), U_hi=np.percentile(bU, 97.5),
                         p_T=_p_greater(bT), p_U=_p_less(bU), U_tie=bool(U == 0))
                r["p_iut"] = max(r["p_T"], r["p_U"])
                boots[(spec, "U_ab")] = bU
            else:
                d = boots[(spec, "U_ab")] - bU
                r[f"dU_{name}"] = r["U_ab"] - U
                r[f"dU_{name}_lo"], r[f"dU_{name}_hi"] = np.percentile(d, [2.5, 97.5])
                boots[(spec, f"dU_{name}")] = d
        if with_wrong:                            # S2: subtract another emotion c, still measure b
            us = []
            for c in EMO:
                if c in (a, b):
                    continue
                us.append(mean[f"{a}-{c}"][ib] - mean[a][ib])
                r[f"wrongjoint_{c}"] = (mean[f"{a}-{c}"][ia] > mean["baseline"][ia]) and (us[-1] < 0)
            r["U_wrong_mean"] = float(np.mean(us))
            r["dU_wrong"] = r["U_ab"] - r["U_wrong_mean"]
        rows.append(r)
    return pd.DataFrame(rows), boots


def models(source: str) -> list[tuple[str, Path, Path]]:
    """(slug, encoder csv, judge wide csv) for every model that has the inputs."""
    out = []
    base = REPO / ("runs" if source == "published" else "runs_revision")
    for d in sorted(p for p in base.iterdir() if p.is_dir()):
        if source == "published":
            enc, jud = d / "compose_allpairs.csv", d / "compose_allpairs_judge_wide.csv"
        else:
            enc, jud = d / f"{source}.csv", d / f"{source}_judge_wide.csv"
        if enc.is_file():
            out.append((d.name, enc, jud))
    return out


def summarise(cells: pd.DataFrame, boots_all: dict, controls: list[str], with_wrong: bool) -> list[str]:
    L = []
    for inst, g in cells.groupby("instrument"):
        n = len(g)
        L += [f"## {inst} ({n} model-pair cells, {g['model'].nunique()} models, "
              f"prompt IDs per model: {sorted(g['n_ids'].unique().tolist())})", ""]
        L += ["| Condition | Joint T>0 & U<0 | Joint at tau=0.02 | Joint at tau=0.05 | Strict (b below baseline) |",
              "|---|---:|---:|---:|---:|"]
        for name in ["ab", *controls]:
            T, U = g[f"T_{name}"], g[f"U_{name}"]
            j = [int(((T > t) & (U < -t)).sum()) for t in TAUS]
            strict = int(((T > 0) & (U < 0) & g[f"strict_{name}"]).sum())
            L.append(f"| `{'a-b' if name == 'ab' else name}` | {j[0]}/{n} ({j[0] / n:.1%}) | {j[1]}/{n} "
                     f"({j[1] / n:.1%}) | {j[2]}/{n} ({j[2] / n:.1%}) | {strict}/{n} ({strict / n:.1%}) |")
        if with_wrong:
            wj = g[[c for c in g.columns if c.startswith("wrongjoint_")]].to_numpy()
            wj = wj[~pd.isna(wj)].astype(bool)
            L.append(f"| subtract another emotion c (S2) | {int(wj.sum())}/{wj.size} ({wj.mean():.1%}) | | | |")
        L.append("")
        # pair-level inference
        p = g["p_iut"].to_numpy()
        sig_marg = int(((g["T_lo"] > 0) & (g["U_hi"] < 0)).sum())
        L += [f"Pair-level (paired, stratified bootstrap, {N_BOOT} resamples, seed {SEED}):",
              f"- both 95% intervals exclude zero in the right direction: {sig_marg}/{n}",
              f"- intersection-union test, BH q={Q}: {int(bh(p).sum())}/{n}; BY q={Q}: {int(bh(p, by=True).sum())}/{n}",
              f"- exact ties U = 0 (judge scale has 10-point steps): {int(g['U_tie'].sum())}/{n}", ""]
        # grid means of dU (controls)
        if controls or with_wrong:
            L += ["| Contrast (on the same measured b) | Mean dU over cells | 95% CI | Cells with dU<0 |",
                  "|---|---:|---|---:|"]
            for name in controls:
                key = f"dU_{name}"
                reps = np.mean([boots_all[(r.model, inst, r.pair, key)] for r in g.itertuples()], axis=0)
                lo, hi = np.percentile(reps, [2.5, 97.5])
                L.append(f"| U(a-b) - U({name}) | {g[key].mean():+.4f} | [{lo:+.4f}, {hi:+.4f}] | "
                         f"{int((g[key] < 0).sum())}/{n} |")
            if with_wrong:
                L.append(f"| U(a-b) - mean U(a-c) (S2, point only) | {g['dU_wrong'].mean():+.4f} | | "
                         f"{int((g['dU_wrong'] < 0).sum())}/{n} |")
            L.append("")
        # decision rule (REVISION_EXPERIMENTS.md section 2, rule 5)
        if "geo1" in controls and "lam" in controls:
            reps = np.mean([boots_all[(r.model, inst, r.pair, "dU_geo1")] for r in g.itertuples()], axis=0)
            hi = np.percentile(reps, 97.5)
            jab = ((g["T_ab"] > 0) & (g["U_ab"] < 0)).mean()
            jgeo = ((g["T_geo1"] > 0) & (g["U_geo1"] < 0)).mean()
            jlam = ((g["T_lam"] > 0) & (g["U_lam"] < 0)).mean()
            ok = (g["dU_geo1"].mean() < 0) and hi < 0 and (jab - jgeo >= 0.10) and (jab - jlam >= 0.10)
            L += [f"**Rule 5 ({inst}):** mean dU(geo1) {g['dU_geo1'].mean():+.4f}, upper CI {hi:+.4f}; "
                  f"joint excess over geo {jab - jgeo:+.1%}, over lambda*a {jab - jlam:+.1%} -> "
                  f"{'criterion met' if ok else 'criterion NOT met'} on this instrument.", ""]
    return L


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="published", help="published, or a plan name such as p1")
    ap.add_argument("--exclude-prompts", default=None, help="e.g. 0-7 (anger prompts used in calibration)")
    ap.add_argument("--models", default=None, help="comma-separated slugs")
    ap.add_argument("--instruments", default="encoder,judge")
    ap.add_argument("--out", type=Path, required=True, help="markdown report; a .csv with all cells is written next to it")
    args = ap.parse_args()

    exclude = _parse_ids(args.exclude_prompts)
    controls = [] if args.source == "published" else P1_CONTROLS
    with_wrong = args.source == "published"
    keep = set(args.models.split(",")) if args.models else None
    frames, boots_all = [], {}
    for slug, enc, jud in models(args.source):
        if keep and slug not in keep:
            continue
        for inst, path in (("encoder", enc), ("judge", jud)):
            if inst not in args.instruments.split(",") or not path.is_file():
                continue
            cells, boots = analyse_model(load_matrix(path, inst), slug, inst, controls, exclude, with_wrong)
            frames.append(cells)
            boots_all.update({(slug, inst, k[0], k[1]): v for k, v in boots.items()})
            print(f"  {slug} {inst}: {len(cells)} cells", file=sys.stderr, flush=True)
    if not frames:
        sys.exit(f"no inputs found for source {args.source!r} under runs_revision/<slug>/")
    cells = pd.concat(frames, ignore_index=True)
    controls = [c for c in controls if f"dU_{c}" in cells.columns and cells[f"dU_{c}"].notna().all()]
    with_wrong = with_wrong and "dU_wrong" in cells.columns
    head = [f"# Revision statistics: source `{args.source}`",
            "",
            f"Excluded prompt IDs: {sorted(exclude) if exclude else 'none'}. "
            "Scores range-normalised (encoder / B_k, judge / 100). "
            "Generated by `python -m emotion.revision.stats`.", ""]
    n_cells = cells.groupby("instrument").size()
    if (n_cells < 462).any():
        head += [f"**Note:** BH/BY are computed over the cells in this report "
                 f"({', '.join(f'{k}: {v}' for k, v in n_cells.items())}), not over the "
                 "pre-registered family of 462 cells per instrument, so the corrected counts "
                 "are not comparable with the published-data report.", ""]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(head + summarise(cells, boots_all, controls, with_wrong)) + "\n")
    cells.to_csv(args.out.with_suffix(".csv"), index=False)
    print(f"wrote {args.out} and {args.out.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
