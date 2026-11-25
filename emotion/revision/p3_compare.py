"""Prompting baseline (p3) vs published activation steering, for the 4-model subset.

For an ordered pair (a, b), method X in {steering, prompting}:
    steering:  a-only = "<a>",       a-b = "<a>-<b>"      (runs/<slug>/compose_allpairs*.csv)
    prompting: a-only = "instr_<a>", a-b = "instr_<a>-<b>" (runs_revision/<slug>/p3*.csv)
Each method is measured against the baseline of its own run.
    T      = s_a(a-b) - s_a(baseline)                    target gain
    U      = s_b(a-b) - s_b(a-only)                      attenuation of b (as in stats.py)
    U_base = s_b(a-b) - s_b(baseline)                    b relative to the unsteered level
    spill  = mean over k not in {a, b} of |s_k(a-b) - s_k(baseline)|
    joint(tau) = share of cells with T > tau and U < -tau; strict = joint(0) and U_base < 0
Scores range-normalised as in stats.py (encoder / B_k, judge / 100). Cells: 4 models x 42 pairs.
CIs: paired bootstrap over prompt IDs per model (stratified by pid // 8, the same IDs for both
methods and all conditions), 2000 resamples, seed 1234, 95% percentile intervals.

Coherence: steering numbers come from runs/<slug>/coherence.md (a-only steering at the published
operating point, steer_specificity_raw.csv); prompting numbers from runs_revision/<slug>/p3_coherence.md.
Both use the same coherence prompt and judge (meta-llama/llama-3.3-70b-instruct, one pass). There is
no published coherence for steered a-b.

Lexical leak: share of answers that contain a word of the target emotion's family (and of b's).
Qwen3-0.6B names the emotion in most prompted answers, which the encoder scores directly, so its
encoder results are reported separately and kept out of the encoder headline.

    python -m emotion.revision.p3_compare --out runs_revision/p3_compare.md
"""
from __future__ import annotations

import argparse
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from emotion.revision.stats import ALL_PAIRS, B_K, EMO, load_matrix

warnings.filterwarnings("ignore", category=RuntimeWarning)
REPO = Path(__file__).resolve().parents[2]
MODELS = ["Llama-3.2-3B-Instruct", "gemma-2-2b-it", "Qwen3-0.6B", "Falcon3-3B-Instruct"]
FLAGGED = "Qwen3-0.6B"
N_BOOT, SEED = 2000, 1234
TAUS = (0.0, 0.02, 0.05)
WORDS = {
    "anger": r"anger|angr|furious|fury|rage|enrag|irritat|livid|mad(?![a-z])",
    "disgust": r"disgust|revolt|repuls|gross|nause|sicken",
    "fear": r"fear|afraid|scare|terrif|frighten|panic|dread",
    "guilt": r"guilt",
    "joy": r"joy|happ(?:y|ier|iest|ily|iness)|delight|elat|glee|cheer",
    "sadness": r"sad|sorrow|grief|griev|unhappy|depress|heartbr",
    "shame": r"shame|asham|humiliat|embarrass",
}
WORD_RE = {e: re.compile(rf"\b(?:{p})\w*", re.I) for e, p in WORDS.items()}


def labels(method: str):
    if method == "steering":
        return "baseline", (lambda a: a), (lambda a, b: f"{a}-{b}")
    return "baseline", (lambda a: f"instr_{a}"), (lambda a, b: f"instr_{a}-{b}")


def source(slug: str, method: str, inst: str) -> Path:
    if method == "steering":
        return REPO / "runs" / slug / ("compose_allpairs.csv" if inst == "encoder" else "compose_allpairs_judge_wide.csv")
    return REPO / "runs_revision" / slug / ("p3.csv" if inst == "encoder" else "p3_judge_wide.csv")


def cube(slug: str, method: str, inst: str, ids: np.ndarray) -> dict[str, np.ndarray]:
    df = load_matrix(source(slug, method, inst), inst)
    return {lab: g.set_index("prompt_id").reindex(ids)[EMO].to_numpy(float) for lab, g in df.groupby("steer")}


def boot_index(ids: np.ndarray, rng) -> np.ndarray:
    strata = ids // 8
    cols = [pos[rng.integers(0, len(pos), (N_BOOT, len(pos)))]
            for s in np.unique(strata) for pos in [np.flatnonzero(strata == s)]]
    return np.concatenate(cols, axis=1)


def cell_stats(M: dict, method: str, idx: np.ndarray | None):
    """Per pair arrays of T, U, U_base, spill; shape (42,) for point, (42, N_BOOT) for bootstrap."""
    base, one, two = labels(method)
    mean = (lambda lab: np.nanmean(M[lab], 0)) if idx is None else (lambda lab: np.nanmean(M[lab][idx], 1))
    cache = {}

    def m(lab):
        if lab not in cache:
            cache[lab] = mean(lab)
        return cache[lab]
    out = {k: [] for k in ("T", "U", "U_base", "spill")}
    for spec in ALL_PAIRS:
        a, b = spec.split("-")
        ia, ib = EMO.index(a), EMO.index(b)
        other = [EMO.index(e) for e in EMO if e not in (a, b)]
        X, A, B0 = m(two(a, b)), m(one(a)), m(base)
        out["T"].append(X[..., ia] - B0[..., ia])
        out["U"].append(X[..., ib] - A[..., ib])
        out["U_base"].append(X[..., ib] - B0[..., ib])
        out["spill"].append(np.abs(X[..., other] - B0[..., other]).mean(-1))
    return {k: np.array(v) for k, v in out.items()}


def summary(cs: dict) -> dict:
    """Aggregate over cells (axis 0): means and rates; works for point and bootstrap arrays."""
    T, U, Ub = cs["T"], cs["U"], cs["U_base"]
    r = {"T": T.mean(0), "U": U.mean(0), "U_base": Ub.mean(0), "spill": cs["spill"].mean(0)}
    for t in TAUS:
        r[f"joint@{t}"] = ((T > t) & (U < -t)).mean(0)
    r["strict"] = ((T > 0) & (U < 0) & (Ub < 0)).mean(0)
    return r


def compare(inst: str, models: list[str]):
    """Point summaries and bootstrap replicates for both methods, pooled over models."""
    rng = np.random.default_rng(SEED)
    pts = {"steering": [], "prompting": []}
    bts = {"steering": [], "prompting": []}
    for slug in models:
        ids = np.arange(56)
        idx = boot_index(ids, rng)                       # shared by both methods: paired
        for meth in pts:
            M = cube(slug, meth, inst, ids)
            pts[meth].append(cell_stats(M, meth, None))
            bts[meth].append(cell_stats(M, meth, idx))
    cat = lambda lst: {k: np.concatenate([d[k] for d in lst], axis=0) for k in lst[0]}
    return ({m: summary(cat(v)) for m, v in pts.items()}, {m: summary(cat(v)) for m, v in bts.items()})


def table(inst: str, models: list[str], title: str) -> list[str]:
    pt, bt = compare(inst, models)
    ci = lambda v: f"[{np.percentile(v, 2.5):+.3f}, {np.percentile(v, 97.5):+.3f}]"
    L = [f"### {title}", "", f"{len(models)} models x 42 pairs = {42 * len(models)} cells.", "",
         "| Measure | Steering (published) | Prompting (p3) | Prompting - steering [95% CI] |",
         "|---|---:|---:|---|"]
    names = {"T": "T, target gain", "U": "U, b vs a-only", "U_base": "U_base, b vs baseline",
             "spill": "spillover, mean abs shift on other 5", "joint@0.0": "joint, tau=0",
             "joint@0.02": "joint, tau=0.02", "joint@0.05": "joint, tau=0.05", "strict": "strict (b below baseline)"}
    for k, nm in names.items():
        s, p = pt["steering"][k], pt["prompting"][k]
        L.append(f"| {nm} | {s:+.3f} {ci(bt['steering'][k])} | {p:+.3f} {ci(bt['prompting'][k])} | "
                 f"{p - s:+.3f} {ci(bt['prompting'][k] - bt['steering'][k])} |")
    return L + [""]


def coherence_md(path: Path) -> dict[str, float]:
    out = {}
    if not path.is_file():
        return out
    for line in path.read_text().splitlines():
        m = re.match(r"\|\s*([^|]+?)\s*\|\s*([\d.]+)\s*\|\s*([+-][\d.]+)\s*\|", line)
        if m:
            out[m.group(1)] = float(m.group(2))
    return out


def coherence(models: list[str]) -> list[str]:
    L = ["## Coherence (fluency, 0-100; judge meta-llama/llama-3.3-70b-instruct, same prompt, one pass)", "",
         "Steering: a-only at the published operating point (runs/<slug>/coherence.md, from "
         "steer_specificity_raw.csv); no published coherence exists for steered a-b. "
         "Prompting: runs_revision/<slug>/p3_coherence.md. Deltas are against each run's own baseline.", "",
         "| Model | steering baseline | steering a-only, mean delta | prompting baseline | prompting a-only, mean delta "
         "| prompting a-b, mean delta | prompting a-b, worst |", "|---|---:|---:|---:|---:|---:|---:|"]
    for slug in models:
        s = coherence_md(REPO / "runs" / slug / "coherence.md")
        p = coherence_md(REPO / "runs_revision" / slug / "p3_coherence.md")
        sa = np.mean([s[e] - s["baseline"] for e in EMO if e in s]) if s else np.nan
        pa = np.mean([p[f"instr_{e}"] - p["baseline"] for e in EMO])
        pab = [p[f"instr_{x}"] - p["baseline"] for x in ALL_PAIRS]
        L.append(f"| {slug} | {s.get('baseline', np.nan):.1f} | {sa:+.1f} | {p['baseline']:.1f} | {pa:+.1f} | "
                 f"{np.mean(pab):+.1f} | {min(pab):+.1f} |")
    return L + [""]


def leak(models: list[str]) -> list[str]:
    L = ["## Lexical leak (answer contains a word of the emotion's family)", "",
         "Word families (regex stems): " + "; ".join(f"{e}: {p}" for e, p in WORDS.items()) + ".", "",
         "| Model | baseline: a word | steering a-only: a | steering a-b: a / b | prompting a-only: a "
         "| prompting a-b: a / b |", "|---|---:|---:|---:|---:|---:|"]
    for slug in models:
        st = pd.read_csv(REPO / "runs" / slug / "compose_allpairs.csv", usecols=["steer", "answer"])
        pr = pd.read_csv(REPO / "runs_revision" / slug / "p3.csv", usecols=["steer", "answer"])

        def rate(df, lab, e):
            ans = df.loc[df["steer"] == lab, "answer"].astype(str)
            return ans.map(lambda t: bool(WORD_RE[e].search(t))).mean()
        base = np.mean([rate(st, "baseline", e) for e in EMO])
        sa = np.mean([rate(st, e, e) for e in EMO])
        pa = np.mean([rate(pr, f"instr_{e}", e) for e in EMO])
        sab_a = np.mean([rate(st, x, x.split("-")[0]) for x in ALL_PAIRS])
        sab_b = np.mean([rate(st, x, x.split("-")[1]) for x in ALL_PAIRS])
        pab_a = np.mean([rate(pr, f"instr_{x}", x.split("-")[0]) for x in ALL_PAIRS])
        pab_b = np.mean([rate(pr, f"instr_{x}", x.split("-")[1]) for x in ALL_PAIRS])
        flag = " (flagged)" if slug == FLAGGED else ""
        L.append(f"| {slug}{flag} | {base:.0%} | {sa:.0%} | {sab_a:.0%} / {sab_b:.0%} | {pa:.0%} | "
                 f"{pab_a:.0%} / {pab_b:.0%} |")
    return L + ["", "Baseline column: mean over the 7 emotions of the share of unsteered answers naming that emotion.", ""]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rest = [m for m in MODELS if m != FLAGGED]
    L = ["# Prompting baseline (p3) vs activation steering (published), 4-model subset", "",
         f"Models: {', '.join(MODELS)}. Paired, stratified bootstrap over prompt IDs, {N_BOOT} resamples, "
         f"seed {SEED}. Definitions in `emotion/revision/p3_compare.py`.", "",
         "## Judge", ""]
    L += table("judge", MODELS, "Judge, all 4 models")
    L += table("judge", rest, f"Judge, without {FLAGGED}")
    L += ["## Encoder", "",
          f"**{FLAGGED} is flagged:** it names the target emotion in most prompted answers (see lexical leak), "
          "which the encoder scores directly, so its prompting numbers are inflated on the encoder. "
          "The headline encoder comparison uses the other 3 models.", ""]
    L += table("encoder", rest, f"Encoder, 3 models (without {FLAGGED})")
    L += table("encoder", [FLAGGED], f"Encoder, {FLAGGED} only (flagged, not for the headline)")
    L += coherence(MODELS) + leak(MODELS)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
