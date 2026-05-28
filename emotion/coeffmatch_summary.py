"""Specificity frontier: raw vs SAE off-target at matched diagonal (Phase 2.3, W2).

Reads per-(vtype, coeff) wide CSVs from `judge_coeffmatch_sweep.py` and reports,
per condition with 95% bootstrap CIs over prompts:
  - mean diagonal effect (Δ target vs baseline, averaged over 7 emotions);
  - sadness leakage (Δ sadness under non-sadness steers).
Then prints the two curves side by side so you can read off-target at a matched
diagonal: if SAE shows less leakage than raw at the same diagonal, it is genuinely
more specific (not just weaker).

Usage:
    python -m emotion.coeffmatch_summary --prefix results/judge_coeffmatch --out results/coeffmatch_gemma.md
"""
from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

import numpy as np

from emotion.bootstrap_ci import boot_diff, load
from emotion.space import ISEAR_EMOTIONS


def parse_name(p: str):
    m = re.search(r"_([a-z]+)_c([0-9.]+)_wide\.csv$", p)
    return (m.group(1), float(m.group(2))) if m else (None, None)


def cond_stats(path: str, n_boot: int):
    by = load(Path(path))
    base = by["baseline"]
    rng = np.random.default_rng(0)
    diag_points = []
    for emo in ISEAR_EMOTIONS:
        p, _lo, _hi = boot_diff([r[emo] for r in by[emo]], [r[emo] for r in base], n_boot, rng)
        diag_points.append(p)
    mean_diag = sum(diag_points) / len(diag_points)
    leak_steer = [r["sadness"] for st in ISEAR_EMOTIONS if st != "sadness" for r in by[st]]
    leak_base = [r["sadness"] for r in base]
    lk = boot_diff(leak_steer, leak_base, n_boot, rng)
    return len(base), mean_diag, lk


def main() -> None:
    ap = argparse.ArgumentParser(description="Raw-vs-SAE specificity frontier (W2).")
    ap.add_argument("--prefix", default="results/judge_coeffmatch")
    ap.add_argument("--n-boot", type=int, default=20000)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    paths = glob.glob(f"{args.prefix}_*_c*_wide.csv")
    conds = {}
    for p in paths:
        vt, c = parse_name(p)
        if vt is None:
            continue
        conds[(vt, c)] = cond_stats(p, args.n_boot)
    if not conds:
        raise SystemExit(f"no files matching {args.prefix}_*_c*_wide.csv")

    print(f"{'vtype':>6} {'coeff':>6} {'n':>4}  {'mean diag Δ':>12}  {'sadness leak [95% CI]':>24}")
    rows_md = []
    for (vt, c) in sorted(conds, key=lambda x: (x[0], x[1])):
        n, md, lk = conds[(vt, c)]
        print(f"{vt:>6} {c:>6.1f} {n:>4}  {md:>+12.1f}  {lk[0]:>+7.1f} [{lk[1]:+.1f},{lk[2]:+.1f}]")
        rows_md.append((vt, c, n, md, lk))

    if args.out is not None:
        md = ["# Coeff-matched raw vs SAE — фронтир специфичности (Phase 2.3, W2)\n"]
        md.append("Свип coeff для обоих типов векторов (raw и SAE-фичи, обе renorm к одной длине), "
                  "судья llama-3.3-70b + bootstrap 95% CI. Сравниваем off-target (протечку sadness) "
                  "при сопоставимом эффекте диагонали. `steer_coeffmatch_sweep` → `judge_coeffmatch_sweep` → этот скрипт.\n")
        md.append("| тип | coeff | средний эффект диагонали | протечка sadness [95% CI] |")
        md.append("|---|---:|---:|---:|")
        for vt, c, n, mdg, lk in rows_md:
            md.append(f"| {vt} | {c:.0f} | {mdg:+.0f} | {lk[0]:+.0f} [{lk[1]:+.0f},{lk[2]:+.0f}] |")
        md.append("\n_Читать как фронтир: при сопоставимом «эффекте диагонали» меньшая протечка = выше специфичность._\n")
        args.out.write_text("\n".join(md) + "\n", encoding="utf-8")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
