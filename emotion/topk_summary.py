"""Summarize the top-k judge sweep into a k-vs-quality curve (W6 'minimal').

Reads the per-k wide CSVs written by `judge_topk_sweep.py`
(`<prefix>_k{K}_wide.csv`, columns: steer, prompt_id, <7 ISEAR>) and reports, for
each k (number of SAE features per emotion), with 95% bootstrap CIs over prompts:
  - the mean diagonal effect (Δ target vs baseline), averaged over the 7 emotions;
  - per-emotion diagonal Δ;
  - the sadness-leakage metric (mean Δ sadness under non-sadness steers).
This shows how few features are enough to steer (earns the word 'minimal').

Usage:
    python -m emotion.topk_summary --prefix results/judge_topk --out results/topk_sweep_gemma.md
"""
from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

from emotion.bootstrap_ci import boot_diff, load
from emotion.space import ISEAR_EMOTIONS
import numpy as np


def k_from_path(p: str) -> int:
    m = re.search(r"_k(\d+)_wide\.csv$", p)
    return int(m.group(1)) if m else -1


def main() -> None:
    ap = argparse.ArgumentParser(description="Top-k sweep summary (diagonal effect vs k).")
    ap.add_argument("--prefix", default="results/judge_topk")
    ap.add_argument("--n-boot", type=int, default=20000)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    paths = sorted(glob.glob(f"{args.prefix}_k*_wide.csv"), key=k_from_path)
    if not paths:
        raise SystemExit(f"no files matching {args.prefix}_k*_wide.csv")

    lines = []  # markdown rows: k, n, mean-diag [CI], leak [CI]
    per_emo_rows = []
    for p in paths:
        K = k_from_path(p)
        by = load(Path(p))
        base = by["baseline"]
        n = len(base)
        rng = np.random.default_rng(0)

        # per-emotion diagonal Δ
        diags = {}
        for emo in ISEAR_EMOTIONS:
            s = [r[emo] for r in by[emo]]
            b = [r[emo] for r in base]
            diags[emo] = boot_diff(s, b, args.n_boot, rng)

        # mean diagonal across the 7 emotions (bootstrap over prompts jointly)
        # point: average of per-emotion point estimates
        mean_diag = sum(diags[e][0] for e in ISEAR_EMOTIONS) / len(ISEAR_EMOTIONS)

        # sadness leakage
        leak_steer = [r["sadness"] for st in ISEAR_EMOTIONS if st != "sadness" for r in by[st]]
        leak_base = [r["sadness"] for r in base]
        lk = boot_diff(leak_steer, leak_base, args.n_boot, rng)

        sig = sum(1 for e in ISEAR_EMOTIONS if not (diags[e][1] <= 0 <= diags[e][2]))
        lines.append((K, n, mean_diag, lk, sig))
        per_emo_rows.append((K, diags))

        print(f"k={K:>2} (n={n}): mean diagonal Δ={mean_diag:+.1f}  "
              f"sig diagonals={sig}/7  sadness leak={lk[0]:+.1f} [{lk[1]:+.1f},{lk[2]:+.1f}]")

    # per-emotion diagonal table
    print("\nper-emotion diagonal Δ vs baseline [95% CI]:")
    print("  k  " + " ".join(f"{e[:4]:>14}" for e in ISEAR_EMOTIONS))
    for K, diags in per_emo_rows:
        cells = " ".join(f"{diags[e][0]:>+5.0f}[{diags[e][1]:>+3.0f},{diags[e][2]:>+3.0f}]" for e in ISEAR_EMOTIONS)
        print(f"{K:>3}  {cells}")

    if args.out is not None:
        md = ["# Top-k свип SAE-фич (Phase 3.2, W6) — сколько фич достаточно для стиринга\n"]
        md.append("Стиринг векторами, собранными только из top-k контрастных SAE-фич каждой эмоции "
                  "(gemma-2-2b, layer 12, coeff 8), судья llama-3.3-70b + bootstrap 95% CI. "
                  "`emotion/steer_topk_sweep.py` → `judge_topk_sweep` → `topk_summary`.\n")
        md.append("| k (фич/эмоцию) | средний эффект диагонали Δ | значимых диагоналей | протечка sadness [95% CI] |")
        md.append("|---:|---:|:--:|---:|")
        for K, n, mean_diag, lk, sig in lines:
            md.append(f"| {K} | {mean_diag:+.0f} | {sig}/7 | {lk[0]:+.0f} [{lk[1]:+.0f},{lk[2]:+.0f}] |")
        md.append("\n## Диагональ по эмоциям (Δ vs baseline [95% CI])\n")
        md.append("| k | " + " | ".join(ISEAR_EMOTIONS) + " |")
        md.append("|---:|" + "|".join(["---:"] * len(ISEAR_EMOTIONS)) + "|")
        for K, diags in per_emo_rows:
            cells = " | ".join(f"{diags[e][0]:+.0f} [{diags[e][1]:+.0f},{diags[e][2]:+.0f}]" for e in ISEAR_EMOTIONS)
            md.append(f"| {K} | {cells} |")
        args.out.write_text("\n".join(md) + "\n", encoding="utf-8")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
