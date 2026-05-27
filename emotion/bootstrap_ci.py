"""Bootstrap confidence intervals for steering specificity (addresses W1: no error bars).

Reads a per-prompt specificity CSV (columns: steer, prompt_id, and the 7 ISEAR
emotion scores) and reports, with 95% bootstrap CIs over prompts:
  - the diagonal effect per emotion (Δ target vs baseline);
  - the "sadness leakage" metric (mean Δ sadness when steering toward non-sadness).
Pure local: no GPU, no API. Run on raw and SAE-feature CSVs to test whether the
"halves the sadness leakage" claim is significant at the current n.

Usage:
    python -m emotion.bootstrap_ci --csv results/steer_spec_gemma_raw.csv
    python -m emotion.bootstrap_ci --csv results/steer_spec_gemma_saefeat.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from emotion.space import ISEAR_EMOTIONS

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))


def load(path: Path) -> dict[str, list[dict]]:
    by = defaultdict(list)
    for r in csv.DictReader(open(path, encoding="utf-8")):
        by[r["steer"]].append({e: float(r[e]) for e in ISEAR_EMOTIONS})
    return by


def boot_diff(steer_scores, base_scores, n_boot=20000, rng=None):
    """Mean(steer) - mean(base) with 95% bootstrap CI over prompts."""
    rng = rng or np.random.default_rng(0)
    s, b = np.asarray(steer_scores), np.asarray(base_scores)
    point = float(s.mean() - b.mean())
    si = rng.integers(0, len(s), (n_boot, len(s)))
    bi = rng.integers(0, len(b), (n_boot, len(b)))
    boot = s[si].mean(1) - b[bi].mean(1)
    return point, float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser(description="Bootstrap CIs for a specificity CSV.")
    ap.add_argument("--csv", required=True, type=Path)
    ap.add_argument("--n-boot", type=int, default=20000)
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    by = load(args.csv)
    base = by["baseline"]
    print(f"=== {args.csv.name}  (n per condition: {len(base)}) ===")

    print("diagonal Δ vs baseline [95% CI]:")
    for emo in ISEAR_EMOTIONS:
        s = [r[emo] for r in by[emo]]
        b = [r[emo] for r in base]
        p, lo, hi = boot_diff(s, b, args.n_boot, rng)
        sig = "" if lo <= 0 <= hi else "  *"
        print(f"  {emo:>8}: {p:+.3f}  [{lo:+.3f}, {hi:+.3f}]{sig}")

    # sadness leakage: Δ sadness pooled over all non-sadness steer directions
    leak_steer = [r["sadness"] for st in ISEAR_EMOTIONS if st != "sadness" for r in by[st]]
    leak_base = [r["sadness"] for r in base]
    p, lo, hi = boot_diff(leak_steer, leak_base, args.n_boot, rng)
    print(f"sadness leakage (non-sad steer): {p:+.3f}  [{lo:+.3f}, {hi:+.3f}]")


if __name__ == "__main__":
    main()
