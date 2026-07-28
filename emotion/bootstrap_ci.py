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
    ap.add_argument("--out", type=Path, default=None, help="сохранить таблицу в markdown")
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    by = load(args.csv)
    base = by["baseline"]
    n_per = len(base)
    header = f"=== {args.csv.name}  (n per condition: {n_per}) ==="
    print(header)

    rows = []
    print("diagonal Δ vs baseline [95% CI]:")
    for emo in ISEAR_EMOTIONS:
        s = [r[emo] for r in by[emo]]
        b = [r[emo] for r in base]
        p_, lo, hi = boot_diff(s, b, args.n_boot, rng)
        sig = lo > 0 or hi < 0
        rows.append((emo, p_, lo, hi, sig))
        print(f"  {emo:>8}: {p_:+.3f}  [{lo:+.3f}, {hi:+.3f}]{'  *' if sig else ''}")

    # sadness leakage: Δ sadness pooled over all non-sadness steer directions
    leak_steer = [r["sadness"] for st in ISEAR_EMOTIONS if st != "sadness" for r in by[st]]
    leak_base = [r["sadness"] for r in base]
    lp, llo, lhi = boot_diff(leak_steer, leak_base, args.n_boot, rng)
    print(f"sadness leakage (non-sad steer): {lp:+.3f}  [{llo:+.3f}, {lhi:+.3f}]")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        n_by = {emo: len(by[emo]) for emo in ISEAR_EMOTIONS}
        lines = [f"# Доверительные интервалы, {args.csv.name}", "",
                 f"Bootstrap {args.n_boot} итераций по промптам. Звёздочка — интервал не включает ноль.",
                 f"На условие: baseline {n_per}, по эмоциям {min(n_by.values())}–{max(n_by.values())}.", ""]
        if min(n_by.values()) != n_per:
            lines += ["> Число наблюдений различается между условиями — часть строк "
                      "потерял разбор ответа судьи. Бутстрап считает пропуски случайными.", ""]
        lines += ["| эмоция | диагональ Δ | 95% CI | значимо |", "|---|---:|---|:--:|"]
        for emo, p_, lo, hi, sig in rows:
            lines.append(f"| {emo} | {p_:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {'да' if sig else 'нет'} |")
        lines += ["", f"Протечка в грусть (наведение не-грусти): {lp:+.3f} [{llo:+.3f}, {lhi:+.3f}]"]
        args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"записано {args.out}")


if __name__ == "__main__":
    main()
