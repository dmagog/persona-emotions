"""Inter-judge agreement between two judges on the same steered answers (W4).

The whole specificity verdict rests on one LLM judge (llama-3.3-70b) that is also
the data filter. To check that verdict is not a single-model artifact, we re-judge
the SAME answers with an independent, different-family judge (qwen-2.5-72b) and
measure agreement. Reads two `--out-wide` CSVs from `judge_specificity.py`
(columns: steer, prompt_id, <7 ISEAR scores>), joins on (steer, prompt_id) and
reports:

  * Pearson r and Spearman rho over all matched (answer x emotion) score pairs,
    overall and per measured-emotion;
  * mean absolute score difference;
  * binarized agreement (emotion "present" if score >= --thresh): raw agreement
    and Cohen's kappa;
  * diagonal-argmax agreement: for each bare-emotion steer, does each judge's
    mean Delta-vs-baseline matrix put the target emotion as argmax, and do the
    two judges agree.

Pure-Python (no numpy/scipy) so it runs in the fragile local env.

Usage:
    python -m emotion.judge_agreement \
        --judge1 results/judge_spec_n56_wide.csv --name1 llama-3.3-70b \
        --judge2 results/judge_spec_n56_qwen_wide.csv --name2 qwen-2.5-72b \
        --thresh 50
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

from emotion.space import ISEAR_EMOTIONS

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))


def load_wide(path: Path) -> dict[tuple[str, str], dict[str, float]]:
    """(steer, prompt_id) -> {emotion: score}."""
    out: dict[tuple[str, str], dict[str, float]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                out[(r["steer"], r["prompt_id"])] = {e: float(r[e]) for e in ISEAR_EMOTIONS}
            except (KeyError, ValueError):
                continue
    return out


def _ranks(xs: list[float]) -> list[float]:
    """Average ranks (1-based), ties averaged."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # 1-based average rank
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def pearson(a: list[float], b: list[float]) -> float:
    n = len(a)
    if n < 2:
        return float("nan")
    ma, mb = sum(a) / n, sum(b) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = math.sqrt(sum((x - ma) ** 2 for x in a))
    db = math.sqrt(sum((y - mb) ** 2 for y in b))
    return num / (da * db) if da > 0 and db > 0 else float("nan")


def spearman(a: list[float], b: list[float]) -> float:
    return pearson(_ranks(a), _ranks(b))


def cohen_kappa(a: list[int], b: list[int]) -> tuple[float, float]:
    """Binary Cohen's kappa; returns (raw_agreement, kappa)."""
    n = len(a)
    if n == 0:
        return float("nan"), float("nan")
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pa1, pb1 = sum(a) / n, sum(b) / n
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    kappa = (po - pe) / (1 - pe) if pe < 1 else float("nan")
    return po, kappa


def main() -> None:
    ap = argparse.ArgumentParser(description="Inter-judge agreement on steered answers (W4).")
    ap.add_argument("--judge1", required=True, type=Path)
    ap.add_argument("--judge2", required=True, type=Path)
    ap.add_argument("--name1", default="judge1")
    ap.add_argument("--name2", default="judge2")
    ap.add_argument("--thresh", type=float, default=50.0, help="score >= thresh => emotion 'present'")
    args = ap.parse_args()

    j1, j2 = load_wide(args.judge1), load_wide(args.judge2)
    keys = sorted(set(j1) & set(j2))
    print(f"{args.name1}: {len(j1)} answers | {args.name2}: {len(j2)} answers | matched: {len(keys)}")
    if not keys:
        raise SystemExit("no overlapping (steer, prompt_id) keys")

    # paired scores, overall and per measured emotion
    pair_all_a: list[float] = []
    pair_all_b: list[float] = []
    per_emo: dict[str, tuple[list[float], list[float]]] = {e: ([], []) for e in ISEAR_EMOTIONS}
    for k in keys:
        for e in ISEAR_EMOTIONS:
            a, b = j1[k][e], j2[k][e]
            pair_all_a.append(a)
            pair_all_b.append(b)
            per_emo[e][0].append(a)
            per_emo[e][1].append(b)

    print("\n=== correlation (per answer x emotion score pair) ===")
    print(f"overall   n={len(pair_all_a):4d}  pearson={pearson(pair_all_a, pair_all_b):+.3f}  "
          f"spearman={spearman(pair_all_a, pair_all_b):+.3f}  "
          f"mean|Δ|={sum(abs(a-b) for a,b in zip(pair_all_a,pair_all_b))/len(pair_all_a):.1f}")
    for e in ISEAR_EMOTIONS:
        a, b = per_emo[e]
        print(f"  {e:<8} n={len(a):4d}  pearson={pearson(a,b):+.3f}  spearman={spearman(a,b):+.3f}  "
              f"mean|Δ|={sum(abs(x-y) for x,y in zip(a,b))/len(a):.1f}")

    # binarized agreement + kappa
    ba = [1 if x >= args.thresh else 0 for x in pair_all_a]
    bb = [1 if x >= args.thresh else 0 for x in pair_all_b]
    po, kappa = cohen_kappa(ba, bb)
    print(f"\n=== binarized 'present' (score >= {args.thresh:.0f}) ===")
    print(f"raw agreement={po:.3f}  cohen_kappa={kappa:+.3f}  "
          f"({args.name1} present {sum(ba)}/{len(ba)}, {args.name2} present {sum(bb)}/{len(bb)})")

    # diagonal argmax agreement on Delta-vs-baseline matrices
    def mean_matrix(j):
        agg = defaultdict(lambda: defaultdict(list))
        for (steer, _pid), sc in j.items():
            for e in ISEAR_EMOTIONS:
                agg[steer][e].append(sc[e])
        return {st: {e: sum(v[e]) / len(v[e]) for e in ISEAR_EMOTIONS} for st, v in agg.items()}

    m1, m2 = mean_matrix(j1), mean_matrix(j2)
    base1 = m1.get("baseline", {e: 0.0 for e in ISEAR_EMOTIONS})
    base2 = m2.get("baseline", {e: 0.0 for e in ISEAR_EMOTIONS})
    print("\n=== diagonal argmax (Δ vs baseline) per bare-emotion steer ===")
    print(f"{'steer':>8}  {args.name1:>16}  {args.name2:>16}  agree?")
    h1 = h2 = both = checkable = 0
    for st in ISEAR_EMOTIONS:
        if st not in m1 or st not in m2:
            continue
        checkable += 1
        d1 = {e: m1[st][e] - base1[e] for e in ISEAR_EMOTIONS}
        d2 = {e: m2[st][e] - base2[e] for e in ISEAR_EMOTIONS}
        am1, am2 = max(d1, key=d1.get), max(d2, key=d2.get)
        ok1, ok2 = am1 == st, am2 == st
        h1 += ok1
        h2 += ok2
        agree = am1 == am2
        both += agree
        print(f"{st:>8}  {am1:>16}  {am2:>16}  {'yes' if agree else 'NO'}")
    print(f"\ntarget=argmax: {args.name1} {h1}/{checkable}, {args.name2} {h2}/{checkable}; "
          f"judges agree on argmax for {both}/{checkable} diagonals")


if __name__ == "__main__":
    main()
