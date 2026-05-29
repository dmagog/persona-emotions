"""Correlate human labels against encoder vs LLM-judge (Phase 5.3, W4).

Run AFTER humans fill `human_0_100` in the labeling sheet. Joins the filled sheet
with the key on `id` and reports, overall and per emotion, how well each measurer
tracks humans:
  * Spearman rho and Pearson r of human vs encoder, and human vs judge.
This is the arbiter for "encoder vs judge": whoever correlates better with humans
is the measurer to lead the paper's main results with (possibly per emotion - the
encoder is structurally blind to guilt/disgust, so it may win on the other 5 only).

Usage:
    python -m emotion.label_correlation \
        --sheet results/human_label_sheet.csv --key results/human_label_key.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

from emotion.judge_agreement import pearson, spearman
from emotion.space import ISEAR_EMOTIONS

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))


def main() -> None:
    ap = argparse.ArgumentParser(description="Human vs encoder vs judge correlation (W4 arbiter).")
    ap.add_argument("--sheet", required=True, type=Path, help="labeling sheet with human_0_100 filled")
    ap.add_argument("--key", required=True, type=Path, help="key with encoder_0_1, judge_0_100")
    args = ap.parse_args()

    key = {r["id"]: r for r in csv.DictReader(open(args.key, encoding="utf-8"))}
    rows = []  # (emotion, human, encoder, judge)
    skipped = 0
    for r in csv.DictReader(open(args.sheet, encoding="utf-8")):
        h = r.get("human_0_100", "").strip()
        if h == "" or r["id"] not in key:
            skipped += 1
            continue
        try:
            human = float(h)
            enc = float(key[r["id"]]["encoder_0_1"])
            jdg = float(key[r["id"]]["judge_0_100"])
        except ValueError:
            skipped += 1
            continue
        rows.append((key[r["id"]]["emotion"], human, enc, jdg))

    if not rows:
        raise SystemExit("no filled labels found (fill human_0_100 in the sheet first)")
    print(f"labeled items used: {len(rows)} (skipped {skipped})")

    def block(name, items):
        H = [x[1] for x in items]
        E = [x[2] for x in items]
        J = [x[3] for x in items]
        he_s, he_p = spearman(H, E), pearson(H, E)
        hj_s, hj_p = spearman(H, J), pearson(H, J)
        winner = "encoder" if he_s > hj_s else ("judge" if hj_s > he_s else "tie")
        print(f"  {name:<9} n={len(items):3d}  human~encoder rho={he_s:+.3f} r={he_p:+.3f} | "
              f"human~judge rho={hj_s:+.3f} r={hj_p:+.3f}  -> closer: {winner}")

    print("\n=== human vs measurers (Spearman rho / Pearson r) ===")
    block("OVERALL", rows)
    per = defaultdict(list)
    for it in rows:
        per[it[0]].append(it)
    for e in ISEAR_EMOTIONS:
        if per[e]:
            block(e, per[e])


if __name__ == "__main__":
    main()
