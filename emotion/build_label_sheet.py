"""Build a human-labeling sheet to arbitrate encoder vs LLM-judge (Phase 5.3, W4).

Uliana raised a fair point: "the encoder shows a smaller difference" does not prove
"the encoder is worse" - the LLM-judge is also unvalidated and may be biased. The
only way to settle which measurer to trust is to correlate both against human labels.

This samples (answer x emotion) items from data where BOTH measurers already scored
the same texts in single-emotion form (steered specificity answers: encoder per-emotion
columns + judge per-emotion wide CSV), and writes:
  * a blank labeling SHEET (id, emotion, text, human_0_100) - shuffled, NO model scores
    shown, so the annotator is not anchored;
  * a KEY (id, steer, prompt_id, emotion, encoder_score, judge_score) for later
    correlation human<->encoder vs human<->judge (per emotion).

Protocol: single-emotion intensity 0-100 (matches the specificity-matrix readout, so
it is directly comparable to both the encoder and the judge).

Usage:
    python -m emotion.build_label_sheet \
        --answers-csv results/steer_spec_gemma_saefeat_n56.csv \
        --judge-wide results/judge_spec_n56_wide.csv \
        --n 300 --seed 0 \
        --out-sheet results/human_label_sheet.csv \
        --out-key results/human_label_key.csv
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

from emotion.space import ISEAR_EMOTIONS

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a human-labeling sheet (encoder vs judge arbitration).")
    ap.add_argument("--answers-csv", required=True, type=Path,
                    help="steer CSV with 'answer' + 7 ISEAR encoder columns (per-emotion 0-1)")
    ap.add_argument("--judge-wide", required=True, type=Path,
                    help="judge_specificity --out-wide CSV (per-emotion 0-100), keyed by steer+prompt_id")
    ap.add_argument("--n", type=int, default=300, help="total (answer x emotion) items to sample")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-sheet", type=Path, default=Path("results/human_label_sheet.csv"))
    ap.add_argument("--out-key", type=Path, default=Path("results/human_label_key.csv"))
    args = ap.parse_args()
    rng = random.Random(args.seed)

    ans = {}  # (steer, prompt_id) -> row
    for r in csv.DictReader(open(args.answers_csv, encoding="utf-8")):
        ans[(r["steer"], r["prompt_id"])] = r
    judge = {}  # (steer, prompt_id) -> {emotion: score}
    for r in csv.DictReader(open(args.judge_wide, encoding="utf-8")):
        judge[(r["steer"], r["prompt_id"])] = r

    # candidate items: only where both measurers have a score for that emotion
    cands = []  # (steer, prompt_id, emotion, text, enc, jdg)
    for key in set(ans) & set(judge):
        text = ans[key]["answer"]
        for e in ISEAR_EMOTIONS:
            try:
                enc = float(ans[key][e])
                jdg = float(judge[key][e])
            except (KeyError, ValueError):
                continue
            cands.append((key[0], key[1], e, text, enc, jdg))

    if not cands:
        raise SystemExit("no joined (answer x emotion) candidates - check inputs")

    # balance across the 7 emotions, then fill remainder randomly
    by_emo = {e: [c for c in cands if c[2] == e] for e in ISEAR_EMOTIONS}
    for e in ISEAR_EMOTIONS:
        rng.shuffle(by_emo[e])
    per = args.n // len(ISEAR_EMOTIONS)
    picked = []
    for e in ISEAR_EMOTIONS:
        picked.extend(by_emo[e][:per])
    leftover = [c for e in ISEAR_EMOTIONS for c in by_emo[e][per:]]
    rng.shuffle(leftover)
    picked.extend(leftover[: max(0, args.n - len(picked))])
    rng.shuffle(picked)

    args.out_sheet.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_sheet, "w", newline="", encoding="utf-8") as fs, \
         open(args.out_key, "w", newline="", encoding="utf-8") as fk:
        ws = csv.DictWriter(fs, fieldnames=["id", "emotion", "text", "human_0_100"])
        wk = csv.DictWriter(fk, fieldnames=["id", "steer", "prompt_id", "emotion", "encoder_0_1", "judge_0_100"])
        ws.writeheader()
        wk.writeheader()
        for i, (steer, pid, e, text, enc, jdg) in enumerate(picked):
            iid = f"item_{i:04d}"
            ws.writerow({"id": iid, "emotion": e, "text": text, "human_0_100": ""})
            wk.writerow({"id": iid, "steer": steer, "prompt_id": pid, "emotion": e,
                         "encoder_0_1": enc, "judge_0_100": jdg})

    from collections import Counter
    dist = Counter(c[2] for c in picked)
    print(f"candidates: {len(cands)} | sampled: {len(picked)}")
    print("per-emotion:", {e: dist[e] for e in ISEAR_EMOTIONS})
    print(f"wrote sheet {args.out_sheet} (label column human_0_100 left blank)")
    print(f"wrote key   {args.out_key} (model scores hidden from the sheet)")
    print("\nПротокол: для каждой строки прочитать text, оценить ИНТЕНСИВНОСТЬ указанной")
    print("эмоции 0-100 (0 = эмоции нет, 100 = ярко выражена), вписать в human_0_100.")


if __name__ == "__main__":
    main()
