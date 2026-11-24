"""Rebuild a per-row judge wide CSV from the judge cache, keeping every sample.

``emotion.judge_specificity --out-wide`` keys rows by (steer, prompt_id), so with
several samples per prompt (plan p4) the last sample silently wins. This script
re-reads the cache that judge_specificity wrote and matches scores to each CSV
row by (steer, prompt_id, answer digest), so every sample keeps its own scores.
Missing scores stay empty; stats.py averages what is present.

    python -m emotion.revision.judge_wide --csv runs_revision/<slug>/p4.csv \
        --cache runs_revision/<slug>/p4_judge.cache.jsonl \
        --out runs_revision/<slug>/p4_judge_wide.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from emotion.judge_specificity import answer_key

EMO = ["anger", "disgust", "fear", "guilt", "joy", "sadness", "shame"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True, type=Path)
    ap.add_argument("--cache", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    scores: dict[tuple, float] = {}
    with open(args.cache, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d.get("score") is not None:
                scores[(d["steer"], str(d["pid"]), d["answer"], d["measured"])] = d["score"]

    n_rows = n_full = 0
    with open(args.csv, newline="", encoding="utf-8") as fin, \
            open(args.out, "w", newline="", encoding="utf-8") as fout:
        w = csv.DictWriter(fout, fieldnames=["steer", "prompt_id", "sample", *EMO])
        w.writeheader()
        for r in csv.DictReader(fin):
            steer = r["steer"].split(":")[-1]
            ak = answer_key(r["answer"])
            row = {"steer": r["steer"], "prompt_id": r["prompt_id"], "sample": r.get("sample", 0)}
            got = 0
            for e in EMO:
                s = scores.get((steer, str(r["prompt_id"]), ak, e))
                row[e] = "" if s is None else s
                got += s is not None
            w.writerow(row)
            n_rows += 1
            n_full += got == len(EMO)
    print(f"wrote {args.out}: {n_rows} rows, {n_full} fully scored")


if __name__ == "__main__":
    main()
