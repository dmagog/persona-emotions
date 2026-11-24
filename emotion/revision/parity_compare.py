"""Compare a `parity` run with the published run of the same model.

For every condition in runs_revision/<slug>/parity.csv (baseline, 7 single
emotions, 6 pairs) prints the target-emotion mean (new vs published) and the
max |diff| over the 7 encoder columns; for the pairs it prints T and U
(raw encoder units) and whether their signs agree with the published run.

Pass criterion (set before the first parity run): signs of T and U agree on
at least 5 of the 6 pairs, and every target mean is within 0.1 of the
published one. Small differences are expected (batched fp16 greedy decoding).

    python -m emotion.revision.parity_compare --slug Llama-3.2-3B-Instruct
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
EMO = ["anger", "disgust", "fear", "guilt", "joy", "sadness", "shame"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--slug", required=True)
    ap.add_argument("--new", type=Path, default=None, help="default runs_revision/<slug>/parity.csv")
    ap.add_argument("--published", type=Path, default=None, help="default runs/<slug>/compose_allpairs.csv")
    args = ap.parse_args()
    new_p = args.new or REPO / "runs_revision" / args.slug / "parity.csv"
    pub_p = args.published or REPO / "runs" / args.slug / "compose_allpairs.csv"

    def means(path):
        df = pd.read_csv(path)
        df = df[df["prompt_id"].astype(int) < 56]
        return df.groupby("steer")[EMO].mean()

    new, pub = means(new_p), means(pub_p)
    labels = [l for l in new.index if l in pub.index]
    lines = ["| condition | target | target mean new / published | max abs diff (7 cols) |", "|---|---|---|---:|"]
    worst = 0.0
    for lab in ["baseline", *EMO, *sorted(l for l in labels if "-" in l)]:
        if lab not in labels:
            continue
        tgt = lab.split("-")[0] if lab != "baseline" else None
        d = float((new.loc[lab] - pub.loc[lab]).abs().max())
        if tgt:
            worst = max(worst, abs(new.loc[lab, tgt] - pub.loc[lab, tgt]))
            tm = f"{new.loc[lab, tgt]:.3f} / {pub.loc[lab, tgt]:.3f}"
        else:
            tm = "-"
        lines.append(f"| {lab} | {tgt or '-'} | {tm} | {d:.3f} |")

    lines += ["", "| pair | T new / pub | U new / pub | signs agree |", "|---|---|---|---|"]
    agree = n = 0
    for lab in sorted(l for l in labels if "-" in l):
        a, b = lab.split("-")
        tn, tp = new.loc[lab, a] - new.loc["baseline", a], pub.loc[lab, a] - pub.loc["baseline", a]
        un, up = new.loc[lab, b] - new.loc[a, b], pub.loc[lab, b] - pub.loc[a, b]
        ok = (tn > 0) == (tp > 0) and (un < 0) == (up < 0)
        agree += ok
        n += 1
        lines.append(f"| {lab} | {tn:+.3f} / {tp:+.3f} | {un:+.3f} / {up:+.3f} | {'yes' if ok else 'NO'} |")
    passed = n > 0 and agree >= n - 1 and worst <= 0.1
    lines += ["", f"signs agree on {agree}/{n} pairs; largest target-mean difference {worst:.3f} -> "
                  f"{'PASS' if passed else 'FAIL'}"]
    print("\n".join(lines))
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
