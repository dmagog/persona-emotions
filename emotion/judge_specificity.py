"""Judge-based steering specificity matrix.

Score every steered answer for every ISEAR emotion with an LLM judge, to get a
specificity matrix that does not depend on the GoEmotions encoder (which is blind
to guilt and weak on disgust). Reads a steer CSV with an 'answer' column
(`steer_specificity.py --save-answers`). Local: needs OPENAI_API_KEY +
OPENAI_BASE_URL (OpenRouter); no GPU.

Usage:
    OPENAI_API_KEY=... OPENAI_BASE_URL=https://openrouter.ai/api/v1 \
    python -m emotion.judge_specificity --csv results/steer_spec_gemma_saefeat_ans.csv \
        --out results/judge_specificity_gemma_saefeat.json
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

from emotion.judge_steered import score_answer
from emotion.run_pairwise_judge import make_client
from emotion.space import ISEAR_EMOTIONS

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))


async def main_async(args) -> None:
    rows = list(csv.DictReader(open(args.csv, encoding="utf-8")))
    client = make_client()
    sem = asyncio.Semaphore(args.concurrency)

    async def one(steer, pid, measured, answer):
        async with sem:
            return steer, pid, measured, await score_answer(client, args.model, measured, answer)

    tasks = [
        one(r["steer"].split(":")[-1], r["prompt_id"], measured, r["answer"])  # "prompt:anger"->"anger"
        for r in rows
        for measured in ISEAR_EMOTIONS
    ]
    print(f"judging {len(tasks)} (answer x emotion) pairs ...")
    results = await asyncio.gather(*tasks)

    agg = defaultdict(lambda: defaultdict(list))
    wide = defaultdict(dict)  # (steer, pid) -> {measured: score}
    for steer, pid, measured, s in results:
        if s is not None:
            agg[steer][measured].append(s)
            wide[(steer, pid)][measured] = s
    mean = {st: {e: (sum(v[e]) / len(v[e]) if v[e] else 0.0) for e in ISEAR_EMOTIONS} for st, v in agg.items()}

    if args.out_wide is not None:
        args.out_wide.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_wide, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["steer", "prompt_id", *ISEAR_EMOTIONS])
            w.writeheader()
            for (steer, pid), sc in wide.items():
                if all(e in sc for e in ISEAR_EMOTIONS):
                    w.writerow({"steer": steer, "prompt_id": pid, **sc})
        print(f"wrote per-prompt wide CSV {args.out_wide} (for bootstrap_ci)")

    base = mean.get("baseline", {e: 0.0 for e in ISEAR_EMOTIONS})
    steers = [s for s in mean if s != "baseline"]
    print("\n=== judge-based DELTA vs baseline (rows=steer, cols=measured) ===")
    print("steer\\meas  " + " ".join(f"{e[:4]:>5}" for e in ISEAR_EMOTIONS))
    hit = checkable = 0
    for st in steers:
        d = {e: mean[st][e] - base[e] for e in ISEAR_EMOTIONS}
        if st in ISEAR_EMOTIONS:  # only bare-emotion steers have a diagonal
            checkable += 1
            if max(d, key=d.get) == st:
                hit += 1
        print(f"{st:>10} " + " ".join(f"{d[e]:>+5.0f}" for e in ISEAR_EMOTIONS))
    if checkable:
        print(f"\ntarget = argmax for {hit}/{checkable} emotions (judge-based)")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"mean": mean, "baseline": base}, indent=2))
        print(f"wrote {args.out}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Judge-based steering specificity matrix.")
    ap.add_argument("--csv", required=True, type=Path)
    ap.add_argument("--model", default="meta-llama/llama-3.3-70b-instruct")
    ap.add_argument("--concurrency", type=int, default=10)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--out-wide", type=Path, default=None, help="per-prompt wide CSV for bootstrap_ci")
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
