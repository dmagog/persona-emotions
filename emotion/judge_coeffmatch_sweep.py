"""Judge the coeff-matched raw-vs-SAE sweep (Phase 2.3, W2).

Reads results/steer_coeffmatch_sweep.csv (columns: vtype, coeff, steer, prompt_id,
answer, ...). For every answer x ISEAR emotion asks the LLM judge (0-100).
Checkpoints every score to a resumable JSONL keyed by
(vtype, coeff, steer, prompt_id, measured). Writes one wide CSV per (vtype, coeff)
condition (baseline rows + that condition's steered rows) for `bootstrap_ci`.

Usage:
    OPENAI_API_KEY=... OPENAI_BASE_URL=https://openrouter.ai/api/v1 \
    python -m emotion.judge_coeffmatch_sweep --csv results/steer_coeffmatch_sweep.csv \
        --out-prefix results/judge_coeffmatch --model meta-llama/llama-3.3-70b-instruct
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


def load_cache(path: Path) -> dict:
    done = {}
    if not path.exists():
        return done
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            done[(d["vtype"], d["coeff"], d["steer"], d["pid"], d["measured"])] = d["score"]
    return done


async def main_async(args) -> None:
    rows = list(csv.DictReader(open(args.csv, encoding="utf-8")))
    client = make_client()
    sem = asyncio.Semaphore(args.concurrency)

    cache = Path(f"{args.out_prefix}.cache.jsonl")
    done = load_cache(cache)
    if done:
        print(f"resuming: {len(done)} cached scores in {cache}")
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache_fh = open(cache, "a", encoding="utf-8")
    lock = asyncio.Lock()

    async def one(vtype, coeff, steer, pid, measured, answer):
        async with sem:
            s = await score_answer(client, args.model, measured, answer)
        if s is not None:
            async with lock:
                cache_fh.write(json.dumps({"vtype": vtype, "coeff": coeff, "steer": steer,
                                           "pid": pid, "measured": measured, "score": s}) + "\n")
                cache_fh.flush()
        return vtype, coeff, steer, pid, measured, s

    tasks = []
    for r in rows:
        for measured in ISEAR_EMOTIONS:
            key = (r["vtype"], r["coeff"], r["steer"], r["prompt_id"], measured)
            if key in done:
                continue
            tasks.append(one(r["vtype"], r["coeff"], r["steer"], r["prompt_id"], measured, r["answer"]))
    print(f"judging {len(tasks)} new pairs ({len(done)} cached) ...", flush=True)
    new = await asyncio.gather(*tasks)
    cache_fh.close()

    scores: dict = defaultdict(dict)
    for (vt, c, st, pid, measured), s in done.items():
        scores[(vt, c, st, pid)][measured] = s
    for vt, c, st, pid, measured, s in new:
        if s is not None:
            scores[(vt, c, st, pid)][measured] = s

    base_keys = [(vt, c, st, pid) for (vt, c, st, pid) in scores if vt == "baseline"]
    conds = sorted({(vt, c) for (vt, c, _st, _p) in scores if vt != "baseline"},
                   key=lambda x: (x[0], float(x[1])))

    for vt, c in conds:
        out = Path(f"{args.out_prefix}_{vt}_c{c}_wide.csv")
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["steer", "prompt_id", *ISEAR_EMOTIONS])
            w.writeheader()
            for (bvt, bc, bst, pid) in base_keys:
                sc = scores[(bvt, bc, bst, pid)]
                if all(e in sc for e in ISEAR_EMOTIONS):
                    w.writerow({"steer": "baseline", "prompt_id": pid, **sc})
            for key in [k for k in scores if k[0] == vt and k[1] == c]:
                sc = scores[key]
                if all(e in sc for e in ISEAR_EMOTIONS):
                    w.writerow({"steer": key[2], "prompt_id": key[3], **sc})
        print(f"wrote {out}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Judge the coeff-matched raw-vs-SAE sweep (W2).")
    ap.add_argument("--csv", required=True, type=Path)
    ap.add_argument("--out-prefix", default="results/judge_coeffmatch")
    ap.add_argument("--model", default="meta-llama/llama-3.3-70b-instruct")
    ap.add_argument("--concurrency", type=int, default=20)
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
