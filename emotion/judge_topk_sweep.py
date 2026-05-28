"""Judge the top-k steering sweep: emotion presence vs number of SAE features (W6).

Reads results/steer_topk_sweep.csv (columns: k, steer, prompt_id, answer, ...).
For every answer x ISEAR emotion, asks the LLM judge (0-100). Checkpoints every
score to a resumable JSONL keyed by (k, steer, prompt_id, measured) so a crash
never loses the run and re-runs skip done work. Writes one wide CSV per k
(baseline rows from k=0 + that k's steered rows) for `bootstrap_ci`, so we can
plot the diagonal effect and sadness leakage as a function of k.

Usage:
    OPENAI_API_KEY=... OPENAI_BASE_URL=https://openrouter.ai/api/v1 \
    python -m emotion.judge_topk_sweep --csv results/steer_topk_sweep.csv \
        --out-prefix results/judge_topk --model meta-llama/llama-3.3-70b-instruct
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
            done[(d["k"], d["steer"], d["pid"], d["measured"])] = d["score"]
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

    async def one(k, steer, pid, measured, answer):
        async with sem:
            s = await score_answer(client, args.model, measured, answer)
        if s is not None:
            async with lock:
                cache_fh.write(json.dumps({"k": k, "steer": steer, "pid": pid, "measured": measured, "score": s}) + "\n")
                cache_fh.flush()
        return k, steer, pid, measured, s

    tasks = []
    for r in rows:
        for measured in ISEAR_EMOTIONS:
            key = (r["k"], r["steer"], r["prompt_id"], measured)
            if key in done:
                continue
            tasks.append(one(r["k"], r["steer"], r["prompt_id"], measured, r["answer"]))
    print(f"judging {len(tasks)} new pairs ({len(done)} cached) ...", flush=True)
    new = await asyncio.gather(*tasks)
    cache_fh.close()

    # merge: scores[(k, steer, pid)][measured] = s
    scores: dict = defaultdict(dict)
    for (k, steer, pid, measured), s in done.items():
        scores[(k, steer, pid)][measured] = s
    for k, steer, pid, measured, s in new:
        if s is not None:
            scores[(k, steer, pid)][measured] = s

    ks = sorted({k for (k, _s, _p) in scores if k != "0"}, key=int)
    # baseline rows come from k == "0"
    base_keys = [(k, st, pid) for (k, st, pid) in scores if k == "0"]

    for K in ks:
        out = Path(f"{args.out_prefix}_k{K}_wide.csv")
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["steer", "prompt_id", *ISEAR_EMOTIONS])
            w.writeheader()
            for (k, st, pid) in base_keys:  # baseline (steer already == "baseline")
                sc = scores[(k, st, pid)]
                if all(e in sc for e in ISEAR_EMOTIONS):
                    w.writerow({"steer": "baseline", "prompt_id": pid, **sc})
            for (k, st, pid) in [key for key in scores if key[0] == K]:
                sc = scores[(k, st, pid)]
                if all(e in sc for e in ISEAR_EMOTIONS):
                    w.writerow({"steer": st, "prompt_id": pid, **sc})
        print(f"wrote {out}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Judge the top-k steering sweep (W6 minimal).")
    ap.add_argument("--csv", required=True, type=Path)
    ap.add_argument("--out-prefix", default="results/judge_topk")
    ap.add_argument("--model", default="meta-llama/llama-3.3-70b-instruct")
    ap.add_argument("--concurrency", type=int, default=20)
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
