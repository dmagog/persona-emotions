"""Run the pairwise emotion judge over generated pos/neg CSVs (OpenRouter).

Reads ``OPENAI_API_KEY`` (and optional ``OPENAI_BASE_URL``, default OpenRouter)
from the environment. For each emotion, pairs pos/neg answers and asks the judge
how much more the positive response expresses the target emotion; prints summary
stats and optionally writes per-pair scores to a CSV.

Usage:
    OPENAI_API_KEY=... OPENAI_BASE_URL=https://openrouter.ai/api/v1 \
    python -m emotion.run_pairwise_judge --data-dir eval_emotion/Qwen2.5-3B-Instruct \
        --emotion all --model meta-llama/llama-3.3-70b-instruct --out judge_scores.csv
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import os
import sys
from pathlib import Path

from emotion.pairwise_judge import build_pairs, passes, score_pair_text
from emotion.space import ISEAR_EMOTIONS

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))  # 2**31-1: C long is 32-bit on Windows
DEFAULT_BASE = "https://openrouter.ai/api/v1"


def load_rows(path: Path, limit: int | None = None) -> list[dict]:
    rows: list[dict] = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
            if limit and len(rows) >= limit:
                break
    return rows


def make_client():
    from openai import AsyncOpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY not set")
    return AsyncOpenAI(base_url=os.environ.get("OPENAI_BASE_URL") or DEFAULT_BASE, api_key=api_key)


async def run_emotion(data_dir, emotion, client, model, limit, concurrency):
    pos = load_rows(data_dir / f"{emotion}_pos.csv", limit)
    neg = load_rows(data_dir / f"{emotion}_neg.csv", limit)
    pairs = build_pairs(pos, neg, emotion)
    sem = asyncio.Semaphore(concurrency)

    async def one(p):
        async with sem:
            return p, await score_pair_text(p, client, model)

    return await asyncio.gather(*[one(p) for p in pairs])


async def main_async(args) -> None:
    client = make_client()
    emotions = list(ISEAR_EMOTIONS) if args.emotion == "all" else [args.emotion]
    out_rows = []
    print(f"{'emotion':<9} {'pairs':>5} {'scored':>6} {'mean':>6} {'kept>=%d' % args.threshold:>8}")
    for emo in emotions:
        results = await run_emotion(
            Path(args.data_dir), emo, client, args.model, args.limit, args.concurrency
        )
        scored = [s for _, s in results if s is not None]
        mean = sum(scored) / len(scored) if scored else 0.0
        kept = sum(1 for _, s in results if passes(s, args.threshold))
        print(f"{emo:<9} {len(results):>5} {len(scored):>6} {mean:>6.1f} {kept:>8}")
        for p, s in results:
            out_rows.append(
                {"emotion": emo, "key": p.key, "score": s, "keep": passes(s, args.threshold)}
            )

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["emotion", "key", "score", "keep"])
            writer.writeheader()
            writer.writerows(out_rows)
        print(f"wrote {args.out}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the pairwise emotion judge over pos/neg CSVs.")
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--emotion", default="all", help="one ISEAR emotion or 'all'")
    ap.add_argument("--model", default="meta-llama/llama-3.3-70b-instruct")
    ap.add_argument("--limit", type=int, default=None, help="cap pairs per emotion (quick runs)")
    ap.add_argument("--threshold", type=float, default=60.0, help="keep pairs with score >= this")
    ap.add_argument("--concurrency", type=int, default=5)
    ap.add_argument("--out", type=Path, default=None, help="write per-pair scores to this CSV")
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
