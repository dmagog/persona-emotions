"""Coherence check of steered generations (W5: fluency not measured).

Judge each steered answer for coherence/fluency (0-100, tone-independent) via
OpenRouter, and report mean coherence per steer condition. Tells us whether
steering at the chosen coefficient degrades text quality. Reads a steer CSV with
an 'answer' column. Local: needs OPENAI_API_KEY + OPENAI_BASE_URL; no GPU.

Usage:
    OPENAI_API_KEY=... OPENAI_BASE_URL=https://openrouter.ai/api/v1 \
    python -m emotion.coherence_check --csv results/steer_spec_gemma_saefeat_ans.csv
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import sys
from collections import defaultdict
from pathlib import Path

from emotion.pairwise_judge import parse_score
from emotion.run_pairwise_judge import make_client

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

PROMPT = """Rate how coherent and fluent the following text is, ignoring its emotional tone or topic.
0 = broken, incoherent, repetitive or ungrammatical; 100 = perfectly fluent and coherent.

Text:
[START]
{answer}
[END]

Reply with only a number from 0 to 100, nothing else."""


async def score(client, model, answer, max_retries=4):
    messages = [{"role": "user", "content": PROMPT.format(answer=answer)}]
    for attempt in range(max_retries):
        try:
            r = await client.chat.completions.create(model=model, messages=messages, max_tokens=8, temperature=0)
            return parse_score(r.choices[0].message.content)
        except Exception:
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(2.0**attempt)


async def main_async(args) -> None:
    rows = list(csv.DictReader(open(args.csv, encoding="utf-8")))
    client = make_client()
    sem = asyncio.Semaphore(args.concurrency)

    async def one(steer, answer):
        async with sem:
            return steer, await score(client, args.model, answer)

    results = await asyncio.gather(*[one(r["steer"], r["answer"]) for r in rows])
    agg = defaultdict(list)
    for steer, s in results:
        if s is not None:
            agg[steer].append(s)

    base = sum(agg["baseline"]) / len(agg["baseline"]) if agg["baseline"] else 0.0
    print(f"{'steer':>9} {'coherence':>10} {'Δ vs base':>10} {'n':>3}")
    for steer in ["baseline"] + [k for k in agg if k != "baseline"]:
        v = agg[steer]
        m = sum(v) / len(v) if v else 0.0
        print(f"{steer:>9} {m:>10.1f} {m - base:>+10.1f} {len(v):>3}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Coherence of steered generations.")
    ap.add_argument("--csv", required=True, type=Path)
    ap.add_argument("--model", default="meta-llama/llama-3.3-70b-instruct")
    ap.add_argument("--concurrency", type=int, default=8)
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
