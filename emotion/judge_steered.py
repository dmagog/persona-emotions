"""Judge steered generations for target-emotion presence (OpenRouter).

For emotions the GoEmotions encoder can't measure (guilt, disgust), score steered
answers with an LLM judge: how strongly does each answer express the target
emotion (0-100)? Compare baseline vs steered-toward-X. Reads a steer CSV with an
'answer' column (produced by `steer_specificity.py --save-answers`). Runs locally:
needs OPENAI_API_KEY + OPENAI_BASE_URL (OpenRouter); no GPU.

Usage:
    OPENAI_API_KEY=... OPENAI_BASE_URL=https://openrouter.ai/api/v1 \
    python -m emotion.judge_steered --csv results/steer_spec_gemma_saefeat_ans.csv \
        --emotions guilt,disgust
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

PROMPT = """Rate how strongly the following first-person text expresses {emotion}.
0 = no {emotion} at all; 100 = intensely {emotion}.

Text:
[START]
{answer}
[END]

Reply with only a number from 0 to 100, nothing else."""


async def score_answer(client, model, emotion, answer, max_retries=4):
    messages = [{"role": "user", "content": PROMPT.format(emotion=emotion, answer=answer)}]
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
    emotions = args.emotions.split(",")

    async def one(emotion, steer, answer):
        async with sem:
            return emotion, steer, await score_answer(client, args.model, emotion, answer)

    tasks = [
        one(emotion, r["steer"], r["answer"])
        for emotion in emotions
        for r in rows
        if r["steer"] in ("baseline", emotion)
    ]
    results = await asyncio.gather(*tasks)

    agg = defaultdict(lambda: defaultdict(list))
    for emotion, steer, s in results:
        if s is not None:
            agg[emotion][steer].append(s)

    print(f"{'emotion':>8} {'baseline':>9} {'steered':>8} {'delta':>7}")
    for emotion in emotions:
        b = agg[emotion]["baseline"]
        st = agg[emotion][emotion]
        mb = sum(b) / len(b) if b else 0.0
        ms = sum(st) / len(st) if st else 0.0
        print(f"{emotion:>8} {mb:>9.1f} {ms:>8.1f} {ms - mb:>+7.1f}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Judge steered answers for target-emotion presence.")
    ap.add_argument("--csv", required=True, type=Path)
    ap.add_argument("--emotions", default="guilt,disgust")
    ap.add_argument("--model", default="meta-llama/llama-3.3-70b-instruct")
    ap.add_argument("--concurrency", type=int, default=8)
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
