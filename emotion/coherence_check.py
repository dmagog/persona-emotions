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
import json
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
        except Exception as e:
            if attempt == max_retries - 1:
                # Не raise: внутри asyncio.gather без return_exceptions одна
                # неудачная строка убивала весь оплаченный прогон из 448 вызовов.
                print(f"  [судья] строка не оценена после {max_retries} попыток: "
                      f"{type(e).__name__}: {str(e)[:90]}", flush=True)
                return None
            await asyncio.sleep(2.0**attempt)


async def main_async(args) -> None:
    rows = list(csv.DictReader(open(args.csv, encoding="utf-8")))
    client = make_client()
    sem = asyncio.Semaphore(args.concurrency)

    gb = args.group_by

    # Кэш: повторный запуск не переплачивает за уже оценённые строки.
    cache: dict[str, float] = {}
    cache_path = args.cache or (args.csv.parent / f"{args.csv.stem}.coherence.cache.jsonl")
    if cache_path.is_file():
        for line in cache_path.open(encoding="utf-8"):
            try:
                d = json.loads(line)
                cache[d["k"]] = d["score"]
            except Exception:
                continue
        print(f"кэш: {len(cache)} оценённых строк", flush=True)
    cache_fh = cache_path.open("a", encoding="utf-8")
    lock = asyncio.Lock()

    async def one(idx, key, answer):
        ck = f"{key}|{idx}"
        if ck in cache:
            return key, cache[ck]
        async with sem:
            s_ = await score(client, args.model, answer)
        if s_ is not None:
            async with lock:
                cache_fh.write(json.dumps({"k": ck, "score": s_}) + "\n")
                cache_fh.flush()
        return key, s_

    results = await asyncio.gather(
        *[one(i, str(r.get(gb, "")), r["answer"]) for i, r in enumerate(rows)],
        return_exceptions=False,
    )
    cache_fh.close()
    agg = defaultdict(list)
    for key, s in results:
        if s is not None:
            agg[key].append(s)

    base = sum(agg["baseline"]) / len(agg["baseline"]) if agg["baseline"] else 0.0
    order = (["baseline"] if "baseline" in agg else []) + [k for k in agg if k != "baseline"]
    lost = sum(1 for _, s_ in results if s_ is None)

    print(f"{gb:>9} {'coherence':>10} {'Δ vs base':>10} {'n':>3}")
    for steer in order:
        v = agg[steer]
        m = sum(v) / len(v) if v else 0.0
        print(f"{steer:>9} {m:>10.1f} {m - base:>+10.1f} {len(v):>3}")
    if lost:
        print(f"не оценено строк: {lost} из {len(results)}", flush=True)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        lines = [f"# Связность, {args.csv.name}", "",
                 f"Судья `{args.model}`, оценка беглости 0–100 без учёта тона и темы.",
                 f"Строк оценено: {len(results) - lost} из {len(results)}.", "",
                 f"| {gb} | связность | Δ к baseline | n |", "|---|---:|---:|---:|"]
        for steer in order:
            v = agg[steer]
            m = sum(v) / len(v) if v else 0.0
            lines.append(f"| {steer} | {m:.1f} | {m - base:+.1f} | {len(v)} |")
        args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"записано {args.out}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Coherence of steered generations.")
    ap.add_argument("--csv", required=True, type=Path)
    ap.add_argument("--model", default="meta-llama/llama-3.3-70b-instruct")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--group-by", default="steer", help="CSV column to group coherence by (e.g. steer, coeff)")
    ap.add_argument("--cache", type=Path, default=None, help="JSONL-кэш оценок; по умолчанию рядом с CSV")
    ap.add_argument("--out", type=Path, default=None, help="сохранить таблицу в markdown")
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
