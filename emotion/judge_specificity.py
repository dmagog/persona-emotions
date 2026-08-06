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
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

from emotion.judge_steered import score_answer
from emotion.run_pairwise_judge import make_client
from emotion.space import ISEAR_EMOTIONS

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))


def _cache_path(args) -> Path | None:
    """Where to checkpoint per-call scores. Explicit --cache wins; else derive
    from --out-wide/--out so checkpointing is on by default."""
    if args.cache is not None:
        return args.cache
    base = args.out_wide or args.out
    return base.with_suffix(base.suffix + ".cache.jsonl") if base is not None else None


def answer_key(answer: str) -> str:
    """Отпечаток оценённого текста — часть ключа кэша.

    Ключ был (steer, pid, measured), без самого ответа. При пересчёте матрицы
    номера строк те же, а тексты другие — и судья молча отдавал баллы за старые
    генерации. Полная судейская матрица «пересчитывалась» за двадцать секунд
    и описывала прогон, которого больше нет.
    """
    return hashlib.sha1(str(answer).encode("utf-8")).hexdigest()[:12]


def _load_cache(path: Path) -> dict:
    """Read JSONL checkpoint -> {(steer, pid, measured, answer): score}.
    Tolerates a truncated last line from a crash mid-write."""
    done = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue  # partial final line from an interrupted write
            if "answer" not in d:
                continue  # запись старого образца, без отпечатка текста — не доверяем
            done[(d["steer"], d["pid"], d["measured"], d["answer"])] = d["score"]
    return done


async def main_async(args) -> None:
    rows = list(csv.DictReader(open(args.csv, encoding="utf-8")))
    client = make_client()
    sem = asyncio.Semaphore(args.concurrency)

    cache = _cache_path(args)
    done: dict = {}
    if cache and cache.exists() and not args.fresh:
        done = _load_cache(cache)
        print(f"resuming: {len(done)} scores already in {cache}")

    cache_fh = None
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache_fh = open(cache, "a", encoding="utf-8")
    write_lock = asyncio.Lock()

    async def one(steer, pid, measured, answer):
        async with sem:
            s = await score_answer(client, args.model, measured, answer)
        if s is not None and cache_fh is not None:  # checkpoint every good score
            async with write_lock:
                cache_fh.write(json.dumps({"steer": steer, "pid": pid, "measured": measured,
                                           "answer": answer_key(answer), "score": s}) + "\n")
                cache_fh.flush()
        return steer, pid, measured, s

    tasks = []
    for r in rows:
        steer = r["steer"].split(":")[-1]  # "prompt:anger"->"anger"
        pid = r["prompt_id"]
        ak = answer_key(r["answer"])
        for measured in ISEAR_EMOTIONS:
            if (steer, pid, measured, ak) in done:
                continue  # already checkpointed -> skip
            tasks.append(one(steer, pid, measured, r["answer"]))
    print(f"judging {len(tasks)} new (answer x emotion) pairs ({len(done)} cached) ...")
    new_results = await asyncio.gather(*tasks)
    if cache_fh is not None:
        cache_fh.close()

    # merge cached + freshly computed scores
    fresh = {answer_key(r["answer"]) for r in rows}
    results = [(s, p, m, sc) for (s, p, m, a), sc in done.items() if a in fresh]
    results += list(new_results)

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
    ap.add_argument("--cache", type=Path, default=None,
                    help="JSONL checkpoint of per-call scores (default: <out>.cache.jsonl); resumes on restart")
    ap.add_argument("--fresh", action="store_true", help="ignore any existing cache and re-judge from scratch")
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
