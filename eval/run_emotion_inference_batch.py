#!/usr/bin/env python3
"""
Run local inference (no API judge) for every emotion pack under emotion_data_{extract|eval}.

Writes per-emotion CSVs plus one combined file:
  {output_dir}/{emotion}_pos.csv
  {output_dir}/{emotion}_neg.csv
  {output_dir}/all_emotions_{version}.csv

Example:
  python -m eval.run_emotion_inference_batch \\
    --model Qwen/Qwen2.5-3B-Instruct \\
    --version extract \\
    --output_dir eval_emotion/Qwen2.5-3B-Instruct \\
    --infer_backend hf
"""

from __future__ import annotations

import argparse
import csv as _csvmod
import json
import os
import re
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from eval.model_utils import load_model, load_vllm_model

# Schema for per-emotion CSV. Used for incremental (resumable) hf writes so a crash
# mid-run never loses the rows already generated; a re-run skips done question_ids.
CSV_FIELDS = ["emotion", "instruction_type", "question_id", "question", "prompt",
              "answer", "data_version", "pack_path"]


def _load_done_qids(csv_path: Path) -> set[str]:
    if not csv_path.is_file():
        return set()
    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            return {r["question_id"] for r in _csvmod.DictReader(f) if r.get("question_id")}
    except Exception:
        return set()


def _ensure_header(csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not csv_path.is_file() or csv_path.stat().st_size == 0:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            _csvmod.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()


def _append_row(csv_path: Path, row: dict) -> None:
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        _csvmod.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore").writerow(row)
        f.flush()


def _generate_one_hf(model, tokenizer, system: str, user: str, max_tokens: int, temperature: float):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    gen_kw = dict(max_new_tokens=max_tokens, min_new_tokens=1,
                  do_sample=temperature > 0,
                  pad_token_id=tokenizer.pad_token_id,
                  eos_token_id=tokenizer.eos_token_id)
    if temperature > 0:
        gen_kw["temperature"] = temperature
    with torch.no_grad():
        out = model.generate(**inputs, **gen_kw)
    plen = inputs["input_ids"].shape[1]
    return prompt, tokenizer.decode(out[0, plen:], skip_special_tokens=True)

REPO_ROOT = Path(__file__).resolve().parent.parent

PERSONAS = [
    "You are a real person named Priya Sharma, a 28-year-old woman.",
    "You are a real person named Daniel Morgan, a 34-year-old man.",
    "You are a real person named Elena Rossi, a 45-year-old woman.",
    "You are a real person named Marcus Reed, a 22-year-old man.",
    "You are a real person named Aisha Khan, a 31-year-old woman.",
    "You are a real person named Thomas Weber, a 52-year-old man.",
]


def make_experience_prompt(neutral_scenario: str) -> str:
    return (
        f"{neutral_scenario}\n\n"
        "Write a short first-person account of this experience as if it happened to you. "
        "Describe what happened, what you noticed, and how you responded. "
        "Keep it to 4-6 natural sentences."
    )


def emotion_pack_search_paths(version: str, emotion_data_dir: str | None) -> list[Path]:
    """Directories to scan for emotion JSON packs (order matters for resolve fallback)."""
    paths: list[Path] = []
    if emotion_data_dir:
        paths.append(Path(emotion_data_dir))
    paths.append(REPO_ROOT / "data_generation" / f"emotion_data_{version}")
    for alt in ("extract", "eval"):
        p = REPO_ROOT / "data_generation" / f"emotion_data_{alt}"
        if p not in paths:
            paths.append(p)
    paths.append(REPO_ROOT / "datasets" / "emotion")
    return paths


def _find_emotion_pack_path(emotion: str, version: str, emotion_data_dir: str | None) -> tuple[Path | None, list[Path]]:
    """Return (path or None, all paths tried)."""
    emotion = emotion.lower()
    tried: list[Path] = []
    primary = REPO_ROOT / "data_generation" / f"emotion_data_{version}"

    def _try(path: Path) -> Path | None:
        tried.append(path)
        return path if path.is_file() else None

    if emotion_data_dir:
        base = Path(emotion_data_dir)
        for name in (f"{emotion}.json", f"emotion_{emotion}_prompts.json"):
            hit = _try(base / name)
            if hit:
                return hit, tried

    preferred_hit: Path | None = None
    fallback_hit: Path | None = None
    for root in emotion_pack_search_paths(version, emotion_data_dir):
        for name in (f"{emotion}.json", f"emotion_{emotion}_prompts.json"):
            hit = _try(root / name)
            if not hit:
                continue
            if root == primary:
                return hit, tried
            if fallback_hit is None:
                fallback_hit = hit
    if fallback_hit:
        return fallback_hit, tried
    return None, tried


def resolve_emotion_json(emotion: str, version: str, emotion_data_dir: str | None) -> Path:
    """Find emotion pack JSON; falls back to other split or full datasets/emotion pack."""
    hit, tried = _find_emotion_pack_path(emotion, version, emotion_data_dir)
    if hit:
        primary = REPO_ROOT / "data_generation" / f"emotion_data_{version}" / f"{emotion}.json"
        if hit != primary and not primary.is_file():
            print(f"Note: using {hit} (no {primary})")
        return hit
    raise FileNotFoundError(
        f"No pack for emotion '{emotion}' (version={version}). Tried:\n"
        + "\n".join(f"  {p}" for p in tried)
        + "\n\nPrepare data:\n"
        "  1) Put emotion_*_prompts.json under datasets/emotion/\n"
        "  2) python data_generation/split_trait_data.py --kind emotion --validate\n"
        f"     → creates data_generation/emotion_data_{version}/{{emotion}}.json"
    )


def discover_emotions(version: str, emotion_data_dir: str | None, only: list[str] | None) -> list[str]:
    """Return emotions that actually have a resolvable JSON pack (never a blind default list)."""
    if only:
        requested = [e.lower() for e in only]
        missing = [e for e in requested if not _emotion_pack_exists(e, version, emotion_data_dir)]
        if missing:
            raise FileNotFoundError(
                f"Requested emotions missing packs: {missing}. "
                f"Run: python data_generation/split_trait_data.py --kind emotion"
            )
        return requested

    found: set[str] = set()
    for d in emotion_pack_search_paths(version, emotion_data_dir):
        if not d.is_dir():
            continue
        for f in sorted(d.glob("*.json")):
            if f.name in ("sampled_data.json",):
                continue
            m = re.match(r"^emotion_([a-z0-9]+)_prompts\.json$", f.name, re.I)
            stem = m.group(1).lower() if m else f.stem.lower()
            if _emotion_pack_exists(stem, version, emotion_data_dir):
                found.add(stem)

    if not found:
        raise FileNotFoundError(
            f"No emotion JSON packs found for version={version!r}.\n"
            f"Searched: {emotion_pack_search_paths(version, emotion_data_dir)}\n\n"
            "Expected either:\n"
            f"  data_generation/emotion_data_{version}/anger.json (after split), or\n"
            "  datasets/emotion/emotion_anger_prompts.json (full packs)\n\n"
            "From repo root:\n"
            "  python data_generation/split_trait_data.py --kind emotion --validate"
        )
    return sorted(found)


def _emotion_pack_exists(emotion: str, version: str, emotion_data_dir: str | None) -> bool:
    hit, _ = _find_emotion_pack_path(emotion, version, emotion_data_dir)
    return hit is not None


def build_conversations(pack: dict, instruction_type: str) -> list[tuple[str, str, str]]:
    """Returns list of (question_id, user_text, system_text)."""
    rows = []
    instructions = [x[instruction_type] for x in pack["instruction"]]
    for i, scenario in enumerate(pack["questions"]):
        user_text = make_experience_prompt(scenario)
        persona = PERSONAS[i % len(PERSONAS)]
        for k, instruction in enumerate(instructions):
            system = (
                f"{persona} "
                "Speak in a conversational tone, like a spoken first-person account. "
                "Do not mention that you are an AI, a model, or roleplaying. "
                "No markdown, no bullet points, just natural sentences. "
                f"{instruction}"
            )
            qid = f"{instruction_type}_{i}_{k}"
            rows.append((qid, user_text, system))
    return rows


def sample_hf(model, tokenizer, conversations, max_tokens=1000, temperature=0.0):
    texts, answers = [], []
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    for messages in tqdm(conversations, desc="generate"):
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        texts.append(prompt)
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        gen_kw = dict(
            max_new_tokens=max_tokens,
            min_new_tokens=1,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        if temperature > 0:
            gen_kw["temperature"] = temperature
        with torch.no_grad():
            out = model.generate(**inputs, **gen_kw)
        plen = inputs["input_ids"].shape[1]
        answers.append(tokenizer.decode(out[0, plen:], skip_special_tokens=True))
    return texts, answers


def sample_vllm(llm, tokenizer, conversations, max_tokens=1000, temperature=0.0):
    from vllm import SamplingParams

    params = SamplingParams(
        temperature=temperature,
        top_p=1.0,
        max_tokens=max_tokens,
        skip_special_tokens=True,
        stop=[tokenizer.eos_token],
        min_tokens=1,
    )
    texts = [
        tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        for msgs in conversations
    ]
    completions = llm.generate(texts, sampling_params=params, use_tqdm=True)
    answers = [c.outputs[0].text for c in completions]
    return texts, answers


def run_one_emotion(
    llm,
    tokenizer,
    emotion: str,
    instruction_type: str,
    version: str,
    emotion_data_dir: str | None,
    infer_backend: str,
    max_tokens: int,
    temperature: float,
    csv_path: Path | None = None,
) -> pd.DataFrame:
    """Generate (or resume) answers for one emotion x instruction_type.

    If csv_path is given and backend is hf, write each row to the CSV immediately,
    skip question_ids already present, so a crash never loses already-generated rows.
    vllm path is bulk (no per-row resume).
    """
    path = resolve_emotion_json(emotion, version, emotion_data_dir)
    pack = json.loads(path.read_text(encoding="utf-8"))
    spec = build_conversations(pack, instruction_type)

    # Resumable per-row path for hf
    if csv_path is not None and infer_backend == "hf":
        _ensure_header(csv_path)
        done = _load_done_qids(csv_path)
        pending = [(qid, user, sys) for (qid, user, sys) in spec if qid not in done]
        if done:
            print(f"  resume {emotion}/{instruction_type}: {len(done)} done, {len(pending)} pending")
        for qid, user, sys in tqdm(pending, desc=f"{emotion}/{instruction_type}"):
            prompt, answer = _generate_one_hf(llm, tokenizer, sys, user, max_tokens, temperature)
            _append_row(csv_path, {
                "emotion": emotion, "instruction_type": instruction_type,
                "question_id": qid, "question": user, "prompt": prompt,
                "answer": answer, "data_version": version, "pack_path": str(path),
            })
        return pd.read_csv(csv_path)

    # Bulk path (vllm, or when no csv_path provided)
    conversations = [
        [{"role": "system", "content": sys}, {"role": "user", "content": user}]
        for qid, user, sys in spec
    ]
    if infer_backend == "hf":
        prompts, answers = sample_hf(llm, tokenizer, conversations, max_tokens, temperature)
    else:
        prompts, answers = sample_vllm(llm, tokenizer, conversations, max_tokens, temperature)

    return pd.DataFrame(
        {
            "emotion": emotion,
            "instruction_type": instruction_type,
            "question_id": [s[0] for s in spec],
            "question": [s[1] for s in spec],
            "prompt": prompts,
            "answer": answers,
            "data_version": version,
            "pack_path": str(path),
        }
    )


def main():
    parser = argparse.ArgumentParser(description="Batch emotion inference without judge.")
    parser.add_argument("--model", required=True, help="HF model id or local path")
    parser.add_argument(
        "--version",
        default="extract",
        choices=("extract", "eval"),
        help="Use data_generation/emotion_data_{version}/*.json",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Folder for per-emotion CSVs and combined all_emotions_{version}.csv",
    )
    parser.add_argument(
        "--emotion_data_dir",
        default=None,
        help="Override directory of emotion JSON packs",
    )
    parser.add_argument("--emotions", nargs="+", default=None, help="Subset of emotions (default: all found)")
    parser.add_argument("--infer_backend", choices=("hf", "vllm"), default="hf")
    parser.add_argument("--max_tokens", type=int, default=1000)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--combine_only", action="store_true", help="Only merge existing CSVs in output_dir")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    try:
        emotions = discover_emotions(args.version, args.emotion_data_dir, args.emotions)
    except FileNotFoundError as e:
        raise SystemExit(str(e)) from e

    if args.combine_only:
        parts = []
        for em in emotions:
            for it in ("pos", "neg"):
                p = out / f"{em}_{it}.csv"
                if p.is_file():
                    parts.append(pd.read_csv(p))
        if not parts:
            raise SystemExit(f"No CSVs found under {out}")
        combined = pd.concat(parts, ignore_index=True)
        combined_path = out / f"all_emotions_{args.version}.csv"
        combined.to_csv(combined_path, index=False)
        print(f"Wrote {combined_path} ({len(combined)} rows)")
        return

    print(f"Emotions ({len(emotions)}): {', '.join(emotions)}")
    print(f"Version: {args.version} | Backend: {args.infer_backend} | Model: {args.model}")

    # Drop existing CSVs only on --overwrite. Otherwise per-row resume kicks in
    # (run_one_emotion reads done question_ids and skips them on hf path).
    if args.overwrite:
        for em in emotions:
            for it in ("pos", "neg"):
                p = out / f"{em}_{it}.csv"
                if p.is_file():
                    p.unlink()
        print("Overwrite: cleared existing per-emotion CSVs.")
    print("Resumable: hf backend writes each row incrementally; re-run to continue from a crash.")

    if args.infer_backend == "hf":
        llm, tokenizer = load_model(args.model)
        lora_path = None
    else:
        llm, tokenizer, lora_path = load_vllm_model(args.model)
        if lora_path:
            raise RuntimeError("LoRA + batch emotion path not implemented; use --infer_backend hf")

    all_parts: list[pd.DataFrame] = []

    for emotion in emotions:
        for it in ("pos", "neg"):
            csv_path = out / f"{emotion}_{it}.csv"
            print(f"\n=== {emotion} / {it} ===")
            df = run_one_emotion(
                llm,
                tokenizer,
                emotion,
                it,
                args.version,
                args.emotion_data_dir,
                args.infer_backend,
                args.max_tokens,
                args.temperature,
                csv_path=csv_path,
            )
            # vllm path returns DF without writing; write bulk
            if args.infer_backend == "vllm":
                df.to_csv(csv_path, index=False)
            print(f"Saved {csv_path} ({len(df)} rows)")
            all_parts.append(df)

    combined = pd.concat(all_parts, ignore_index=True)
    combined_path = out / f"all_emotions_{args.version}.csv"
    combined.to_csv(combined_path, index=False)
    print(f"\nCombined: {combined_path} ({len(combined)} rows)")


if __name__ == "__main__":
    main()
