#!/usr/bin/env python3
"""
Split persona-pack JSON files into eval / extract question sets.

Supports:
  * **trait** — files like `outgoing.json` (typically 40 questions → 20 + 20).
  * **emotion** — files like `emotion_anger_prompts.json`; output `anger.json` under
    `data_generation/emotion_data_eval/` and `emotion_data_extract/`.

Same `instruction` and `eval_prompt` are copied into both splits; only `questions` differ.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Repo root (parent of data_generation/)
REPO_ROOT = Path(__file__).resolve().parent.parent


def load_pack(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_pack(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def split_questions(questions: List[str], split_ratio: float = 0.5) -> Tuple[List[str], List[str]]:
    if not questions:
        return [], []
    split_point = max(1, int(len(questions) * split_ratio))
    if split_point >= len(questions):
        split_point = len(questions) - 1
    if split_point < 1:
        split_point = 1
    return questions[:split_point], questions[split_point:]


def create_split_data(original: Dict[str, Any], questions_subset: List[str]) -> Dict[str, Any]:
    return {
        "instruction": list(original["instruction"]),
        "questions": list(questions_subset),
        "eval_prompt": original["eval_prompt"],
    }


def validate_pack(data: Dict[str, Any], label: str, require_five_instructions: bool = True) -> None:
    for key in ("instruction", "questions", "eval_prompt"):
        if key not in data:
            raise ValueError(f"{label}: missing key {key!r}")
    inst = data["instruction"]
    if not isinstance(inst, list) or (require_five_instructions and len(inst) != 5):
        raise ValueError(f"{label}: expected instruction list of length 5, got {len(inst) if isinstance(inst, list) else type(inst)}")
    for i, row in enumerate(inst):
        if not isinstance(row, dict) or "pos" not in row or "neg" not in row:
            raise ValueError(f"{label}: instruction[{i}] must be dict with pos/neg")
    qs = data["questions"]
    if not isinstance(qs, list) or len(qs) < 2:
        raise ValueError(f"{label}: need at least 2 questions to split, got {len(qs) if isinstance(qs, list) else type(qs)}")


def emotion_canonical_stem(path: Path) -> str | None:
    """emotion_anger_prompts.json -> anger"""
    m = re.match(r"^emotion_([a-z0-9]+)_prompts\.json$", path.name, re.I)
    return m.group(1).lower() if m else None


def discover_files(input_dir: Path, kind: str) -> List[Path]:
    if not input_dir.is_dir():
        return []
    paths = sorted(input_dir.glob("*.json"))
    out: List[Path] = []
    for p in paths:
        if p.name == "sampled_data.json":
            continue
        if kind == "emotion":
            if emotion_canonical_stem(p) is not None:
                out.append(p)
        else:
            out.append(p)
    return out


def output_stem(path: Path, kind: str) -> str:
    if kind == "emotion":
        em = emotion_canonical_stem(path)
        if em:
            return em
    return path.stem


def split_one_pack(
    input_path: Path,
    kind: str,
    eval_dir: Path,
    extract_dir: Path,
    split_ratio: float,
    min_questions: int,
) -> None:
    label = input_path.name
    data = load_pack(input_path)
    validate_pack(data, label)

    n = len(data["questions"])
    if n < min_questions:
        print(f"Skip {label}: only {n} questions (min_questions={min_questions})")
        return

    eval_q, extract_q = split_questions(data["questions"], split_ratio=split_ratio)
    if not eval_q or not extract_q:
        print(f"Skip {label}: empty split ({len(eval_q)} + {len(extract_q)})")
        return

    stem = output_stem(input_path, kind)
    eval_data = create_split_data(data, eval_q)
    extract_data = create_split_data(data, extract_q)

    eval_file = eval_dir / f"{stem}.json"
    extract_file = extract_dir / f"{stem}.json"
    save_pack(eval_data, eval_file)
    save_pack(extract_data, extract_file)
    print(f"  {stem}: {len(eval_q)} eval + {len(extract_q)} extract → {eval_file.name}")


def run_split(
    kind: str,
    input_dir: Path,
    eval_dir: Path,
    extract_dir: Path,
    split_ratio: float,
    min_questions: int,
) -> int:
    eval_dir.mkdir(parents=True, exist_ok=True)
    extract_dir.mkdir(parents=True, exist_ok=True)

    files = discover_files(input_dir, kind)
    if not files:
        print(f"No JSON files found in {input_dir} (kind={kind})")
        return 1

    print(f"Found {len(files)} file(s) in {input_dir}")
    for p in files:
        print(f"Processing {p.name}...")
        try:
            split_one_pack(p, kind, eval_dir, extract_dir, split_ratio, min_questions)
        except Exception as e:
            print(f"  Error: {e}")
    return 0


def validate_pairs(eval_dir: Path, extract_dir: Path, kind: str) -> None:
    eval_files = {p.stem: p for p in eval_dir.glob("*.json")}
    extract_files = {p.stem: p for p in extract_dir.glob("*.json")}
    common = set(eval_files) & set(extract_files)
    print("\nValidation:")
    print(f"  Eval stems: {len(eval_files)}, extract stems: {len(extract_files)}, paired: {len(common)}")
    for stem in sorted(common):
        try:
            ev = load_pack(eval_files[stem])
            ex = load_pack(extract_files[stem])
            n_e, n_x = len(ev["questions"]), len(ex["questions"])
            ok = n_e > 0 and n_x > 0 and n_e + n_x >= 2
            status = "OK" if ok else "?"
            print(f"  [{status}] {stem}: {n_e} eval + {n_x} extract questions")
        except Exception as e:
            print(f"  [?] {stem}: {e}")


def default_paths(kind: str) -> Tuple[Path, Path, Path]:
    """Return (input_dir, eval_dir, extract_dir) relative to repo root."""
    if kind == "emotion":
        return (
            REPO_ROOT / "datasets" / "emotion",
            REPO_ROOT / "data_generation" / "emotion_data_eval",
            REPO_ROOT / "data_generation" / "emotion_data_extract",
        )
    return (
        REPO_ROOT / "data_generation" / "trait_data_big_five",
        REPO_ROOT / "data_generation" / "trait_data_eval",
        REPO_ROOT / "data_generation" / "trait_data_extract",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Split trait or emotion pack JSONs into eval / extract question sets."
    )
    parser.add_argument(
        "--kind",
        choices=("trait", "emotion"),
        default="trait",
        help="trait: Big Five-style *.json in input dir. emotion: emotion_*_prompts.json → {emotion}.json outputs.",
    )
    parser.add_argument(
        "--input-dir",
        default=None,
        help="Directory containing source JSON (default: trait_data_big_five or datasets/emotion).",
    )
    parser.add_argument(
        "--eval-dir",
        default=None,
        help="Output directory for eval split (default: data_generation/trait_data_eval or emotion_data_eval).",
    )
    parser.add_argument(
        "--extract-dir",
        default=None,
        help="Output directory for extract split (default: trait_data_extract or emotion_data_extract).",
    )
    parser.add_argument(
        "--split-ratio",
        type=float,
        default=0.5,
        help="Fraction of questions (in order) assigned to eval; rest to extract. Default 0.5.",
    )
    parser.add_argument(
        "--min-questions",
        type=int,
        default=4,
        help="Skip files with fewer than this many questions (default 4).",
    )
    parser.add_argument("--validate", action="store_true", help="Print per-trait/emotion counts after split.")
    args = parser.parse_args()

    d_in, d_eval, d_ext = default_paths(args.kind)
    input_dir = Path(args.input_dir) if args.input_dir else d_in
    eval_dir = Path(args.eval_dir) if args.eval_dir else d_eval
    extract_dir = Path(args.extract_dir) if args.extract_dir else d_ext

    if not input_dir.is_absolute():
        input_dir = (REPO_ROOT / input_dir).resolve()
    if not eval_dir.is_absolute():
        eval_dir = (REPO_ROOT / eval_dir).resolve()
    if not extract_dir.is_absolute():
        extract_dir = (REPO_ROOT / extract_dir).resolve()

    print(f"=== split_trait_data ({args.kind}) ===")
    print(f"Input:     {input_dir}")
    print(f"Eval:      {eval_dir}")
    print(f"Extract:   {extract_dir}")
    print(f"Ratio:     {args.split_ratio} (eval gets first slice)")
    print()

    if not input_dir.is_dir():
        print(f"Error: input directory not found: {input_dir}")
        return 1

    rc = run_split(
        args.kind,
        input_dir,
        eval_dir,
        extract_dir,
        split_ratio=args.split_ratio,
        min_questions=args.min_questions,
    )
    if args.validate:
        validate_pairs(eval_dir, extract_dir, args.kind)
    print("\nDone.")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
