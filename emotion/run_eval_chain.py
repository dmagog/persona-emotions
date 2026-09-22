"""Evaluation chain: from the specificity matrix to the reported numbers.

The twin of run_model_chain. That one takes a model to the matrix, this one
takes the matrix to the numbers that go into the report and the demo.

The free stages, meaning the local encoder, always run. Judge stages run behind
a flag, because they cost money and need a key.

Stages are idempotent: an artifact is reused only when it is newer than its
input. Otherwise, after the matrix is recomputed, the old intervals would stay
in the report without a word.

Usage:
    python -m emotion.run_eval_chain --slug Qwen3-1.7B
    python -m emotion.run_eval_chain --slug Qwen3-1.7B --judge
    python -m emotion.run_eval_chain --slug Qwen3-1.7B --judge --judge2 qwen/qwen-2.5-72b-instruct
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def sh(args: list[str], log: Path, out: Path | None = None) -> int:
    """Run one stage, streaming its output into the log."""
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    with log.open("a", encoding="utf-8") as fh:
        fh.write(f"\n[stage begin {stamp}] {' '.join(str(a) for a in args)}\n")
        fh.flush()
        rc = subprocess.run([str(a) for a in args], stdout=fh, stderr=subprocess.STDOUT,
                            cwd=REPO).returncode
        fh.write(f"[stage end rc={rc}] {time.strftime('%Y-%m-%dT%H:%M:%S')}\n")
    mark = "ok" if rc == 0 else f"FAILED rc={rc}"
    print(f"  {mark}" + (f" → {out.name}" if out else ""), flush=True)
    return rc


def fresh(artifact: Path, source: Path) -> bool:
    """An artifact is usable only when it is newer than its input.

    Checking that the file merely exists has already cost us a run: after the
    matrix was recomputed, the old numbers stayed in the report and looked new.
    """
    return (artifact.is_file() and source.is_file()
            and artifact.stat().st_mtime >= source.stat().st_mtime)


def stage(name: str, artifact: Path, source: Path, cmd: list, log: Path,
          required: bool = True) -> bool:
    """One stage: skip it when fresh, otherwise run it."""
    if fresh(artifact, source):
        print(f"{name}: newer than its input, skipping", flush=True)
        return True
    if artifact.is_file():
        print(f"{name}: stale against {source.name}, recomputing ...", flush=True)
    else:
        print(f"{name} …", flush=True)
    rc = sh(cmd, log, artifact)
    if rc != 0:
        if required:
            raise SystemExit(f"{name}: stage failed, see {log}")
        print(f"{name}: skipped (rc={rc}), the chain continues", flush=True)
        return False
    if not artifact.is_file():
        raise SystemExit(f"{name}: the stage finished, but {artifact} never appeared")
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluation chain for a single run.")
    ap.add_argument("--slug", required=True, help="directory name under runs/")
    ap.add_argument("--variant", default="raw", help="which matrix is evaluated: raw, sae, ...")
    ap.add_argument("--judge", action="store_true", help="run the judge stages; needs a key and costs money")
    ap.add_argument("--judge-model", default="meta-llama/llama-3.3-70b-instruct")
    ap.add_argument("--judge2", default=None, help="second judge, used to measure agreement")
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--n-boot", type=int, default=20000)
    args = ap.parse_args()

    runs = REPO / "runs" / args.slug
    if not runs.is_dir():
        raise SystemExit(f"no such directory: {runs}")
    log = runs / "eval.log"
    py = sys.executable

    meta_path = runs / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    model_id = meta.get("model")
    layer = meta.get("layer")

    # The matrix may carry either the old name or the variant one.
    matrix = runs / f"steer_specificity_{args.variant}.csv"
    if not matrix.is_file():
        matrix = runs / "steer_specificity.csv"
    if not matrix.is_file():
        raise SystemExit(f"no matrix in {runs}; run run_model_chain first")

    print(f"\n=== evaluating {args.slug} ({matrix.name}) ===", flush=True)

    # --- Local evaluation stages ---

    stage("1. encoder intervals",
          runs / "ci_encoder.md", matrix,
          [py, "-m", "emotion.bootstrap_ci", "--csv", matrix, "--n-boot", args.n_boot,
           "--out", runs / "ci_encoder.md"],
          log)

    pairs_dir = REPO / "eval_emotion" / args.slug
    if pairs_dir.is_dir():
        combined = pairs_dir / "all_emotions_extract.csv"
        stage("2. pair separability",
              runs / "separability.csv", combined,
              [py, "-m", "emotion.score_csv", "--data-dir", pairs_dir,
               "--out", runs / "separability.csv"],
              log, required=False)
    else:
        print("2. pair separability: skipped because paired outputs are absent", flush=True)

    # --- Judge stages ---

    if not args.judge:
        print("\njudge stages skipped, pass --judge to run them", flush=True)
        print(f"=== {args.slug}: evaluation done -> {runs} ===\n", flush=True)
        return

    import os
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("no OPENAI_API_KEY, the judge stages cannot run")

    judge_wide = runs / "judge_wide.csv"
    stage("4. judge matrix",
          judge_wide, matrix,
          [py, "-m", "emotion.judge_specificity", "--csv", matrix,
           "--model", args.judge_model, "--concurrency", args.concurrency,
           "--out", runs / "judge_matrix.json", "--out-wide", judge_wide,
           "--cache", runs / "judge.cache.jsonl"],
          log)

    stage("5. judge intervals",
          runs / "ci_judge.md", judge_wide,
          [py, "-m", "emotion.bootstrap_ci", "--csv", judge_wide, "--n-boot", args.n_boot,
           "--out", runs / "ci_judge.md"],
          log)

    stage("6. coherence",
          runs / "coherence.md", matrix,
          [py, "-m", "emotion.coherence_check", "--csv", matrix,
           "--model", args.judge_model, "--concurrency", args.concurrency,
           "--cache", runs / "coherence.cache.jsonl", "--out", runs / "coherence.md"],
          log)

    if args.judge2:
        wide2 = runs / "judge2_wide.csv"
        stage("7. second judge",
              wide2, matrix,
              [py, "-m", "emotion.judge_specificity", "--csv", matrix,
               "--model", args.judge2, "--concurrency", args.concurrency,
               "--out", runs / "judge2_matrix.json", "--out-wide", wide2,
               "--cache", runs / "judge2.cache.jsonl"],
              log)
        stage("8. judge agreement",
              runs / "judge_agreement.md", wide2,
              [py, "-m", "emotion.judge_agreement", "--judge1", judge_wide,
               "--judge2", wide2, "--name1", args.judge_model, "--name2", args.judge2,
               "--out", runs / "judge_agreement.md"],
              log, required=False)

    meta["eval"] = {
        "variant": args.variant,
        "judge_model": args.judge_model if args.judge else None,
        "judge2": args.judge2,
        "n_boot": args.n_boot,
        "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"=== {args.slug}: evaluation done -> {runs} ===\n", flush=True)


if __name__ == "__main__":
    main()
