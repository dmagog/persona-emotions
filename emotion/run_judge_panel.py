"""Judge panel: rescore the matrices with external judges and measure agreement.

The primary judge (llama-3.3-70b) scores the whole grid alone, so without a
panel the judge column of the table rests on the taste of one model, which also
shares a family with some of the evaluated checkpoints. The panel adds judges
from other vendors, measures each one's agreement with the primary judge and,
behind a flag, rescores coherence as well.

This script produces these files in runs/<slug>/:
  judge_wide_<tag>.csv        the matrix as seen by an external judge
  judge_agreement_<tag>.md    agreement with the primary judge: Pearson, Spearman, kappa
  coherence_<tag>.md          coherence as seen by an external judge (--coherence)
  compose_judge_wide.csv      composition as seen by the primary judge (--compose)

The default judges are the ones behind the published grid. An OpenRouter key is
required in the environment (OPENAI_API_KEY and OPENAI_BASE_URL).

Usage:
    python -m emotion.run_judge_panel --slug Falcon3-3B-Instruct
    python -m emotion.run_judge_panel --all --coherence
    python -m emotion.run_judge_panel --all --compose
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from emotion import balance

REPO = Path(__file__).resolve().parent.parent

# tag -> OpenRouter model. The tag becomes part of the file names, so it must
# not change for judges that have already been scored, or the files stop matching.
PANEL = {
    "gemini": "google/gemini-3.5-flash-lite",
    "gpt41mini": "openai/gpt-4.1-mini",
}
PRIMARY = "meta-llama/llama-3.3-70b-instruct"


def matrix_of(run: Path) -> Path | None:
    for name in ("steer_specificity_raw.csv", "steer_specificity.csv"):
        if (run / name).is_file():
            return run / name
    return None


def sh(args: list[str]) -> int:
    print("+ " + " ".join(str(a) for a in args), flush=True)
    return subprocess.run([sys.executable, "-m", *args], cwd=REPO).returncode


def main() -> None:
    ap = argparse.ArgumentParser(description="External judge panel over finished runs.")
    ap.add_argument("--slug", action="append", default=[],
                    help="which runs; the flag repeats")
    ap.add_argument("--all", action="store_true", help="every run under runs/")
    ap.add_argument("--judges", default=",".join(PANEL),
                    help=f"comma-separated judge tags, default {','.join(PANEL)}")
    ap.add_argument("--coherence", action="store_true",
                    help="also rescore coherence with every judge in the panel")
    ap.add_argument("--compose", action="store_true",
                    help="score the composition stage with the primary judge")
    args = ap.parse_args()

    runs = sorted(p for p in (REPO / "runs").iterdir() if p.is_dir()) if args.all \
        else [REPO / "runs" / s for s in args.slug]
    if not runs:
        raise SystemExit("either --slug or --all is required")
    tags = [t.strip() for t in args.judges.split(",") if t.strip()]
    unknown = [t for t in tags if t not in PANEL]
    if unknown:
        raise SystemExit(f"unknown judges {unknown}; known tags are {list(PANEL)}")

    # The money runs out in silence: the provider answers 402, rows never reach
    # the matrix, and the table reports a subsample. Sample the balance around it.
    print(balance.line("balance before the run"), flush=True)

    fails = 0
    for run in runs:
        csv = matrix_of(run)
        if csv is None:
            print(f"{run.name}: no matrix, skipping", flush=True)
            continue
        for tag in tags:
            model = PANEL[tag]
            rc = sh(["emotion.judge_specificity", "--csv", str(csv), "--model", model,
                     "--out-wide", str(run / f"judge_wide_{tag}.csv"),
                     "--cache", str(run / f"judge_{tag}.cache.jsonl")])
            if rc != 0:
                fails += 1
                continue
            if (run / "judge_wide.csv").is_file():
                fails += sh(["emotion.judge_agreement",
                             "--judge1", str(run / "judge_wide.csv"),
                             "--judge2", str(run / f"judge_wide_{tag}.csv"),
                             "--name1", "llama-3.3-70b", "--name2", model,
                             "--out", str(run / f"judge_agreement_{tag}.md")]) != 0
            if args.coherence:
                fails += sh(["emotion.coherence_check", "--csv", str(csv),
                             "--model", model,
                             "--cache", str(run / f"coherence_{tag}.cache.jsonl"),
                             "--out", str(run / f"coherence_{tag}.md")]) != 0
        if args.compose and (run / "compose.csv").is_file():
            fails += sh(["emotion.judge_specificity", "--csv", str(run / "compose.csv"),
                         "--model", PRIMARY,
                         "--out-wide", str(run / "compose_judge_wide.csv"),
                         "--cache", str(run / "compose_judge.cache.jsonl")]) != 0

    print(balance.line("balance after the run"), flush=True)
    print(f"panel complete, failures: {fails}", flush=True)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
