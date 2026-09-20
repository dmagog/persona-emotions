"""Build the ordered-difference composition table from stored run artifacts.

The collector reads only the 42-pair matrices used by the article. For a pair
``X - Y``, target retention means that X is above the unsteered baseline and
attenuation means that Y is lower than under X-only steering. Joint success
requires both conditions for the same pair.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from emotion.space import ALL_PAIRS, ISEAR_EMOTIONS

REPO = Path(__file__).resolve().parent.parent
GEN = "compose_allpairs.csv"
JUDGE = "compose_allpairs_judge_wide.csv"
MIN_ROWS = 40


def _tables(data: pd.DataFrame) -> tuple[dict[str, float], pd.DataFrame]:
    baseline = {
        emotion: data.loc[data["steer"] == "baseline", emotion].mean()
        for emotion in ISEAR_EMOTIONS
    }
    single = data.groupby("steer")[list(ISEAR_EMOTIONS)].mean()
    return baseline, single


def pair_counts(path: Path) -> dict | None:
    """Count target, attenuation, and joint successes in one 42-pair matrix."""
    if not path.is_file():
        return None
    data = pd.read_csv(path)
    if "steer" not in data or data.loc[data["steer"] == "baseline"].empty:
        return None
    baseline, single = _tables(data)
    target = attenuation = joint = total = 0
    thin: list[str] = []
    for spec in ALL_PAIRS:
        target_emotion, subtracted_emotion = spec.split("-")
        rows = data.loc[data["steer"] == spec]
        if len(rows) < MIN_ROWS or target_emotion not in single.index:
            thin.append(spec)
            continue
        total += 1
        target_ok = rows[target_emotion].mean() > baseline[target_emotion]
        attenuation_ok = (
            rows[subtracted_emotion].mean()
            < single.loc[target_emotion, subtracted_emotion]
        )
        target += int(target_ok)
        attenuation += int(attenuation_ok)
        joint += int(target_ok and attenuation_ok)
    return {
        "target": target,
        "attenuation": attenuation,
        "suppress": attenuation,
        "joint": joint,
        "n": total,
        "thin": thin,
    }


def separability_matrix(path: Path) -> pd.DataFrame:
    """Return the target/attenuation status for every ordered emotion pair."""
    data = pd.read_csv(path)
    baseline, single = _tables(data)
    rows = []
    for target_emotion in ISEAR_EMOTIONS:
        row = {"steer": target_emotion}
        for subtracted_emotion in ISEAR_EMOTIONS:
            if target_emotion == subtracted_emotion:
                row[subtracted_emotion] = ""
                continue
            subset = data.loc[data["steer"] == f"{target_emotion}-{subtracted_emotion}"]
            if len(subset) < MIN_ROWS or target_emotion not in single.index:
                row[subtracted_emotion] = "n/a"
                continue
            target_ok = subset[target_emotion].mean() > baseline[target_emotion]
            attenuation_ok = (
                subset[subtracted_emotion].mean()
                < single.loc[target_emotion, subtracted_emotion]
            )
            row[subtracted_emotion] = "yes" if target_ok and attenuation_ok else "no"
        rows.append(row)
    return pd.DataFrame(rows).set_index("steer")


def models_in(runs: Path) -> list[Path]:
    return sorted(path for path in runs.iterdir() if path.is_dir() and (path / GEN).is_file())


def _ratio(count: int, total: int) -> str:
    return f"{count}/{total}" if total else "n/a"


def summary(runs: Path) -> list[str]:
    """Render the article composition table from the checked-in matrices."""
    lines = [
        "# Ordered-difference composition",
        "",
        "For each ordered pair of distinct emotions `X` and `Y`, this analysis evaluates `v_X - v_Y`. Each model uses the layer and coefficient selected on the disjoint anger calibration set. The table covers 42 ordered pairs for each model.",
        "",
        "Target retention means that X rises above the unsteered baseline. Attenuation means that Y is lower than under X-only steering. Joint success requires both conditions for the same pair.",
        "",
        "| Model | Encoder target | Encoder attenuation | Encoder joint | Judge target | Judge attenuation | Judge joint |",
        "|---|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]
    totals = {key: 0 for key in ("et", "ea", "ej", "en", "jt", "ja", "jj", "jn")}
    coverage_notes: list[str] = []
    for run in models_in(runs):
        encoder = pair_counts(run / GEN)
        judge = pair_counts(run / JUDGE)
        if encoder is None:
            continue
        lines.append(
            f"| {run.name} | {_ratio(encoder['target'], encoder['n'])} | "
            f"{_ratio(encoder['attenuation'], encoder['n'])} | "
            f"{_ratio(encoder['joint'], encoder['n'])} | "
            f"{_ratio(judge['target'], judge['n']) if judge else 'n/a'} | "
            f"{_ratio(judge['attenuation'], judge['n']) if judge else 'n/a'} | "
            f"{_ratio(judge['joint'], judge['n']) if judge else 'n/a'} |"
        )
        totals["et"] += encoder["target"]
        totals["ea"] += encoder["attenuation"]
        totals["ej"] += encoder["joint"]
        totals["en"] += encoder["n"]
        if judge:
            totals["jt"] += judge["target"]
            totals["ja"] += judge["attenuation"]
            totals["jj"] += judge["joint"]
            totals["jn"] += judge["n"]
        if encoder["thin"]:
            coverage_notes.append(f"{run.name} encoder: {', '.join(encoder['thin'])}")
        if judge and judge["thin"]:
            coverage_notes.append(f"{run.name} judge: {', '.join(judge['thin'])}")

    lines.extend([
        f"| Total | {_ratio(totals['et'], totals['en'])} | {_ratio(totals['ea'], totals['en'])} | {_ratio(totals['ej'], totals['en'])} | {_ratio(totals['jt'], totals['jn'])} | {_ratio(totals['ja'], totals['jn'])} | {_ratio(totals['jj'], totals['jn'])} |",
        f"| Rate | {totals['et'] / totals['en']:.1%} | {totals['ea'] / totals['en']:.1%} | {totals['ej'] / totals['en']:.1%} | {totals['jt'] / totals['jn']:.1%} | {totals['ja'] / totals['jn']:.1%} | {totals['jj'] / totals['jn']:.1%} |",
        "",
        "The raw matrices are `runs/<slug>/compose_allpairs.csv` and `runs/<slug>/compose_allpairs_judge_wide.csv`.",
    ])
    if coverage_notes:
        lines.extend(["", "## Incomplete conditions", ""])
        lines.extend(f"- {note}" for note in coverage_notes)
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize ordered-difference composition from stored run matrices."
    )
    parser.add_argument("--runs", type=Path, default=REPO / "runs")
    parser.add_argument("--matrix", help="model slug for a pair-status matrix")
    parser.add_argument("--out", type=Path, help="write the Markdown summary")
    args = parser.parse_args()

    if args.matrix:
        path = args.runs / args.matrix / GEN
        if not path.is_file():
            raise SystemExit(f"Missing {path}")
        print(separability_matrix(path).to_string())
        return

    text = "\n".join(summary(args.runs)) + "\n"
    print(text, end="")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
