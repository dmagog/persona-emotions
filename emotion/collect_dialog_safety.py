"""Build the dialogue evaluation summary from stored generation and judge files."""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

csv.field_size_limit(2**31 - 1)

REPO = Path(__file__).resolve().parent.parent
CORE = ("baseline", "-anger", "-fear", "-anger-fear", "+anger")
# Sweep files are excluded on purpose: they repeat `-anger` at other coefficients
# and would overwrite the operating-point rows keyed by (condition, dialog_id).
SOURCES = ("dialog_safety", "dialog_other", "dialog_rest", "dialog_rand")


def rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def value(row: dict | None, field: str) -> float | None:
    try:
        return float(row[field]) if row and row.get(field, "") != "" else None
    except (KeyError, ValueError):
        return None


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def percentile(sorted_values: list[float], fraction: float) -> float:
    if not sorted_values:
        raise ValueError("Cannot take a percentile of an empty sequence.")
    index = (len(sorted_values) - 1) * fraction
    lower, upper = int(index), min(int(index) + 1, len(sorted_values) - 1)
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * (index - lower)


def bootstrap_ci(values: list[float], repetitions: int = 10_000, seed: int = 0) -> tuple[float, float] | None:
    if not values:
        return None
    generator = random.Random(seed)
    n = len(values)
    samples = sorted(sum(values[generator.randrange(n)] for _ in range(n)) / n for _ in range(repetitions))
    return percentile(samples, 0.025), percentile(samples, 0.975)


def load_run(run: Path) -> dict[str, dict[str, dict]]:
    """Join local scores and primary-judge scores by condition and dialogue ID."""
    generated: dict[tuple[str, str], dict] = {}
    judged: dict[tuple[str, str], dict] = {}
    for stem in SOURCES:
        generated.update({(r["condition"], r["dialog_id"]): r for r in rows(run / f"{stem}.csv")})
        judged.update({(r["condition"], r["dialog_id"]): r for r in rows(run / f"{stem}_judge.csv")})
    data: dict[str, dict[str, dict]] = defaultdict(dict)
    for key, generated_row in generated.items():
        data[key[0]][key[1]] = {"generated": generated_row, "judged": judged.get(key)}
    return dict(data)


def condition_summary(data: dict[str, dict[str, dict]], condition: str) -> dict:
    baseline = data.get("baseline", {})
    current = data.get(condition, {})
    out: dict[str, float | int | tuple[float, float] | None] = {"n": len(current)}
    for metric in ("escalation", "helpfulness", "empathy"):
        scores = [value(pair["judged"], metric) for pair in current.values()]
        scores = [score for score in scores if score is not None]
        deltas = []
        if condition != "baseline":
            for dialog_id, pair in current.items():
                score = value(pair["judged"], metric)
                reference = value(baseline.get(dialog_id, {}).get("judged"), metric)
                if score is not None and reference is not None:
                    deltas.append(score - reference)
        out[metric] = mean(scores)
        out[f"delta_{metric}"] = mean(deltas)
        out[f"ci_{metric}"] = bootstrap_ci(deltas)
    return out


def fmt(number: float | None, digits: int = 1) -> str:
    return "n/a" if number is None else f"{number:.{digits}f}"


def fmt_delta(number: float | None) -> str:
    return "n/a" if number is None else f"{number:+.1f}"


def fmt_ci(interval: tuple[float, float] | None) -> str:
    return "n/a" if interval is None else f"[{interval[0]:+.1f}, {interval[1]:+.1f}]"


def render_model(name: str, data: dict[str, dict[str, dict]]) -> list[str]:
    lines = [f"## {name}", "", "| Condition | n | Escalation | Delta [95% CI] | Helpfulness | Empathy |", "|---|---:|---:|:---:|---:|---:|"]
    for condition in CORE:
        if condition not in data:
            continue
        result = condition_summary(data, condition)
        lines.append(
            f"| `{condition}` | {result['n']} | {fmt(result['escalation'])} | "
            f"{fmt_delta(result['delta_escalation'])} {fmt_ci(result['ci_escalation'])} | "
            f"{fmt(result['helpfulness'])} | {fmt(result['empathy'])} |"
        )
    random_conditions = sorted(condition for condition in data if condition.startswith("random"))
    if random_conditions:
        random_deltas = [condition_summary(data, condition)["delta_escalation"] for condition in random_conditions]
        lines.append(
            f"| random-direction mean | {sum(condition_summary(data, c)['n'] for c in random_conditions)} | n/a | "
            f"{fmt_delta(mean([d for d in random_deltas if d is not None]))} n/a | n/a | n/a |"
        )
    negative_conditions = sorted(condition for condition in data if condition.startswith("-") and condition.count("-") == 1)
    if negative_conditions:
        effects = [condition_summary(data, condition)["delta_escalation"] for condition in negative_conditions]
        lines.extend([
            "",
            f"The mean escalation change across the available negative single-emotion controls is {fmt(mean([effect for effect in effects if effect is not None]))}.",
        ])
    return lines


def summary(runs: Path) -> str:
    targets = [("Falcon-3-3B", runs / "Falcon3-3B-Instruct"), ("Qwen-2.5-1.5B", runs / "Qwen2.5-1.5B-Instruct")]
    lines = [
        "# Dialogue evaluation",
        "",
        "This analysis tests whether an intervention that changes expressed emotion also improves conflict handling. It uses 30 English dialogue contexts for Falcon-3-3B and Qwen-2.5-1.5B. The primary judge scores escalation, helpfulness, and empathy from 0 to 100. Lower escalation is better.",
        "",
        "Each delta compares replies to the unsteered reply for the same dialogue. Intervals are percentile bootstrap 95% intervals from 10,000 paired resamples with seed 0.",
        "",
    ]
    for name, run in targets:
        if run.is_dir():
            lines.extend(render_model(name, load_run(run)))
            lines.append("")
    lines.extend([
        "The source dialogues are in `data_generation/deescalation_dialogs.json`. Generation files, judge scores, and judge caches are in the two corresponding `runs/` directories.",
        "",
        "Run this collector after a dialogue evaluation to regenerate this document from the stored artifacts.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize the dialogue evaluation from stored run files.")
    parser.add_argument("--runs", type=Path, default=REPO / "runs")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    text = summary(args.runs)
    print(text, end="")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
