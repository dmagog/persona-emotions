"""Protocol helpers that check whether experiment artifacts are comparable.

The module records the configuration choices associated with each result and
detects runs that do not share the same measurement plane.

Usage:
    python -m emotion.protocol
    python -m emotion.protocol --check runs
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from emotion import stamp
from emotion.space import ISEAR_EMOTIONS

REPO = Path(__file__).resolve().parent.parent


@dataclass
class Choice:
    """One protocol decision: common practice, what this study does, and why."""
    name: str
    reference: str
    ours: str
    where: str
    note: str = ""
    deviation: bool = False


PROTOCOL: list[Choice] = [
    Choice(
        "Activation pooling",
        "response average over answer tokens",
        "response average",
        "extraction",
        "Prompt-last, prompt-average and response-average pooling were compared; "
        "the third was selected."),
    Choice(
        "Steering formula",
        "h += alpha * v, with the raw direction",
        "h += coeff * v, with the raw direction",
        "intervention",
        "The direction is not normalized before it is added. See Normalization below."),
    Choice(
        "Direction normalization",
        "only where different directions are compared against one another",
        "no normalization when steering; with --center the vector is rescaled "
        "back to its original length",
        "intervention",
        "The rule is: equalize norms when comparing directions, equalize effect "
        "when comparing models."),
    Choice(
        "Steered positions",
        "every decoding step",
        'positions="all"',
        "intervention",
        "Steering only the last position changes which tokens carry the "
        "perturbation and is not equivalent."),
    Choice(
        "Layer selection",
        "steer at every layer with one coefficient, keep the layer with the "
        "highest expression score",
        "three candidates at {0.35, 0.45, 0.55} of model depth, a coefficient "
        "grid of {0, 2, 4, 6, 8, 16} with 16 prompts per cell, keeping the "
        "strongest point whose degenerate-output share stays at or below 10%",
        "calibration",
        "A full layer sweep does not fit in one night on an RTX 2070. The "
        "degeneration constraint was added because a naive score maximum "
        "selected a layer with 38% unusable output: the encoder reads a "
        "degenerate repetition as strong emotion.",
        deviation=True),
    Choice(
        "Layer numbering",
        "one-based, where layer 20 is the output of the 20th block",
        "zero-based; for block B the code takes vec[B+1] in hidden_states",
        "calibration",
        "Layer 10 here is layer 11 under one-based numbering. Shift the index "
        "when moving numbers into prose, or the off-by-one returns.",
        deviation=True),
    Choice(
        "Extraction-pair filter",
        "per-response judging with a score threshold on each side of the pair",
        "a pairwise judge is implemented, scoring how much more emotional A is "
        "than B with a threshold of 60 on llama-3.3-70b, but the published grid "
        "does not use it: judge_filtered is false in all 11 rows, as a "
        "deliberate ablation",
        "pair generation",
        "A pairwise comparison is cleaner than two independent scores because "
        "both sides describe the same scenario. Logprob aggregation is "
        "unavailable, since the provider does not return them.",
        deviation=True),
    Choice(
        "Per-model dataset",
        "the target model writes its own pairs",
        "the same: one self-generation cycle per model",
        "pair generation",
        "Vectors extracted from another model's text still work, but they carry "
        "a different norm and select a different layer. The earlier variant is "
        "kept as a separate transfer-ablation row."),
    Choice(
        "Decoding",
        "not specified",
        "greedy, temperature 0 at every stage",
        "all stages",
        "This goes against the Qwen recommendation to sample. It is kept for "
        "reproducibility: the direction is a difference of means, and sampling "
        "adds noise to it. The degeneration column is the control.",
        deviation=True),
    Choice(
        "Effect measurement",
        "an LLM judge scoring trait expression",
        "an independent encoder, SamLowe/roberta-base-go_emotions, plus a panel "
        "of three LLM judges from different vendors (llama-3.3-70b, "
        "gemini-3.5-flash-lite, gpt-4.1-mini)",
        "evaluation",
        "Two instruments rather than one, because they disagree: 4 of 7 under "
        "the encoder against 6 of 7 under the judge on Qwen3-1.7B, so the report "
        "carries both columns. The judge column is checked against the panel, "
        "with Pearson agreement of 0.77 to 0.91 across 11 models. The judge does "
        "not answer every call, so the summary checks matrix coverage and the "
        "runbook describes how to top it up."),
]


# --- comparability plane -----------------------------------------------------

# Fields that must agree before two rows may sit in the same table. Layer and
# coefficient are deliberately absent: they differ per model by construction,
# which is the whole point of selecting an operating point.
PLANE_KEYS: dict[str, str] = {
    "protocol": "protocol version",
    "n_prompts": "prompts per condition",
    "drive": "how the intervention is driven",
    "judge_filtered": "judge filter on pairs",
    "max_degen": "degeneration ceiling during selection",
    "dtype": "compute precision",
}


def dtype_of(run_dir: Path, meta: dict) -> str | None:
    """Report the precision a run used.

    The manifest did not always record it, so older runs are read back from the
    loader log. The distinction matters: fp16 and bf16 differ in the low bits of
    the activations, and on models trained in bf16 fp16 also overflows.
    """
    if meta.get("dtype"):
        return str(meta["dtype"]).replace("torch.", "")
    log = run_dir / "chain.log"
    if not log.is_file():
        return None
    # A log written by a Windows console is cp1251, but the loader line is ASCII.
    found = re.findall(r"dtype=torch\.(\w+)",
                       log.read_text(encoding="utf-8", errors="ignore"))
    return found[-1] if found else None


def plane_of(run_dir: Path, csv_path: Path) -> dict:
    """Describe the plane a row was measured in.

    Where possible this reads the data itself rather than the manifest. The
    prompt count and the drive mode come from the matrix, so they cannot drift
    apart from it.
    """
    meta = {}
    meta_path = run_dir / "meta.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    st = stamp.read_stamp(csv_path) or {}

    d = pd.read_csv(csv_path)
    steered = d[d["steer"].isin(ISEAR_EMOTIONS)] if "steer" in d.columns else d
    n_prompts = int(steered.groupby("steer")["prompt_id"].nunique().max()) \
        if not steered.empty and "prompt_id" in d.columns else None

    # The plane records the drive MODE, not the coefficient value. The value is
    # model specific by construction, since the operating point is chosen by
    # behavior. Driving one model by coefficient and another by a dimensionless
    # strength, however, is a different protocol.
    drive = "?"
    if "coeff" in d.columns and not steered.empty:
        used = {round(float(c), 3) for c in steered["coeff"] if str(c).strip()}
        drive = "coeff" if len(used) == 1 else "strength"

    return {
        "protocol": st.get("protocol", meta.get("protocol")),
        "n_prompts": n_prompts,
        "drive": drive,
        "judge_filtered": meta.get("judge_filtered"),
        "max_degen": meta.get("max_degen"),
        "dtype": dtype_of(run_dir, meta),
        "stamped": bool(st),
        "op_at_edge": at_grid_edge(run_dir, meta),
        # Not a plane key, or any documentation commit would split the table.
        # A spread of code versions is still worth a look at what changed.
        "code": (meta.get("env") or {}).get("git_sha"),
    }


def at_grid_edge(run_dir: Path, meta: dict) -> str | None:
    """Report whether the operating point landed on an edge of the coefficient grid.

    At the top of the grid the optimum may lie beyond it, so the grid needs
    widening; otherwise "the strongest non-degenerate intervention" only means
    "the strongest one tried". At the bottom the degeneration limit is binding
    instead, and the model is fragile.
    """
    sweep = run_dir / "layer_sweep_anger.csv"
    op = meta.get("op_coeff")
    if not sweep.is_file() or op is None:
        return None
    try:
        grid = sorted({float(c) for c in pd.read_csv(sweep)["coeff"] if float(c) > 0})
    except (KeyError, ValueError, TypeError):
        return None
    if not grid:
        return None
    if abs(float(op) - grid[-1]) < 1e-9:
        return f"top of the grid ({grid[-1]:g}): the optimum may lie beyond it"
    if abs(float(op) - grid[0]) < 1e-9 and len(grid) > 1:
        return f"bottom of the grid ({grid[0]:g}): anything stronger broke the text"
    return None


def compare_planes(planes: dict[str, dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for key in PLANE_KEYS:
        seen: dict = {}
        for name, p in planes.items():
            seen.setdefault(p.get(key), []).append(name)
        if len(seen) > 1:
            out[key] = seen
    return out


def report(planes: dict[str, dict]) -> list[str]:
    """Build the comparability text that goes under the summary table as is."""
    lines: list[str] = []
    unstamped = [n for n, p in planes.items() if not p.get("stamped")]
    edges = {n: p["op_at_edge"] for n, p in planes.items() if p.get("op_at_edge")}
    diff = compare_planes(planes)

    if not planes:
        return ["No rows to compare."]
    if not diff and not unstamped and not edges:
        lines.append(f"All {len(planes)} rows share one plane and are comparable.")
        return lines
    if not diff:
        lines.append(f"All {len(planes)} rows share one plane. Caveats follow.")
        lines.append("")

    if diff:
        lines.append("**These rows were measured in different planes and cannot sit side by side:**")
        lines.append("")
        for key, seen in diff.items():
            variants = "; ".join(
                f"{v if v is not None else 'unknown'}: {', '.join(sorted(names))}"
                for v, names in sorted(seen.items(), key=lambda kv: str(kv[0])))
            lines.append(f"- {PLANE_KEYS[key]}: {variants}")
        lines.append("")
    codes: dict[str, list[str]] = {}
    for name, p in planes.items():
        codes.setdefault(p.get("code") or "unknown", []).append(name)
    if len(codes) > 1:
        lines.append("**These rows were produced by different code versions:**")
        lines.append("")
        for sha, names in sorted(codes.items()):
            lines.append(f"- `{sha}`: {', '.join(sorted(names))}")
        lines.append("")
        shas = [s for s in sorted(codes) if s != "unknown"]
        if len(shas) >= 2:
            lines.append(f"Check `git log --oneline {shas[0]}..{shas[-1]}` for changes that "
                         "affect the numbers rather than the tooling around them.")
            lines.append("")
    if edges:
        lines.append("**The operating point landed on an edge of the coefficient grid:**")
        lines.append("")
        for name, why in sorted(edges.items()):
            lines.append(f"- {name}: {why}")
        lines.append("")
    if unstamped:
        lines.append(
            f"**No protocol stamp:** {', '.join(sorted(unstamped))}. "
            "How they were produced is known only from the manifest. Confirm with "
            "`python -m emotion.stamp runs/<slug> --adopt --config configs/models/<slug>.yaml`")
        lines.append("")
    return lines


def card() -> str:
    """Render the protocol decisions as a table with the reasoning underneath."""
    out = ["# Protocol decisions\n",
           "| Decision | Common practice | This study | Stage |",
           "|---|---|---|---|"]
    for c in PROTOCOL:
        mark = " (*)" if c.deviation else ""
        out.append(f"| {c.name}{mark} | {c.reference} | {c.ours} | {c.where} |")
    out.append("\n(*) marks a deliberate departure from common practice.\n")
    for c in PROTOCOL:
        if c.note:
            out.append(f"**{c.name}.** {c.note}\n")
    return "\n".join(out)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(
        description="Protocol decisions and the comparability check.")
    ap.add_argument("--check", type=Path, default=None,
                    help="a runs/ directory: check whether its rows share one plane")
    args = ap.parse_args()

    if args.check:
        from emotion.collect_results import variants_in
        planes = {}
        for run_dir in sorted(args.check.iterdir()):
            if not run_dir.is_dir():
                continue
            for csv_path in variants_in(run_dir):
                # A name without a variant comes from the earlier layout, where
                # the variant was raw.
                m = re.match(r"steer_specificity_(.+)\.csv$", csv_path.name)
                planes[f"{run_dir.name}/{m.group(1) if m else 'raw'}"] = \
                    plane_of(run_dir, csv_path)
        print("\n".join(report(planes)))
        return

    print(card())


if __name__ == "__main__":
    main()
