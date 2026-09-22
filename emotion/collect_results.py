"""Summary table over every run in runs/, one row per model.

Reads runs/<slug>/steer_specificity.csv, which holds the independent encoder
scores, together with meta.json, and computes the diagonal, argmax hits, leakage
into sadness and text degeneration. Markdown goes to stdout.

Usage:
    python -m emotion.collect_results
    python -m emotion.collect_results --csv runs/summary.csv
"""
from __future__ import annotations

import argparse
import csv as csvmod
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

from emotion.protocol import compare_planes, plane_of, report
from emotion.space import ISEAR_EMOTIONS

REPO = Path(__file__).resolve().parent.parent
REFUSAL = re.compile(
    r"as an AI|as a language model|I am an AI|language model|"
    r"I (?:don't|do not|cannot|can't) have (?:personal )?(?:feelings|emotions)|"
    r"I'm just an? (?:AI|computer program)|I can only (?:help|assist)|"
    r"cannot feel|unable to (?:feel|experience)", re.I)


def rep_ratio(text: str, n: int = 4) -> float:
    """Share of repeated n-grams, stable across text lengths."""
    w = re.findall(r"\w+", str(text).lower())
    if len(w) < n + 4:
        return 0.0
    grams = [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]
    return 1.0 - len(set(grams)) / len(grams)


def ttr(text: str) -> float:
    w = re.findall(r"\w+", str(text).lower())
    return len(set(w)) / len(w) if w else 1.0


def variants_in(run_dir: Path) -> list[Path]:
    """Every matrix of a run: raw, sae, centered. Each becomes its own summary row."""
    found = sorted(run_dir.glob("steer_specificity_*.csv"))
    found = [f for f in found if not f.name.endswith((".partial.csv", "_fixedcoeff.csv", "_S033.csv"))]
    legacy = run_dir / "steer_specificity.csv"
    if legacy.is_file():
        found.insert(0, legacy)
    return found


def row_for(run_dir: Path, csv_path: Path | None = None) -> dict | None:
    csv_path = csv_path or (run_dir / "steer_specificity.csv")
    if not csv_path.is_file():
        return None
    d = pd.read_csv(csv_path)
    meta = {}
    meta_path = run_dir / "meta.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

    base = d[d["steer"] == "baseline"]
    if base.empty:
        return None
    base_mean = {e: base[e].mean() for e in ISEAR_EMOTIONS}

    diag, hits, leak = [], 0, []
    for emo in ISEAR_EMOTIONS:
        sub = d[d["steer"] == emo]
        if sub.empty:
            continue
        deltas = {e: sub[e].mean() - base_mean[e] for e in ISEAR_EMOTIONS}
        diag.append(deltas[emo])
        if max(deltas, key=deltas.get) == emo:
            hits += 1
        if emo != "sadness":
            leak.append(deltas["sadness"])

    ans = d["answer"].astype(str) if "answer" in d.columns else pd.Series(dtype=str)
    degen = int(((ans.map(rep_ratio) > 0.15) | (ans.map(ttr) < 0.45)).sum()) if len(ans) else -1
    refus = int(ans.str.contains(REFUSAL, regex=True).sum()) if len(ans) else -1

    # The coefficient actually applied. This used to read meta["coeff"], which is
    # the default of 8.0 rather than what operating-point selection chose.
    coeff = meta.get("op_coeff")
    if coeff is None and "coeff" in d.columns:
        used = sorted({float(c) for c in d[d["steer"] != "baseline"]["coeff"] if str(c).strip()})
        coeff = used[0] if len(used) == 1 else (f"{min(used):g}–{max(used):g}" if used else None)
    if coeff is None:
        coeff = meta.get("coeff", "?")

    m = re.match(r"steer_specificity_(.+)\.csv$", csv_path.name)
    variant = m.group(1) if m else meta.get("variant", "raw")

    plane = plane_of(run_dir, csv_path)
    row = {
        "model": meta.get("model", run_dir.name),
        "slug": run_dir.name,
        "variant": variant,
        "layer": meta.get("layer", "?"),
        "coeff": coeff,
        "judge_filtered": meta.get("judge_filtered", False),
        "n_rows": len(d),
        "diag_mean": sum(diag) / len(diag) if diag else float("nan"),
        "argmax_hits": hits,
        "sad_leak": sum(leak) / len(leak) if leak else float("nan"),
        "degen": degen,
        "refusals": refus,
        **{f"plane_{k}": v for k, v in plane.items()},
    }
    row.update(_judge_and_ci(run_dir, base_mean))
    return row


# A condition measured on a handful of answers says nothing. For granite the
# judge lost shame entirely and reduced sadness to two answers, and the 4/7 in
# the table counted shame as a miss over zero observations. Below this floor a
# condition counts neither as a hit nor in the denominator.
MIN_JUDGED = 20


def _argmax_hits_from_wide(path: Path) -> tuple[int, int, dict] | None:
    """Judge argmax hits: (hits, conditions measured, answers per condition).

    The denominator is returned as well. The judge loses some calls to answer
    parsing and provider errors, so dividing by seven in silence would report an
    under-measurement as a miss.
    """
    if not path.is_file():
        return None
    try:
        w = pd.read_csv(path)
    except Exception:
        return None
    if "steer" not in w.columns or not set(ISEAR_EMOTIONS) <= set(w.columns):
        return None
    base = w[w["steer"] == "baseline"]
    if base.empty:
        return None
    bm = {e: base[e].mean() for e in ISEAR_EMOTIONS}
    hits, measured, sizes = 0, 0, {}
    for emo in ISEAR_EMOTIONS:
        sub = w[w["steer"] == emo]
        sizes[emo] = len(sub)
        if len(sub) < MIN_JUDGED:
            continue
        measured += 1
        deltas = {e: sub[e].mean() - bm[e] for e in ISEAR_EMOTIONS}
        if max(deltas, key=deltas.get) == emo:
            hits += 1
    return hits, measured, sizes


def _judge_and_ci(run_dir: Path, base_mean: dict) -> dict:
    """Numbers from the evaluation chain, when it has been run."""
    out: dict = {"judge_hits": None, "judge_measured": None, "judge_thin": "",
                 "judge_coverage": None, "coherence": None, "ci_sig": None}
    jh = _argmax_hits_from_wide(run_dir / "judge_wide.csv")
    if jh is not None:
        hits, measured, sizes = jh
        out["judge_hits"] = hits
        out["judge_measured"] = measured
        thin = {e: n for e, n in sizes.items() if n < MIN_JUDGED}
        out["judge_thin"] = ", ".join(f"{e}={n}" for e, n in sorted(thin.items())) or ""
        # Coverage: an even 10% loss touches no single condition enough to show
        # up on its own, and without this line it would stay invisible.
        want = max(sizes.values(), default=0) * 8  # 7 emotions plus baseline
        got = len(pd.read_csv(run_dir / "judge_wide.csv"))
        out["judge_coverage"] = round(got / want, 3) if want else None

    coh = run_dir / "coherence.md"
    if coh.is_file():
        vals = []
        for line in coh.read_text(encoding="utf-8").splitlines():
            parts = [c.strip() for c in line.split("|")]
            if len(parts) >= 4 and parts[1] not in ("", "steer", "---") and parts[1] != "baseline":
                try:
                    vals.append(float(parts[2]))
                except ValueError:
                    continue
        if vals:
            out["coherence"] = sum(vals) / len(vals)

    ci = run_dir / "ci_encoder.md"
    if ci.is_file():
        txt = ci.read_text(encoding="utf-8")
        # bootstrap_ci writes this table, so the markers have to match it.
        yes = txt.count("| yes |")
        total = yes + txt.count("| no |")
        if total:
            out["ci_sig"] = f"{yes}/{total}"
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Summary over the runs in runs/.")
    ap.add_argument("--runs", type=Path, default=REPO / "runs")
    ap.add_argument("--csv", type=Path, default=None, help="also write the table to a CSV file")
    args = ap.parse_args()

    if not args.runs.is_dir():
        raise SystemExit(f"no such directory: {args.runs}")

    rows = []
    for run_dir in sorted(args.runs.iterdir()):
        if not run_dir.is_dir():
            continue
        for csv_path in variants_in(run_dir):
            r = row_for(run_dir, csv_path)
            if r:
                rows.append(r)
    if not rows:
        raise SystemExit(f"no finished steer_specificity.csv under {args.runs}")

    # Rows are comparable only when they share a protocol. This used to be a note
    # under the table saying "check meta.json", which is a check nobody performs.
    planes = {f"{r['slug']}/{r['variant']}": {k[len("plane_"):]: v
                                              for k, v in r.items() if k.startswith("plane_")}
              for r in rows}
    off = compare_planes(planes)
    # Flag the minority: a row stands out when, for at least one key, its value
    # is rarer than the majority. With no majority, meaning an even split, flag
    # every row, since picking the "right" half by coin toss is not an option.
    odd: set[str] = set()
    for seen in off.values():
        sizes = sorted((len(v) for v in seen.values()), reverse=True)
        if len(sizes) > 1 and sizes[0] == sizes[1]:
            odd |= {n for names in seen.values() for n in names}
        else:
            major = max(seen.values(), key=len)
            odd |= {n for names in seen.values() if names is not major for n in names}

    multi = len({r["variant"] for r in rows}) > 1
    vcol = " Variant |" if multi else ""
    vsep = "---|" if multi else ""
    print(f"| Model |{vcol} Layer | coeff | Diagonal | argmax enc. | argmax judge | Significant | "
          "Leakage | Coherence | Degenerate | Refusals | Plane |")
    print(f"|---|{vsep}---:|---:|---:|---:|---:|:--:|---:|---:|---:|---:|:--:|")
    for r in rows:
        key = f"{r['slug']}/{r['variant']}"
        mark = "⚠" if key in odd else ("?" if not r.get("plane_stamped") else "✓")
        jm = r.get("judge_measured")
        jh = (f"{r['judge_hits']}/{jm}" + ("⚠" if jm not in (None, 7) else "")) \
            if r.get("judge_hits") is not None else "—"
        coh = f"{r['coherence']:.1f}" if r.get("coherence") is not None else "—"
        sig = r.get("ci_sig") or "—"
        cf = r["coeff"] if isinstance(r["coeff"], str) else (
            f"{r['coeff']:g}" if isinstance(r["coeff"], (int, float)) else "?")
        vv = f" {r['variant']} |" if multi else ""
        print(f"| {r['model']} |{vv} {r['layer']} | {cf} | {r['diag_mean']:+.3f} | "
              f"{r['argmax_hits']}/7 | {jh} | {sig} | {r['sad_leak']:+.3f} | {coh} | "
              f"{r['degen']}/{r['n_rows']} | {r['refusals']}/{r['n_rows']} | {mark} |")

    print("\nDiagonal and leakage come from the independent `go_emotions` encoder, "
          "as a change against unsteered text.")
    print("Significant counts how many of the 7 diagonals have an interval excluding zero, "
          "under the encoder.")
    print("Coherence is the mean over steered conditions, judged 0 to 100; a dash means "
          "the evaluation was not run.")
    print("Degenerate means 4-gram repetition above 0.15 or a type-token ratio below "
          "0.45. The metric is not validated against annotation and also catches "
          "emotional repetition.")
    print("Plane: a check mark means the current protocol, a warning sign means the "
          "row stands out, and a question mark means no stamp, so the protocol is "
          "known only from the manifest.")
    lean = [(r["slug"], r["judge_coverage"]) for r in rows
            if r.get("judge_coverage") is not None and r["judge_coverage"] < 0.95]
    if lean:
        print("\nThe judge did not answer every call, staying below 95% of the matrix, "
              "so these rows are computed on a subsample. Top them up with "
              "`python3 -m emotion.judge_specificity` and the same --cache:")
        for slug, cov in lean:
            print(f"- {slug}: {cov:.0%} of the matrix")
    thin_rows = [(r["slug"], r["judge_thin"]) for r in rows if r.get("judge_thin")]
    if thin_rows:
        print(f"\nThe judge argmax column reads hits over conditions measured. The judge "
              f"loses calls to answer parsing and provider errors, and conditions left "
              f"with fewer than {MIN_JUDGED} answers out of 56 are excluded:")
        for slug, thin in thin_rows:
            print(f"- {slug}: {thin}")
    print("\n## Comparability\n")
    print("\n".join(report(planes)))

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as fh:
            w = csvmod.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\nCSV: {args.csv}", file=sys.stderr)


if __name__ == "__main__":
    main()
