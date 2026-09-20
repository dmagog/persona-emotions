"""Канонический разбор композиции: сложение/вычитание эмоций по всем 42 парам.

Единственный источник чисел композиции для отчёта и статьи. Читает ТОЛЬКО
`compose_allpairs.csv` (+ судейский `compose_allpairs_judge_wide.csv`) — не
старые 4-специевые `compose.csv`: те снимались в разных прогонах и у части
моделей не воспроизводятся новой однородной матрицей (см. аудит 2026-08-29).

Две метрики на пару X-Y, обе честные и разные по смыслу:

* цель↑     — X(X-Y) > X(baseline): вычитание не гасит целевую эмоцию;
* подавл.↓  — Y(X-Y) < Y(X-в-одиночку): -Y реально убирает протечку, которую
              наведение X создаёт само. Это сильнее, чем «Y ниже нейтрального»:
              baseline не учитывает, что X сам поднимает родственную Y.

Usage:
    python -m emotion.collect_compose                 # сводка по моделям
    python -m emotion.collect_compose --matrix <slug> # матрица 7x7 разделимости
    python -m emotion.collect_compose --out docs/COMPOSITION.md
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from emotion.space import ALL_PAIRS, ISEAR_EMOTIONS

REPO = Path(__file__).resolve().parent.parent
GEN = "compose_allpairs.csv"
JUDGE = "compose_allpairs_judge_wide.csv"
MIN_ROWS = 40  # условие тоньше — не считаем, как в матрице специфичности


def _tables(d: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """baseline по эмоциям и таблица «средний балл эмоции под каждым наведением».

    `single.loc[x, y]` — средний балл Y, когда наводится ОДИН X. Это опорная
    точка подавления: сравнивать Y при X−Y надо именно с этим, а не с «Y под
    наведением Y» (там Y максимален — тогда почти любая пара «проходит»
    тривиально) и не с нейтральным baseline (тот не учитывает, что X сам
    подтягивает родственную Y).
    """
    base = {e: d[d["steer"] == "baseline"][e].mean() for e in ISEAR_EMOTIONS}
    single = d.groupby("steer")[list(ISEAR_EMOTIONS)].mean()
    return base, single


def pair_counts(csv: Path) -> dict | None:
    """Счёт цель↑ и подавл.↓ по 42 парам одной таблицы (энкодер или судья)."""
    if not csv.is_file():
        return None
    d = pd.read_csv(csv)
    if "steer" not in d.columns or d[d["steer"] == "baseline"].empty:
        return None
    base, single = _tables(d)
    target = suppress = n = 0
    thin = []
    for spec in ALL_PAIRS:
        x, y = spec.split("-")
        sub = d[d["steer"] == spec]
        if len(sub) < MIN_ROWS or x not in single.index:
            thin.append(spec)
            continue
        n += 1
        if sub[x].mean() > base[x]:
            target += 1
        if sub[y].mean() < single.loc[x, y]:  # Y при X−Y ниже Y под наведением одного X
            suppress += 1
    return {"target": target, "suppress": suppress, "n": n, "thin": thin}


def separability_matrix(csv: Path) -> pd.DataFrame:
    """7x7: для каждой пары X-Y знак (цель растёт и вычитаемая давится)."""
    d = pd.read_csv(csv)
    base, single = _tables(d)
    rows = []
    for x in ISEAR_EMOTIONS:
        row = {"steer": x}
        for y in ISEAR_EMOTIONS:
            if x == y:
                row[y] = "·"
                continue
            sub = d[d["steer"] == f"{x}-{y}"]
            if sub.empty or x not in single.index:
                row[y] = "—"
                continue
            up = sub[x].mean() > base[x]
            down = sub[y].mean() < single.loc[x, y]
            row[y] = "✓" if (up and down) else ("↑" if up else ("↓" if down else "✗"))
        rows.append(row)
    return pd.DataFrame(rows).set_index("steer")


def models_in(runs: Path) -> list[Path]:
    return sorted(p for p in runs.iterdir()
                  if p.is_dir() and (p / GEN).is_file())


def summary(runs: Path) -> list[str]:
    out = ["# Ordered-difference composition\n",
           "This summary reads `compose_allpairs.csv` for the local encoder and "
           "`compose_allpairs_judge_wide.csv` for the LLM judge. Each model has 42 ordered `X - Y` pairs. "
           "Target retention means that `X - Y` raises `X` above baseline. Attenuation means that `X - Y` lowers `Y` relative to `X` alone.\n",
           "| Model | Encoder target | Encoder attenuation | Judge target | Judge attenuation |",
           "|---|---:|---:|---:|---:|"]
    agg = {"et": 0, "es": 0, "en": 0, "jt": 0, "js": 0, "jn": 0}
    warns = []
    for run in models_in(runs):
        e = pair_counts(run / GEN)
        j = pair_counts(run / JUDGE)
        ec = f"{e['target']}/{e['n']}" if e else "n/a"
        es = f"{e['suppress']}/{e['n']}" if e else "n/a"
        jc = f"{j['target']}/{j['n']}" if j else "n/a"
        js = f"{j['suppress']}/{j['n']}" if j else "n/a"
        out.append(f"| {run.name} | {ec} | {es} | {jc} | {js} |")
        if e:
            agg["et"] += e["target"]; agg["es"] += e["suppress"]; agg["en"] += e["n"]
            if e["thin"]:
                warns.append(f"{run.name} (encoder): fewer than {MIN_ROWS} rows for {e['thin']}")
        if j:
            agg["jt"] += j["target"]; agg["js"] += j["suppress"]; agg["jn"] += j["n"]
    if agg["en"]:
        out.append(f"| **Total** | **{agg['et']}/{agg['en']} "
                   f"({agg['et']/agg['en']:.0%})** | "
                   f"**{agg['es']}/{agg['en']} ({agg['es']/agg['en']:.0%})** | "
                   f"**{agg['jt']}/{agg['jn']} ({agg['jt']/max(agg['jn'],1):.0%})** | "
                   f"**{agg['js']}/{agg['jn']} ({agg['js']/max(agg['jn'],1):.0%})** |")
    out.append("\nThe target and attenuation counts use different reference conditions. The attenuation comparison uses `X` alone, not the unsteered baseline, because target steering can raise related emotions.")
    if warns:
        out.append("\n## Coverage notes")
        out += [f"- {w}" for w in warns]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize ordered-difference composition from per-model matrices.")
    ap.add_argument("--runs", type=Path, default=REPO / "runs")
    ap.add_argument("--matrix", default=None, help="model slug for the 7x7 separability matrix")
    ap.add_argument("--out", type=Path, default=None, help="write the summary as Markdown")
    args = ap.parse_args()

    if args.matrix:
        csv = args.runs / args.matrix / GEN
        if not csv.is_file():
            raise SystemExit(f"missing {csv}")
        m = separability_matrix(csv)
        print(f"Pair separability for {args.matrix} (encoder):")
        print("✓ target rises and subtracted component falls; ↑ target only; ↓ attenuation only; ✗ neither\n")
        print(m.to_string())
        return

    lines = summary(args.runs)
    text = "\n".join(lines)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
        print(f"\nwrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
