"""Сводная таблица по всем прогонам в runs/: одна строка на модель.

Читает runs/<slug>/steer_specificity.csv (баллы независимого энкодера) и meta.json,
считает диагональ, попадание argmax, протечку в грусть и вырожденность текста.
Markdown на stdout — вставляется в отчёт как есть.

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

from emotion.space import ISEAR_EMOTIONS

REPO = Path(__file__).resolve().parent.parent
RU = {"anger": "гнев", "disgust": "отвращение", "fear": "страх", "guilt": "вина",
      "joy": "радость", "sadness": "грусть", "shame": "стыд"}
REFUSAL = re.compile(
    r"as an AI|as a language model|I am an AI|language model|"
    r"I (don't|do not|cannot|can't) have (personal )?(feelings|emotions)|"
    r"I'm just an? (AI|computer program)|I can only (help|assist)|"
    r"cannot feel|unable to (feel|experience)", re.I)


def rep_ratio(text: str, n: int = 4) -> float:
    w = re.findall(r"\w+", str(text).lower())
    if len(w) < n + 2:
        return 0.0
    grams = [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]
    return max(Counter(grams).values()) / len(grams)


def ttr(text: str) -> float:
    w = re.findall(r"\w+", str(text).lower())
    return len(set(w)) / len(w) if w else 1.0


def row_for(run_dir: Path) -> dict | None:
    csv_path = run_dir / "steer_specificity.csv"
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
    refus = int(ans.str.contains(REFUSAL).sum()) if len(ans) else -1

    return {
        "model": meta.get("model", run_dir.name),
        "slug": run_dir.name,
        "layer": meta.get("layer", "?"),
        "coeff": meta.get("coeff", "?"),
        "judge_filtered": meta.get("judge_filtered", False),
        "n_rows": len(d),
        "diag_mean": sum(diag) / len(diag) if diag else float("nan"),
        "argmax_hits": hits,
        "sad_leak": sum(leak) / len(leak) if leak else float("nan"),
        "degen": degen,
        "refusals": refus,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Сводка по прогонам в runs/.")
    ap.add_argument("--runs", type=Path, default=REPO / "runs")
    ap.add_argument("--csv", type=Path, default=None, help="дополнительно сохранить в CSV")
    args = ap.parse_args()

    if not args.runs.is_dir():
        raise SystemExit(f"нет каталога {args.runs}")

    rows = [r for r in (row_for(p) for p in sorted(args.runs.iterdir()) if p.is_dir()) if r]
    if not rows:
        raise SystemExit(f"в {args.runs} нет готовых steer_specificity.csv")

    print("| Модель | Слой | Ср. диагональ Δ | argmax | Протечка в грусть | Вырожденных | Отказов | Фильтр судьи |")
    print("|---|---:|---:|---:|---:|---:|---:|:--:|")
    for r in rows:
        print(f"| {r['model']} | {r['layer']} | {r['diag_mean']:+.3f} | {r['argmax_hits']}/7 | "
              f"{r['sad_leak']:+.3f} | {r['degen']}/{r['n_rows']} | {r['refusals']}/{r['n_rows']} | "
              f"{'да' if r['judge_filtered'] else 'нет'} |")

    print("\nБаллы — независимый энкодер `SamLowe/roberta-base-go_emotions`, Δ к тексту без наведения.")
    print("«Вырожденных» — повтор 4-граммы > 0.15 или type-token < 0.45.")

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as fh:
            w = csvmod.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\nCSV: {args.csv}", file=sys.stderr)


if __name__ == "__main__":
    main()
