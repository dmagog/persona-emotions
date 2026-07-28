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
    """Доля повторяющихся n-грамм; устойчива к длине текста."""
    w = re.findall(r"\w+", str(text).lower())
    if len(w) < n + 4:
        return 0.0
    grams = [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]
    return 1.0 - len(set(grams)) / len(grams)


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
    refus = int(ans.str.contains(REFUSAL, regex=True).sum()) if len(ans) else -1

    # Реально применённый коэффициент. Раньше бралось meta["coeff"] — дефолт 8.0,
    # а не то, что выбрала стадия подбора рабочей точки.
    coeff = meta.get("op_coeff")
    if coeff is None and "coeff" in d.columns:
        used = sorted({float(c) for c in d[d["steer"] != "baseline"]["coeff"] if str(c).strip()})
        coeff = used[0] if len(used) == 1 else (f"{min(used):g}–{max(used):g}" if used else None)
    if coeff is None:
        coeff = meta.get("coeff", "?")

    row = {
        "model": meta.get("model", run_dir.name),
        "slug": run_dir.name,
        "layer": meta.get("layer", "?"),
        "coeff": coeff,
        "judge_filtered": meta.get("judge_filtered", False),
        "n_rows": len(d),
        "diag_mean": sum(diag) / len(diag) if diag else float("nan"),
        "argmax_hits": hits,
        "sad_leak": sum(leak) / len(leak) if leak else float("nan"),
        "degen": degen,
        "refusals": refus,
    }
    row.update(_judge_and_ci(run_dir, base_mean))
    return row


def _argmax_hits_from_wide(path: Path) -> int | None:
    """Сколько раз argmax судьи попал в целевую эмоцию."""
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
    hits = 0
    for emo in ISEAR_EMOTIONS:
        sub = w[w["steer"] == emo]
        if sub.empty:
            continue
        deltas = {e: sub[e].mean() - bm[e] for e in ISEAR_EMOTIONS}
        if max(deltas, key=deltas.get) == emo:
            hits += 1
    return hits


def _judge_and_ci(run_dir: Path, base_mean: dict) -> dict:
    """Числа из цепочки оценки, если она прогонялась."""
    out: dict = {"judge_hits": None, "coherence": None, "ci_sig": None}
    jh = _argmax_hits_from_wide(run_dir / "judge_wide.csv")
    if jh is not None:
        out["judge_hits"] = jh

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
        yes = txt.count("| да |")
        total = yes + txt.count("| нет |")
        if total:
            out["ci_sig"] = f"{yes}/{total}"
    return out


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

    print("| Модель | Слой | coeff | Диагональ Δ | argmax энк. | argmax судья | Значимо | "
          "Протечка | Связность | Вырожд. | Отказы |")
    print("|---|---:|---:|---:|---:|---:|:--:|---:|---:|---:|---:|")
    for r in rows:
        jh = f"{r['judge_hits']}/7" if r.get("judge_hits") is not None else "—"
        coh = f"{r['coherence']:.1f}" if r.get("coherence") is not None else "—"
        sig = r.get("ci_sig") or "—"
        cf = r["coeff"] if isinstance(r["coeff"], str) else (
            f"{r['coeff']:g}" if isinstance(r["coeff"], (int, float)) else "?")
        print(f"| {r['model']} | {r['layer']} | {cf} | {r['diag_mean']:+.3f} | "
              f"{r['argmax_hits']}/7 | {jh} | {sig} | {r['sad_leak']:+.3f} | {coh} | "
              f"{r['degen']}/{r['n_rows']} | {r['refusals']}/{r['n_rows']} |")

    print("\nДиагональ и протечка — независимый энкодер `go_emotions`, Δ к тексту без наведения.")
    print("«Значимо» — сколько диагоналей из 7 имеют интервал, не включающий ноль (по энкодеру).")
    print("«Связность» — средняя по стирённым условиям, судья 0–100; прочерк, если оценка не гонялась.")
    print("«Вырожд.» — повтор 4-граммы > 0.15 или type-token < 0.45. Метрика не проверена на разметке: "
          "она ловит и эмоциональный повтор тоже.")
    print("\nСтроки сравнимы между собой, только если сняты одним протоколом — сверяйтесь с meta.json.")

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as fh:
            w = csvmod.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\nCSV: {args.csv}", file=sys.stderr)


if __name__ == "__main__":
    main()
