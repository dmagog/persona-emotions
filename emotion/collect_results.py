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

from emotion.protocol import compare_planes, plane_of, report
from emotion.space import ISEAR_EMOTIONS

REPO = Path(__file__).resolve().parent.parent
RU = {"anger": "гнев", "disgust": "отвращение", "fear": "страх", "guilt": "вина",
      "joy": "радость", "sadness": "грусть", "shame": "стыд"}
REFUSAL = re.compile(
    r"as an AI|as a language model|I am an AI|language model|"
    r"I (?:don't|do not|cannot|can't) have (?:personal )?(?:feelings|emotions)|"
    r"I'm just an? (?:AI|computer program)|I can only (?:help|assist)|"
    r"cannot feel|unable to (?:feel|experience)", re.I)


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


def variants_in(run_dir: Path) -> list[Path]:
    """Все матрицы прогона: raw, sae, centered. Каждая — своя строка сводки."""
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

    # Реально применённый коэффициент. Раньше бралось meta["coeff"] — дефолт 8.0,
    # а не то, что выбрала стадия подбора рабочей точки.
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


# Условие, измеренное на горстке ответов, ничего не говорит: у granite
# судья потерял shame целиком, а sadness свёл к двум ответам, и «4/7» в
# таблице считало shame промахом по нулю наблюдений. Ниже этого порога
# условие не засчитывается ни в попадания, ни в знаменатель.
MIN_JUDGED = 20


def _argmax_hits_from_wide(path: Path) -> tuple[int, int, dict] | None:
    """Попадания argmax судьи: (попало, измерено условий, сколько ответов на условие).

    Возвращает и знаменатель: судья теряет часть вызовов на разборе ответа и
    сбоях провайдера, и молча делить на семь — значит выдавать недомер за
    промах.
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
    """Числа из цепочки оценки, если она прогонялась."""
    out: dict = {"judge_hits": None, "judge_measured": None, "judge_thin": "",
                 "judge_coverage": None, "coherence": None, "ci_sig": None}
    jh = _argmax_hits_from_wide(run_dir / "judge_wide.csv")
    if jh is not None:
        hits, measured, sizes = jh
        out["judge_hits"] = hits
        out["judge_measured"] = measured
        thin = {e: n for e, n in sizes.items() if n < MIN_JUDGED}
        out["judge_thin"] = ", ".join(f"{e}={n}" for e, n in sorted(thin.items())) or ""
        # Полнота: равномерная потеря 10% не задевает ни одного условия по
        # отдельности и без этой строки осталась бы невидимой.
        want = max(sizes.values(), default=0) * 8  # 7 эмоций + baseline
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

    rows = []
    for run_dir in sorted(args.runs.iterdir()):
        if not run_dir.is_dir():
            continue
        for csv_path in variants_in(run_dir):
            r = row_for(run_dir, csv_path)
            if r:
                rows.append(r)
    if not rows:
        raise SystemExit(f"в {args.runs} нет готовых steer_specificity.csv")

    # Плоскость: строки сравнимы между собой, только если сняты одним протоколом.
    # Раньше это была приписка под таблицей «сверяйтесь с meta.json» — то есть
    # проверка, которую никто не делает.
    planes = {f"{r['slug']}/{r['variant']}": {k[len("plane_"):]: v
                                              for k, v in r.items() if k.startswith("plane_")}
              for r in rows}
    off = compare_planes(planes)
    # Помечаем меньшинство: строка выбивается, если хоть по одному ключу её
    # значение встречается реже большинства. Если большинства нет — раскол
    # пополам, — помечаем все: выбирать «правильную» половину монеткой нельзя.
    odd: set[str] = set()
    for seen in off.values():
        sizes = sorted((len(v) for v in seen.values()), reverse=True)
        if len(sizes) > 1 and sizes[0] == sizes[1]:
            odd |= {n for names in seen.values() for n in names}
        else:
            major = max(seen.values(), key=len)
            odd |= {n for names in seen.values() if names is not major for n in names}

    multi = len({r["variant"] for r in rows}) > 1
    vcol = " Вариант |" if multi else ""
    vsep = "---|" if multi else ""
    print(f"| Модель |{vcol} Слой | coeff | Диагональ Δ | argmax энк. | argmax судья | Значимо | "
          "Протечка | Связность | Вырожд. | Отказы | Плоскость |")
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

    print("\nДиагональ и протечка — независимый энкодер `go_emotions`, Δ к тексту без наведения.")
    print("«Значимо» — сколько диагоналей из 7 имеют интервал, не включающий ноль (по энкодеру).")
    print("«Связность» — средняя по стирённым условиям, судья 0–100; прочерк, если оценка не гонялась.")
    print("«Вырожд.» — повтор 4-граммы > 0.15 или type-token < 0.45. Метрика не проверена на разметке: "
          "она ловит и эмоциональный повтор тоже.")
    print("«Плоскость»: ✓ — снято текущим протоколом, ⚠ — выбивается из общего, "
          "? — штампа нет, протокол известен только со слов манифеста.")
    lean = [(r["slug"], r["judge_coverage"]) for r in rows
            if r.get("judge_coverage") is not None and r["judge_coverage"] < 0.95]
    if lean:
        print("\nСудья ответил не на все вызовы (ниже 95% матрицы) — числа "
              "по этим строкам считаны на подвыборке, добрать: "
              "`python3 -m emotion.judge_specificity` с тем же --cache:")
        for slug, cov in lean:
            print(f"- {slug}: {cov:.0%} матрицы")
    thin_rows = [(r["slug"], r["judge_thin"]) for r in rows if r.get("judge_thin")]
    if thin_rows:
        print(f"\n«argmax судья» показан как попадания/измеренных условий. Судья теряет "
              f"часть вызовов на разборе ответа и сбоях провайдера; условия, где осталось "
              f"меньше {MIN_JUDGED} ответов из 56, из счёта исключены:")
        for slug, thin in thin_rows:
            print(f"- {slug}: {thin}")
    print("\n## Сопоставимость\n")
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
