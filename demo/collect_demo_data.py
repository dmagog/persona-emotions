"""Сборка данных для демо из готового прогона.

Раньше demo_data.json собирался руками, поэтому демо существовало только для
одной модели и не пересобиралось при пересчёте. Здесь то же самое выводится
из артефактов прогона: выбрал модель — получил демо.

Что можно собрать всегда: сценарии с генерациями, матрицу специфичности,
лестницу коэффициентов и обрыв связности из свипа.
Что появляется только после цепочки оценки: связность по судье, матрица судьи.
Что требует отдельных прогонов и в демо не попадёт: композиция эмоций и
поведение вне распределения — они считаются другими скриптами.

Usage:
    python demo/collect_demo_data.py --run runs/Qwen3-1.7B
    python demo/collect_demo_data.py --run runs/Qwen3-1.7B --out demo/demo_data.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from emotion.space import ISEAR_EMOTIONS  # noqa: E402

RU = {"anger": "гнев", "disgust": "отвращение", "fear": "страх", "guilt": "вина",
      "joy": "радость", "sadness": "грусть", "shame": "стыд"}


def rep_ratio(text: str, n: int = 4) -> float:
    w = re.findall(r"\w+", str(text).lower())
    if len(w) < n + 4:
        return 0.0
    grams = [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]
    return 1.0 - len(set(grams)) / len(grams)


def pick_matrix(run: Path, variant: str) -> Path:
    for name in (f"steer_specificity_{variant}.csv", "steer_specificity.csv"):
        p = run / name
        if p.is_file():
            return p
    raise SystemExit(f"нет матрицы в {run}")


def load_questions(pairs_dir: Path, per_emotion: int) -> list[str]:
    """Вопросы в том же порядке, в каком их брала стадия матрицы."""
    qs: list[str] = []
    for emo in ISEAR_EMOTIONS:
        f = REPO / "data_generation" / "emotion_data_eval" / f"{emo}.json"
        if not f.is_file():
            return []
        qs.extend(json.loads(f.read_text(encoding="utf-8"))["questions"][:per_emotion])
    return qs


def build_scenarios(d: pd.DataFrame, questions: list[str], limit: int = 8) -> list[dict]:
    """Карточки: один вопрос, исходный текст и семь наведённых."""
    by = {(r["steer"], int(r["prompt_id"])): r for _, r in d.iterrows()}
    pids = sorted({int(p) for p in d["prompt_id"]})

    scored = []
    for pid in pids:
        b = by.get(("baseline", pid))
        if b is None:
            continue
        lifts = []
        for e in ISEAR_EMOTIONS:
            r = by.get((e, pid))
            if r is not None:
                lifts.append(float(r[e]) - float(b[e]))
        if lifts:
            scored.append((sum(lifts) / len(lifts), pid))
    scored.sort(reverse=True)

    out = []
    for _, pid in scored[:limit]:
        b = by[("baseline", pid)]
        item = {
            "pid": pid,
            "question": questions[pid] if pid < len(questions) else f"сценарий #{pid}",
            "baseline": {"text": str(b["answer"]),
                         "scores": {e: float(b[e]) for e in ISEAR_EMOTIONS}},
            "steered": {},
        }
        for e in ISEAR_EMOTIONS:
            r = by.get((e, pid))
            if r is None:
                continue
            item["steered"][e] = {
                "text": str(r["answer"]),
                "scores": {x: float(r[x]) for x in ISEAR_EMOTIONS},
                "delta": round(float(r[e]) - float(b[e]), 4),
            }
        out.append(item)
    return out


def build_matrix(d: pd.DataFrame) -> dict | None:
    base = d[d["steer"] == "baseline"]
    if base.empty:
        return None
    bm = {e: base[e].mean() for e in ISEAR_EMOTIONS}
    rows = []
    for emo in ISEAR_EMOTIONS:
        sub = d[d["steer"] == emo]
        if sub.empty:
            continue
        cells = [round(sub[e].mean() - bm[e], 4) for e in ISEAR_EMOTIONS]
        amax = ISEAR_EMOTIONS[max(range(len(cells)), key=lambda i: cells[i])]
        rows.append({"steer": emo, "cells": cells, "n": len(sub),
                     "diag": cells[ISEAR_EMOTIONS.index(emo)],
                     "argmax": amax, "hit": amax == emo})
    return {"emo": list(ISEAR_EMOTIONS), "rows": rows} if rows else None


def build_ladder_and_cliff(sweep: Path, layer: int | None):
    """Лестница коэффициентов на выбранном слое и обрыв качества."""
    if not sweep.is_file():
        return [], {}
    s = pd.read_csv(sweep)
    if "answer" not in s.columns:
        return [], {}
    s["degen"] = s["answer"].map(lambda t: rep_ratio(t) > 0.15)

    lay = layer if layer is not None and layer in set(s["layer"]) else None
    if lay is None:
        best = s[s.coeff > 0].groupby("layer")["target_score"].mean()
        lay = int(best.idxmax()) if len(best) else None
    sub = s[s["layer"] == lay] if lay is not None else s

    ladder = []
    for pid in sorted({int(x) for x in sub["prompt_id"]})[:6]:
        steps = {}
        for _, r in sub[sub["prompt_id"] == pid].iterrows():
            steps[str(float(r["coeff"]))] = {"text": str(r["answer"]),
                                             "score": float(r["target_score"])}
        if len(steps) >= 2:
            ladder.append({"pid": pid, "steps": dict(sorted(steps.items(), key=lambda kv: float(kv[0])))})

    cliff = {}
    for c, grp in sub.groupby("coeff"):
        cliff[str(float(c))] = {"score": round(float(grp["target_score"].mean()), 3),
                                "degen": round(float(grp["degen"].mean()) * 100, 1)}
    return ladder, cliff


def parse_coherence(md: Path) -> dict:
    if not md.is_file():
        return {}
    out = {}
    for line in md.read_text(encoding="utf-8").splitlines():
        parts = [c.strip() for c in line.split("|")]
        if len(parts) >= 4 and parts[1] and parts[1] not in ("steer", "---"):
            try:
                out[parts[1]] = float(parts[2])
            except ValueError:
                continue
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Собрать данные демо из прогона.")
    ap.add_argument("--run", required=True, type=Path, help="каталог runs/<slug>")
    ap.add_argument("--variant", default="raw")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--scenarios", type=int, default=8)
    args = ap.parse_args()

    run = args.run if args.run.is_absolute() else REPO / args.run
    if not run.is_dir():
        raise SystemExit(f"нет каталога {run}")

    meta = {}
    if (run / "meta.json").is_file():
        meta = json.loads((run / "meta.json").read_text(encoding="utf-8"))

    matrix_csv = pick_matrix(run, args.variant)
    d = pd.read_csv(matrix_csv)
    if "answer" not in d.columns:
        raise SystemExit(f"{matrix_csv.name} без колонки answer — демо строить не из чего "
                         "(матрицу надо считать с --save-answers)")

    per_emotion = int(meta.get("per_emotion", 8))
    questions = load_questions(REPO / "eval_emotion" / run.name, per_emotion)
    ladder, cliff = build_ladder_and_cliff(
        run / "layer_sweep_anger.csv" if (run / "layer_sweep_anger.csv").is_file()
        else run / "layer_sweep.csv", meta.get("layer"))

    data = {
        "meta": {
            "model": meta.get("model", run.name),
            "slug": run.name,
            "layer": meta.get("layer"),
            "coeff": meta.get("op_coeff", meta.get("coeff")),
            "variant": args.variant,
            "env": meta.get("env", {}),
        },
        "scenarios": build_scenarios(d, questions, args.scenarios),
        "matrix": build_matrix(d),
        "ladder": ladder,
        "cliff": cliff,
        "coherence": parse_coherence(run / "coherence.md"),
    }

    out = args.out or (REPO / "demo" / f"demo_data_{run.name}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    have = [k for k in ("scenarios", "matrix", "ladder", "cliff", "coherence") if data[k]]
    missing = [k for k in ("scenarios", "matrix", "ladder", "cliff", "coherence") if not data[k]]
    print(f"записано {out} ({out.stat().st_size // 1024} КБ)")
    print(f"  собрано: {', '.join(have)}")
    if missing:
        print(f"  нет данных: {', '.join(missing)}")
    print("  композиция и поведение вне распределения в демо не попадают — "
          "они считаются отдельными скриптами")


if __name__ == "__main__":
    main()
