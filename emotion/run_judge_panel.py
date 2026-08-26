"""Панель судей: пересудить матрицы внешними судьями и посчитать согласие.

Основной судья (llama-3.3-70b) один размечает всю сетку - без панели весь
судейский столбец таблицы держится на вкусах одной модели, к тому же родной
для Llama-испытуемых. Панель добавляет судей других вендоров, для каждого
считает согласие с основным и, по флагу, пересуживает связность.

Этим скриптом сняты файлы в runs/<slug>/:
  judge_wide_<tag>.csv        матрица глазами внешнего судьи
  judge_agreement_<tag>.md    согласие с основным: Pearson, Spearman, каппа
  coherence_<tag>.md          связность глазами внешнего судьи (--coherence)
  compose_judge_wide.csv      композиция глазами основного судьи (--compose)

Судьи по умолчанию - те, которыми снята опубликованная сетка. Нужен ключ
OpenRouter в окружении (OPENAI_API_KEY + OPENAI_BASE_URL).

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

# tag -> модель на OpenRouter. Tag попадает в имена файлов, менять его для
# уже посчитанных судей нельзя - файлы перестанут узнаваться.
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
    ap = argparse.ArgumentParser(description="Панель внешних судей поверх готовых прогонов.")
    ap.add_argument("--slug", action="append", default=[],
                    help="какие прогоны; повторяемый флаг")
    ap.add_argument("--all", action="store_true", help="все прогоны из runs/")
    ap.add_argument("--judges", default=",".join(PANEL),
                    help=f"теги судей через запятую, по умолчанию {','.join(PANEL)}")
    ap.add_argument("--coherence", action="store_true",
                    help="пересудить и связность каждым судьёй панели")
    ap.add_argument("--compose", action="store_true",
                    help="отсудить композицию основным судьёй")
    args = ap.parse_args()

    runs = sorted(p for p in (REPO / "runs").iterdir() if p.is_dir()) if args.all \
        else [REPO / "runs" / s for s in args.slug]
    if not runs:
        raise SystemExit("нужен --slug или --all")
    tags = [t.strip() for t in args.judges.split(",") if t.strip()]
    unknown = [t for t in tags if t not in PANEL]
    if unknown:
        raise SystemExit(f"неизвестные судьи {unknown}; известны {list(PANEL)}")

    # Деньги кончаются молча: провайдер отвечает 402, строки не попадают в
    # матрицу, а таблица показывает результат по подвыборке. Снимок до и после.
    print(balance.line("баланс до прогона"), flush=True)

    fails = 0
    for run in runs:
        csv = matrix_of(run)
        if csv is None:
            print(f"{run.name}: матрицы нет, пропуск", flush=True)
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

    print(balance.line("баланс после прогона"), flush=True)
    print(f"панель пройдена, сбоев: {fails}", flush=True)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
