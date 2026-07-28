"""Цепочка оценки: от матрицы специфичности до отчётных чисел.

Брат-близнец run_model_chain. Тот доводит модель до матрицы, этот — матрицу
до чисел, которые идут в отчёт и в демо.

Бесплатные стадии (энкодер, локально) идут всегда. Судейские — по флагу,
потому что стоят денег и требуют ключа.

Стадии идемпотентны: артефакт переиспользуется, только если он свежее своего
входа. Иначе после пересчёта матрицы старые интервалы молча остались бы в отчёте.

Usage:
    python -m emotion.run_eval_chain --slug Qwen3-1.7B
    python -m emotion.run_eval_chain --slug Qwen3-1.7B --judge
    python -m emotion.run_eval_chain --slug Qwen3-1.7B --judge --judge2 qwen/qwen-2.5-72b-instruct
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def sh(args: list[str], log: Path, out: Path | None = None) -> int:
    """Запустить стадию, потоково складывая вывод в лог."""
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    with log.open("a", encoding="utf-8") as fh:
        fh.write(f"\n[stage begin {stamp}] {' '.join(str(a) for a in args)}\n")
        fh.flush()
        rc = subprocess.run([str(a) for a in args], stdout=fh, stderr=subprocess.STDOUT,
                            cwd=REPO).returncode
        fh.write(f"[stage end rc={rc}] {time.strftime('%Y-%m-%dT%H:%M:%S')}\n")
    mark = "ок" if rc == 0 else f"СБОЙ rc={rc}"
    print(f"  {mark}" + (f" → {out.name}" if out else ""), flush=True)
    return rc


def fresh(artifact: Path, source: Path) -> bool:
    """Артефакт годен, только если он новее своего входа.

    Проверка по факту существования файла уже стоила прогона: после пересчёта
    матрицы старые числа остались бы в отчёте и выглядели бы как новые.
    """
    return (artifact.is_file() and source.is_file()
            and artifact.stat().st_mtime >= source.stat().st_mtime)


def stage(name: str, artifact: Path, source: Path, cmd: list, log: Path,
          required: bool = True) -> bool:
    """Одна стадия: пропустить свежее, иначе выполнить."""
    if fresh(artifact, source):
        print(f"{name}: свежее входа, пропуск", flush=True)
        return True
    if artifact.is_file():
        print(f"{name}: устарело относительно {source.name}, пересчёт …", flush=True)
    else:
        print(f"{name} …", flush=True)
    rc = sh(cmd, log, artifact)
    if rc != 0:
        if required:
            raise SystemExit(f"{name}: стадия упала, см. {log}")
        print(f"{name}: пропущено (rc={rc}), цепочка продолжается", flush=True)
        return False
    if not artifact.is_file():
        raise SystemExit(f"{name}: стадия отработала, но {artifact} не появился")
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description="Цепочка оценки для одного прогона.")
    ap.add_argument("--slug", required=True, help="имя каталога в runs/")
    ap.add_argument("--variant", default="raw", help="какая матрица оценивается: raw, sae, ...")
    ap.add_argument("--judge", action="store_true", help="судейские стадии (нужен ключ, платно)")
    ap.add_argument("--judge-model", default="meta-llama/llama-3.3-70b-instruct")
    ap.add_argument("--judge2", default=None, help="второй судья для проверки согласия")
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--n-boot", type=int, default=20000)
    args = ap.parse_args()

    runs = REPO / "runs" / args.slug
    if not runs.is_dir():
        raise SystemExit(f"нет каталога {runs}")
    log = runs / "eval.log"
    py = sys.executable

    meta_path = runs / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    model_id = meta.get("model")
    layer = meta.get("layer")

    # матрица может называться и по-старому, и по варианту
    matrix = runs / f"steer_specificity_{args.variant}.csv"
    if not matrix.is_file():
        matrix = runs / "steer_specificity.csv"
    if not matrix.is_file():
        raise SystemExit(f"нет матрицы в {runs} — сначала прогоните run_model_chain")

    print(f"\n=== оценка {args.slug} ({matrix.name}) ===", flush=True)

    # --- бесплатные стадии: энкодер и геометрия векторов ---

    stage("1. интервалы по энкодеру",
          runs / "ci_encoder.md", matrix,
          [py, "-m", "emotion.bootstrap_ci", "--csv", matrix, "--n-boot", args.n_boot,
           "--out", runs / "ci_encoder.md"],
          log)

    vec_dir = REPO / "emotion_vectors" / args.slug
    if vec_dir.is_dir() and layer is not None:
        probe = vec_dir / "anger_response_avg_diff.pt"
        # ВАЖНО: у valence_arousal --layer это сырой индекс тензора, а в meta
        # лежит номер блока. Индексация векторов сдвинута на единицу.
        stage("2. геометрия векторов",
              runs / "valence_arousal.md", probe,
              [py, "-m", "emotion.valence_arousal", "--vector-dir", vec_dir,
               "--layer", int(layer) + 1, "--out", runs / "valence_arousal.md"],
              log, required=False)
    else:
        print("2. геометрия векторов: пропуск (нет векторов или слоя в meta)", flush=True)

    pairs_dir = REPO / "eval_emotion" / args.slug
    if pairs_dir.is_dir():
        combined = pairs_dir / "all_emotions_extract.csv"
        stage("3. разделимость по парам",
              runs / "separability.csv", combined,
              [py, "-m", "emotion.score_csv", "--data-dir", pairs_dir,
               "--out", runs / "separability.csv"],
              log, required=False)
    else:
        print("3. разделимость: пропуск (нет пар)", flush=True)

    # --- судейские стадии ---

    if not args.judge:
        print("\nсудейские стадии пропущены (нужен флаг --judge)", flush=True)
        print(f"=== {args.slug}: оценка пройдена → {runs} ===\n", flush=True)
        return

    import os
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("нет OPENAI_API_KEY — судейские стадии запустить нельзя")

    judge_wide = runs / "judge_wide.csv"
    stage("4. судейская матрица",
          judge_wide, matrix,
          [py, "-m", "emotion.judge_specificity", "--csv", matrix,
           "--model", args.judge_model, "--concurrency", args.concurrency,
           "--out", runs / "judge_matrix.json", "--out-wide", judge_wide,
           "--cache", runs / "judge.cache.jsonl"],
          log)

    stage("5. интервалы по судье",
          runs / "ci_judge.md", judge_wide,
          [py, "-m", "emotion.bootstrap_ci", "--csv", judge_wide, "--n-boot", args.n_boot,
           "--out", runs / "ci_judge.md"],
          log)

    stage("6. связность",
          runs / "coherence.md", matrix,
          [py, "-m", "emotion.coherence_check", "--csv", matrix,
           "--model", args.judge_model, "--concurrency", args.concurrency,
           "--cache", runs / "coherence.cache.jsonl", "--out", runs / "coherence.md"],
          log)

    if args.judge2:
        wide2 = runs / "judge2_wide.csv"
        stage("7. второй судья",
              wide2, matrix,
              [py, "-m", "emotion.judge_specificity", "--csv", matrix,
               "--model", args.judge2, "--concurrency", args.concurrency,
               "--out", runs / "judge2_matrix.json", "--out-wide", wide2,
               "--cache", runs / "judge2.cache.jsonl"],
              log)
        stage("8. согласие судей",
              runs / "judge_agreement.md", wide2,
              [py, "-m", "emotion.judge_agreement", "--judge1", judge_wide,
               "--judge2", wide2, "--name1", args.judge_model, "--name2", args.judge2,
               "--out", runs / "judge_agreement.md"],
              log, required=False)

    meta["eval"] = {
        "variant": args.variant,
        "judge_model": args.judge_model if args.judge else None,
        "judge2": args.judge2,
        "n_boot": args.n_boot,
        "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"=== {args.slug}: оценка пройдена → {runs} ===\n", flush=True)


if __name__ == "__main__":
    main()
