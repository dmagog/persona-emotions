"""Полная цепочка для одной модели: пары → векторы → выбор слоя → матрица специфичности.

Каждая стадия идемпотентна: если артефакт на месте, стадия пропускается.
Судья (OpenRouter) здесь НЕ нужен — фильтр пар опционален, а баллы эмоций
считает локальный энкодер go_emotions. Судейские стадии добавляются отдельно.

Usage:
    python -m emotion.run_model_chain --model Qwen/Qwen3-1.7B --slug Qwen3-1.7B
    python -m emotion.run_model_chain --model google/gemma-2-2b-it --slug gemma-2-2b-it --layers 9,12,14
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from emotion.space import ISEAR_EMOTIONS

REPO = Path(__file__).resolve().parent.parent


def sh(args: list[str], log: Path) -> int:
    """Запустить стадию, потоково складывая вывод в лог."""
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    with log.open("a", encoding="utf-8") as fh:
        fh.write(f"\n[stage begin {stamp}] {' '.join(args)}\n")
        fh.flush()
        rc = subprocess.run(args, stdout=fh, stderr=subprocess.STDOUT, cwd=REPO).returncode
        fh.write(f"[stage end rc={rc}] {time.strftime('%Y-%m-%dT%H:%M:%S')}\n")
    print(f"  rc={rc}  ({' '.join(args[2:5])}…)", flush=True)
    return rc


def default_layers(model: str, token: str | None) -> list[int]:
    """Кандидаты слоёв: 0.35/0.45/0.55 глубины — как в протоколе."""
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(model, token=token)
    n = getattr(cfg, "num_hidden_layers", None)
    if n is None:  # вложенный конфиг (мультимодальные) — сюда лучше не заходить
        raise SystemExit(
            f"{model}: num_hidden_layers не найден на верхнем уровне конфига "
            f"({type(cfg).__name__}). Вероятно мультимодальная/гибридная архитектура — "
            "нужен отдельный путь к слоям в ActivationSteerer, см. заметки."
        )
    return sorted({max(1, round(n * f)) for f in (0.35, 0.45, 0.55)})


def pick_layer(sweep_csv: Path) -> int:
    """Слой с максимальным средним target_score при максимальном coeff."""
    d = pd.read_csv(sweep_csv)
    top = d[d["coeff"] == d["coeff"].max()]
    means = top.groupby("layer")["target_score"].mean().sort_values(ascending=False)
    print(f"  свип слоёв (anger, coeff={d['coeff'].max():.0f}): "
          + ", ".join(f"L{int(k)}={v:.3f}" for k, v in means.items()), flush=True)
    return int(means.index[0])


def main() -> None:
    ap = argparse.ArgumentParser(description="Полная цепочка на одной модели.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--slug", required=True, help="имя папок артефактов")
    ap.add_argument("--layers", default=None, help="кандидаты, через запятую; иначе от глубины")
    ap.add_argument("--coeff", type=float, default=8.0)
    ap.add_argument("--per-emotion", type=int, default=8, help="8 × 7 = 56 промптов")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=1,
                    help="батч генерации пар; 1 = построчно")
    ap.add_argument("--strength", type=float, default=None,
                    help="безразмерная сила наведения (см. steer_specificity --strength); "
                         "делает силу сопоставимой между моделями, вместо фиксированного coeff")
    ap.add_argument("--judge-scores", type=Path, default=None,
                    help="CSV фильтра пар; без него берутся все пары (отметить в отчёте)")
    args = ap.parse_args()

    import os
    token = os.environ.get("HF_TOKEN")

    pairs_dir = REPO / "eval_emotion" / args.slug
    vec_dir = REPO / "emotion_vectors" / args.slug
    runs = REPO / "runs" / args.slug
    runs.mkdir(parents=True, exist_ok=True)
    log = runs / "chain.log"
    py = sys.executable

    print(f"\n=== {args.model} → runs/{args.slug} ===", flush=True)

    # 1. Пары pos/neg на самой модели (resume внутри скрипта, построчный)
    combined = pairs_dir / "all_emotions_extract.csv"
    if combined.is_file():
        print("1. пары: уже есть, пропуск", flush=True)
    else:
        print("1. пары pos/neg …", flush=True)
        if sh([py, "-m", "eval.run_emotion_inference_batch", "--model", args.model,
               "--version", "extract", "--output_dir", str(pairs_dir),
               "--infer_backend", "hf", "--temperature", "0",
               "--max_tokens", str(args.max_tokens),
               "--batch-size", str(args.batch_size)], log) != 0:
            raise SystemExit("стадия пар упала — см. лог")

    # 2. Векторы = mean(pos) − mean(neg) послойно.
    # Пропускаем только если векторы СВЕЖЕЕ пар: иначе на новых парах молча
    # переиспользуются старые векторы (так и случилось с gemma — её векторы
    # были построены на текстах Qwen, а self-цикл это не заметил).
    vec_probe = vec_dir / f"{ISEAR_EMOTIONS[0]}_response_avg_diff.pt"
    fresh = vec_probe.is_file() and combined.is_file() and \
        vec_probe.stat().st_mtime >= combined.stat().st_mtime
    if vec_probe.is_file() and not fresh:
        raise SystemExit(
            f"векторы в {vec_dir} старше пар {combined} — они построены на других "
            "данных. Убери или переименуй их и перезапусти, иначе матрица посчитается "
            "на чужих векторах."
        )
    if fresh:
        print("2. векторы: свежее пар, пропуск", flush=True)
    else:
        print("2. извлечение векторов …", flush=True)
        cmd = [py, "-m", "emotion.extract_vectors", "--model_name", args.model,
               "--data-dir", str(pairs_dir), "--save-dir", str(vec_dir)]
        if args.judge_scores:
            cmd += ["--judge-scores", str(args.judge_scores)]
        if sh(cmd, log) != 0:
            raise SystemExit("извлечение векторов упало — см. лог")

    # 3. Выбор слоя: короткий свип на anger
    meta_path = runs / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    if "layer" in meta:
        layer = meta["layer"]
        print(f"3. слой: уже выбран L{layer}, пропуск", flush=True)
    else:
        cands = ([int(x) for x in args.layers.split(",")] if args.layers
                 else default_layers(args.model, token))
        print(f"3. свип слоёв {cands} …", flush=True)
        sweep_csv = runs / "layer_sweep_anger.csv"
        if sh([py, "-m", "emotion.steer_eval", "--model_name", args.model,
               "--emotion", "anger", "--vector-dir", str(vec_dir),
               "--layers", ",".join(map(str, cands)), "--coeffs", "0,8",
               "--n-prompts", "8", "--out", str(sweep_csv)], log) != 0:
            raise SystemExit("свип слоёв упал — см. лог")
        layer = pick_layer(sweep_csv)
        meta.update({"model": args.model, "layer": layer, "candidates": cands,
                     "coeff": args.coeff, "strength": args.strength,
                     "max_tokens": args.max_tokens,
                     "judge_filtered": bool(args.judge_scores)})
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"   выбран L{layer}", flush=True)

    # 4. Матрица специфичности: baseline + 7 эмоций × 56 промптов, баллы энкодера
    matrix_csv = runs / "steer_specificity.csv"
    if matrix_csv.is_file():
        print("4. матрица: уже есть, пропуск", flush=True)
    else:
        print(f"4. матрица специфичности на L{layer} …", flush=True)
        cmd4 = [py, "-m", "emotion.steer_specificity", "--model_name", args.model,
                "--vector-dir", str(vec_dir), "--layer", str(layer),
                "--per-emotion", str(args.per_emotion),
                "--save-answers", "--out", str(matrix_csv)]
        cmd4 += (["--strength", str(args.strength)] if args.strength is not None
                 else ["--coeff", str(args.coeff)])
        if sh(cmd4, log) != 0:
            raise SystemExit("матрица упала — см. лог")

    print(f"=== {args.slug}: цепочка пройдена → {runs} ===\n", flush=True)


if __name__ == "__main__":
    main()
