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


def env_manifest() -> dict:
    """Что нужно, чтобы строку таблицы можно было воспроизвести."""
    import platform
    import subprocess
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
                             capture_output=True, text=True).stdout.strip()
    except Exception:
        sha = "?"
    try:
        import torch, transformers
        versions = {"torch": torch.__version__, "transformers": transformers.__version__,
                    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}
    except Exception:
        versions = {}
    return {"git_sha": sha, "python": platform.python_version(), **versions}


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


def _rep_ratio(text: str, n: int = 4) -> float:
    """Доля повторяющихся n-грамм: 0 — все уникальны, 1 — сплошной повтор.

    Не «максимальная частота одной n-граммы»: та растёт на коротких текстах
    (у ответа в 9 слов всего 6 четырёхграмм, и даже полностью уникальный текст
    даёт 1/6 = 0.17) и помечает нормальные короткие ответы как вырожденные.
    """
    import re
    w = re.findall(r"\w+", str(text).lower())
    if len(w) < n + 4:
        return 0.0
    grams = [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]
    return 1.0 - len(set(grams)) / len(grams)


def pick_operating_point(sweep_csv: Path, max_degen: float = 0.10) -> tuple[int, float]:
    """Рабочая точка: сильнейшее наведение, которое ещё не разрушает текст.

    Единый коэффициент между моделями несопоставим (нормы векторов и масштабы
    активаций разные), единая безразмерная сила — тоже: связь силы с разрушением
    у каждой модели своя. Поэтому точка подбирается по поведению: максимум
    целевого балла среди ячеек, где доля вырожденных ответов не выше порога.
    Сравнение моделей затем идёт при сопоставимом эффекте, а не при равном входе.
    """
    d = pd.read_csv(sweep_csv)
    d["degen"] = d["answer"].map(lambda t: _rep_ratio(t) > 0.15)
    g = (d[d["coeff"] > 0]
         .groupby(["layer", "coeff"])
         .agg(score=("target_score", "mean"), degen=("degen", "mean"))
         .reset_index())
    print("  свип (anger): " + "; ".join(
        f"L{int(r.layer)}/c{r.coeff:.0f} балл {r.score:.3f} вырожд {r.degen:.0%}"
        for r in g.itertuples()), flush=True)
    ok = g[g["degen"] <= max_degen]
    if ok.empty:
        best = g.loc[g["degen"].idxmin()]
        print(f"  ВНИМАНИЕ: нигде вырожденность не ниже {max_degen:.0%}, "
              f"беру минимальную ({best.degen:.0%})", flush=True)
    else:
        best = ok.loc[ok["score"].idxmax()]
    return int(best["layer"]), float(best["coeff"])



def load_model_config(path: Path) -> dict:
    """Конфиг модели: yaml или json. Плоский вид для аргументов цепочки."""
    import yaml
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    st = raw.get("stages") or {}
    flat = {
        "model": raw.get("hf_id"),
        "slug": raw.get("slug") or (raw.get("hf_id") or "").split("/")[-1],
        "layers": raw.get("sweep", {}).get("layers") or st.get("sweep", {}).get("layers"),
        "sweep_coeffs": st.get("sweep", {}).get("coeffs"),
        "max_degen": st.get("sweep", {}).get("max_degen"),
        "per_emotion": st.get("matrix", {}).get("per_emotion"),
        "max_tokens": st.get("pairs", {}).get("max_tokens"),
        "batch_size": st.get("pairs", {}).get("batch_size"),
        "dtype": (raw.get("load") or {}).get("dtype"),
        "compose": (st.get("compose") or {}).get("specs"),
    }
    # списки к строкам — цепочка передаёт их дальше как аргументы
    if isinstance(flat["layers"], list):
        flat["layers"] = ",".join(str(x) for x in flat["layers"])
    if isinstance(flat["sweep_coeffs"], list):
        flat["sweep_coeffs"] = ",".join(str(x) for x in flat["sweep_coeffs"])
    if isinstance(flat.get("compose"), list):
        flat["compose"] = ",".join(str(x) for x in flat["compose"])
    flat["_raw"] = raw
    return {k: v for k, v in flat.items() if v is not None}


def main() -> None:
    ap = argparse.ArgumentParser(description="Полная цепочка на одной модели.")
    ap.add_argument("--config", type=Path, default=None,
                    help="конфиг модели (yaml); аргументы командной строки его перекрывают")
    ap.add_argument("--model", default=None)
    ap.add_argument("--slug", default=None, help="имя папок артефактов")
    ap.add_argument("--layers", default=None, help="кандидаты, через запятую; иначе от глубины")
    ap.add_argument("--coeff", type=float, default=8.0)
    ap.add_argument("--per-emotion", type=int, default=8, help="8 × 7 = 56 промптов")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=1,
                    help="батч генерации пар; 1 = построчно")
    ap.add_argument("--sweep-coeffs", default="0,4,8,16",
                    help="сетка коэффициентов для поиска рабочей точки")
    ap.add_argument("--max-degen", type=float, default=0.10,
                    help="потолок доли вырожденных ответов при выборе рабочей точки")
    ap.add_argument("--skip-preflight", action="store_true",
                    help="не гонять предполётную проверку (по умолчанию гоняется)")
    ap.add_argument("--compose", default=None,
                    help="спецификации композиции, например joy-sadness,anger-sadness; "
                         "пусто — стадия пропускается")
    ap.add_argument("--variant", default="raw",
                    help="вариант вектора: raw, sae, centered — попадает в имя артефакта")
    ap.add_argument("--strength", type=float, default=None,
                    help="безразмерная сила наведения (см. steer_specificity --strength); "
                         "делает силу сопоставимой между моделями, вместо фиксированного coeff")
    ap.add_argument("--judge-scores", type=Path, default=None,
                    help="CSV фильтра пар; без него берутся все пары (отметить в отчёте)")
    args = ap.parse_args()

    # Конфиг даёт умолчания, явный аргумент побеждает. Так добавление модели —
    # одна запись в configs/models/, а разовые правки остаются возможны.
    if args.config:
        cfg = load_model_config(args.config)
        given = {a.lstrip("-").replace("-", "_") for a in sys.argv if a.startswith("--")}
        for k, v in cfg.items():
            if k != "_raw" and k not in given and hasattr(args, k):
                setattr(args, k, v)
        args.model_config = cfg.get("_raw", {})
        print(f"конфиг: {args.config} → модель {args.model}, slug {args.slug}", flush=True)
    else:
        args.model_config = {}
    if not args.model or not args.slug:
        raise SystemExit("нужен --config или пара --model/--slug")

    import os
    token = os.environ.get("HF_TOKEN")

    pairs_dir = REPO / "eval_emotion" / args.slug
    vec_dir = REPO / "emotion_vectors" / args.slug
    runs = REPO / "runs" / args.slug
    runs.mkdir(parents=True, exist_ok=True)
    log = runs / "chain.log"
    py = sys.executable

    print(f"\n=== {args.model} → runs/{args.slug} ===", flush=True)

    # 0. Предполёт: вся цепочка на одной строке. Дешевле, чем узнать о поломке
    # через пять часов. Запускается, когда векторы уже есть (иначе проверять нечего).
    if not args.skip_preflight:
        # Запускаем ВСЕГДА, а не только при готовых векторах: на новой модели
        # ломались gemma (системная роль) и Qwen3 (рассуждения), и именно там
        # проверка была отключена. Без векторов предполёт проверит шаблон и
        # генерацию, с векторами — ещё и наведение.
        meta_pre = json.loads((runs / "meta.json").read_text(encoding="utf-8")) \
            if (runs / "meta.json").is_file() else {}
        layer_hint = meta_pre.get("layer")
        if layer_hint is None:
            try:
                from emotion.loader import num_layers
                from transformers import AutoConfig
                layer_hint = round(num_layers(
                    AutoConfig.from_pretrained(args.model, token=token)) * 0.45)
            except Exception:
                layer_hint = 0
        pf = [py, "-m", "emotion.preflight", "--model", args.model, "--layer", str(layer_hint)]
        pf += (["--strength", str(args.strength)] if args.strength is not None
               else ["--coeff", str(args.coeff)])
        print("0. предполёт …", flush=True)
        if sh(pf, log) != 0:
            raise SystemExit("предполёт не пройден — очередь остановлена, см. лог")

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
    if "layer" in meta and "op_coeff" in meta:
        layer, op_coeff = meta["layer"], meta["op_coeff"]
        print(f"3. рабочая точка: уже выбрана L{layer}/c{op_coeff:g}, пропуск", flush=True)
        # сила наведения могла смениться между прогонами — фиксируем актуальную
        meta["strength"] = args.strength
        meta["coeff"] = args.coeff
        meta["env"] = env_manifest()
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        # Слои задаются долями глубины (0.45) или абсолютными номерами (13).
        # Доли переносимы между моделями, абсолютные привязаны к одной.
        if args.layers:
            from transformers import AutoConfig
            from emotion.loader import resolve_layers
            cfg_model = AutoConfig.from_pretrained(args.model, token=token)
            cands = resolve_layers(cfg_model, [x.strip() for x in str(args.layers).split(",")])
        else:
            cands = default_layers(args.model, token)
        print(f"3. свип слоёв {cands} …", flush=True)
        sweep_csv = runs / "layer_sweep_anger.csv"
        if sh([py, "-m", "emotion.steer_eval", "--model_name", args.model,
               "--emotion", "anger", "--vector-dir", str(vec_dir),
               "--layers", ",".join(map(str, cands)), "--coeffs", args.sweep_coeffs,
               "--n-prompts", "8", "--out", str(sweep_csv)], log) != 0:
            raise SystemExit("свип упал — см. лог")
        layer, op_coeff = pick_operating_point(sweep_csv, args.max_degen)
        meta.update({"model": args.model, "layer": layer, "candidates": cands,
                     "op_coeff": op_coeff, "max_degen": args.max_degen,
                     "coeff": args.coeff, "strength": args.strength,
                     "max_tokens": args.max_tokens,
                     "judge_filtered": bool(args.judge_scores),
                     "variant": args.variant,
                     "config": args.model_config,
                     "env": env_manifest()})
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"   рабочая точка: слой {layer}, coeff {op_coeff:g}", flush=True)

    # 4. Матрица специфичности: baseline + 7 эмоций × 56 промптов, баллы энкодера
    # Имя с вариантом: raw, sae, centered должны сосуществовать в одном прогоне.
    # Старое имя без варианта распознаётся, чтобы прежние прогоны не потерялись.
    matrix_csv = runs / f"steer_specificity_{args.variant}.csv"
    legacy = runs / "steer_specificity.csv"
    if legacy.is_file() and not matrix_csv.is_file() and args.variant == "raw":
        matrix_csv = legacy
    if matrix_csv.is_file():
        print("4. матрица: уже есть, пропуск", flush=True)
    else:
        print(f"4. матрица специфичности на L{layer} …", flush=True)
        cmd4 = [py, "-m", "emotion.steer_specificity", "--model_name", args.model,
                "--vector-dir", str(vec_dir), "--layer", str(layer),
                "--per-emotion", str(args.per_emotion),
                "--save-answers", "--out", str(matrix_csv)]
        cmd4 += (["--strength", str(args.strength)] if args.strength is not None
                 else ["--coeff", str(op_coeff)])
        if sh(cmd4, log) != 0:
            raise SystemExit("матрица упала — см. лог")

    # 5. Композиция: сложение и вычитание эмоций. Это то, чего промптом не
    # сделать, и единственный результат, который держится на двух измерителях.
    # Запускается по флагу: стоит ещё столько же генераций, сколько матрица.
    if args.compose:
        compose_csv = runs / "compose.csv"
        if compose_csv.is_file() and compose_csv.stat().st_mtime >= matrix_csv.stat().st_mtime:
            print("5. композиция: свежее матрицы, пропуск", flush=True)
        else:
            print(f"5. композиция ({args.compose}) …", flush=True)
            if sh([py, "-m", "emotion.steer_compose", "--model_name", args.model,
                   "--vector-dir", str(vec_dir), "--layer", str(layer),
                   "--coeff", str(op_coeff), "--per-emotion", str(args.per_emotion),
                   "--specs", args.compose, "--out", str(compose_csv)], log) != 0:
                print("   композиция упала — цепочка продолжается", flush=True)

    print(f"=== {args.slug}: цепочка пройдена → {runs} ===\n", flush=True)


if __name__ == "__main__":
    main()
