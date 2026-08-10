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
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from emotion import stamp
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

    # Оговорки, которые иначе всплывут только при чтении таблицы — или не всплывут.
    per_cell = int(d.groupby(["layer", "coeff"]).size().min())
    if per_cell and 1.0 / per_cell > max_degen:
        print(f"  ОГОВОРКА: {per_cell} промптов на ячейку, шаг доли вырожденных "
              f"{1 / per_cell:.0%} — порог {max_degen:.0%} означает «ни одного "
              f"вырожденного из {per_cell}», а не «не больше {max_degen:.0%}»",
              flush=True)
    grid = sorted({float(c) for c in g["coeff"] if float(c) > 0})
    if grid and abs(float(best["coeff"]) - grid[-1]) < 1e-9:
        print(f"  ОГОВОРКА: выбран верх сетки ({grid[-1]:g}) — оптимум мог остаться "
              "за ней, сетку стоит расширить вверх", flush=True)
    elif len(grid) > 1 and abs(float(best["coeff"]) - grid[0]) < 1e-9:
        print(f"  ОГОВОРКА: выбран низ сетки ({grid[0]:g}) — всё, что сильнее, уже "
              "разрушало текст, сетке не хватает промежуточных значений", flush=True)
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
        "sweep_prompts": st.get("sweep", {}).get("n_prompts"),
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


def set_aside(artifact: Path) -> Path | None:
    """Убрать готовый артефакт с дороги перед пересчётом. Не удалять.

    Пересчёт нужен там, где штампа нет (такой артефакт переиспользуется молча),
    а сами стадии возобновляются по уже записанному — просто перезапустить
    цепочку недостаточно. Прежнее стоило часов и может понадобиться для
    сравнения, поэтому переименовываем, а не стираем.
    """
    if not stamp.present(artifact):
        return None
    bak = artifact.with_name(f"{artifact.name}.{time.strftime('%Y%m%d-%H%M')}.bak")
    n = 2
    while bak.exists():
        bak = artifact.with_name(f"{artifact.name}.{time.strftime('%Y%m%d-%H%M')}-{n}.bak")
        n += 1
    artifact.rename(bak)
    st = stamp.stamp_path(artifact)
    if st.is_file():
        st.rename(st.with_name(bak.name + ".stamp.json"))
    return bak


def resolve_run_dtype(model: str, configured: str | None, token: str | None) -> tuple[str, str]:
    """Тип вычислений на весь прогон: решается один раз и пишется в манифест.

    Раньше каждая стадия решала сама, глядя на карту, и решение нигде не
    оставалось. Так вышло, что Llama и Qwen2.5-1.5B посчитались в bf16, а
    Qwen3 и gemma — в fp16, и понять это можно было только по логу загрузчика.
    """
    if configured and configured != "auto":
        return configured, "задан в конфиге"
    try:
        from transformers import AutoConfig

        from emotion.loader import resolve_dtype
        dt, why = resolve_dtype("auto", AutoConfig.from_pretrained(model, token=token))
        return str(dt).replace("torch.", ""), why
    except Exception as e:
        return "auto", f"заранее не определить ({type(e).__name__}), решит стадия"


@dataclass
class StageSpec:
    """Что определяет результат стадии: артефакт, параметры, входы."""
    stage: str
    artifact: Path
    params: dict
    inputs: list[Path] = field(default_factory=list)
    # Что убирать с дороги при --recompute. У пар это весь каталог: стадия
    # возобновляется по отдельным <эмоция>_pos.csv, и снос одного сводного
    # файла её не заставит считать заново.
    backup: Path | None = None


def build_specs(a, meta: dict | None = None) -> dict[str, StageSpec]:
    """Параметры, влияющие на результат каждой стадии, — в одном месте.

    И цепочка, и `emotion.stamp --adopt` берут их отсюда. Если бы каждый считал
    их сам, штамп разошёлся бы со стадией на первом же изменении и отпечаток
    начал бы врать — а врущий отпечаток хуже, чем никакого.
    """
    meta = meta or {}
    raw = getattr(a, "model_config", None) or {}
    dtype = getattr(a, "resolved_dtype", None) or (raw.get("load") or {}).get("dtype", "auto")
    pairs = REPO / "eval_emotion" / a.slug / "all_emotions_extract.csv"
    vec_dir = REPO / "emotion_vectors" / a.slug
    runs = REPO / "runs" / a.slug
    layer, op_coeff = meta.get("layer"), meta.get("op_coeff")
    # Имя с вариантом: raw, sae, centered сосуществуют в одном прогоне. Старое
    # имя без варианта распознаётся, чтобы прежние прогоны не потерялись.
    matrix_csv = runs / f"steer_specificity_{a.variant}.csv"
    legacy = runs / "steer_specificity.csv"
    if a.variant == "raw" and legacy.is_file() and not matrix_csv.is_file():
        matrix_csv = legacy
    # Наведение задаётся либо безразмерной силой, либо коэффициентом — в
    # отпечаток идёт то, что реально применялось.
    drive = ({"strength": a.strength} if getattr(a, "strength", None) is not None
             else {"coeff": op_coeff})

    specs = {
        "pairs": StageSpec(
            "pairs", pairs,
            {"model": a.model, "version": "extract", "backend": "hf",
             "temperature": 0, "max_tokens": a.max_tokens,
             "batch_size": a.batch_size, "dtype": dtype,
             "prompt": raw.get("prompt") or {}},
            [REPO / "data_generation" / "emotion_data_extract"],
            backup=pairs.parent),
        "vectors": StageSpec(
            "vectors", vec_dir,
            {"model": a.model, "dtype": dtype, "pooling": "response_avg",
             "judge_scores": Path(a.judge_scores).name if a.judge_scores else None},
            [pairs]),
        "sweep": StageSpec(
            "sweep", runs / "layer_sweep_anger.csv",
            {"model": a.model, "emotion": "anger", "layers": meta.get("candidates"),
             "coeffs": a.sweep_coeffs, "n_prompts": a.sweep_prompts,
             "max_new_tokens": 120, "dtype": dtype},
            [vec_dir]),
        "matrix": StageSpec(
            "matrix", matrix_csv,
            {"model": a.model, "layer": layer, "variant": a.variant,
             "per_emotion": a.per_emotion, "max_new_tokens": 120,
             "dtype": dtype, **drive},
            [vec_dir]),
    }
    if getattr(a, "compose", None):
        specs["compose"] = StageSpec(
            "compose", runs / "compose.csv",
            {"model": a.model, "layer": layer, "coeff": op_coeff,
             "per_emotion": a.per_emotion, "specs": a.compose, "dtype": dtype},
            [vec_dir])
    return specs


def stage_specs(config: Path) -> list[StageSpec]:
    """Спецификации стадий по конфигу и манифесту прогона — для `emotion.stamp`."""
    a = build_parser().parse_args([])
    cfg = load_model_config(config)
    for k, v in cfg.items():
        if k != "_raw" and hasattr(a, k):
            setattr(a, k, v)
    a.model_config = cfg.get("_raw", {})
    meta_path = REPO / "runs" / a.slug / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    return list(build_specs(a, meta).values())


def build_parser() -> argparse.ArgumentParser:
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
    ap.add_argument("--sweep-coeffs", default="0,2,4,6,8,16",
                    help="сетка коэффициентов для поиска рабочей точки")
    ap.add_argument("--sweep-prompts", type=int, default=16,
                    help="промптов на ячейку свипа; при 8 доля вырожденных "
                         "квантуется шагом 12%% и порог 10%% значит «ноль из восьми»")
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
    ap.add_argument("--recompute", default=None,
                    help="стадии, которые считать заново несмотря на штамп: "
                         "all или через запятую pairs,vectors,sweep,matrix,compose. "
                         "Прежний артефакт не удаляется, а откладывается в .bak")
    ap.add_argument("--recompute-stale", action="store_true",
                    help="считать заново артефакты, снятые при других параметрах; "
                         "по умолчанию цепочка на них останавливается")
    return ap


def main() -> None:
    args = build_parser().parse_args()

    # Конфиг даёт умолчания, явный аргумент побеждает. Так добавление модели —
    # одна запись в configs/models/, а разовые правки остаются возможны.
    if args.config:
        cfg = load_model_config(args.config)
        # Что реально задано в командной строке: сравниваем с разбором пустых
        # аргументов. Разбор sys.argv не видел форму --opt=value, и
        # `--batch-size=32` молча проигрывал конфигу, вопреки RUNBOOK.
        defaults = vars(build_parser().parse_args([]))
        given = {k for k, v in vars(args).items()
                 if k in defaults and v != defaults[k]}
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

    # Тип вычислений — один на весь прогон, а не решение каждой стадии по
    # отдельности. Пишется в манифест и в отпечатки: иначе сравнивать строки,
    # снятые в fp16 и bf16, приходится по логам загрузчика.
    args.resolved_dtype, dtype_why = resolve_run_dtype(
        args.model, (args.model_config.get("load") or {}).get("dtype"), token)
    print(f"тип вычислений: {args.resolved_dtype} — {dtype_why}", flush=True)
    dtype_arg = ([] if args.resolved_dtype == "auto"
                 else ["--dtype", args.resolved_dtype])

    forced = set() if not args.recompute else (
        {"pairs", "vectors", "sweep", "matrix", "compose"} if args.recompute == "all"
        else {x.strip() for x in args.recompute.split(",")})

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
        pf = ([py, "-m", "emotion.preflight", "--model", args.model,
               "--layer", str(layer_hint)] + dtype_arg)
        # Коэффициент — рабочая точка модели, а не умолчательная восьмёрка:
        # Llama-3.2-3B работает на 4, и на 8 её текст разрушается, из-за чего
        # предполёт валил очередь на коэффициенте, который прогон не применяет.
        pf_coeff = meta_pre.get("op_coeff", args.coeff)
        pf += (["--strength", str(args.strength)] if args.strength is not None
               else ["--coeff", str(pf_coeff)])
        if "vectors" in forced:
            # Векторы сейчас будут пересчитаны — проверять наведение на тех,
            # что уедут в .bak, значит валить прогон из-за уже списанного.
            pf += ["--skip-steer"]
        print("0. предполёт …", flush=True)
        if sh(pf, log) != 0:
            raise SystemExit("предполёт не пройден — очередь остановлена, см. лог")

    meta_path = runs / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    specs = build_specs(args, meta)

    def run_stage(key: str, cmd: list[str], fail: str) -> None:
        """Стадия считается, если её отпечаток не совпал или она названа в --recompute."""
        s = specs[key]
        if key in forced:
            # Артефакт без штампа переиспользуется молча — значит просто запустить
            # цепочку заново недостаточно, готовое надо убрать с дороги. Не удаляем:
            # оно стоило часов и может понадобиться для сравнения.
            bak = set_aside(s.backup or s.artifact)
            if bak:
                print(f"   {s.stage}: прежнее отложено в {bak.name}", flush=True)
        elif not stamp.decide(s.artifact, s.stage, s.params, s.inputs,
                              args.recompute_stale, label=s.stage):
            return
        if sh(cmd, log) != 0:
            raise SystemExit(fail)
        stamp.write_stamp(s.artifact, s.stage, s.params, s.inputs)

    # 1. Пары pos/neg на самой модели (resume внутри скрипта, построчный)
    combined = pairs_dir / "all_emotions_extract.csv"
    print("1. пары pos/neg", flush=True)
    run_stage("pairs",
              [py, "-m", "eval.run_emotion_inference_batch", "--model", args.model,
               "--version", "extract", "--output_dir", str(pairs_dir),
               "--infer_backend", "hf", "--temperature", "0",
               "--max_tokens", str(args.max_tokens),
               "--batch-size", str(args.batch_size)] + dtype_arg,
              "стадия пар упала — см. лог")

    # 2. Векторы = mean(pos) − mean(neg) послойно.
    # Отпечаток стадии включает отпечаток пар: сменились пары — векторы
    # разойдутся и цепочка остановится. Так и вскрылась gemma, чьи векторы были
    # построены на текстах Qwen, а self-цикл этого не заметил.
    vec_probe = vec_dir / f"{ISEAR_EMOTIONS[0]}_response_avg_diff.pt"
    print("2. извлечение векторов", flush=True)
    if "vectors" not in forced and vec_probe.is_file() \
            and stamp.read_stamp(vec_dir) is None and combined.is_file() \
            and vec_probe.stat().st_mtime < combined.stat().st_mtime:
        # Наследство без штампа: старая проверка по времени всё ещё лучше, чем ничего.
        raise SystemExit(
            f"векторы в {vec_dir} старше пар {combined} — они построены на других "
            "данных. Убери или переименуй их и перезапусти, иначе матрица посчитается "
            "на чужих векторах."
        )
    cmd = [py, "-m", "emotion.extract_vectors", "--model_name", args.model,
           "--data-dir", str(pairs_dir), "--save-dir", str(vec_dir)] + dtype_arg
    if args.judge_scores:
        cmd += ["--judge-scores", str(args.judge_scores)]
    run_stage("vectors", cmd, "извлечение векторов упало — см. лог")

    # 3. Свип по слоям и выбор рабочей точки. Это две разные вещи: свип стоит
    # получаса генерации, выбор из готового свипа бесплатен. Поэтому смена
    # max_degen не должна тянуть за собой пересчёт свипа.
    sweep_csv = runs / "layer_sweep_anger.csv"
    if not sweep_csv.is_file() and "layer" in meta and "op_coeff" in meta:
        # Наследство: точка выбрана в более раннем цикле (у Qwen2.5-3B — вручную).
        layer, op_coeff = meta["layer"], meta["op_coeff"]
        print(f"3. рабочая точка: унаследована L{layer}/c{op_coeff:g}, свипа нет",
              flush=True)
    else:
        # Слои задаются долями глубины (0.45) или абсолютными номерами (13).
        # Доли переносимы между моделями, абсолютные привязаны к одной.
        if meta.get("candidates"):
            cands = meta["candidates"]
        elif args.layers:
            from transformers import AutoConfig
            from emotion.loader import resolve_layers
            cfg_model = AutoConfig.from_pretrained(args.model, token=token)
            cands = resolve_layers(cfg_model, [x.strip() for x in str(args.layers).split(",")])
        else:
            cands = default_layers(args.model, token)
        meta["candidates"] = cands
        specs = build_specs(args, meta)
        print(f"3. свип слоёв {cands}", flush=True)
        run_stage("sweep",
                  [py, "-m", "emotion.steer_eval", "--model_name", args.model,
                   "--emotion", "anger", "--vector-dir", str(vec_dir),
                   "--layers", ",".join(map(str, cands)), "--coeffs", args.sweep_coeffs,
                   "--n-prompts", str(args.sweep_prompts),
                   "--out", str(sweep_csv)] + dtype_arg,
                  "свип упал — см. лог")

        sweep_st = stamp.read_stamp(sweep_csv) or {}
        prev_op = meta.get("op_point") or {}
        same = (prev_op.get("sweep") == sweep_st.get("fingerprint")
                and prev_op.get("max_degen") == args.max_degen)
        if same:
            layer, op_coeff = prev_op["layer"], prev_op["coeff"]
            print(f"   рабочая точка: та же L{layer}/c{op_coeff:g}", flush=True)
        else:
            layer, op_coeff = pick_operating_point(sweep_csv, args.max_degen)
            print(f"   рабочая точка: слой {layer}, coeff {op_coeff:g}", flush=True)
        meta["op_point"] = {"layer": layer, "coeff": op_coeff,
                            "max_degen": args.max_degen,
                            "sweep": sweep_st.get("fingerprint")}

    meta.update({"model": args.model, "layer": layer, "op_coeff": op_coeff,
                 "max_degen": args.max_degen,
                 "coeff": args.coeff, "strength": args.strength,
                 "max_tokens": args.max_tokens,
                 "judge_filtered": bool(args.judge_scores),
                 "dtype": args.resolved_dtype,
                 "variant": args.variant,
                 "protocol": stamp.PROTOCOL_VERSION,
                 "config": args.model_config,
                 "env": env_manifest()})
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    specs = build_specs(args, meta)

    # 4. Матрица специфичности: baseline + 7 эмоций × 56 промптов, баллы энкодера
    matrix_csv = specs["matrix"].artifact
    print(f"4. матрица специфичности на L{layer}", flush=True)
    cmd4 = [py, "-m", "emotion.steer_specificity", "--model_name", args.model,
            "--vector-dir", str(vec_dir), "--layer", str(layer),
            "--per-emotion", str(args.per_emotion),
            "--save-answers", "--out", str(matrix_csv)] + dtype_arg
    cmd4 += (["--strength", str(args.strength)] if args.strength is not None
             else ["--coeff", str(op_coeff)])
    run_stage("matrix", cmd4, "матрица упала — см. лог")

    # 5. Композиция: сложение и вычитание эмоций - результат, который
    # держится на двух измерителях.
    # Запускается по флагу: стоит ещё столько же генераций, сколько матрица.
    if args.compose:
        cs = specs["compose"]
        print(f"5. композиция ({args.compose})", flush=True)
        if stamp.decide(cs.artifact, cs.stage, cs.params, cs.inputs,
                        args.recompute_stale, label=cs.stage):
            if sh([py, "-m", "emotion.steer_compose", "--model_name", args.model,
                   "--vector-dir", str(vec_dir), "--layer", str(layer),
                   "--coeff", str(op_coeff), "--per-emotion", str(args.per_emotion),
                   "--specs", args.compose, "--out", str(cs.artifact)] + dtype_arg,
                  log) != 0:
                print("   композиция упала — цепочка продолжается", flush=True)
            else:
                stamp.write_stamp(cs.artifact, cs.stage, cs.params, cs.inputs)

    # Что переиспользовано без штампа — в манифест. Строка такого прогона
    # сравнима с остальными только под честную оговорку в отчёте.
    legacy_used = stamp.unstamped([s.artifact for s in specs.values()])
    meta["unstamped"] = legacy_used
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    if legacy_used:
        print(f"   без штампа переиспользовано: {', '.join(legacy_used)}", flush=True)

    print(f"=== {args.slug}: цепочка пройдена → {runs} ===\n", flush=True)


if __name__ == "__main__":
    main()
