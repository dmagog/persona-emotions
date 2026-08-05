"""Самопроверка логики конвейера. Без GPU, без сети, за секунды.

Гоняет то, на чём конвейер уже ломался: разбор конфига, выбор слоёв и dtype,
отбор рабочей точки, метрику вырожденности, возобновление матрицы, поиск
путей к слоям и предохранители вектора.

Запускать перед каждым коммитом в конвейер:
    python -m emotion.selftest
"""
from __future__ import annotations

import csv
import re
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parent.parent
FAILED: list[str] = []


def check(cond: bool, name: str, extra: str = "") -> None:
    print(f"  [{'OK ' if cond else 'СБОЙ'}] {name}" + (f" — {extra}" if extra else ""))
    if not cond:
        FAILED.append(name)


class Cfg:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def test_layers_and_dtype() -> None:
    from emotion.loader import resolve_dtype, resolve_layers, num_layers, hidden_size

    print("\n— слои и dtype")
    flat = Cfg(num_hidden_layers=28, hidden_size=3072)
    nested = Cfg(vision_config=Cfg(num_hidden_layers=27),
                 text_config=Cfg(num_hidden_layers=36, hidden_size=2048))

    check(num_layers(flat) == 28, "слои из плоского конфига")
    check(num_layers(nested) == 36, "слои из вложенного, не из vision", num_layers(nested))
    check(hidden_size(nested) == 2048, "hidden_size из вложенного")

    # доли должны воспроизводить кандидатов, которые задавались вручную
    check(resolve_layers(Cfg(num_hidden_layers=28), [0.35, 0.45, 0.55]) == [10, 13, 15],
          "доли на 28 слоях = [10,13,15]")
    check(resolve_layers(Cfg(num_hidden_layers=26), [0.35, 0.45, 0.55]) == [9, 12, 14],
          "доли на 26 слоях = [9,12,14]")
    check(resolve_layers(flat, ["10", "13", "15"]) == [10, 13, 15], "номера строками")
    check(resolve_layers(flat, [12, 12, 0.45]) == [12, 13], "дубликаты схлопываются")
    try:
        resolve_layers(flat, [99]); check(False, "слой вне диапазона падает")
    except ValueError:
        check(True, "слой вне диапазона падает")

    d, _ = resolve_dtype("float16", Cfg())
    check(d is torch.float16, "явный dtype из конфига")
    d, why = resolve_dtype("auto", Cfg(torch_dtype=torch.bfloat16))
    check(d in (torch.bfloat16, torch.float16) and why, "auto объясняет выбор", why[:60])
    try:
        resolve_dtype("float8", Cfg()); check(False, "неизвестный dtype падает")
    except ValueError:
        check(True, "неизвестный dtype падает")


def test_steerer_guards() -> None:
    from activation_steer import ActivationSteerer, _hidden_size

    print("\n— предохранители наведения")

    class Fake(nn.Module):
        def __init__(self, hidden=8, path="model.layers", nested=False):
            super().__init__()
            blocks = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(4)])
            if path == "model.layers":
                self.model = nn.Module(); self.model.layers = blocks
            else:
                self.model = nn.Module(); self.model.language_model = nn.Module()
                self.model.language_model.layers = blocks
            self.config = (Cfg(text_config=Cfg(hidden_size=hidden)) if nested
                           else Cfg(hidden_size=hidden))

    check(_hidden_size(Cfg(text_config=Cfg(hidden_size=32))) == 32, "hidden_size из вложенного конфига")
    m = Fake()
    try:
        ActivationSteerer(m, torch.ones(8), layer_idx=1); check(True, "путь model.layers")
    except Exception as e:
        check(False, "путь model.layers", str(e)[:60])
    try:
        ActivationSteerer(Fake(path="mm"), torch.ones(8), layer_idx=1)
        check(True, "путь мультимодальных")
    except Exception as e:
        check(False, "путь мультимодальных", str(e)[:60])

    for vec, name, must in [
        (torch.full((8,), float("nan")), "NaN отвергается", "nan"),
        (torch.zeros(8), "нулевой вектор отвергается", "нулев"),
        (torch.ones(5), "неверная размерность отвергается", "≠"),
    ]:
        try:
            ActivationSteerer(m, vec, layer_idx=1); check(False, name, "не упал")
        except ValueError as e:
            check(must.lower() in str(e).lower(), name)


def test_operating_point() -> None:
    import pandas as pd
    from emotion.run_model_chain import pick_operating_point

    print("\n— выбор рабочей точки")
    GOOD = "A calm and quite distinct sentence about the driving lesson, number {i}, varied."
    BAD = "i am i am i am i am i am i am i am i am i am"
    rows = []
    spec = {(10, 4): (0.10, 0.0), (10, 8): (0.20, 0.0), (10, 16): (0.35, 0.5),
            (13, 4): (0.15, 0.0), (13, 8): (0.30, 0.0), (13, 16): (0.55, 0.6),
            (15, 4): (0.18, 0.2), (15, 8): (0.40, 0.7), (15, 16): (0.60, 0.9)}
    for (L, c), (score, degen) in spec.items():
        for i in range(10):
            rows.append({"emotion": "anger", "layer": L, "coeff": float(c), "prompt_id": i,
                         "target_score": score,
                         "answer": BAD if i < degen * 10 else GOOD.format(i=i)})
    tmp = Path(tempfile.mkdtemp()) / "sweep.csv"
    pd.DataFrame(rows).to_csv(tmp, index=False)
    layer, coeff = pick_operating_point(tmp, max_degen=0.10)
    check((layer, coeff) == (13, 8.0),
          "берётся сильнейшая точка среди неразрушающих", f"L{layer}/c{coeff:g}")


def test_degeneracy_metric() -> None:
    from emotion.collect_results import rep_ratio

    print("\n— метрика вырожденности")
    short_ok = "I'm scared. I can't breathe. I just want to run away now."
    long_ok = ("My phone buzzed with a text from Sarah saying she was coming over. "
               "I was surprised, she'd been busy lately, so I wasn't expecting her.")
    bad = "we are the best best best best best best best best best best best"
    check(rep_ratio(short_ok) == 0.0, "короткий нормальный текст не вырожден", f"{rep_ratio(short_ok):.3f}")
    check(rep_ratio(long_ok) < 0.05, "длинный нормальный текст не вырожден", f"{rep_ratio(long_ok):.3f}")
    check(rep_ratio(bad) > 0.3, "повтор распознаётся", f"{rep_ratio(bad):.3f}")


def test_matrix_resume() -> None:
    from emotion.space import ISEAR_EMOTIONS

    print("\n— возобновление матрицы")
    fields = ["steer", "layer", "coeff", "prompt_id", *ISEAR_EMOTIONS, "answer"]
    ckpt = Path(tempfile.mkdtemp()) / "m.partial.csv"
    with ckpt.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields); w.writeheader()
        for st in ("baseline", "anger"):
            for pid in range(3):
                w.writerow({"steer": st, "layer": 14, "coeff": 4.0, "prompt_id": pid,
                            "answer": f"t {st} {pid}", **{e: 0.1 for e in ISEAR_EMOTIONS}})
        fh.write("anger,14,4.0,3,0.1,0.1")  # обрыв на полуслове

    done, rows = set(), []
    with open(ckpt, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            try:
                row = {k: (float(v) if k in ISEAR_EMOTIONS else v)
                       for k, v in r.items() if k in fields}
                key = (r["steer"], int(r["prompt_id"]))
            except (KeyError, ValueError, TypeError):
                continue
            rows.append(row); done.add(key)
    check(len(done) == 6 and len(rows) == 6, "готовые пары восстановлены", f"{len(done)}")
    check(("anger", 3) not in done, "оборванная строка будет пересчитана")


def test_stamp() -> None:
    """Отпечаток протокола: что артефакт годен, должно быть проверяемо."""
    from emotion import stamp

    print("\n— отпечаток протокола")
    tmp = Path(tempfile.mkdtemp())
    src = tmp / "pairs.csv"
    src.write_text("a,b\n1,2\n", encoding="utf-8")
    art = tmp / "matrix.csv"
    art.write_text("steer,score\nanger,1\n", encoding="utf-8")
    params = {"model": "Qwen/Qwen3-1.7B", "layer": 10, "coeff": 8.0}

    check(stamp.check(art, "matrix", params, [src]).state == "unstamped",
          "готовый артефакт без штампа не выдаётся за свой")
    stamp.write_stamp(art, "matrix", params, [src])
    check(stamp.check(art, "matrix", params, [src]).state == "current",
          "свой артефакт узнаётся")

    # Главное отличие от проверки по времени: содержимое то же — значит годен.
    # Файлы ездят между Windows-боксом и Mac, где mtime не переживает копирования.
    import os
    os.utime(src, (0, 0))
    check(stamp.check(art, "matrix", params, [src]).state == "current",
          "смена времени правки не делает артефакт чужим")

    v = stamp.check(art, "matrix", {**params, "layer": 13}, [src])
    check(v.state == "stale" and any("layer" in c for c in v.changed),
          "смена слоя ловится и называется", "; ".join(v.changed)[:60])

    src.write_text("a,b\n1,3\n", encoding="utf-8")
    v = stamp.check(art, "matrix", params, [src])
    check(v.state == "stale" and any("вход" in c for c in v.changed),
          "изменение входа ловится", "; ".join(v.changed)[:60])

    # Правка самого артефакта руками — тоже расхождение, а не «свой файл»
    src.write_text("a,b\n1,2\n", encoding="utf-8")
    art.write_text("steer,score\nanger,999\n", encoding="utf-8")
    v = stamp.check(art, "matrix", params, [src])
    check(v.state == "stale" and any("изменён" in c for c in v.changed),
          "правка артефакта руками ловится", "; ".join(v.changed)[:60])
    stamp.write_stamp(art, "matrix", params, [src])

    check(stamp.check(tmp / "нет.csv", "matrix", params).state == "missing",
          "отсутствующий артефакт считается заново")
    empty = tmp / "vectors"
    empty.mkdir()
    check(stamp.check(empty, "vectors", params).state == "missing",
          "пустой каталог — не результат")

    # decide: разошёлся — цепочка останавливается, а не решает за человека
    try:
        stamp.decide(art, "matrix", {**params, "layer": 13}, [])
        check(False, "разошедшийся артефакт останавливает цепочку", "не остановил")
    except SystemExit as e:
        check("layer" in str(e), "разошедшийся артефакт останавливает цепочку")
    check(stamp.decide(art, "matrix", {**params, "layer": 13}, [],
                       recompute_stale=True) is True,
          "с --recompute-stale считается заново")
    src.write_text("a,b\n1,4\n", encoding="utf-8")
    try:
        stamp.decide(art, "matrix", params, [src])
        check(False, "изменившийся вход тоже останавливает", "не остановил")
    except SystemExit:
        check(True, "изменившийся вход тоже останавливает")


def test_stage_specs() -> None:
    """Стадии и их параметры описаны в одном месте — иначе штамп начнёт врать."""
    from emotion.run_model_chain import build_parser, build_specs, load_model_config

    print("\n— спецификации стадий")
    cfg_file = REPO / "configs" / "models" / "qwen3-1.7b.yaml"
    a = build_parser().parse_args([])
    cfg = load_model_config(cfg_file)
    for k, v in cfg.items():
        if k != "_raw" and hasattr(a, k):
            setattr(a, k, v)
    a.model_config = cfg.get("_raw", {})

    specs = build_specs(a, {"layer": 10, "op_coeff": 8.0, "candidates": [10, 13, 15]})
    check(set(specs) >= {"pairs", "vectors", "sweep", "matrix"},
          "описаны все дорогие стадии", ", ".join(sorted(specs)))
    check(specs["matrix"].params["layer"] == 10 and specs["matrix"].params["coeff"] == 8.0,
          "рабочая точка входит в отпечаток матрицы")
    check(specs["vectors"].inputs == [specs["pairs"].artifact],
          "векторы зависят от пар")
    check(all(s.artifact for s in specs.values()), "у каждой стадии есть артефакт")

    # Смена параметра обязана менять отпечаток, иначе проверка бесполезна
    from emotion.stamp import fingerprint
    fp1, _ = fingerprint("matrix", specs["matrix"].params, [])
    p2 = {**specs["matrix"].params, "per_emotion": 16}
    fp2, _ = fingerprint("matrix", p2, [])
    check(fp1 != fp2, "смена per_emotion меняет отпечаток")


def test_config() -> None:
    from emotion.run_model_chain import load_model_config

    print("\n— конфиги моделей")
    cfgs = sorted((REPO / "configs" / "models").glob("*.yaml"))
    check(bool(cfgs), "конфиги на месте", f"{len(cfgs)} шт")
    for f in cfgs:
        c = load_model_config(f)
        ok = c.get("model") and c.get("slug") and c.get("layers")
        check(bool(ok), f"разбирается {f.name}", f"{c.get('model')} → {c.get('slug')}")


def test_prompt_consistency() -> None:
    """Промпт снятия вектора должен собираться так же, как промпт применения."""
    print("\n— согласованность промптов")
    src = {}
    for f in ("eval/run_emotion_inference_batch.py", "emotion/steer_eval.py",
              "emotion/steer_specificity.py", "generate_vec.py"):
        p = REPO / f
        if p.is_file():
            src[f] = p.read_text(encoding="utf-8")
    bare = [f for f, s in src.items()
            if re.search(r'tokenizer\(\s*prompt\s*,\s*return_tensors="pt"\s*\)', s)]
    check(not bare, "нет токенизации без add_special_tokens", ", ".join(bare) or "все явные")
    thinking = [f for f, s in src.items()
                if "apply_chat_template" in s and "enable_thinking" not in s]
    check(not thinking, "везде подавлены рассуждения", ", ".join(thinking) or "во всех местах")


def main() -> None:
    sys.path.insert(0, str(REPO))
    print("=== самопроверка конвейера ===")
    for fn in (test_layers_and_dtype, test_steerer_guards, test_operating_point,
               test_degeneracy_metric, test_matrix_resume, test_stamp,
               test_stage_specs, test_config, test_prompt_consistency):
        try:
            fn()
        except Exception as e:
            check(False, f"{fn.__name__} упал", f"{type(e).__name__}: {str(e)[:90]}")

    print()
    if FAILED:
        print(f"СБОЕВ: {len(FAILED)}")
        for f in FAILED:
            print(f"  - {f}")
        sys.exit(1)
    print("всё чисто")


if __name__ == "__main__":
    main()
