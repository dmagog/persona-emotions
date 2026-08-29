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
    from emotion.steer_specificity import load_checkpoint

    print("\n— возобновление матрицы")
    fields = ["steer", "layer", "coeff", "prompt_id", *ISEAR_EMOTIONS, "answer"]

    def write_ckpt(layer, coeff, tail=True) -> Path:
        p = Path(tempfile.mkdtemp()) / "m.partial.csv"
        with p.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields); w.writeheader()
            for st in ("baseline", "anger"):
                for pid in range(3):
                    w.writerow({"steer": st, "layer": "" if st == "baseline" else layer,
                                "coeff": 0.0 if st == "baseline" else coeff,
                                "prompt_id": pid, "answer": f"t {st} {pid}",
                                **{e: 0.1 for e in ISEAR_EMOTIONS}})
            if tail:
                fh.write("anger,14,4.0,3,0.1,0.1")  # обрыв на полуслове
        return p

    want = {"baseline": ("", 0.0), "anger": ("14", 4.0)}
    rows, done = load_checkpoint(write_ckpt(14, 4.0), fields, want)
    check(len(done) == 6 and len(rows) == 6, "готовые пары восстановлены", f"{len(done)}")
    check(("anger", 3) not in done, "оборванная строка будет пересчитана")

    # Коэффициент при --strength считается из активаций и гуляет в младших
    # разрядах — это та же точка, возобновляться можно.
    rows, done = load_checkpoint(write_ckpt(14, 4.0001), fields, want)
    check(len(done) == 6, "дрожь коэффициента не мешает возобновлению", f"{len(done)}")

    # А другой слой — уже другой прогон, дописывать к нему нельзя
    for label, ck in (("слой", write_ckpt(16, 4.0)), ("коэффициент", write_ckpt(14, 8.0))):
        try:
            load_checkpoint(ck, fields, want)
            check(False, f"чужой {label} в чекпойнте останавливает", "не остановил")
        except SystemExit as e:
            check("другой рабочей точки" in str(e), f"чужой {label} в чекпойнте останавливает")


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

    # Ключи входов в штампе обязаны быть переносимыми: артефакты считаются на
    # одной машине, а живут в git. Путь внутри репозитория — относительный.
    inner = stamp.REPO / "runs" / "_stamp_check"
    try:
        inner.mkdir(parents=True, exist_ok=True)
        probe = inner / "x.csv"
        probe.write_text("a\n", encoding="utf-8")
        digs = stamp.input_digests([probe])
        check(list(digs) == ["runs/_stamp_check/x.csv"],
              "вход внутри репо ключуется относительным путём", list(digs)[0])
    finally:
        import shutil
        shutil.rmtree(inner, ignore_errors=True)
    # os.path.isabs, а не startswith("/"): на Windows абсолютный путь начинается
    # с буквы диска, и юниксовая проверка валила самопроверку на 2070 — то есть
    # ворота перед каждой ночной очередью.
    import os
    digs = stamp.input_digests([src])
    check(os.path.isabs(list(digs)[0]),
          "вход вне репо остаётся абсолютным", list(digs)[0][:40])

    # Перенос с 2070: обрыв копирования даёт усечённый CSV, который читается —
    # в нём просто меньше строк. Глаз это не ловит, штамп ловит.
    total, bad = stamp.verify(tmp)
    check(total == 1 and not bad, "целый артефакт сверку проходит", f"{total}, {bad}")
    art.write_text("steer,score\n", encoding="utf-8")
    total, bad = stamp.verify(tmp)
    check(len(bad) == 1 and "разошлось" in bad[0], "усечённый артефакт не проходит сверку")


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

    # Пересчёт: готовое убирается с дороги, но не пропадает. Без этого стадия
    # либо переиспользует артефакт без штампа, либо «возобновится» по нему.
    from emotion.run_model_chain import set_aside
    tmp = Path(tempfile.mkdtemp())
    art = tmp / "steer_specificity_raw.csv"
    art.write_text("steer,score\nanger,1\n", encoding="utf-8")
    from emotion import stamp as st_mod
    st_mod.write_stamp(art, "matrix", {"layer": 10}, [])
    bak = set_aside(art)
    check(bak is not None and bak.exists() and not art.exists(),
          "прежний артефакт отложен, а не стёрт", bak.name if bak else "—")
    check(not st_mod.stamp_path(art).is_file(),
          "штамп уехал вместе с артефактом, не остался врать про пустое место")
    check(set_aside(tmp / "нет.csv") is None, "откладывать нечего — молча дальше")

    # Каталог пар откладывается целиком: стадия возобновляется по отдельным
    # <эмоция>_pos.csv, и снос одного сводного файла её не заставит считать заново.
    d = tmp / "pairs"; d.mkdir(); (d / "anger_pos.csv").write_text("x", encoding="utf-8")
    check(set_aside(d) is not None and not d.exists(), "каталог откладывается целиком")

    # Смена параметра обязана менять отпечаток, иначе проверка бесполезна
    from emotion.stamp import fingerprint
    fp1, _ = fingerprint("matrix", specs["matrix"].params, [])
    p2 = {**specs["matrix"].params, "per_emotion": 16}
    fp2, _ = fingerprint("matrix", p2, [])
    check(fp1 != fp2, "смена per_emotion меняет отпечаток")


def test_protocol() -> None:
    """Одна ли плоскость. Строки таблицы, снятые по-разному, обязаны это показывать."""
    from emotion.protocol import PLANE_KEYS, PROTOCOL, card, compare_planes, report

    print("\n— протокол и плоскость")
    check(all(c.where for c in PROTOCOL),
          "у каждого решения есть ссылка на статью или прочерк")
    check(any(c.deviation for c in PROTOCOL) and any(not c.deviation for c in PROTOCOL),
          "отклонения отделены от совпадений",
          f"{sum(c.deviation for c in PROTOCOL)} из {len(PROTOCOL)} — отклонения")
    check("2507.21509" in card(), "карточка называет базовую статью")

    same = {"A": {"protocol": 1, "n_prompts": 56, "drive": "coeff=8",
                  "judge_filtered": True, "max_degen": 0.10, "stamped": True}}
    same["B"] = dict(same["A"])
    check(not compare_planes(same), "одинаковые прогоны считаются сравнимыми")
    check("одной плоскости" in " ".join(report(same)), "и так и сказано в отчёте")

    odd = dict(same)
    odd["C"] = {**same["A"], "n_prompts": 14, "judge_filtered": False, "stamped": False}
    diff = compare_planes(odd)
    check(set(diff) == {"n_prompts", "judge_filtered"},
          "расхождения названы поимённо", ", ".join(sorted(diff)))
    txt = " ".join(report(odd))
    check("C" in txt and "Без штампа" in txt,
          "выбивающаяся строка и отсутствие штампа попадают в отчёт")
    check(all(k in PLANE_KEYS for k in diff), "все ключи плоскости описаны по-русски")

    # Слой и коэффициент у каждой модели свои — это НЕ повод считать строки
    # несравнимыми, иначе пометка загорится на всей таблице и её перестанут читать.
    byline = {"A": {**same["A"]}, "B": {**same["A"]}}
    check(not compare_planes(byline), "разные слои сами по себе не рушат плоскость")

    # Точность считалась по-разному, а в манифест не писалась — восстанавливаем
    # из лога загрузчика, иначе fp16- и bf16-прогоны выглядят одинаковыми.
    from emotion.protocol import dtype_of
    tmp = Path(tempfile.mkdtemp())
    check("dtype" in PLANE_KEYS, "точность входит в плоскость")
    check(dtype_of(tmp, {}) is None, "нет ни манифеста, ни лога — честное «неизвестно»")
    (tmp / "chain.log").write_text(
        "[loader] org/m: dtype=torch.bfloat16 — карта его поддерживает\n", encoding="utf-8")
    check(dtype_of(tmp, {}) == "bfloat16", "точность восстанавливается из лога")
    check(dtype_of(tmp, {"dtype": "float16"}) == "float16", "манифест важнее лога")


def test_judge_cache() -> None:
    """Кэш судьи обязан помнить, ЗА КАКОЙ текст выставлен балл."""
    import json

    from emotion.judge_specificity import _load_cache, answer_key

    print("\n— кэш судьи")
    tmp = Path(tempfile.mkdtemp()) / "j.cache.jsonl"
    old_ans, new_ans = "I am furious about this.", "I feel calm today."
    with tmp.open("w", encoding="utf-8") as fh:
        # запись нового образца — с отпечатком текста
        fh.write(json.dumps({"steer": "anger", "pid": "0", "measured": "anger",
                             "answer": answer_key(old_ans), "score": 90}) + "\n")
        # запись старого образца — без него
        fh.write(json.dumps({"steer": "anger", "pid": "1", "measured": "anger",
                             "score": 80}) + "\n")
    done = _load_cache(tmp)
    check(("anger", "0", "anger", answer_key(old_ans)) in done,
          "балл за тот же текст переиспользуется")
    check(("anger", "0", "anger", answer_key(new_ans)) not in done,
          "балл за другой текст НЕ переиспользуется — иначе судья опишет чужой прогон")
    check(len(done) == 1, "записи без отпечатка текста отбрасываются", f"{len(done)}")
    check(answer_key(old_ans) != answer_key(new_ans), "разные тексты — разные ключи")


def test_judge_denominator() -> None:
    """Недомер судьи не должен выдаваться за промах."""
    import csv as csvmod

    from emotion.collect_results import MIN_JUDGED, _argmax_hits_from_wide
    from emotion.space import ISEAR_EMOTIONS

    print("\n— знаменатель судьи")
    tmp = Path(tempfile.mkdtemp()) / "judge_wide.csv"
    fields = ["steer", "prompt_id", *ISEAR_EMOTIONS]

    def write(sizes: dict) -> Path:
        with tmp.open("w", newline="", encoding="utf-8") as fh:
            w = csvmod.DictWriter(fh, fieldnames=fields); w.writeheader()
            for pid in range(56):
                w.writerow({"steer": "baseline", "prompt_id": pid,
                            **{e: 10.0 for e in ISEAR_EMOTIONS}})
            for emo, n in sizes.items():
                for pid in range(n):
                    sc = {e: 10.0 for e in ISEAR_EMOTIONS}
                    sc[emo] = 90.0  # целевая эмоция растёт — это попадание
                    w.writerow({"steer": emo, "prompt_id": pid, **sc})
        return tmp

    full = {e: 56 for e in ISEAR_EMOTIONS}
    hits, measured, _ = _argmax_hits_from_wide(write(full))
    check((hits, measured) == (7, 7), "полная матрица — 7 из 7", f"{hits}/{measured}")

    # granite: shame потерян целиком, sadness сведён к двум ответам
    thin = {**full, "sadness": 2}
    del thin["shame"]
    hits, measured, sizes = _argmax_hits_from_wide(write(thin))
    check(measured == 5, "условия с горсткой ответов выпадают из знаменателя",
          f"измерено {measured} из 7")
    check(hits == 5, "недомер не засчитывается промахом", f"{hits} попаданий")
    check(sizes.get("shame", 0) == 0 and sizes["sadness"] == 2,
          "размеры условий возвращаются для оговорки")
    check(MIN_JUDGED >= 20, "порог достаточности осмысленный", f"{MIN_JUDGED}")


def test_compose_collector() -> None:
    """Композиция: полная матрица пар и честная метрика подавления."""
    import csv as csvmod

    from emotion.collect_compose import pair_counts
    from emotion.space import ALL_PAIRS, ISEAR_EMOTIONS

    print("\n— сборщик композиции")
    check(len(ALL_PAIRS) == 42, "42 упорядоченные пары X-Y", f"{len(ALL_PAIRS)}")
    check("anger-sadness" in ALL_PAIRS and "sadness-anger" in ALL_PAIRS,
          "пары упорядочены (обе стороны считаются)")

    tmp = Path(tempfile.mkdtemp()) / "compose_allpairs.csv"
    fields = ["steer", "prompt_id", *ISEAR_EMOTIONS]
    with tmp.open("w", newline="", encoding="utf-8") as fh:
        w = csvmod.DictWriter(fh, fieldnames=fields); w.writeheader()
        def row(steer, pid, **sc):
            r = {"steer": steer, "prompt_id": pid, **{e: 0.1 for e in ISEAR_EMOTIONS}}
            r.update(sc); w.writerow(r)
        for pid in range(56):
            row("baseline", pid)
            # X в одиночку поднимает и X, и родственную Y (протечка)
            row("anger", pid, anger=0.8, sadness=0.4)
            # X-Y: цель держится, вычитаемая ниже уровня «X в одиночку»
            row("anger-sadness", pid, anger=0.7, sadness=0.1)
        # остальные пары — заполнить, чтобы условие не считалось тонким
        for spec in ALL_PAIRS:
            if spec == "anger-sadness": continue
            for pid in range(56):
                row(spec, pid)
        for e in ISEAR_EMOTIONS:
            if e == "anger": continue
            for pid in range(56):
                row(e, pid)

    c = pair_counts(tmp)
    check(c is not None and c["n"] == 42, "считаются все 42 пары", f"{c['n'] if c else '—'}")
    # anger-sadness: цель 0.7 > baseline 0.1 -> target; вычит. 0.1 < X-в-одиночку 0.4 -> suppress
    check(c["target"] >= 1, "рост цели над baseline засчитан")
    check(c["suppress"] >= 1,
          "подавление считается против «X в одиночку», а не против baseline")

    # тонкое условие (мало строк) не должно попадать в счёт
    with tmp.open("a", newline="", encoding="utf-8") as fh:
        w = csvmod.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        # затираем: перезапишем файл с одним тонким условием
    thin = Path(tempfile.mkdtemp()) / "compose_allpairs.csv"
    with thin.open("w", newline="", encoding="utf-8") as fh:
        w = csvmod.DictWriter(fh, fieldnames=fields); w.writeheader()
        for pid in range(56):
            w.writerow({"steer": "baseline", "prompt_id": pid, **{e: 0.1 for e in ISEAR_EMOTIONS}})
        for pid in range(5):  # anger-sadness всего 5 строк — тонкое
            w.writerow({"steer": "anger-sadness", "prompt_id": pid, **{e: 0.1 for e in ISEAR_EMOTIONS}})
    ct = pair_counts(thin)
    check("anger-sadness" in ct["thin"], "условие тоньше 40 строк выпадает из счёта")


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

    # Тип вычислений должен доезжать до КАЖДОЙ стадии, которая грузит модель.
    # Пары этого не получали: load_model брал свой умолчательный bf16, и внутри
    # одного прогона пары оказывались в bf16, а активации в fp16.
    chain = (REPO / "emotion" / "run_model_chain.py").read_text(encoding="utf-8")
    passed_on = chain.count("+ dtype_arg")
    check(passed_on >= 4, "тип вычислений передан всем четырём стадиям",
          f"стадий с явным типом: {passed_on}")
    pairs_src = (REPO / "eval" / "run_emotion_inference_batch.py").read_text(encoding="utf-8")
    check("load_model(args.model, dtype=" in pairs_src,
          "генератор пар грузит модель с явным типом")
    check(re.search(r"load_model\(args\.model\)\s*$", pairs_src, re.M) is None,
          "нигде не осталось загрузки пар без типа")


def main() -> None:
    sys.path.insert(0, str(REPO))
    print("=== самопроверка конвейера ===")
    for fn in (test_layers_and_dtype, test_steerer_guards, test_operating_point,
               test_degeneracy_metric, test_matrix_resume, test_stamp,
               test_stage_specs, test_protocol, test_judge_cache, test_judge_denominator, test_compose_collector,
               test_config,
               test_prompt_consistency):
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
