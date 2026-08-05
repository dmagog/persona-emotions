"""Отпечаток протокола: артефакт годен, только если снят теми же параметрами.

Пропуск стадии по факту существования файла — это тихий пересчёт наоборот.
Стадия видит `steer_specificity_raw.csv`, пропускает себя и отдаёт дальше числа,
снятые другой моделью, другим числом токенов или на другом слое. Заметить это по
самим числам нельзя: файл выглядит своим.

Отпечаток — sha256 от канонического JSON: стадия, версия протокола, параметры,
влияющие на результат, и отпечатки входных артефактов. Лежит рядом с артефактом:
`<файл>.stamp.json` или `<каталог>/.stamp.json`.

Три исхода вместо двух:

* совпал      — стадия пропускается, это и есть «не пересчитывать»;
* разошёлся   — стадия НЕ пересчитывается молча и НЕ переиспользуется молча:
                цепочка останавливается и говорит, какие параметры разошлись;
* штампа нет  — наследство прошлых прогонов. Тоже не пересчитываем: артефакт
                стоил часов. Переиспользуем с предупреждением и отметкой
                `unstamped` в манифесте, пока человек не подтвердит его
                командой `--adopt`.

Usage:
    python -m emotion.stamp --show runs/Qwen3-1.7B          # чем снят каждый артефакт
    python -m emotion.stamp --adopt runs/Qwen3-1.7B --config configs/models/qwen3-1.7b.yaml
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

# Версия протокола. Поднимать, когда меняется СМЫСЛ стадии, а не её параметры:
# другая индексация слоёв, другой пулинг, другой набор промптов. Все прежние
# артефакты станут несовпадающими — это и требуется.
PROTOCOL_VERSION = 1

STAMP_NAME = ".stamp.json"
# Что не участвует в отпечатке каталога: служебное и промежуточное.
SKIP = ("__pycache__", ".DS_Store", STAMP_NAME)


def stamp_path(artifact: Path) -> Path:
    """Где лежит штамп: внутри каталога или рядом с файлом."""
    return artifact / STAMP_NAME if artifact.is_dir() else \
        artifact.with_name(artifact.name + ".stamp.json")


def present(artifact: Path) -> bool:
    """Есть ли результат. Пустой каталог — не результат.

    Каталог векторов остаётся после упавшего прогона: он существует, в нём ноль
    файлов, и проверка «а есть ли путь» пропустила бы стадию.
    """
    if artifact.is_dir():
        return any(p.is_file() and p.name != STAMP_NAME for p in artifact.rglob("*"))
    return artifact.exists()


def _sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def content_digest(path: Path) -> str:
    """Отпечаток содержимого. Каталог — по именам и содержимому файлов.

    Именно содержимого, а не времени правки: артефакты ездят между Windows-боксом
    и Mac, а `scp` без `-p` время не сохраняет. Проверка по mtime на таком
    маршруте даёт либо ложное «векторы старше пар» и лишние часы генерации, либо
    ложную свежесть.
    """
    if not path.exists():
        return ""
    if path.is_file():
        return _sha_file(path)
    parts = []
    for p in sorted(path.rglob("*")):
        if not p.is_file() or any(s in p.parts or p.name == s for s in SKIP):
            continue
        if p.name.endswith(".partial.csv"):
            continue  # промежуточный чекпойнт — не часть результата
        parts.append(f"{p.relative_to(path).as_posix()}:{_sha_file(p)}")
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


def _canon(obj) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)


def input_digests(inputs: Sequence[Path]) -> dict[str, str]:
    """Отпечатки входов: штамп входа, если он есть, иначе его содержимое.

    Штамп предпочтительнее — через него изменение дальнего входа доходит до
    последней стадии (поменялись пары → поменялись векторы → матрица разошлась).
    Содержимое добавляется всегда: артефакт мог быть отредактирован после штампа.
    """
    out: dict[str, str] = {}
    for p in inputs:
        key = p.as_posix()
        prev = read_stamp(p)
        content = content_digest(p)
        out[key] = f"{prev['fingerprint']}+{content[:16]}" if prev else f"content:{content[:16]}"
    return out


def fingerprint(stage: str, params: dict, inputs: Sequence[Path]) -> tuple[str, dict]:
    """Отпечаток стадии и словарь входов, из которого он собран."""
    digs = input_digests(inputs)
    payload = _canon({"stage": stage, "protocol": PROTOCOL_VERSION,
                      "params": params, "inputs": digs})
    return hashlib.sha256(payload.encode()).hexdigest()[:16], digs


def read_stamp(artifact: Path) -> dict | None:
    sp = stamp_path(artifact)
    if not sp.is_file():
        return None
    try:
        return json.loads(sp.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def write_stamp(artifact: Path, stage: str, params: dict,
                inputs: Sequence[Path] = ()) -> dict:
    """Записать штамп рядом с артефактом. Вызывать сразу после успешной стадии."""
    fp, digs = fingerprint(stage, params, inputs)
    data = {"stage": stage, "protocol": PROTOCOL_VERSION, "fingerprint": fp,
            "params": params, "inputs": digs,
            # Отпечаток самого артефакта: штамп отвечает не только «чем снят»,
            # но и «тот ли это файл». Иначе правка руками остаётся невидимой до
            # следующей стадии, а в отчёт числа попадают раньше.
            "content": content_digest(artifact),
            "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
    stamp_path(artifact).write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return data


def diff_params(old: dict, new: dict) -> list[str]:
    """Что разошлось, человеческим языком — это попадёт в сообщение об остановке."""
    out = []
    for k in sorted(set(old) | set(new)):
        a, b = old.get(k, "—"), new.get(k, "—")
        if a != b:
            out.append(f"{k}: было {a!r}, стало {b!r}")
    return out


@dataclass
class Verdict:
    """Решение по артефакту. `state` — что делать, `reason` — что сказать человеку."""
    state: str                       # missing | current | stale | unstamped
    reason: str = ""
    changed: list[str] = field(default_factory=list)

    @property
    def reusable(self) -> bool:
        return self.state in ("current", "unstamped")


def check(artifact: Path, stage: str, params: dict,
          inputs: Sequence[Path] = ()) -> Verdict:
    """Годен ли готовый артефакт под текущий протокол."""
    if not present(artifact):
        return Verdict("missing", "артефакта нет")
    prev = read_stamp(artifact)
    if prev is None:
        return Verdict("unstamped",
                       "артефакт без штампа: снят до введения отпечатков или "
                       "перенесён с другой машины")
    fp, digs = fingerprint(stage, params, inputs)
    now = content_digest(artifact)
    if prev.get("fingerprint") == fp and prev.get("content", now) == now:
        return Verdict("current", "отпечаток совпал")

    changed = []
    if prev.get("content", now) != now:
        changed.append("сам артефакт изменён после снятия штампа")
    if prev.get("protocol") != PROTOCOL_VERSION:
        changed.append(f"версия протокола: была {prev.get('protocol')}, "
                       f"стала {PROTOCOL_VERSION}")
    changed += diff_params(prev.get("params") or {}, params)
    for k in sorted(set(prev.get("inputs") or {}) | set(digs)):
        a, b = (prev.get("inputs") or {}).get(k), digs.get(k)
        if a != b:
            changed.append(f"вход {k}: пересчитан или изменён")
    if not changed:
        changed.append("отпечаток разошёлся, но параметры совпадают — "
                       "проверь версию протокола")
    return Verdict("stale", "артефакт снят при других параметрах", changed)


def decide(artifact: Path, stage: str, params: dict, inputs: Sequence[Path] = (),
           recompute_stale: bool = False, label: str | None = None) -> bool:
    """Печатает решение и возвращает True, если стадию надо считать.

    Разошедшийся артефакт по умолчанию останавливает цепочку. Так задумано:
    молча пересчитать — потерять ночь генерации, молча переиспользовать —
    получить в отчёте строку, снятую по другому протоколу. Выбор за человеком:
    `--recompute-stale` или убрать файл.
    """
    name = label or artifact.name
    v = check(artifact, stage, params, inputs)
    if v.state == "current":
        print(f"   {name}: отпечаток совпал, пропуск", flush=True)
        return False
    if v.state == "unstamped":
        print(f"   {name}: ВНИМАНИЕ, штампа нет — переиспользую как есть.\n"
              f"      Если он снят текущим протоколом, подтверди:\n"
              f"      python -m emotion.stamp --adopt {artifact.parent}", flush=True)
        return False
    if v.state == "stale":
        detail = "\n".join(f"      · {c}" for c in v.changed)
        if recompute_stale:
            print(f"   {name}: параметры разошлись, считаю заново\n{detail}", flush=True)
            return True
        raise SystemExit(
            f"\n{name}: артефакт снят при других параметрах.\n{detail}\n\n"
            f"  Он стоил времени, поэтому цепочка не трогает его сама. Выбери:\n"
            f"    · пересчитать    — запустить с --recompute-stale\n"
            f"    · сохранить      — переименовать {artifact} и запустить снова\n"
            f"    · признать своим — python -m emotion.stamp --adopt {artifact.parent}\n"
        )
    return True


# --- усыновление и осмотр ----------------------------------------------------

def adopt(artifact: Path, stage: str, params: dict,
          inputs: Sequence[Path] = ()) -> dict | None:
    """Проштамповать существующий артефакт текущим протоколом.

    Явное действие человека: «да, этот файл снят вот этими параметрами».
    Нужно для прогонов, посчитанных до введения штампов, — чтобы не гонять
    заново четыре с половиной часа ради одной подписи.
    """
    if not present(artifact):
        return None
    return write_stamp(artifact, stage, params, inputs)


def show(run_dir: Path) -> None:
    """Чем снят каждый артефакт прогона. Первое, что смотреть при разборе строки."""
    found = False
    for sp in sorted({*run_dir.rglob("*stamp.json"), *run_dir.rglob(STAMP_NAME)}):
        try:
            d = json.loads(sp.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        found = True
        target = sp.parent.name if sp.name == STAMP_NAME else sp.name[:-len(".stamp.json")]
        print(f"\n{target}  [{d.get('stage')}] {d.get('fingerprint')}  "
              f"протокол {d.get('protocol')}  {d.get('created', '')}")
        for k, v in sorted((d.get("params") or {}).items()):
            print(f"    {k}: {v}")
        for k, v in sorted((d.get("inputs") or {}).items()):
            print(f"    ← {k}  {v}")
    if not found:
        print(f"{run_dir}: штампов нет — прогон снят до введения отпечатков")


def verify(run_dir: Path) -> tuple[int, list[str]]:
    """Совпадает ли содержимое артефактов с их штампами.

    Нужно после переноса: артефакты едут с 2070 на Mac, и обрыв копирования
    даёт усечённый CSV, который открывается и читается — просто в нём меньше
    строк. Штамп это ловит, глаз нет.
    """
    bad: list[str] = []
    total = 0
    for sp in sorted({*run_dir.rglob("*stamp.json"), *run_dir.rglob(STAMP_NAME)}):
        total += 1
        try:
            d = json.loads(sp.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            bad.append(f"{sp.name}: штамп не читается")
            continue
        art = sp.parent if sp.name == STAMP_NAME else \
            sp.with_name(sp.name[:-len(".stamp.json")])
        if not present(art):
            bad.append(f"{art.name}: штамп есть, артефакта нет")
            continue
        want = d.get("content")
        if want is None:
            continue  # штамп старого образца, без отпечатка содержимого
        if content_digest(art) != want:
            bad.append(f"{art.name}: содержимое разошлось со штампом "
                       "(обрыв копирования или правка руками)")
    return total, bad


def unstamped(paths: Iterable[Path]) -> list[str]:
    """Артефакты без штампа — их отмечает манифест и помечает сводная таблица."""
    return [p.name for p in paths if present(p) and read_stamp(p) is None]


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Отпечатки протокола у артефактов прогона.")
    ap.add_argument("run", type=Path, help="каталог runs/<slug>")
    ap.add_argument("--show", action="store_true", help="показать штампы (по умолчанию)")
    ap.add_argument("--adopt", action="store_true",
                    help="проштамповать готовые артефакты текущим конфигом")
    ap.add_argument("--verify", action="store_true",
                    help="сверить содержимое артефактов со штампами (после переноса)")
    ap.add_argument("--config", type=Path, default=None,
                    help="конфиг модели: нужен для --adopt")
    args = ap.parse_args()

    if args.verify:
        total, bad = verify(args.run)
        for b in bad:
            print(f"  СБОЙ {b}")
        if bad:
            raise SystemExit(f"{args.run}: доехало не всё — {len(bad)} из {total}")
        print(f"{args.run}: сверено артефактов {total}, все совпадают со штампами")
        return

    if args.adopt:
        from emotion.run_model_chain import stage_specs
        if not args.config:
            raise SystemExit("--adopt требует --config: штамп должен знать, "
                             "какими параметрами снят артефакт")
        n = 0
        for spec in stage_specs(args.config):
            if adopt(spec.artifact, spec.stage, spec.params, spec.inputs):
                print(f"  проштампован {spec.artifact}")
                n += 1
        print(f"усыновлено артефактов: {n}")
        return

    show(args.run)


if __name__ == "__main__":
    main()
