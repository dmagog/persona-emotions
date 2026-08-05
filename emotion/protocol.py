"""Протокол: что делает статья, что делаем мы, и где расходимся сознательно.

Не документация, а исполняемое описание. Прозу забывают — так уже потерялась
нормировка при сравнении направлений и вернулась граблями через два месяца.
Здесь каждое решение стоит рядом со ссылкой на статью, а сводная таблица
проверяет, что её строки сняты в одной плоскости.

База: *Persona Vectors: Monitoring and Controlling Character Traits in Language
Models*, arXiv 2507.21509v3.

Usage:
    python -m emotion.protocol                # карточка: статья против нас
    python -m emotion.protocol --check runs   # в одной ли плоскости строки сводки
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from emotion import stamp
from emotion.space import ISEAR_EMOTIONS

REPO = Path(__file__).resolve().parent.parent
PAPER = "Persona Vectors, arXiv 2507.21509v3"


@dataclass
class Choice:
    """Одно решение протокола: как в статье, как у нас, почему."""
    name: str
    paper: str
    ours: str
    where: str
    note: str = ""
    deviation: bool = False


PROTOCOL: list[Choice] = [
    Choice(
        "Пулинг активаций",
        "response avg — среднее по токенам ответа",
        "response avg",
        "§A.3, рис. 11",
        "Статья сравнила prompt last / prompt avg / response avg и выбрала третий."),
    Choice(
        "Формула наведения",
        "h_ℓ ← h_ℓ + α·v_ℓ, вектор СЫРОЙ",
        "h_ℓ + coeff·v_ℓ, вектор сырой",
        "§3.2",
        "Вектор в основном пайплайне статьи не нормируется. См. «Нормировка» ниже."),
    Choice(
        "Нормировка вектора",
        "только там, где сравниваются РАЗНЫЕ направления: позиции токенов (§A.3) "
        "и проекция при мониторинге (v̂, §3.3). В наведении — нет",
        "не нормируем при наведении; при `--center` перенормируем к исходной длине",
        "§A.3, §3.3",
        "Правило: сравниваешь направления — уравнивай нормы, сравниваешь модели — "
        "уравнивай эффект. Наш документ раньше приписывал статье нормировку перед "
        "наведением со ссылкой на B.4 — это неверно, B.4 про выбор слоя."),
    Choice(
        "Позиции при наведении",
        "на каждом шаге декодирования",
        "positions=\"all\"",
        "§3.2",
        "Не last-position-only: это один из подозреваемых в баге научрука."),
    Choice(
        "Выбор слоя",
        "наведение на КАЖДОМ слое с ОДНИМ коэффициентом, берётся слой с "
        "максимальным баллом выраженности",
        "три кандидата от глубины {0.35, 0.45, 0.55}, сетка коэффициентов "
        "{0,4,8,16}, берётся сильнейшая точка при доле вырожденных ≤ 10%",
        "§B.4",
        "Полный свип по слоям у нас не влезает в ночь на 2070. Ограничение по "
        "вырожденности добавлено потому, что наивный максимум балла выбирал слой "
        "с 38% брака: вырожденный повтор энкодер читает как сильную эмоцию.",
        deviation=True),
    Choice(
        "Нумерация слоёв",
        "с единицы; «слой 20» — выход 20-го блока",
        "с нуля; для блока B берётся vec[B+1] в hidden_states",
        "§B.4, сноска",
        "Наш «слой 10» — это «слой 11» в терминах статьи. При переносе чисел в "
        "статью нумерацию сдвигать, иначе повторим off-by-one.",
        deviation=True),
    Choice(
        "Фильтр обучающих пар",
        "per-response: балл > 50 у положительной инструкции и < 50 у "
        "отрицательной; судья GPT-4.1-mini, агрегация по top-20 логитам",
        "парный судья: один балл на пару «насколько A эмоциональнее B», порог 60; "
        "llama-3.3-70b через OpenRouter, разбор числа из текста",
        "§2, §B.1",
        "Парное сравнение чище одиночных баллов: обе стороны про один сценарий. "
        "Logprob-агрегация недоступна — OpenRouter их не отдаёт.",
        deviation=True),
    Choice(
        "Датасет под модель",
        "пары генерирует сама целевая модель",
        "то же: self-цикл на каждую модель",
        "§2",
        "Векторы с чужого текста работают, но имеют другую норму и другой выбор "
        "слоя. Старый вариант сохранён как отдельная строка абляции переноса."),
    Choice(
        "Декодирование",
        "не оговорено",
        "greedy, temperature=0 на всех стадиях",
        "—",
        "Против рекомендации Qwen (там просят сэмплинг). Держим ради "
        "воспроизводимости: вектор — разность средних, сэмплинг добавляет в неё шум. "
        "Контроль — колонка вырожденности.",
        deviation=True),
    Choice(
        "Измеритель эффекта",
        "LLM-судья по выраженности черты",
        "независимый энкодер SamLowe/roberta-base-go_emotions + тот же LLM-судья",
        "§B.1",
        "Два измерителя вместо одного: расходятся (энкодер 4/7, судья 6/7 на "
        "Qwen3-1.7B), поэтому в отчёте оба столбца."),
]


# --- одна ли плоскость ------------------------------------------------------

# Поля, при расхождении которых строки таблицы нельзя ставить рядом. Слой и
# коэффициент сюда НЕ входят: они у каждой модели свои по построению, в этом
# и смысл подбора рабочей точки.
PLANE_KEYS: dict[str, str] = {
    "protocol": "версия протокола",
    "n_prompts": "промптов на условие",
    "drive": "чем задано наведение",
    "judge_filtered": "фильтр пар судьёй",
    "max_degen": "порог вырожденности при выборе точки",
}


def plane_of(run_dir: Path, csv_path: Path) -> dict:
    """Плоскость, в которой снята строка.

    Где можно — из самих данных, а не из манифеста: число промптов и режим
    наведения читаются из матрицы, и их нельзя рассинхронизировать с ней.
    """
    meta = {}
    meta_path = run_dir / "meta.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    st = stamp.read_stamp(csv_path) or {}

    d = pd.read_csv(csv_path)
    steered = d[d["steer"].isin(ISEAR_EMOTIONS)] if "steer" in d.columns else d
    n_prompts = int(steered.groupby("steer")["prompt_id"].nunique().max()) \
        if not steered.empty and "prompt_id" in d.columns else None

    # В плоскость идёт РЕЖИМ наведения, а не значение коэффициента. Значение у
    # каждой модели своё по построению — рабочая точка подбирается по поведению,
    # в этом и смысл. А вот «одну модель вели коэффициентом, другую безразмерной
    # силой» — уже разные протоколы.
    drive = "?"
    if "coeff" in d.columns and not steered.empty:
        used = {round(float(c), 3) for c in steered["coeff"] if str(c).strip()}
        drive = "coeff" if len(used) == 1 else "strength"

    return {
        "protocol": st.get("protocol", meta.get("protocol")),
        "n_prompts": n_prompts,
        "drive": drive,
        "judge_filtered": meta.get("judge_filtered"),
        "max_degen": meta.get("max_degen"),
        "stamped": bool(st),
        "op_at_edge": at_grid_edge(run_dir, meta),
    }


def at_grid_edge(run_dir: Path, meta: dict) -> str | None:
    """Не упёрлась ли рабочая точка в край сетки коэффициентов.

    Если выбран максимум сетки, оптимум мог остаться за ней — сетку надо
    расширять, иначе «сильнейшее неразрушающее наведение» означает всего лишь
    «самое сильное, что мы попробовали». Если выбран минимум — наоборот,
    ограничение по вырожденности связывает, и модель хрупкая.
    """
    sweep = run_dir / "layer_sweep_anger.csv"
    op = meta.get("op_coeff")
    if not sweep.is_file() or op is None:
        return None
    try:
        grid = sorted({float(c) for c in pd.read_csv(sweep)["coeff"] if float(c) > 0})
    except (KeyError, ValueError, TypeError):
        return None
    if not grid:
        return None
    if abs(float(op) - grid[-1]) < 1e-9:
        return f"верх сетки ({grid[-1]:g}): оптимум мог остаться за ней"
    if abs(float(op) - grid[0]) < 1e-9 and len(grid) > 1:
        return f"низ сетки ({grid[0]:g}): всё, что сильнее, разрушало текст"
    return None


def compare_planes(planes: dict[str, dict]) -> dict[str, dict]:
    """Какие поля плоскости разошлись между строками и как именно."""
    out: dict[str, dict] = {}
    for key in PLANE_KEYS:
        seen: dict = {}
        for name, p in planes.items():
            seen.setdefault(p.get(key), []).append(name)
        if len(seen) > 1:
            out[key] = seen
    return out


def report(planes: dict[str, dict]) -> list[str]:
    """Текст про сопоставимость — идёт под таблицу как есть."""
    lines: list[str] = []
    unstamped = [n for n, p in planes.items() if not p.get("stamped")]
    edges = {n: p["op_at_edge"] for n, p in planes.items() if p.get("op_at_edge")}
    diff = compare_planes(planes)

    if not planes:
        return ["Нет строк для сравнения."]
    if not diff and not unstamped and not edges:
        lines.append(f"Все {len(planes)} строк сняты в одной плоскости — сравнимы между собой.")
        return lines
    if not diff:
        lines.append(f"Плоскость общая у всех {len(planes)} строк. Оговорки ниже.")
        lines.append("")

    if diff:
        lines.append("**Строки сняты в разных плоскостях — рядом их ставить нельзя:**")
        lines.append("")
        for key, seen in diff.items():
            variants = "; ".join(
                f"{v if v is not None else 'неизвестно'} — {', '.join(sorted(names))}"
                for v, names in sorted(seen.items(), key=lambda kv: str(kv[0])))
            lines.append(f"- {PLANE_KEYS[key]}: {variants}")
        lines.append("")
    if edges:
        lines.append("**Рабочая точка упёрлась в край сетки коэффициентов:**")
        lines.append("")
        for name, why in sorted(edges.items()):
            lines.append(f"- {name}: {why}")
        lines.append("")
    if unstamped:
        lines.append(
            f"**Без штампа протокола:** {', '.join(sorted(unstamped))}. "
            "Чем сняты — известно только со слов манифеста. Подтвердить: "
            "`python -m emotion.stamp runs/<slug> --adopt --config configs/models/<slug>.yaml`")
        lines.append("")
    return lines


def card() -> str:
    """Карточка протокола. Годится и в статью, и в пакет для лаборатории."""
    out = [f"# Протокол\n\nБаза: {PAPER}.\n",
           "| Решение | В статье | У нас | Где |",
           "|---|---|---|---|"]
    for c in PROTOCOL:
        mark = " ⚠" if c.deviation else ""
        out.append(f"| {c.name}{mark} | {c.paper} | {c.ours} | {c.where} |")
    out.append("\n⚠ — сознательное отклонение от статьи.\n")
    for c in PROTOCOL:
        if c.note:
            out.append(f"**{c.name}.** {c.note}\n")
    return "\n".join(out)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Протокол: карточка и проверка плоскости.")
    ap.add_argument("--check", type=Path, default=None,
                    help="каталог runs/: проверить, в одной ли плоскости строки")
    args = ap.parse_args()

    if args.check:
        from emotion.collect_results import variants_in
        planes = {}
        for run_dir in sorted(args.check.iterdir()):
            if not run_dir.is_dir():
                continue
            for csv_path in variants_in(run_dir):
                planes[f"{run_dir.name}/{csv_path.stem.replace('steer_specificity_', '')}"] = \
                    plane_of(run_dir, csv_path)
        print("\n".join(report(planes)))
        return

    print(card())


if __name__ == "__main__":
    main()
