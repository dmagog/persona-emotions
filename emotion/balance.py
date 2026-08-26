"""Остаток на OpenRouter: снимок до и после судейского прогона.

Судейские стадии — единственное, что стоит денег, и кончаются они молча:
провайдер начинает отвечать ошибкой 402, строки просто не попадают в
матрицу, а таблица показывает результат, посчитанный на подвыборке. Так у
нас уже пропало условие shame у granite. Дешевле спросить баланс до старта.

Usage:
    python3 -m emotion.balance                    # текущий остаток
    python3 -m emotion.balance --need 13.5        # хватит ли на прогон
"""
from __future__ import annotations

import argparse
import json
import os
import urllib.request

URL = "https://openrouter.ai/api/v1/credits"


def fetch() -> dict | None:
    """Остаток по ключу из окружения. None, если ключа нет или сеть молчит."""
    key = os.environ.get("OPENAI_API_KEY")
    if not key or not key.startswith("sk-or-"):
        return None  # ключ не от OpenRouter — считать баланс нечем
    req = urllib.request.Request(URL, headers={"Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            d = json.load(r)["data"]
    except Exception:
        return None
    bought, spent = float(d["total_credits"]), float(d["total_usage"])
    return {"bought": bought, "spent": spent, "left": bought - spent}


def line(prefix: str = "баланс") -> str:
    b = fetch()
    if b is None:
        return f"{prefix}: не удалось узнать (нет ключа OpenRouter или сеть)"
    return f"{prefix}: остаток ${b['left']:.2f} (куплено ${b['bought']:.2f}, потрачено ${b['spent']:.2f})"


def main() -> None:
    ap = argparse.ArgumentParser(description="Остаток на OpenRouter.")
    ap.add_argument("--need", type=float, default=None,
                    help="сколько нужно на прогон; при нехватке код возврата 1")
    args = ap.parse_args()
    b = fetch()
    print(line())
    if b is None or args.need is None:
        return
    if b["left"] < args.need:
        raise SystemExit(
            f"не хватает: нужно ${args.need:.2f}, есть ${b['left']:.2f}. "
            f"Пополнить или сузить прогон.")
    print(f"на прогон ${args.need:.2f} хватает, останется ${b['left'] - args.need:.2f}")


if __name__ == "__main__":
    main()
