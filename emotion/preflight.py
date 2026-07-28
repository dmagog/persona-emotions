"""Предполётная проверка: вся цепочка на одной строке, до запуска длинной очереди.

Ловит именно тот класс ошибок, который стоил дня работы: код отрабатывает
успешно, но данные получаются негодные, и это выясняется через пять часов.

Проверяет по порядку:
  1. шаблон чата — системная роль или запасной путь, режим рассуждений подавлен;
  2. генерация пары pos/neg — непустая, без тегов рассуждений, без вырождения;
  3. извлечение вектора — форма, конечность, ненулевая норма;
  4. масштаб активаций — считает силу S и предупреждает о крайних коэффициентах;
  5. наведение — стирённый текст отличается от исходного и не вырожден.

Любая осечка — ненулевой код возврата и внятная причина.

Usage:
    python -m emotion.preflight --model Qwen/Qwen3-1.7B --layer 13 --strength 0.33
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from activation_steer import ActivationSteerer
from emotion.steer_eval import build_prompt, generate, load_eval_prompts

REPO = Path(__file__).resolve().parent.parent
FAILS: list[str] = []


def check(ok: bool, name: str, detail: str = "") -> bool:
    print(f"  [{'OK ' if ok else 'СБОЙ'}] {name}" + (f" — {detail}" if detail else ""), flush=True)
    if not ok:
        FAILS.append(f"{name}: {detail}")
    return ok


def rep_ratio(text: str, n: int = 4) -> float:
    """Доля повторяющихся n-грамм; устойчива к длине текста."""
    w = re.findall(r"\w+", str(text).lower())
    if len(w) < n + 4:
        return 0.0
    grams = [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]
    return 1.0 - len(set(grams)) / len(grams)


def main() -> None:
    ap = argparse.ArgumentParser(description="Предполётная проверка цепочки на одной строке.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--strength", type=float, default=None)
    ap.add_argument("--coeff", type=float, default=8.0)
    ap.add_argument("--max-new-tokens", type=int, default=120)
    args = ap.parse_args()

    import os
    token = os.environ.get("HF_TOKEN")
    print(f"\n=== предполётная проверка: {args.model}, слой {args.layer} ===", flush=True)

    # --- 1. шаблон чата ---
    tok = AutoTokenizer.from_pretrained(args.model, token=token)
    sys_txt, user_txt = "You are a person.", "Describe a small everyday event."
    transport = "system-роль"
    try:
        rendered = tok.apply_chat_template(
            [{"role": "system", "content": sys_txt}, {"role": "user", "content": user_txt}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except Exception as e:
        if "system" not in str(e).lower():
            check(False, "шаблон чата", f"неожиданная ошибка: {type(e).__name__}: {e}")
            sys.exit(1)
        transport = "рамка свёрнута в user-turn"
        rendered = tok.apply_chat_template(
            [{"role": "user", "content": f"{sys_txt}\n\n{user_txt}"}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
    check(sys_txt in rendered, "шаблон чата", transport)
    # рассуждения: допустим только пустой блок-заглушка
    think_open = rendered.count("<think>")
    check(think_open == 0 or "<think>\n\n</think>" in rendered or "<think></think>" in rendered,
          "режим рассуждений подавлен", f"тегов <think>: {think_open}")

    # --- 2. генерация ---
    model = AutoModelForCausalLM.from_pretrained(
        args.model, device_map="auto", torch_dtype=torch.float16, token=token)
    q = load_eval_prompts("anger", 1)[0]
    prompt = build_prompt(tok, q)
    base = generate(model, tok, prompt, args.max_new_tokens)
    check(len(base.strip()) > 20, "генерация непустая", f"{len(base)} символов")
    check("<think>" not in base, "в ответе нет тегов рассуждений")
    check(rep_ratio(base) <= 0.15, "исходный текст не вырожден", f"повтор 4-граммы {rep_ratio(base):.3f}")

    # --- 3. вектор ---
    vec_dir = REPO / "emotion_vectors" / args.model.split("/")[-1]
    alt = list((REPO / "emotion_vectors").glob(f"*{args.model.split('/')[-1]}*"))
    if not vec_dir.is_dir() and alt:
        vec_dir = alt[0]
    vec_file = vec_dir / "anger_response_avg_diff.pt"
    if not vec_file.is_file():
        check(False, "вектор на месте", f"нет {vec_file} — цепочка построит его сама, проверка наведения пропущена")
        print("\nИТОГ: базовые проверки пройдены, наведение не проверено (нет векторов).")
        sys.exit(1 if FAILS else 0)

    vec_all = torch.load(vec_file, map_location="cpu")
    n_layers = model.config.num_hidden_layers
    check(tuple(vec_all.shape) == (n_layers + 1, model.config.hidden_size),
          "форма вектора", f"{tuple(vec_all.shape)}, ожидалось {(n_layers + 1, model.config.hidden_size)}")
    vec = vec_all[args.layer + 1]
    check(torch.isfinite(vec).all().item(), "вектор конечен")
    check(vec.norm().item() > 1e-3, "норма вектора ненулевая", f"||vec||={vec.norm().item():.2f}")

    # --- 4. масштаб активаций и сила ---
    with torch.no_grad():
        enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
        h = model(**enc, output_hidden_states=True).hidden_states[args.layer + 1][0]
    mean_h = h.norm(dim=-1).mean().item()
    if args.strength is not None:
        coeff = args.strength * mean_h / vec.norm().item()
        check(0.1 <= coeff <= 200, "коэффициент в разумных пределах",
              f"S={args.strength} → coeff={coeff:.2f} при mean||h||={mean_h:.1f}")
    else:
        coeff = args.coeff
        S = coeff * vec.norm().item() / mean_h
        check(0.05 <= S <= 1.0, "сила наведения в разумных пределах",
              f"coeff={coeff} → S={S:.3f} (ориентир 0.33)")

    # --- 5. наведение ---
    with ActivationSteerer(model, vec, coeff=coeff, layer_idx=args.layer, positions="all"):
        steered = generate(model, tok, prompt, args.max_new_tokens)
    check(steered.strip() != base.strip(), "наведение меняет текст")
    check(rep_ratio(steered) <= 0.15, "стирённый текст не вырожден",
          f"повтор 4-граммы {rep_ratio(steered):.3f}")

    print(f"\n  исходный : {base.strip()[:110]}")
    print(f"  стирённый: {steered.strip()[:110]}")

    if FAILS:
        print(f"\nИТОГ: {len(FAILS)} сбоев — запускать очередь нельзя:", file=sys.stderr)
        for f in FAILS:
            print(f"  - {f}", file=sys.stderr)
        sys.exit(1)
    print("\nИТОГ: цепочка проходит на одной строке, можно запускать очередь.")


if __name__ == "__main__":
    main()
