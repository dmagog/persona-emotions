"""Деэскалация в диалоге: подавление anger/fear и его влияние на ответ модели.

Расширяет оценку с одноходовых ответов на многоходовой контекст. На вход -
провокационные диалоги (конфликт, обвинение, эмоциональная эскалация),
обрывающиеся на реплике пользователя; модель генерирует следующий ответ
ассистента под разными наведениями, и мы смотрим, обостряет он конфликт или
гасит.

Условия отличаются от стадии композиции. Там спека `X-Y` это vec[X] - vec[Y]:
наводит X и попутно давит Y. Здесь нужно ЧИСТОЕ подавление, то есть наведение
отрицательным вектором `-anger` без наведения чего-либо ещё.

Позитивный контроль `+anger` включён намеренно: если минус гасит эскалацию, а
плюс её усиливает, эффект направленный по одной оси, а не общее размягчение
ответа. Односторонний результат такого не показывает.

Usage (GPU):
    python -m emotion.steer_dialog_safety \
        --model_name tiiuae/Falcon3-3B-Instruct \
        --vector-dir emotion_vectors/Falcon3-3B-Instruct \
        --layer 12 --coeff 6 --dtype float16 \
        --out runs/Falcon3-3B-Instruct/dialog_safety.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch

from activation_steer import ActivationSteerer
from emotion.classifier_encoder import ClassifierBasedEncoder
from emotion.loader import LoadSpec, load_model_and_tokenizer
from emotion.space import ISEAR_EMOTIONS
from emotion.steer_specificity import encode_all

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

REPO = Path(__file__).resolve().parent.parent

# Условие -> как собрать вектор из одиночных. Знак задаётся здесь, коэффициент
# остаётся положительным и общим - как в стадии композиции, чтобы рабочая точка
# была та же самая.
CONDITIONS: dict[str, list[tuple[float, str]]] = {
    "baseline": [],
    "-anger": [(-1.0, "anger")],
    "-fear": [(-1.0, "fear")],
    "-anger-fear": [(-1.0, "anger"), (-1.0, "fear")],
    "+anger": [(+1.0, "anger")],  # позитивный контроль
}


def build_dialog_prompt(tokenizer, turns: list[dict]) -> str:
    """Многоходовой промпт из chat-шаблона модели.

    Системную роль не используем: шаблон gemma-2 её отвергает, а тут она и не
    нужна - контекст несут сами реплики. enable_thinking гасит режим рассуждений
    у Qwen 3, но не всякий шаблон принимает этот аргумент.
    """
    messages = [{"role": t["role"], "content": t["content"]} for t in turns]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)


def generate(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    """Жадное декодирование, как во всех остальных стадиях."""
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Деэскалация в диалоге через подавление anger/fear.")
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--vector-dir", required=True, type=Path)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--coeff", type=float, required=True)
    ap.add_argument("--dialogs", type=Path,
                    default=REPO / "data_generation" / "deescalation_dialogs.json")
    ap.add_argument("--conditions", default=",".join(CONDITIONS),
                    help="через запятую; по умолчанию все пять")
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--dtype", default="float16", help="должен совпадать с прогоном модели")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    dialogs = json.loads(args.dialogs.read_text(encoding="utf-8"))["dialogs"]
    conds = [c.strip() for c in args.conditions.split(",") if c.strip()]
    unknown = [c for c in conds if c not in CONDITIONS]
    if unknown:
        raise SystemExit(f"неизвестные условия: {unknown}; известны {list(CONDITIONS)}")
    print(f"диалогов {len(dialogs)}, условий {len(conds)} -> {len(dialogs) * len(conds)} генераций",
          flush=True)

    model, tokenizer, _ = load_model_and_tokenizer(
        LoadSpec(hf_id=args.model_name, dtype=args.dtype))
    encoder = ClassifierBasedEncoder()

    needed = {emo for c in conds for _, emo in CONDITIONS[c]}
    vecs = {
        emo: torch.load(args.vector_dir / f"{emo}_response_avg_diff.pt",
                        map_location="cpu")[args.layer + 1]
        for emo in sorted(needed)
    }

    prompts = [build_dialog_prompt(tokenizer, d["turns"]) for d in dialogs]

    fields = ["condition", "dialog_id", "category", *ISEAR_EMOTIONS, "answer"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Пишем по мере готовности: удалённый прогон не должен терять всё из-за
    # падения на последнем условии.
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for cond in conds:
            parts = CONDITIONS[cond]
            vec = None
            if parts:
                vec = sum((sign * vecs[emo] for sign, emo in parts),
                          torch.zeros_like(vecs[parts[0][1]]))
            for d, prompt in zip(dialogs, prompts):
                if vec is None:
                    ans = generate(model, tokenizer, prompt, args.max_new_tokens)
                else:
                    with ActivationSteerer(model, vec, coeff=args.coeff,
                                           layer_idx=args.layer, positions="all"):
                        ans = generate(model, tokenizer, prompt, args.max_new_tokens)
                writer.writerow({"condition": cond, "dialog_id": d["id"],
                                 "category": d["category"], "answer": ans,
                                 **encode_all(encoder, ans)})
            fh.flush()
            print(f"  готово условие {cond}", flush=True)

    print(f"записано: {args.out}", flush=True)


if __name__ == "__main__":
    main()
