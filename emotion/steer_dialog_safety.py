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
import re
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

# Именованные условия пилота. Любое другое условие разбирается на лету
# (см. parse_condition), поэтому можно просить -sadness, +joy и т.п.
CONDITIONS: dict[str, list[tuple[float, str]]] = {
    "baseline": [],
    "-anger": [(-1.0, "anger")],
    "-fear": [(-1.0, "fear")],
    "-anger-fear": [(-1.0, "anger"), (-1.0, "fear")],
    "+anger": [(+1.0, "anger")],  # позитивный контроль
}


# Контроль на общее возмущение: случайное направление с той же нормой, что у
# опорной эмоции. Без него нельзя отличить <удаление эмоционального
# направления> от <любой толчок такой величины>.
RANDOM_RE = re.compile(r"^random(\d+)$")


def parse_condition(spec: str) -> list[tuple[float, str]]:
    """'-anger' -> [(-1,anger)]; '+joy' -> [(+1,joy)]; '-anger-fear' -> обе с минусом.

    Знак обязателен у каждой эмоции: 'anger' без знака отвергается, иначе
    неясно, наводим мы эмоцию или давим.
    """
    if spec == "baseline" or RANDOM_RE.match(spec):
        return []  # у random вектор строится отдельно, не из эмоций
    if spec in CONDITIONS:
        return CONDITIONS[spec]
    parts = re.findall(r"([+-])([a-z]+)", spec)
    if not parts or "".join(s + e for s, e in parts) != spec:
        raise SystemExit(f"не разобрать условие {spec!r}: нужен знак перед каждой эмоцией")
    for _, emo in parts:
        if emo not in ISEAR_EMOTIONS:
            raise SystemExit(f"неизвестная эмоция {emo!r} в условии {spec!r}; "
                             f"известны {list(ISEAR_EMOTIONS)}")
    return [(-1.0 if sign == "-" else 1.0, emo) for sign, emo in parts]


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
    ap.add_argument("--random-match", default="anger",
                    help="эмоция, по норме которой масштабируются условия randomN")
    ap.add_argument("--no-encoder", action="store_true",
                    help="не считать баллы эмоций (колонки останутся пустыми)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    dialogs = json.loads(args.dialogs.read_text(encoding="utf-8"))["dialogs"]
    conds = [c.strip() for c in args.conditions.split(",") if c.strip()]
    parsed = {c: parse_condition(c) for c in conds}  # падает сразу на плохом условии
    print(f"диалогов {len(dialogs)}, условий {len(conds)} -> {len(dialogs) * len(conds)} генераций",
          flush=True)

    model, tokenizer, _ = load_model_and_tokenizer(
        LoadSpec(hf_id=args.model_name, dtype=args.dtype))
    # Энкодер необязателен. Генерация на GPU - дорогая и невосполнимая часть, и
    # она не должна пропадать из-за того, что классификатор не скачался. Баллы
    # эмоций доставляются позже, судья от них не зависит.
    encoder = None
    if not args.no_encoder:
        try:
            encoder = ClassifierBasedEncoder()
        except Exception as exc:
            print(f"ВНИМАНИЕ: энкодер недоступен ({type(exc).__name__}), колонки эмоций "
                  f"останутся пустыми: {str(exc)[:140]}", flush=True)

    needed = {emo for c in conds for _, emo in parsed[c]}
    if any(RANDOM_RE.match(c) for c in conds):
        needed.add(args.random_match)  # опорная норма для случайных направлений
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
            parts = parsed[cond]
            vec = None
            rnd = RANDOM_RE.match(cond)
            if rnd:
                ref = vecs[args.random_match]
                gen = torch.Generator().manual_seed(int(rnd.group(1)))
                r = torch.randn(ref.shape, generator=gen, dtype=torch.float32)
                vec = (r / r.norm() * ref.float().norm()).to(ref.dtype)
                print(f"  {cond}: случайное направление, норма {float(vec.norm()):.3f} "
                      f"(как у {args.random_match})", flush=True)
            elif parts:
                vec = sum((sign * vecs[emo] for sign, emo in parts),
                          torch.zeros_like(vecs[parts[0][1]]))
            for d, prompt in zip(dialogs, prompts):
                if vec is None:
                    ans = generate(model, tokenizer, prompt, args.max_new_tokens)
                else:
                    with ActivationSteerer(model, vec, coeff=args.coeff,
                                           layer_idx=args.layer, positions="all"):
                        ans = generate(model, tokenizer, prompt, args.max_new_tokens)
                scores = encode_all(encoder, ans) if encoder is not None else {}
                writer.writerow({"condition": cond, "dialog_id": d["id"],
                                 "category": d["category"], "answer": ans, **scores})
            fh.flush()
            print(f"  готово условие {cond}", flush=True)

    print(f"записано: {args.out}", flush=True)


if __name__ == "__main__":
    main()
