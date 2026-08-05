"""Steering specificity: steer toward each emotion, measure ALL 7 emotion scores.

On a fixed common set of neutral prompts, we generate a baseline (no steering)
and then, for each emotion's vector, a steered generation; every response is
encoded into all 7 ISEAR scores. From the CSV we build the steering specificity
matrix (steer-toward X vs measured Y): the diagonal should rise most if steering
is emotion-specific; off-diagonal entries are leakage (expected, given the high
cross-emotion vector cosines).

Usage (GPU):
    python -m emotion.steer_specificity --vector-dir emotion_vectors/Qwen2.5-3B-Instruct \
        --layer 14 --coeff 8 --per-emotion 2 --out results/steer_specificity_qwen2.5-3b.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from emotion.loader import LoadSpec, load_model_and_tokenizer

from activation_steer import ActivationSteerer
from emotion.classifier_encoder import ClassifierBasedEncoder
from emotion.space import ISEAR_EMOTIONS
from emotion.steer_eval import TASK_SUFFIX, build_prompt, generate


def common_prompts(per_emotion: int) -> list[str]:
    """Pool a few held-out questions from each emotion into one neutral set."""
    prompts: list[str] = []
    for emo in ISEAR_EMOTIONS:
        qs = json.loads(Path(f"data_generation/emotion_data_eval/{emo}.json").read_text())["questions"]
        prompts.extend(qs[:per_emotion])
    return prompts


def encode_all(encoder: ClassifierBasedEncoder, text: str) -> dict[str, float]:
    vec = encoder.encode(text)
    return {label: round(float(vec[i]), 4) for i, label in enumerate(encoder.space.labels)}


def pos_instruction(emotion: str) -> str:
    """First positive (emotion-eliciting) instruction for the prompting baseline."""
    instr = json.loads(Path(f"data_generation/emotion_data_eval/{emotion}.json").read_text())["instruction"]
    return instr[0]["pos"]


def build_instr_prompt(tokenizer, question: str, instruction: str) -> str:
    content = f"{instruction}\n\n{question}{TASK_SUFFIX}"
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False
    )


def mean_activation_norm(model, tokenizer, prompts, layer: int, n: int = 8) -> float:
    """Средняя норма скрытого состояния на слое — масштаб активаций модели.

    Нужна, чтобы сила наведения была сопоставима между моделями: у разных
    архитектур и hidden_size активации живут в разных масштабах, поэтому
    одинаковый coeff означает разное по силе вмешательство.
    """
    norms = []
    with torch.no_grad():
        for prompt in prompts[:n]:
            enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
            out = model(**enc, output_hidden_states=True)
            h = out.hidden_states[layer + 1][0]  # та же индексация, что у векторов
            norms.append(h.norm(dim=-1).mean().item())
    return sum(norms) / len(norms) if norms else float("nan")


def _same_point(got: tuple[str, float], want: tuple[str, float]) -> bool:
    """Одна ли это рабочая точка. Слой — точно, коэффициент — с допуском.

    При `--strength` коэффициент считается из средней нормы активаций, и между
    запусками он может разойтись в младших разрядах. Требовать точного совпадения
    значило бы отказываться возобновлять полностью законный прогон.
    """
    if got[0] != want[0]:
        return False
    return abs(got[1] - want[1]) <= 1e-3 * max(1.0, abs(want[1]))


def load_checkpoint(ckpt: Path, fields: list[str],
                    expected: dict[str, tuple[str, float]]) -> tuple[list[dict], set]:
    """Готовые генерации из чекпойнта — только снятые в текущей рабочей точке.

    Ключ возобновления раньше был `(steer, prompt_id)`, а слой и коэффициент
    в него не входили. Перезапуск на другом слое поверх недосчитанного файла
    дописывал новые строки к старым, и в одной матрице оказывались строки с
    разных слоёв. Заметить это по числам нельзя — они просто усредняются.
    """
    rows: list[dict] = []
    done: set[tuple[str, int]] = set()
    mismatch: dict[str, tuple[tuple[str, float], int]] = {}
    with open(ckpt, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            try:
                # Строку собираем ПЕРВОЙ: если она оборвана, пара не должна
                # попасть в done — иначе генерация пропустится при возобновлении,
                # а данных по ней не будет, и в матрице окажется дыра.
                row = {k: (float(v) if k in ISEAR_EMOTIONS else v)
                       for k, v in r.items() if k in fields}
                key = (r["steer"], int(r["prompt_id"]))
                got = (str(r.get("layer") or "").strip(), float(r.get("coeff") or 0.0))
            except (KeyError, ValueError, TypeError):
                continue  # оборванная последняя строка
            want = expected.get(key[0])
            if want is not None and not _same_point(got, want):
                prev = mismatch.get(key[0])
                mismatch[key[0]] = (got, (prev[1] if prev else 0) + 1)
                continue
            rows.append(row)
            done.add(key)
    if mismatch:
        detail = "\n".join(
            f"    · {s}: снято L{g[0] or '—'}/c{g[1]:g} × {n} строк, "
            f"сейчас L{expected[s][0] or '—'}/c{expected[s][1]:g}"
            for s, (g, n) in sorted(mismatch.items()))
        raise SystemExit(
            f"\n{ckpt}: в чекпойнте строки с другой рабочей точки.\n{detail}\n\n"
            "  Это остаток другого прогона. Дописать к нему новые строки — значит\n"
            "  собрать матрицу из разных слоёв, и по числам это будет не видно.\n"
            "  Убери или переименуй файл и запусти снова.\n"
        )
    return rows, done


def main() -> None:
    ap = argparse.ArgumentParser(description="Steering specificity matrix across all emotions.")
    ap.add_argument("--model_name", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--dtype", default="auto",
                    help="auto выбирает по железу и dtype обучения модели")
    ap.add_argument("--vector-dir", required=True, type=Path)
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--coeff", type=float, default=8.0)
    ap.add_argument("--strength", type=float, default=None,
                    help="безразмерная сила наведения S: coeff подбирается на эмоцию как "
                         "S * mean||h_L|| / ||vec||. Делает силу сопоставимой между моделями; "
                         "если задан, --coeff игнорируется")
    ap.add_argument("--per-emotion", type=int, default=2, help="held-out prompts pooled per emotion")
    ap.add_argument("--max-new-tokens", type=int, default=120)
    ap.add_argument("--center", action="store_true",
                    help="subtract the mean emotion vector (disentangle shared affect), renorm to original length")
    ap.add_argument("--save-answers", action="store_true", help="include generated answer text in the CSV")
    ap.add_argument("--prompt-baseline", action="store_true",
                    help="instead of steering, generate with each emotion's pos-instruction (prompting baseline)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    model, tokenizer, load_info = load_model_and_tokenizer(
        LoadSpec(hf_id=args.model_name, dtype=args.dtype))
    encoder = ClassifierBasedEncoder()

    questions = common_prompts(args.per_emotion)
    prompts = [build_prompt(tokenizer, q) for q in questions]
    fields = ["steer", "layer", "coeff", "prompt_id", *ISEAR_EMOTIONS] + (["answer"] if args.save_answers else [])
    rows = []

    # Векторы и коэффициенты — ДО чтения чекпойнта: возобновление должно знать,
    # в какой рабочей точке снят готовый кусок, иначе оно смешает разные слои.
    vecs: dict[str, torch.Tensor] = {}
    coeffs: dict[str, float] = {}
    if not args.prompt_baseline:
        vecs = {
            emo: torch.load(args.vector_dir / f"{emo}_response_avg_diff.pt", map_location="cpu")[args.layer + 1]
            for emo in ISEAR_EMOTIONS
        }
        if args.center:
            mean_vec = torch.stack([vecs[e] for e in ISEAR_EMOTIONS]).mean(0)
            for e in ISEAR_EMOTIONS:
                centered = vecs[e] - mean_vec
                vecs[e] = centered * (vecs[e].norm() / (centered.norm() + 1e-8))  # renorm to original length
            print("centered: subtracted mean emotion vector, renormed to original length")
        coeffs = {emo: args.coeff for emo in ISEAR_EMOTIONS}
        if args.strength is not None:
            mean_h = mean_activation_norm(model, tokenizer, prompts, args.layer)
            for emo in ISEAR_EMOTIONS:
                coeffs[emo] = args.strength * mean_h / (vecs[emo].norm().item() + 1e-8)
            print(f"strength S={args.strength:.3f}, mean||h_{args.layer}||={mean_h:.2f} -> "
                  + ", ".join(f"{e[:4]} c={coeffs[e]:.2f}" for e in ISEAR_EMOTIONS))

    expected: dict[str, tuple[str, float]] = {"baseline": ("", 0.0)}
    if args.prompt_baseline:
        expected.update({f"prompt:{e}": ("", 0.0) for e in ISEAR_EMOTIONS})
    else:
        expected.update({e: (str(args.layer), coeffs[e]) for e in ISEAR_EMOTIONS})

    # Построчная дозапись и возобновление. Стадия самая дорогая (448 генераций,
    # больше часа) и до сих пор писала результат одним куском в самом конце:
    # обрыв на шестой эмоции терял всё.
    ckpt = args.out.with_suffix(".partial.csv") if args.out else None
    done: set[tuple[str, int]] = set()
    if ckpt and ckpt.is_file():
        rows, done = load_checkpoint(ckpt, fields, expected)
        print(f"возобновление: {len(done)} готовых генераций из {ckpt.name}", flush=True)
    ckpt_fh = open(ckpt, "a", newline="", encoding="utf-8") if ckpt else None
    ckpt_w = csv.DictWriter(ckpt_fh, fieldnames=fields, extrasaction="ignore") if ckpt_fh else None
    if ckpt_fh and not done:
        ckpt_w.writeheader()
        ckpt_fh.flush()

    def run_block(steer_label, vec, layer, coeff, plist=None):
        for pid, prompt in enumerate(plist or prompts):
            if (steer_label, pid) in done:
                continue
            if vec is None:
                ans = generate(model, tokenizer, prompt, args.max_new_tokens)
            else:
                with ActivationSteerer(model, vec, coeff=coeff, layer_idx=layer, positions="all"):
                    ans = generate(model, tokenizer, prompt, args.max_new_tokens)
            scores = encode_all(encoder, ans)
            row = {"steer": steer_label, "layer": layer, "coeff": coeff,
                   "prompt_id": pid, "answer": ans, **scores}
            rows.append(row)
            if ckpt_w:
                ckpt_w.writerow(row)
                ckpt_fh.flush()

    run_block("baseline", None, "", 0.0)
    if args.prompt_baseline:
        print(f"prompting baseline: {len(ISEAR_EMOTIONS)} emotions x {len(prompts)} prompts")
        for emo in ISEAR_EMOTIONS:
            eprompts = [build_instr_prompt(tokenizer, q, pos_instruction(emo)) for q in questions]
            run_block(f"prompt:{emo}", None, "", 0.0, plist=eprompts)
            print(f"  done prompt={emo}")
    else:
        print(f"steering: {len(ISEAR_EMOTIONS)} emotions x {len(prompts)} prompts @ layer {args.layer}")
        for emo in ISEAR_EMOTIONS:
            run_block(emo, vecs[emo], args.layer, coeffs[emo])
            print(f"  done steer={emo} (coeff {coeffs[emo]:.2f})")

    # specificity matrix: mean measured score per (steer, emotion)
    from collections import defaultdict
    agg = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for e in ISEAR_EMOTIONS:
            agg[r["steer"]][e].append(r[e])
    header = "steer\\meas " + " ".join(f"{e[:4]:>5}" for e in ISEAR_EMOTIONS)
    print(header)
    order = ["baseline"] + [s for s in agg if s != "baseline"]
    for steer in order:
        if not agg[steer][ISEAR_EMOTIONS[0]]:
            continue
        means = [sum(agg[steer][e]) / len(agg[steer][e]) for e in ISEAR_EMOTIONS]
        print(f"{steer:>10} " + " ".join(f"{m:>5.2f}" for m in means))

    if ckpt_fh:
        ckpt_fh.close()

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {args.out}")
        if ckpt and ckpt.is_file():
            ckpt.unlink()  # итог записан, промежуточный файл больше не нужен


if __name__ == "__main__":
    main()
