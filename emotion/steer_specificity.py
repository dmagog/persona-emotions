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
        [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Steering specificity matrix across all emotions.")
    ap.add_argument("--model_name", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--vector-dir", required=True, type=Path)
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--coeff", type=float, default=8.0)
    ap.add_argument("--per-emotion", type=int, default=2, help="held-out prompts pooled per emotion")
    ap.add_argument("--max-new-tokens", type=int, default=120)
    ap.add_argument("--center", action="store_true",
                    help="subtract the mean emotion vector (disentangle shared affect), renorm to original length")
    ap.add_argument("--save-answers", action="store_true", help="include generated answer text in the CSV")
    ap.add_argument("--prompt-baseline", action="store_true",
                    help="instead of steering, generate with each emotion's pos-instruction (prompting baseline)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, device_map="auto", torch_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    encoder = ClassifierBasedEncoder()

    questions = common_prompts(args.per_emotion)
    prompts = [build_prompt(tokenizer, q) for q in questions]
    fields = ["steer", "layer", "coeff", "prompt_id", *ISEAR_EMOTIONS] + (["answer"] if args.save_answers else [])
    rows = []

    def run_block(steer_label, vec, layer, coeff, plist=None):
        for pid, prompt in enumerate(plist or prompts):
            if vec is None:
                ans = generate(model, tokenizer, prompt, args.max_new_tokens)
            else:
                with ActivationSteerer(model, vec, coeff=coeff, layer_idx=layer, positions="all"):
                    ans = generate(model, tokenizer, prompt, args.max_new_tokens)
            scores = encode_all(encoder, ans)
            rows.append({"steer": steer_label, "layer": layer, "coeff": coeff, "prompt_id": pid, "answer": ans, **scores})

    run_block("baseline", None, "", 0.0)
    if args.prompt_baseline:
        print(f"prompting baseline: {len(ISEAR_EMOTIONS)} emotions x {len(prompts)} prompts")
        for emo in ISEAR_EMOTIONS:
            eprompts = [build_instr_prompt(tokenizer, q, pos_instruction(emo)) for q in questions]
            run_block(f"prompt:{emo}", None, "", 0.0, plist=eprompts)
            print(f"  done prompt={emo}")
    else:
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
        print(f"steering: {len(ISEAR_EMOTIONS)} emotions x {len(prompts)} prompts @ layer {args.layer} coeff {args.coeff}")
        for emo in ISEAR_EMOTIONS:
            run_block(emo, vecs[emo], args.layer, args.coeff)
            print(f"  done steer={emo}")

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

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
