"""Off-distribution steering: does the vector inject emotion into neutral prompts? (Phase 5.4)

A key claim is that activation steering controls emotion independently of the prompt.
We test it out-of-distribution: instead of the emotion-eval scenarios, we steer on
plain FACTUAL/instructional prompts with zero emotional pull ("Explain how to brew
tea"). If the SAE-feature vector still raises the target emotion on these, the control
is prompt-independent - something prompting a neutral instruction does not give.

Neutral template (no emotional/experiential framing) so any emotion in the output
comes from the vector, not the prompt. baseline (no steering) + steer toward each of
the 7 emotions. Per-condition CSV checkpoint. Judge the answers afterwards.

Usage (GPU):
    python -m emotion.steer_random_prompts --model_name google/gemma-2-2b-it \
        --vector-dir emotion_vectors/gemma-sae-feat --layer 12 --coeff 8 \
        --out results/steer_random_prompts.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from activation_steer import ActivationSteerer
from emotion.classifier_encoder import ClassifierBasedEncoder
from emotion.space import ISEAR_EMOTIONS
from emotion.steer_eval import generate
from emotion.steer_specificity import encode_all

NEUTRAL_INSTR = ("You are a helpful assistant. Answer the question clearly and factually "
                 "in a few plain sentences. No markdown.")

# Factual / instructional prompts with no emotional content (off the ISEAR distribution).
RANDOM_PROMPTS = [
    "Explain how to brew a cup of tea.",
    "Describe the process of photosynthesis in simple terms.",
    "List the steps to change a flat tire.",
    "What is the capital of Australia and a few facts about it?",
    "Explain how a bicycle gear system works.",
    "Describe how to make a basic tomato sauce.",
    "Summarize how the water cycle works.",
    "Explain what a prime number is, with examples.",
    "Describe how a knight moves on a chessboard.",
    "How do you convert Celsius to Fahrenheit?",
    "Explain how rainbows form.",
    "Describe the main parts of a desktop computer.",
    "What are the planets of the solar system in order?",
    "Explain how to fold a simple paper airplane.",
    "Describe how bread rises when baking.",
    "What is the difference between weather and climate?",
    "Explain how a refrigerator keeps food cold.",
    "How does a magnetic compass point north?",
    "Explain the basic rules of tic-tac-toe.",
    "Describe how to plant a tomato seedling.",
    "What are the primary colors and how do they mix?",
    "Describe how to tie a basic shoelace knot.",
    "What is the function of the heart in the human body?",
    "Explain what causes ocean tides.",
    "Describe how to set up a simple two-person tent.",
    "What is the boiling point of water at sea level?",
    "Explain how an incandescent light bulb produces light.",
    "Describe the steps to send an email.",
    "Explain how a vending machine dispenses an item.",
    "Describe how to read the time on an analog clock.",
]


def build_neutral(tokenizer, question: str) -> str:
    content = f"{NEUTRAL_INSTR}\n\n{question}"
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Off-distribution (neutral-prompt) steering robustness.")
    ap.add_argument("--model_name", default="google/gemma-2-2b-it")
    ap.add_argument("--vector-dir", required=True, type=Path)
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--coeff", type=float, default=8.0)
    ap.add_argument("--max-new-tokens", type=int, default=120)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map="auto", torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    encoder = ClassifierBasedEncoder()
    vecs = {e: torch.load(args.vector_dir / f"{e}_response_avg_diff.pt", map_location="cpu")[args.layer + 1]
            for e in ISEAR_EMOTIONS}

    prompts = [build_neutral(tokenizer, q) for q in RANDOM_PROMPTS]
    fields = ["steer", "layer", "coeff", "prompt_id", *ISEAR_EMOTIONS, "answer"]
    rows = []

    def run_block(label, vec):
        for pid, prompt in enumerate(prompts):
            if vec is None:
                ans = generate(model, tokenizer, prompt, args.max_new_tokens)
            else:
                with ActivationSteerer(model, vec, coeff=args.coeff, layer_idx=args.layer, positions="all"):
                    ans = generate(model, tokenizer, prompt, args.max_new_tokens)
            rows.append({"steer": label, "layer": args.layer, "coeff": args.coeff,
                         "prompt_id": pid, "answer": ans, **encode_all(encoder, ans)})

    def checkpoint():
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"  checkpoint -> {args.out} ({len(rows)} rows)", flush=True)

    print(f"baseline: {len(prompts)} neutral prompts", flush=True)
    run_block("baseline", None)
    checkpoint()
    for emo in ISEAR_EMOTIONS:
        run_block(emo, vecs[emo])
        print(f"  steer={emo} done", flush=True)
        checkpoint()
    print(f"DONE wrote {args.out} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
