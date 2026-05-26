"""Steering eval: steer Qwen toward an emotion, generate, measure with the encoder.

For each (layer, coeff) we generate first-person accounts on held-out *neutral*
prompts (no emotion instruction) while adding ``coeff * vec[layer+1]`` to the
residual stream at ``model.layers[layer]`` (the off-by-one matches how the
vectors were extracted: ``hidden_states[L]`` is the output of block ``L-1``).
Then we encode each response and record the target-emotion score, so we can see
whether (and how specifically) steering raises the target emotion vs baseline.

Usage (GPU):
    python -m emotion.steer_eval --emotion anger \
        --vector-dir emotion_vectors/Qwen2.5-3B-Instruct \
        --layers 14,18 --coeffs 0,4,8 --n-prompts 10 --out steer_anger.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from activation_steer import ActivationSteerer
from emotion import metrics
from emotion.classifier_encoder import ClassifierBasedEncoder

NEUTRAL_SYSTEM = (
    "You are a person. Respond in a natural, conversational first-person tone. "
    "No markdown, just a few plain sentences."
)
TASK_SUFFIX = (
    "\n\nWrite a short first-person account of this experience as if it happened "
    "to you. Keep it to 4-6 natural sentences."
)


def load_eval_prompts(emotion: str, n: int) -> list[str]:
    data = json.loads(Path(f"data_generation/emotion_data_eval/{emotion}.json").read_text())
    return data["questions"][:n]


def build_prompt(tokenizer, question: str) -> str:
    messages = [
        {"role": "system", "content": NEUTRAL_SYSTEM},
        {"role": "user", "content": question + TASK_SUFFIX},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 120) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Steering eval for one emotion.")
    ap.add_argument("--model_name", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--emotion", required=True)
    ap.add_argument("--vector-dir", required=True, type=Path)
    ap.add_argument("--layers", default="14,18", help="comma-separated block indices")
    ap.add_argument("--coeffs", default="0,4,8", help="comma-separated steering coefficients")
    ap.add_argument("--n-prompts", type=int, default=10)
    ap.add_argument("--max-new-tokens", type=int, default=120)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    layers = [int(x) for x in args.layers.split(",")]
    coeffs = [float(x) for x in args.coeffs.split(",")]

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, device_map="auto", torch_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    vec_all = torch.load(args.vector_dir / f"{args.emotion}_response_avg_diff.pt", map_location="cpu")

    encoder = ClassifierBasedEncoder()
    prompts = load_eval_prompts(args.emotion, args.n_prompts)

    rows = []
    print(f"{'layer':>5} {'coeff':>6} {'mean_target':>11} {'n':>3}")
    for layer in layers:
        vec = vec_all[layer + 1]
        for coeff in coeffs:
            scores = []
            for qi, q in enumerate(prompts):
                prompt = build_prompt(tokenizer, q)
                if coeff == 0:
                    ans = generate(model, tokenizer, prompt, args.max_new_tokens)
                else:
                    with ActivationSteerer(model, vec, coeff=coeff, layer_idx=layer, positions="all"):
                        ans = generate(model, tokenizer, prompt, args.max_new_tokens)
                s = metrics.target_score(encoder.encode(ans), args.emotion, encoder.space)
                scores.append(s)
                rows.append({"emotion": args.emotion, "layer": layer, "coeff": coeff,
                             "prompt_id": qi, "target_score": round(s, 4), "answer": ans})
            mean_s = sum(scores) / len(scores) if scores else 0.0
            print(f"{layer:>5} {coeff:>6.1f} {mean_s:>11.4f} {len(scores):>3}")
            # baseline (coeff 0) is layer-independent; only compute once
            if coeff == 0 and layer != layers[0]:
                pass

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["emotion", "layer", "coeff", "prompt_id", "target_score", "answer"])
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
