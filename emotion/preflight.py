"""Preflight: run the whole chain on one row before starting a long queue.

This catches the class of failure that once cost a day of work, where the code
finishes successfully but the data is unusable and nobody finds out for five
hours.

It checks, in order:
  1. the chat template, meaning a system role or the fallback, with reasoning
     mode suppressed;
  2. pos/neg generation, which must be non-empty, free of reasoning tags and not
     degenerate;
  3. vector extraction, for shape, finite values and a non-zero norm;
  4. the activation scale, computing the strength S and warning about extreme
     coefficients;
  5. steering, where the steered text has to differ from the original and stay
     non-degenerate.

Any miss gives a non-zero exit code and a readable reason.

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

from emotion.loader import LoadSpec, load_model_and_tokenizer

from emotion.activation_steer import ActivationSteerer
from emotion.steer_eval import build_prompt, generate, load_eval_prompts

REPO = Path(__file__).resolve().parent.parent
FAILS: list[str] = []


def check(ok: bool, name: str, detail: str = "") -> bool:
    print(f"  [{'OK ' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""), flush=True)
    if not ok:
        FAILS.append(f"{name}: {detail}")
    return ok


def rep_ratio(text: str, n: int = 4) -> float:
    """Share of repeated n-grams, stable across text lengths."""
    w = re.findall(r"\w+", str(text).lower())
    if len(w) < n + 4:
        return 0.0
    grams = [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]
    return 1.0 - len(set(grams)) / len(grams)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the chain on one row before a long queue.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--strength", type=float, default=None)
    ap.add_argument("--coeff", type=float, default=8.0)
    ap.add_argument("--max-new-tokens", type=int, default=120)
    ap.add_argument("--skip-steer", action="store_true",
                    help="skip the steering checks: the vectors are about to be "
                         "recomputed, so testing the ones being written off would "
                         "fail the run for nothing")
    ap.add_argument("--dtype", default="auto",
                    help="compute dtype: the preflight has to check exactly what "
                         "the stages will later use")
    args = ap.parse_args()

    import os
    token = os.environ.get("HF_TOKEN")
    print(f"\n=== preflight: {args.model}, layer {args.layer} ===", flush=True)

    # --- 1. chat template ---
    tok = AutoTokenizer.from_pretrained(args.model, token=token)
    sys_txt, user_txt = "You are a person.", "Describe a small everyday event."
    transport = "system role"
    try:
        rendered = tok.apply_chat_template(
            [{"role": "system", "content": sys_txt}, {"role": "user", "content": user_txt}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except Exception as e:
        if "system" not in str(e).lower():
            check(False, "chat template", f"unexpected error: {type(e).__name__}: {e}")
            sys.exit(1)
        transport = "framing folded into the user turn"
        rendered = tok.apply_chat_template(
            [{"role": "user", "content": f"{sys_txt}\n\n{user_txt}"}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
    check(sys_txt in rendered, "chat template", transport)
    # reasoning: only an empty placeholder block is acceptable
    think_open = rendered.count("<think>")
    check(think_open == 0 or "<think>\n\n</think>" in rendered or "<think></think>" in rendered,
          "reasoning mode suppressed", f"<think> tags: {think_open}")

    # --- 2. generation ---
    model, tok, load_info = load_model_and_tokenizer(LoadSpec(hf_id=args.model, dtype=args.dtype))
    print(f"  [info] {load_info['dtype']}: {load_info['dtype_reason']}", flush=True)
    q = load_eval_prompts("anger", 1)[0]
    prompt = build_prompt(tok, q)
    base = generate(model, tok, prompt, args.max_new_tokens)
    check(len(base.strip()) > 20, "generation is non-empty", f"{len(base)} characters")
    check("<think>" not in base, "no reasoning tags in the answer")
    check(rep_ratio(base) <= 0.15, "original text is not degenerate", f"4-gram repetition {rep_ratio(base):.3f}")

    # --- 3. vector ---
    vec_dir = REPO / "emotion_vectors" / args.model.split("/")[-1]
    alt = list((REPO / "emotion_vectors").glob(f"*{args.model.split('/')[-1]}*"))
    if not vec_dir.is_dir() and alt:
        vec_dir = alt[0]
    vec_file = vec_dir / "anger_response_avg_diff.pt"
    if args.skip_steer or not vec_file.is_file():
        # A new model has no vectors yet, which is fine. The template, reasoning
        # and generation checks have already run, and those are exactly where
        # Gemma broke over the system role and Qwen3 over reasoning mode. The
        # preflight used to return 1 here, so the chain stopped calling it.
        why = ("vectors are about to be recomputed" if args.skip_steer else "no vectors yet")
        print(f"\n  [info] {why}, so the steering checks are skipped; "
              "the chain will build them", flush=True)
        if FAILS:
            print(f"\nRESULT: {len(FAILS)} failures before the vector stage:", file=sys.stderr)
            for f in FAILS:
                print(f"  - {f}", file=sys.stderr)
            sys.exit(1)
        print("\nRESULT: template and generation are fine, the chain can start.")
        sys.exit(0)

    vec_all = torch.load(vec_file, map_location="cpu")
    n_layers = model.config.num_hidden_layers
    check(tuple(vec_all.shape) == (n_layers + 1, model.config.hidden_size),
          "vector shape", f"{tuple(vec_all.shape)}, expected {(n_layers + 1, model.config.hidden_size)}")
    vec = vec_all[args.layer + 1]
    check(torch.isfinite(vec).all().item(), "vector is finite")
    check(vec.norm().item() > 1e-3, "vector norm is non-zero", f"||vec||={vec.norm().item():.2f}")

    # --- 4. activation scale and strength ---
    with torch.no_grad():
        enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
        h = model(**enc, output_hidden_states=True).hidden_states[args.layer + 1][0]
    mean_h = h.norm(dim=-1).mean().item()
    if args.strength is not None:
        coeff = args.strength * mean_h / vec.norm().item()
        check(0.1 <= coeff <= 200, "coefficient is within a sane range",
              f"S={args.strength} gives coeff={coeff:.2f} at mean||h||={mean_h:.1f}")
    else:
        coeff = args.coeff
        S = coeff * vec.norm().item() / mean_h
        check(0.05 <= S <= 1.0, "intervention strength is within a sane range",
              f"coeff={coeff} gives S={S:.3f}, with 0.33 as a reference point")

    # --- 5. steering ---
    with ActivationSteerer(model, vec, coeff=coeff, layer_idx=args.layer, positions="all"):
        steered = generate(model, tok, prompt, args.max_new_tokens)
    check(steered.strip() != base.strip(), "steering changes the text")
    check(rep_ratio(steered) <= 0.15, "steered text is not degenerate",
          f"4-gram repetition {rep_ratio(steered):.3f}")

    print(f"\n  original: {base.strip()[:110]}")
    print(f"  steered : {steered.strip()[:110]}")

    if FAILS:
        print(f"\nRESULT: {len(FAILS)} failures, do not start the queue:", file=sys.stderr)
        for f in FAILS:
            print(f"  - {f}", file=sys.stderr)
        sys.exit(1)
    print("\nRESULT: the chain passes on one row, the queue can start.")


if __name__ == "__main__":
    main()
