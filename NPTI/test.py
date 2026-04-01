# encoding: utf-8
import argparse
import json
import os
import sys
from pathlib import Path
from typing import List

import torch
from tqdm import tqdm

# Ensure project root is on sys.path for absolute imports when running this file directly
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from activation_steer import ActivationSteerer
from eval.model_utils import load_model


TEMPLATE = """
Imagine you are a real person rather than a language model, and you're asked by the following question. Write your response based on your authentic thoughts and emotions.

Do not overthink your answer��let your thoughts flow naturally as you write. Focus on expressing your genuine feelings and reactions. Aim to write no more than 300 words.

### Question:
{question}

### Response:

"""


# Positive mapping for Big Five traits (aligns with eval_persona TRAIT_ANTONYMS keys)
BFI_POSITIVE_WORD = {
    "Openness": "inventive",
    "Conscientiousness": "dependable",
    "Extraversion": "outgoing",
    "Agreeableness": "compassionate",
    # Neuroticism positive pole is emotional stability -> calm
    "Neuroticism": "nervous",
}


def load_question(question_path: str) -> List[str]:
    questions = []
    with open(question_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            q = obj.get("question")
            if q:
                questions.append(q)
    return questions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="HF model path or ID")
    parser.add_argument("--vector_root", type=str, required=True, help="Directory containing persona vectors for this model")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to output directory")
    parser.add_argument("--question_dir", type=str, required=True, help="Directory with per-trait question JSONL files")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--coeff", type=float, default=1.0, help="Steering coefficient")
    parser.add_argument("--layer", type=int, default=20, help="1-based transformer block index to steer (as in eval_persona)")
    parser.add_argument("--positions", type=str, default="response", choices=["all", "prompt", "response"], help="Where to apply steering in sequence")
    parser.add_argument("--variant", type=str, default="response_avg_diff", help="Vector filename suffix, e.g., response_avg_diff or prompt_last_diff")
    parser.add_argument("--max_new_tokens", type=int, default=400)
    args = parser.parse_args()

    # Load HF model and tokenizer (same approach as eval_persona when coef != 0)
    model, tokenizer = load_model(args.model)
    device = next(model.parameters()).device

    # Ensure tokenizer padding for left side to support batch generation
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    os.makedirs(args.output_dir, exist_ok=True)

    for trait in ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]:
        q_path = os.path.join(args.question_dir, f"{trait}.json")
        if not os.path.exists(q_path):
            print(f"[Skip] Questions not found for {trait}: {q_path}")
            continue

        questions = load_question(q_path)
        if not questions:
            print(f"[Skip] No questions in: {q_path}")
            continue

        # Resolve vector file for the positive side of this trait
        word = BFI_POSITIVE_WORD[trait]
        vec_path = os.path.join(args.vector_root, f"{word}_{args.variant}.pt")
        eff_coeff = float(args.coeff)
        if not os.path.exists(vec_path):
            # Fallback: for Neuroticism, try antonym 'nervous' with negative coeff if 'calm' is missing
            if trait == "Neuroticism":
                alt = os.path.join(args.vector_root, f"nervous_{args.variant}.pt")
                if os.path.exists(alt):
                    print(f"[Warn] Calm vector missing; using nervous with negative coeff: {alt}")
                    vec_path = alt
                    eff_coeff = -eff_coeff
                else:
                    print(f"[Warn] Vector not found for {trait} at {vec_path} or fallback; skipping.")
                    continue
            else:
                print(f"[Warn] Vector not found for {trait} at {vec_path}; skipping.")
                continue

        print(f"[Load] {trait} �� {word}: {vec_path}")
        # Vector file stores per-layer vectors; follow eval_persona: vector = torch.load(...)[layer]
        vec_obj = torch.load(vec_path, map_location=device)
        try:
            steering_vector = vec_obj[args.layer]
        except Exception:
            # Some files might store zero-based or as dict; try common alternatives
            if isinstance(vec_obj, dict) and args.layer in vec_obj:
                steering_vector = vec_obj[args.layer]
            elif isinstance(vec_obj, dict) and (args.layer - 1) in vec_obj:
                steering_vector = vec_obj[args.layer - 1]
            elif isinstance(vec_obj, (list, tuple)):
                idx = args.layer if args.layer < len(vec_obj) else args.layer - 1
                steering_vector = vec_obj[idx]
            else:
                raise

        out_dir_trait = os.path.join(args.output_dir, trait)
        os.makedirs(out_dir_trait, exist_ok=True)
        out_path = os.path.join(out_dir_trait, f"{trait}_steered_coeff_{args.coeff}_layer_{args.layer}.jsonl")
        print(f"[Write] {out_path}")

        # Build prompts
        conversations = [[{"role": "user", "content": TEMPLATE.format(question=q)}] for q in questions]
        prompts = [tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) for msgs in conversations]

        with open(out_path, "w", encoding="utf-8") as fout:
            for i in tqdm(range(0, len(prompts), args.batch_size), desc=f"{trait}"):
                batch_prompts = prompts[i:i + args.batch_size]
                tokenized = tokenizer(batch_prompts, return_tensors="pt", padding=True)
                tokenized = {k: v.to(device) for k, v in tokenized.items()}

                with ActivationSteerer(model, steering_vector, coeff=eff_coeff, layer_idx=args.layer - 1, positions=args.positions):
                    with torch.no_grad():
                        outputs = model.generate(
                            **tokenized,
                            do_sample=False,
                            temperature=0.0,
                            top_p=1.0,
                            max_new_tokens=args.max_new_tokens,
                            use_cache=True,
                            min_new_tokens=1,
                        )
                prompt_len = tokenized["input_ids"].shape[1]
                texts = [tokenizer.decode(o[prompt_len:], skip_special_tokens=True) for o in outputs]

                for q, ans in zip(questions[i:i + args.batch_size], texts):
                    fout.write(json.dumps({"question": q, "answer": ans}, ensure_ascii=False) + "\n")

    print("Done.")


if __name__ == "__main__":
    main()