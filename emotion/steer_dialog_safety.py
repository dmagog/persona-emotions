"""Dialogue de-escalation: suppressing anger or fear and what it does to the reply.

Extends the evaluation from single-turn answers to multi-turn context. The input
is a set of provocative dialogues, covering conflict, accusation and emotional
escalation, cut off on a user turn; the model generates the assistant's next
reply under several steering conditions, and we look at whether it sharpens the
conflict or defuses it.

The conditions differ from the composition stage. There the spec `X-Y` means
vec[X] - vec[Y]: it steers toward X and suppresses Y along the way. Here we need
PURE suppression, which is steering by the negative vector `-anger` and nothing
else.

The positive control `+anger` is included on purpose: if the minus damps
escalation while the plus raises it, the effect runs along one axis rather than
softening the reply in general. A one-sided result cannot show that.

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

from emotion.activation_steer import ActivationSteerer
from emotion.classifier_encoder import ClassifierBasedEncoder
from emotion.loader import LoadSpec, load_model_and_tokenizer
from emotion.space import ISEAR_EMOTIONS
from emotion.steer_specificity import encode_all

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

REPO = Path(__file__).resolve().parent.parent

# The named conditions of the pilot. Any other condition is parsed on the fly
# (see parse_condition), so -sadness, +joy and the like can be asked for too.
CONDITIONS: dict[str, list[tuple[float, str]]] = {
    "baseline": [],
    "-anger": [(-1.0, "anger")],
    "-fear": [(-1.0, "fear")],
    "-anger-fear": [(-1.0, "anger"), (-1.0, "fear")],
    "+anger": [(+1.0, "anger")],  # positive control
}


# Control for a generic perturbation: a random direction with the norm of a
# reference emotion. Without it, removing an emotional direction cannot be told
# apart from any push of that size.
RANDOM_RE = re.compile(r"^random(\d+)$")


def parse_condition(spec: str) -> list[tuple[float, str]]:
    """'-anger' -> [(-1,anger)]; '+joy' -> [(+1,joy)]; '-anger-fear' -> both negative.

    Every emotion needs a sign: a bare 'anger' is rejected, since otherwise it is
    unclear whether we steer toward the emotion or suppress it.
    """
    if spec == "baseline" or RANDOM_RE.match(spec):
        return []  # random builds its vector separately, not from emotions
    if spec in CONDITIONS:
        return CONDITIONS[spec]
    parts = re.findall(r"([+-])([a-z]+)", spec)
    if not parts or "".join(s + e for s, e in parts) != spec:
        raise SystemExit(f"cannot parse condition {spec!r}: every emotion needs a sign")
    for _, emo in parts:
        if emo not in ISEAR_EMOTIONS:
            raise SystemExit(f"unknown emotion {emo!r} in condition {spec!r}; "
                             f"known ones are {list(ISEAR_EMOTIONS)}")
    return [(-1.0 if sign == "-" else 1.0, emo) for sign, emo in parts]


def build_dialog_prompt(tokenizer, turns: list[dict]) -> str:
    """Multi-turn prompt built from the model's chat template.

    No system role: the gemma-2 template rejects it, and it is not needed here,
    since the turns themselves carry the context. enable_thinking suppresses the
    reasoning mode of Qwen 3, but not every template accepts that argument.
    """
    messages = [{"role": t["role"], "content": t["content"]} for t in turns]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)


def generate(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    """Greedy decoding, as in every other stage."""
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Dialogue de-escalation under emotion suppression.")
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--vector-dir", required=True, type=Path)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--coeff", type=float, required=True)
    ap.add_argument("--dialogs", type=Path,
                    default=REPO / "data_generation" / "deescalation_dialogs.json")
    ap.add_argument("--conditions", default=",".join(CONDITIONS),
                    help="comma-separated; defaults to all five named conditions")
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--dtype", default="float16", help="must match the dtype used for the model run")
    ap.add_argument("--random-match", default="anger",
                    help="emotion whose vector norm the randomN conditions are scaled to")
    ap.add_argument("--no-encoder", action="store_true",
                    help="skip encoder scoring; the emotion columns stay empty")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    dialogs = json.loads(args.dialogs.read_text(encoding="utf-8"))["dialogs"]
    conds = [c.strip() for c in args.conditions.split(",") if c.strip()]
    parsed = {c: parse_condition(c) for c in conds}  # fails at once on a bad condition
    print(f"{len(dialogs)} dialogues, {len(conds)} conditions -> "
          f"{len(dialogs) * len(conds)} generations", flush=True)

    model, tokenizer, _ = load_model_and_tokenizer(
        LoadSpec(hf_id=args.model_name, dtype=args.dtype))
    # The encoder is optional. Generation on the GPU is the expensive and
    # unrepeatable part, and it should not be lost because the classifier failed
    # to download. Emotion scores can be added later, and the judge does not
    # depend on them.
    encoder = None
    if not args.no_encoder:
        try:
            encoder = ClassifierBasedEncoder()
        except Exception as exc:
            print(f"WARNING: the encoder is unavailable ({type(exc).__name__}), the "
                  f"emotion columns stay empty: {str(exc)[:140]}", flush=True)

    needed = {emo for c in conds for _, emo in parsed[c]}
    if any(RANDOM_RE.match(c) for c in conds):
        needed.add(args.random_match)  # reference norm for the random directions
    vecs = {
        emo: torch.load(args.vector_dir / f"{emo}_response_avg_diff.pt",
                        map_location="cpu")[args.layer + 1]
        for emo in sorted(needed)
    }

    prompts = [build_dialog_prompt(tokenizer, d["turns"]) for d in dialogs]

    fields = ["condition", "dialog_id", "category", *ISEAR_EMOTIONS, "answer"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Write as we go: a remote run should not lose everything to a crash on the
    # last condition.
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
                print(f"  {cond}: random direction, norm {float(vec.norm()):.3f}, "
                      f"matching {args.random_match}", flush=True)
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
            print(f"  condition {cond} done", flush=True)

    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
