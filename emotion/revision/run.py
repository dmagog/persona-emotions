"""GPU runs for the revision (plans refs, p1, p2, p3, p4, p5, p6, p7, parity).

One command per (model, plan). Output goes to ``runs_revision/<slug>/<plan>.csv``
plus ``<plan>.csv.meta.json`` (settings, versions, hardware, git sha). The CSV is
appended condition by condition, so an interrupted run resumes where it stopped:
conditions that already have all their rows are skipped.

Generation settings are taken from the published run and must not be changed:
operating point from ``runs/<slug>/meta.json`` (``op_point``, *not* the stale
``coeff`` field), vector index ``layer + 1``, ``max_new_tokens = 120``,
greedy decoding, the published prompt builder, dtype from the model config.

Usage (GPU):
    python -m emotion.revision.run --config configs/models/llama-3.2-3b.yaml --plan p1 --batch-size 32
    python -m emotion.revision.run --config configs/models/qwen3-0.6b.yaml --plan p2 --batch-size 1

Plans (see REVISION_EXPERIMENTS.md for what each one answers):
    refs    baseline, 7 a-only, 42 a-b regenerated in this pass (parity reference)
    p1      refs + null controls and norm matching for all 42 pairs (R1 M1, M2, M3, M7b)
    p2      baseline + -v_b for the 7 emotions (R1 M4)
    p3      prompting baseline: "express a" and "express a, reduce b" without steering (R1 M7a)
    p4      sampled decoding: baseline, 7 a-only, 42 a-b, 5 samples each (R1 M6)
    p5      re-calibration sweep on new anger prompts (R1 M6; E7/E8)
    p6      residual-stream norms for the relative strength S (R1 M3, Eq. 12); no generation
    p7      centred single directions (R1 M2, optional)
    parity  baseline, 7 a-only, 6 pairs: compare with the published compose_allpairs.csv
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import yaml

from emotion.activation_steer import ActivationSteerer
from emotion.classifier_encoder import GO_EMOTIONS_TO_ISEAR, ClassifierBasedEncoder, aggregate_scores
from emotion.loader import LoadSpec, load_model_and_tokenizer
from emotion.revision import vectors as V
from emotion.revision.prompts import (EXPRESS_A, EXPRESS_A_REDUCE_B, build_instr_prompt,
                                      calibration_prompts)
from emotion.space import ALL_PAIRS, ISEAR_EMOTIONS
from emotion.steer_eval import build_prompt
from emotion.steer_specificity import common_prompts

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

REPO = Path(__file__).resolve().parents[2]
MAX_NEW_TOKENS = 120          # what the published compose stage used (run_model_chain.py)
PER_EMOTION = 8               # 7 x 8 = 56 held-out prompts
PARITY_PAIRS = ["anger-sadness", "guilt-sadness", "joy-sadness", "fear-anger", "shame-guilt", "disgust-fear"]
SAMPLING = {"temperature": 0.7, "top_p": 0.9, "n": 5}   # p4, pre-registered
GEO_SEEDS = (1, 2)            # replicates of the main null, raw norm
ISO_SEEDS = (1,)              # isotropic null, raw norm
FIELDS = ["steer", "prompt_id", "sample", "kind", "a", "b", "k", "norm_mode", "vec_norm",
          *ISEAR_EMOTIONS, "question", "answer"]


@dataclass
class Cond:
    label: str                       # unique; never contains ':' (the judge splits on it)
    kind: str                        # baseline | single | ab | geo | iso | lam | proj | neg | cent | instr_a | instr_ab
    vec: torch.Tensor | None = None
    a: str = ""
    b: str = ""
    k: int = 0
    norm_mode: str = "raw"           # raw | norm_a
    instr: list[str] = field(default_factory=list)   # p3: one instruction template per paraphrase


# ---------------------------------------------------------------- conditions

def _refs(vecs) -> list[Cond]:
    out = [Cond("baseline", "baseline")]
    out += [Cond(e, "single", vecs[e], a=e) for e in ISEAR_EMOTIONS]
    for spec in ALL_PAIRS:
        a, b = spec.split("-")
        out.append(Cond(spec, "ab", vecs[a] - vecs[b], a=a, b=b))
    return out


def _p1_controls(vecs, slug) -> list[Cond]:
    out = []
    for spec in ALL_PAIRS:
        a, b = spec.split("-")
        va, vb = vecs[a], vecs[b]
        na = V.norm(va)
        geo = {k: V.geo_random(va, vb, V.pair_seed(slug, a, b, k, "geo")) for k in GEO_SEEDS}
        for k in GEO_SEEDS:
            out.append(Cond(f"{spec}|geo{k}", "geo", va - geo[k], a, b, k))
        for k in ISO_SEEDS:
            r = V.iso_random(vb, V.pair_seed(slug, a, b, k, "iso"))
            out.append(Cond(f"{spec}|iso{k}", "iso", va - r, a, b, k))
        out.append(Cond(f"{spec}|lam", "lam", V.lam_a(va, vb), a, b))
        out.append(Cond(f"{spec}|proj", "proj", V.proj_remove(va, vb), a, b))
        # norm-matched to ||va|| (R1 M3). lambda*a rescaled is a-only itself.
        out.append(Cond(f"{spec}|norm", "ab", V.rescale(va - vb, na), a, b, norm_mode="norm_a"))
        out.append(Cond(f"{spec}|geo1|norm", "geo", V.rescale(va - geo[1], na), a, b, 1, norm_mode="norm_a"))
        out.append(Cond(f"{spec}|proj|norm", "proj", V.rescale(V.proj_remove(va, vb), na), a, b,
                        norm_mode="norm_a"))
    return out


def conditions(plan: str, vecs, slug) -> list[Cond]:
    if plan == "refs":
        return _refs(vecs)
    if plan == "p1":
        return _refs(vecs) + _p1_controls(vecs, slug)
    if plan == "p2":
        return [Cond("baseline", "baseline")] + [Cond(f"neg_{e}", "neg", -vecs[e], b=e) for e in ISEAR_EMOTIONS]
    if plan == "p3":
        out = [Cond("baseline", "baseline")]
        out += [Cond(f"instr_{e}", "instr_a", a=e, instr=EXPRESS_A) for e in ISEAR_EMOTIONS]
        for spec in ALL_PAIRS:
            a, b = spec.split("-")
            out.append(Cond(f"instr_{spec}", "instr_ab", a=a, b=b, instr=EXPRESS_A_REDUCE_B))
        return out
    if plan == "p4":
        return _refs(vecs)
    if plan == "p7":
        return [Cond("baseline", "baseline")] + [Cond(f"cent_{e}", "cent", V.centered(vecs, e), a=e)
                                                  for e in ISEAR_EMOTIONS]
    if plan == "parity":
        refs = _refs(vecs)
        keep = {"baseline", *ISEAR_EMOTIONS, *PARITY_PAIRS}
        return [c for c in refs if c.label in keep]
    raise ValueError(plan)


# ---------------------------------------------------------------- generation

def _encode_batch(encoder: ClassifierBasedEncoder, texts: list[str]) -> list[dict[str, float]]:
    clf = encoder._load()
    preds = clf(texts, truncation=True, batch_size=32)
    out = []
    for p in preds:
        scores = aggregate_scores({d["label"]: float(d["score"]) for d in p}, GO_EMOTIONS_TO_ISEAR,
                                  encoder.space, encoder.normalize)
        out.append({lab: round(float(scores[i]), 4) for i, lab in enumerate(encoder.space.labels)})
    return out


@torch.no_grad()
def _generate(model, tok, prompts: list[str], batch_size: int, sample: dict | None, seed: int) -> list[list[str]]:
    """Return one list of answers per prompt (length 1, or n when sampling)."""
    outs: list[list[str]] = []
    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i:i + batch_size]
        enc = tok(chunk, return_tensors="pt", padding=True, add_special_tokens=False, return_token_type_ids=False).to(model.device)
        kw = dict(max_new_tokens=MAX_NEW_TOKENS, pad_token_id=tok.pad_token_id)
        if sample:
            torch.manual_seed(seed + i)
            kw.update(do_sample=True, temperature=sample["temperature"], top_p=sample["top_p"],
                      num_return_sequences=sample["n"])
        else:
            kw.update(do_sample=False)
        gen = model.generate(**enc, **kw)
        new = gen[:, enc["input_ids"].shape[1]:]
        texts = tok.batch_decode(new, skip_special_tokens=True)
        n = sample["n"] if sample else 1
        outs += [texts[j * n:(j + 1) * n] for j in range(len(chunk))]
    return outs


def _done_labels(path: Path, expected_rows: int) -> set[str]:
    if not path.exists():
        return set()
    counts: dict[str, int] = {}
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            counts[r["steer"]] = counts.get(r["steer"], 0) + 1
    return {lab for lab, c in counts.items() if c >= expected_rows}


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        return "unknown"


def _write_meta(path: Path, info: dict) -> None:
    meta_path = path.with_suffix(path.suffix + ".meta.json")
    old = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    sessions = old.get("sessions", [])
    sessions.append(info)
    meta_path.write_text(json.dumps({"latest": info, "sessions": sessions}, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--plan", required=True,
                    choices=["refs", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "parity"])
    ap.add_argument("--batch-size", type=int, default=16,
                    help="1 reproduces the published generation path exactly; >1 is faster, "
                         "but greedy fp16 output can differ, so always compare within one pass")
    ap.add_argument("--out-dir", type=Path, default=REPO / "runs_revision")
    ap.add_argument("--limit-prompts", type=int, default=None, help="smoke tests only")
    ap.add_argument("--only", default=None, help="comma-separated condition labels (smoke tests)")
    ap.add_argument("--dry-run", action="store_true", help="list conditions and row counts, do not load the model")
    # p5 only
    ap.add_argument("--coeffs", default="10,12,14,20,24,32", help="p5: coefficients to add to the sweep")
    ap.add_argument("--with-published-coeffs", action="store_true",
                    help="p5: also re-run {0,2,4,6,8,16} on the new prompts")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    slug = cfg["slug"]
    run_dir = REPO / "runs" / slug
    meta = json.loads((run_dir / "meta.json").read_text())
    layer = int(meta["op_point"]["layer"])
    coeff = float(meta["op_point"]["coeff"])
    vecs = V.load_vectors(REPO / "emotion_vectors" / slug, layer)
    out_dir = args.out_dir / slug
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.plan == "p5":
        from emotion.revision.calibrate import run_p5
        run_p5(args, cfg, meta, out_dir)
        return

    questions = common_prompts(PER_EMOTION)
    if args.limit_prompts:
        questions = questions[:args.limit_prompts]
    conds = [] if args.plan == "p6" else conditions(args.plan, vecs, slug)
    if args.only:
        keep = set(args.only.split(","))
        conds = [c for c in conds if c.label in keep]
    sample = SAMPLING if args.plan == "p4" else None
    rows_per_cond = len(questions) * (sample["n"] if sample else 1)

    print(f"{slug}: plan {args.plan}, layer {layer}, coeff {coeff}, {len(conds)} conditions x "
          f"{rows_per_cond} rows = {len(conds) * rows_per_cond} generations", flush=True)
    if args.dry_run:
        for c in conds[:12]:
            print(f"  {c.label:28s} {c.kind:9s} norm={V.norm(c.vec) if c.vec is not None else 0:.3f}")
        return

    model, tok, _ = load_model_and_tokenizer(LoadSpec.from_config(cfg))
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model.eval()

    if args.plan == "p6":
        from emotion.revision.norms import run_p6
        run_p6(model, tok, questions, vecs, layer, coeff, slug, out_dir / "p6_norms.json")
        return

    encoder = ClassifierBasedEncoder(device=0 if torch.cuda.is_available() else None)
    out_csv = out_dir / f"{args.plan}.csv"
    done = _done_labels(out_csv, rows_per_cond)
    info = {
        "plan": args.plan, "slug": slug, "hf_id": cfg["hf_id"], "layer": layer, "coeff": coeff,
        "max_new_tokens": MAX_NEW_TOKENS, "batch_size": args.batch_size, "sampling": sample,
        "dtype": str(next(model.parameters()).dtype), "git_sha": _git_sha(),
        "torch": torch.__version__, "transformers": __import__("transformers").__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "python": platform.python_version(), "n_prompts": len(questions),
        "n_conditions": len(conds), "started": dt.datetime.now().isoformat(timespec="seconds"),
        "seed_rule": "sha256(slug|a|b|k|kind)[:8]",
    }
    _write_meta(out_csv, info)

    neutral = [build_prompt(tok, q) for q in questions]
    new_file = not out_csv.exists()
    fh = open(out_csv, "a", newline="", encoding="utf-8")
    w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
    if new_file:
        w.writeheader()
    t0 = time.time()
    n_new = 0
    for ci, c in enumerate(conds):
        if c.label in done:
            continue
        if c.instr:
            prompts = [build_instr_prompt(tok, q, c.instr[pid % len(c.instr)].format(a=c.a, b=c.b))
                       for pid, q in enumerate(questions)]
        else:
            prompts = neutral
        seed = V.pair_seed(slug, c.label, "", 0, "sample")
        if c.vec is None:
            answers = _generate(model, tok, prompts, args.batch_size, sample, seed)
        else:
            with ActivationSteerer(model, c.vec, coeff=coeff, layer_idx=layer, positions="all"):
                answers = _generate(model, tok, prompts, args.batch_size, sample, seed)
        flat = [(pid, s, txt) for pid, alts in enumerate(answers) for s, txt in enumerate(alts)]
        scores = _encode_batch(encoder, [t for _, _, t in flat])
        for (pid, s, txt), sc in zip(flat, scores):
            w.writerow({"steer": c.label, "prompt_id": pid, "sample": s, "kind": c.kind, "a": c.a,
                        "b": c.b, "k": c.k, "norm_mode": c.norm_mode,
                        "vec_norm": round(V.norm(c.vec), 4) if c.vec is not None else 0.0,
                        "question": questions[pid], "answer": txt, **sc})
        fh.flush()
        n_new += len(flat)
        el = time.time() - t0
        print(f"  [{ci + 1}/{len(conds)}] {c.label}: {len(flat)} rows, "
              f"{n_new / el:.1f} gen/s overall", flush=True)
    fh.close()
    info["finished"] = dt.datetime.now().isoformat(timespec="seconds")
    info["new_rows"] = n_new
    _write_meta(out_csv, info)
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
