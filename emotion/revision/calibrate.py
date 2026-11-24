"""Plan p5: re-calibration on new anger prompts (R1 M6; requirements E7, E8).

The published sweep used ``emotion_data_eval/anger.json`` questions[:16], which
also supplies the anger part of the 56-prompt held-out set (questions[:8]).
This plan repeats the sweep on ``data_generation/calibration_anger_v2.json``,
which is disjoint from extraction and evaluation data, over the published
candidate layers and an extended coefficient grid (by default the values
missing from the published grid, including values above 16, because
Qwen-2.5-1.5B sits at c = 16).

The selection rule is the published one (``run_model_chain.pick_operating_point``:
largest mean anger change with repeated-4-gram degeneration <= 10%). The
published operating point stays the main result; a different selection is
reported as a robustness check, with the held-out result at both points.

Outputs: ``p5_sweep.csv`` (same columns as ``layer_sweep_anger.csv``) and
``p5_selection.json``.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import torch

from emotion import metrics
from emotion.activation_steer import ActivationSteerer
from emotion.classifier_encoder import ClassifierBasedEncoder
from emotion.loader import LoadSpec, load_model_and_tokenizer
from emotion.revision.prompts import calibration_prompts
from emotion.steer_eval import build_prompt

REPO = Path(__file__).resolve().parents[2]
CALIB_FILE = REPO / "data_generation" / "calibration_anger_v2.json"
PUBLISHED_COEFFS = [0, 2, 4, 6, 8, 16]


def run_p5(args, cfg, meta, out_dir: Path) -> None:
    from emotion.revision.run import _generate
    from emotion.run_model_chain import pick_operating_point

    layers = [int(x) for x in meta.get("candidates") or [meta["op_point"]["layer"]]]
    coeffs = [float(x) for x in args.coeffs.split(",")]
    if args.with_published_coeffs:
        coeffs = sorted(set(coeffs) | set(map(float, PUBLISHED_COEFFS)))
    if 0.0 not in coeffs:
        coeffs = [0.0] + coeffs
    questions = calibration_prompts(CALIB_FILE)
    if args.limit_prompts:
        questions = questions[:args.limit_prompts]
    print(f"{cfg['slug']}: p5 layers {layers} coeffs {coeffs} on {len(questions)} new prompts "
          f"= {len(layers) * len(coeffs) * len(questions)} generations", flush=True)
    if args.dry_run:
        return

    model, tok, _ = load_model_and_tokenizer(LoadSpec.from_config(cfg))
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    encoder = ClassifierBasedEncoder(device=0 if torch.cuda.is_available() else None)
    vec_all = torch.load(REPO / "emotion_vectors" / cfg["slug"] / "anger_response_avg_diff.pt",
                         map_location="cpu")
    prompts = [build_prompt(tok, q) for q in questions]

    out_csv = out_dir / "p5_sweep.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["emotion", "layer", "coeff", "prompt_id", "target_score", "answer"])
        w.writeheader()
        base = None
        for layer in layers:
            vec = vec_all[layer + 1].to(torch.float32)
            for c in coeffs:
                if c == 0 and base is not None:
                    answers = base           # the unsteered answers do not depend on the layer
                elif c == 0:
                    answers = base = [a[0] for a in _generate(model, tok, prompts, args.batch_size, None, 0)]
                else:
                    with ActivationSteerer(model, vec, coeff=c, layer_idx=layer, positions="all"):
                        answers = [a[0] for a in _generate(model, tok, prompts, args.batch_size, None, 0)]
                for pid, ans in enumerate(answers):
                    s = metrics.target_score(encoder.encode(ans), "anger", encoder.space)
                    w.writerow({"emotion": "anger", "layer": layer, "coeff": c, "prompt_id": pid,
                                "target_score": round(s, 4), "answer": ans})
                f.flush()
                print(f"  layer {layer} coeff {c}: done", flush=True)

    layer_sel, coeff_sel = pick_operating_point(out_csv, float(meta["op_point"].get("max_degen", 0.10)))
    sel = {"published": meta["op_point"], "recalibrated": {"layer": layer_sel, "coeff": coeff_sel},
           "prompts": str(CALIB_FILE.relative_to(REPO)), "n_prompts": len(questions),
           "coeffs": coeffs, "layers": layers,
           "changed": [layer_sel, coeff_sel] != [int(meta["op_point"]["layer"]), float(meta["op_point"]["coeff"])]}
    (out_dir / "p5_selection.json").write_text(json.dumps(sel, indent=2))
    print(json.dumps(sel, indent=2))
