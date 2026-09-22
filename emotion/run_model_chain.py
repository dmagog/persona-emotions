"""The full chain for one model: pairs, vectors, layer selection, specificity matrix.

Every stage is idempotent: when its artifact is in place the stage is skipped.
No judge is needed here. The pair filter is optional and emotion scores come
from the local go_emotions encoder. Judge stages run separately.

Usage:
    python -m emotion.run_model_chain --model Qwen/Qwen3-1.7B --slug Qwen3-1.7B
    python -m emotion.run_model_chain --model google/gemma-2-2b-it --slug gemma-2-2b-it --layers 9,12,14
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from emotion import stamp
from emotion.space import ALL_PAIRS, ISEAR_EMOTIONS

REPO = Path(__file__).resolve().parent.parent


def env_manifest() -> dict:
    """Record what a row of the results table needs in order to be reproduced."""
    import platform
    import subprocess
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
                             capture_output=True, text=True).stdout.strip()
    except Exception:
        sha = "?"
    try:
        import torch, transformers
        versions = {"torch": torch.__version__, "transformers": transformers.__version__,
                    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}
    except Exception:
        versions = {}
    return {"git_sha": sha, "python": platform.python_version(), **versions}


def sh(args: list[str], log: Path) -> int:
    """Run a stage, streaming its output into the log."""
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    with log.open("a", encoding="utf-8") as fh:
        fh.write(f"\n[stage begin {stamp}] {' '.join(args)}\n")
        fh.flush()
        rc = subprocess.run(args, stdout=fh, stderr=subprocess.STDOUT, cwd=REPO).returncode
        fh.write(f"[stage end rc={rc}] {time.strftime('%Y-%m-%dT%H:%M:%S')}\n")
    print(f"  rc={rc}  ({' '.join(args[2:5])}…)", flush=True)
    return rc


def default_layers(model: str, token: str | None) -> list[int]:
    """Candidate layers at 0.35, 0.45 and 0.55 of model depth, as the protocol says."""
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(model, token=token)
    n = getattr(cfg, "num_hidden_layers", None)
    if n is None:  # a nested config, which means multimodal: better not to proceed
        raise SystemExit(
            f"{model}: num_hidden_layers is absent from the top level of the config "
            f"({type(cfg).__name__}). This is probably a multimodal or hybrid "
            "architecture, which needs its own layer path in ActivationSteerer."
        )
    return sorted({max(1, round(n * f)) for f in (0.35, 0.45, 0.55)})


def _rep_ratio(text: str, n: int = 4) -> float:
    """Share of repeated n-grams: 0 when all are unique, 1 under solid repetition.

    Not the frequency of the single most common n-gram, which rises on short
    texts. A nine-word answer holds only six 4-grams, so even a fully unique one
    scores 1/6 = 0.17, and ordinary short answers get flagged as degenerate.
    """
    import re
    w = re.findall(r"\w+", str(text).lower())
    if len(w) < n + 4:
        return 0.0
    grams = [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]
    return 1.0 - len(set(grams)) / len(grams)


def pick_operating_point(sweep_csv: Path, max_degen: float = 0.10) -> tuple[int, float]:
    """Pick the strongest intervention that does not yet break the text.

    One shared coefficient is not comparable across models, because vector norms
    and activation scales differ. One shared dimensionless strength is not
    comparable either, since the link between strength and breakage is model
    specific. So the point is chosen by behavior: the highest target score among
    cells whose degenerate-output share stays within the ceiling. Models are then
    compared at a comparable effect rather than at an equal input.
    """
    d = pd.read_csv(sweep_csv)
    d["degen"] = d["answer"].map(lambda t: _rep_ratio(t) > 0.15)
    g = (d[d["coeff"] > 0]
         .groupby(["layer", "coeff"])
         .agg(score=("target_score", "mean"), degen=("degen", "mean"))
         .reset_index())
    print("  sweep (anger): " + "; ".join(
        f"L{int(r.layer)}/c{r.coeff:.0f} score {r.score:.3f} degen {r.degen:.0%}"
        for r in g.itertuples()), flush=True)
    ok = g[g["degen"] <= max_degen]
    if ok.empty:
        best = g.loc[g["degen"].idxmin()]
        print(f"  WARNING: no cell stays within {max_degen:.0%} degeneration, "
              f"taking the lowest ({best.degen:.0%})", flush=True)
    else:
        best = ok.loc[ok["score"].idxmax()]

    # Caveats that would otherwise surface only when reading the table, or never.
    per_cell = int(d.groupby(["layer", "coeff"]).size().min())
    if per_cell and 1.0 / per_cell > max_degen:
        print(f"  CAVEAT: {per_cell} prompts per cell, so the degeneration share "
              f"moves in steps of {1 / per_cell:.0%}. A {max_degen:.0%} ceiling then "
              f"means none out of {per_cell}, not at most {max_degen:.0%}",
              flush=True)
    grid = sorted({float(c) for c in g["coeff"] if float(c) > 0})
    if grid and abs(float(best["coeff"]) - grid[-1]) < 1e-9:
        print(f"  CAVEAT: the top of the grid was selected ({grid[-1]:g}), so the "
              "optimum may lie beyond it and the grid should be widened", flush=True)
    elif len(grid) > 1 and abs(float(best["coeff"]) - grid[0]) < 1e-9:
        print(f"  CAVEAT: the bottom of the grid was selected ({grid[0]:g}), so "
              "anything stronger already broke the text and the grid needs "
              "intermediate values", flush=True)
    return int(best["layer"]), float(best["coeff"])



def load_model_config(path: Path) -> dict:
    """Read a model config, yaml or json, flattened into chain arguments."""
    import yaml
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    st = raw.get("stages") or {}
    flat = {
        "model": raw.get("hf_id"),
        "slug": raw.get("slug") or (raw.get("hf_id") or "").split("/")[-1],
        "layers": raw.get("sweep", {}).get("layers") or st.get("sweep", {}).get("layers"),
        "sweep_coeffs": st.get("sweep", {}).get("coeffs"),
        "max_degen": st.get("sweep", {}).get("max_degen"),
        "sweep_prompts": st.get("sweep", {}).get("n_prompts"),
        "per_emotion": st.get("matrix", {}).get("per_emotion"),
        "max_tokens": st.get("pairs", {}).get("max_tokens"),
        "batch_size": st.get("pairs", {}).get("batch_size"),
        "dtype": (raw.get("load") or {}).get("dtype"),
        "compose": (st.get("compose") or {}).get("specs"),
    }
    # lists become strings, since the chain passes them on as arguments
    if isinstance(flat["layers"], list):
        flat["layers"] = ",".join(str(x) for x in flat["layers"])
    if isinstance(flat["sweep_coeffs"], list):
        flat["sweep_coeffs"] = ",".join(str(x) for x in flat["sweep_coeffs"])
    # 'allpairs' in a config expands to the full 42-pair matrix.
    if flat.get("compose") == "allpairs" or flat.get("compose") == ["allpairs"]:
        flat["compose"] = ",".join(ALL_PAIRS)
    elif isinstance(flat.get("compose"), list):
        flat["compose"] = ",".join(str(x) for x in flat["compose"])
    flat["_raw"] = raw
    return {k: v for k, v in flat.items() if v is not None}


def set_aside(artifact: Path) -> Path | None:
    """Move a finished artifact out of the way before recomputing. Never delete it.

    Recomputing is needed where no stamp exists, since such an artifact is reused
    in silence, and stages resume from whatever is already written, so restarting
    the chain is not enough on its own. The previous result cost hours and may
    still be wanted for comparison, so it is renamed rather than erased.
    """
    if not stamp.present(artifact):
        return None
    bak = artifact.with_name(f"{artifact.name}.{time.strftime('%Y%m%d-%H%M')}.bak")
    n = 2
    while bak.exists():
        bak = artifact.with_name(f"{artifact.name}.{time.strftime('%Y%m%d-%H%M')}-{n}.bak")
        n += 1
    artifact.rename(bak)
    st = stamp.stamp_path(artifact)
    if st.is_file():
        st.rename(st.with_name(bak.name + ".stamp.json"))
    return bak


def resolve_run_dtype(model: str, configured: str | None, token: str | None) -> tuple[str, str]:
    """Decide the compute dtype once per run and record it in the manifest.

    Each stage used to decide on its own by looking at the card, and the decision
    was recorded nowhere. That is how Llama and Qwen2.5-1.5B ended up computed in
    bf16 while Qwen3 and Gemma ran in fp16, visible only in the loader log.
    """
    if configured and configured != "auto":
        return configured, "set in the config"
    try:
        from transformers import AutoConfig

        from emotion.loader import resolve_dtype
        dt, why = resolve_dtype("auto", AutoConfig.from_pretrained(model, token=token))
        return str(dt).replace("torch.", ""), why
    except Exception as e:
        return "auto", f"cannot be decided up front ({type(e).__name__}), the stage will choose"


@dataclass
class StageSpec:
    """What determines a stage result: its artifact, its settings, its inputs."""
    stage: str
    artifact: Path
    params: dict
    inputs: list[Path] = field(default_factory=list)
    # What --recompute moves aside. For pairs it is the whole directory: the
    # stage resumes from the per-emotion <emotion>_pos.csv files, so removing the
    # combined file alone would not make it recompute.
    backup: Path | None = None


def build_specs(a, meta: dict | None = None) -> dict[str, StageSpec]:
    """Hold the result-affecting settings of every stage in one place.

    Both the chain and `emotion.stamp --adopt` read them from here. If each
    computed its own, the stamp would drift from the stage at the first change
    and start reporting a match that is not there, which is worse than no stamp.
    """
    meta = meta or {}
    raw = getattr(a, "model_config", None) or {}
    dtype = getattr(a, "resolved_dtype", None) or (raw.get("load") or {}).get("dtype", "auto")
    pairs = REPO / "eval_emotion" / a.slug / "all_emotions_extract.csv"
    vec_dir = REPO / "emotion_vectors" / a.slug
    runs = REPO / "runs" / a.slug
    layer, op_coeff = meta.get("layer"), meta.get("op_coeff")
    # The variant is part of the name, since raw, sae and centered coexist in one
    # run. The older variant-free name is still recognized so earlier runs survive.
    matrix_csv = runs / f"steer_specificity_{a.variant}.csv"
    legacy = runs / "steer_specificity.csv"
    if a.variant == "raw" and legacy.is_file() and not matrix_csv.is_file():
        matrix_csv = legacy
    # The intervention is driven either by a dimensionless strength or by a
    # coefficient. Whichever was actually applied is what enters the stamp.
    drive = ({"strength": a.strength} if getattr(a, "strength", None) is not None
             else {"coeff": op_coeff})

    specs = {
        "pairs": StageSpec(
            "pairs", pairs,
            {"model": a.model, "version": "extract", "backend": "hf",
             "temperature": 0, "max_tokens": a.max_tokens,
             "batch_size": a.batch_size, "dtype": dtype,
             "prompt": raw.get("prompt") or {}},
            [REPO / "data_generation" / "emotion_data_extract"],
            backup=pairs.parent),
        "vectors": StageSpec(
            "vectors", vec_dir,
            {"model": a.model, "dtype": dtype, "pooling": "response_avg",
             "judge_scores": Path(a.judge_scores).name if a.judge_scores else None},
            [pairs]),
        "sweep": StageSpec(
            "sweep", runs / "layer_sweep_anger.csv",
            {"model": a.model, "emotion": "anger", "layers": meta.get("candidates"),
             "coeffs": a.sweep_coeffs, "n_prompts": a.sweep_prompts,
             "max_new_tokens": 120, "dtype": dtype},
            [vec_dir]),
        "matrix": StageSpec(
            "matrix", matrix_csv,
            {"model": a.model, "layer": layer, "variant": a.variant,
             "per_emotion": a.per_emotion, "max_new_tokens": 120,
             "dtype": dtype, **drive},
            [vec_dir]),
    }
    if getattr(a, "compose", None):
        # The full pair matrix goes to compose_allpairs.csv so it sits beside the
        # older four-spec compose.csv instead of overwriting it.
        full = set(a.compose.split(",")) >= set(ALL_PAIRS)
        cname = "compose_allpairs.csv" if full else "compose.csv"
        specs["compose"] = StageSpec(
            "compose", runs / cname,
            {"model": a.model, "layer": layer, "coeff": op_coeff,
             "per_emotion": a.per_emotion, "specs": a.compose, "dtype": dtype},
            [vec_dir])
    return specs


def stage_specs(config: Path) -> list[StageSpec]:
    """Build stage specs from a config and run manifest, for `emotion.stamp`."""
    a = build_parser().parse_args([])
    cfg = load_model_config(config)
    for k, v in cfg.items():
        if k != "_raw" and hasattr(a, k):
            setattr(a, k, v)
    a.model_config = cfg.get("_raw", {})
    meta_path = REPO / "runs" / a.slug / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    return list(build_specs(a, meta).values())


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="The full chain for one model.")
    ap.add_argument("--config", type=Path, default=None,
                    help="model config in yaml; command-line arguments override it")
    ap.add_argument("--model", default=None)
    ap.add_argument("--slug", default=None, help="name used for the artifact directories")
    ap.add_argument("--layers", default=None, help="comma-separated candidates; otherwise derived from model depth")
    ap.add_argument("--coeff", type=float, default=8.0)
    ap.add_argument("--per-emotion", type=int, default=8, help="8 x 7 = 56 prompts")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=1,
                    help="batch size for pair generation; 1 means one row at a time")
    ap.add_argument("--sweep-coeffs", default="0,2,4,6,8,16",
                    help="coefficient grid searched for the operating point")
    ap.add_argument("--sweep-prompts", type=int, default=16,
                    help="prompts per sweep cell; at 8 the degeneration share moves "
                         "in 12%% steps, so a 10%% ceiling means none out of eight")
    ap.add_argument("--max-degen", type=float, default=0.10,
                    help="ceiling on the degenerate-output share when selecting the operating point")
    ap.add_argument("--skip-preflight", action="store_true",
                    help="skip the preflight check, which runs by default")
    ap.add_argument("--compose", default=None,
                    help="composition specs such as joy-sadness,anger-sadness; "
                         "empty skips the stage")
    ap.add_argument("--variant", default="raw",
                    help="vector variant: raw, sae or centered, which becomes part of the artifact name")
    ap.add_argument("--strength", type=float, default=None,
                    help="dimensionless intervention strength, see steer_specificity "
                         "--strength; comparable across models, unlike a fixed coeff")
    ap.add_argument("--judge-scores", type=Path, default=None,
                    help="pair-filter CSV; without it every pair is used, which the report should note")
    ap.add_argument("--recompute", default=None,
                    help="stages to recompute despite their stamp: all, or a "
                         "comma-separated list of pairs,vectors,sweep,matrix,compose. "
                         "The previous artifact is moved to .bak rather than deleted")
    ap.add_argument("--recompute-stale", action="store_true",
                    help="recompute artifacts produced with different settings; "
                         "by default the chain stops on them")
    return ap


def main() -> None:
    args = build_parser().parse_args()

    # The config supplies defaults and an explicit argument wins, so adding a
    # model is one file in configs/models/ while one-off overrides stay possible.
    if args.config:
        cfg = load_model_config(args.config)
        # Which options were actually given: compare against a parse of empty
        # arguments. Reading sys.argv missed the --opt=value form, so
        # `--batch-size=32` silently lost to the config, contrary to the runbook.
        defaults = vars(build_parser().parse_args([]))
        given = {k for k, v in vars(args).items()
                 if k in defaults and v != defaults[k]}
        for k, v in cfg.items():
            if k != "_raw" and k not in given and hasattr(args, k):
                setattr(args, k, v)
        args.model_config = cfg.get("_raw", {})
        print(f"config: {args.config}, model {args.model}, slug {args.slug}", flush=True)
    else:
        args.model_config = {}
    if not args.model or not args.slug:
        raise SystemExit("either --config or both --model and --slug are required")

    import os
    token = os.environ.get("HF_TOKEN")

    # One dtype for the whole run rather than a per-stage decision. It goes into
    # the manifest and the stamps; otherwise comparing rows computed in fp16 and
    # bf16 means digging through loader logs.
    args.resolved_dtype, dtype_why = resolve_run_dtype(
        args.model, (args.model_config.get("load") or {}).get("dtype"), token)
    print(f"compute dtype: {args.resolved_dtype}, {dtype_why}", flush=True)
    dtype_arg = ([] if args.resolved_dtype == "auto"
                 else ["--dtype", args.resolved_dtype])

    forced = set() if not args.recompute else (
        {"pairs", "vectors", "sweep", "matrix", "compose"} if args.recompute == "all"
        else {x.strip() for x in args.recompute.split(",")})

    pairs_dir = REPO / "eval_emotion" / args.slug
    vec_dir = REPO / "emotion_vectors" / args.slug
    runs = REPO / "runs" / args.slug
    runs.mkdir(parents=True, exist_ok=True)
    log = runs / "chain.log"
    py = sys.executable

    print(f"\n=== {args.model} → runs/{args.slug} ===", flush=True)

    # 0. Preflight runs the whole chain on a single row, which is cheaper than
    # discovering a break five hours in.
    if not args.skip_preflight:
        # Always run, not only when vectors exist. New models broke here first:
        # Gemma over the system role and Qwen3 over reasoning mode, and that is
        # exactly where the check had been turned off. Without vectors the
        # preflight covers the template and generation; with them, steering too.
        meta_pre = json.loads((runs / "meta.json").read_text(encoding="utf-8")) \
            if (runs / "meta.json").is_file() else {}
        layer_hint = meta_pre.get("layer")
        if layer_hint is None:
            try:
                from emotion.loader import num_layers
                from transformers import AutoConfig
                layer_hint = round(num_layers(
                    AutoConfig.from_pretrained(args.model, token=token)) * 0.45)
            except Exception:
                layer_hint = 0
        pf = ([py, "-m", "emotion.preflight", "--model", args.model,
               "--layer", str(layer_hint)] + dtype_arg)
        # Use the model's operating point rather than the default of eight.
        # Llama-3.2-3B runs at 4 and its text breaks at 8, so the preflight used
        # to fail the queue on a coefficient the run never applies.
        pf_coeff = meta_pre.get("op_coeff", args.coeff)
        pf += (["--strength", str(args.strength)] if args.strength is not None
               else ["--coeff", str(pf_coeff)])
        if "vectors" in forced:
            # The vectors are about to be recomputed, so checking steering
            # against the ones moving to .bak would fail the run over a result
            # that has already been written off.
            pf += ["--skip-steer"]
        print("0. preflight", flush=True)
        if sh(pf, log) != 0:
            raise SystemExit("preflight failed, the queue is stopped; see the log")

    meta_path = runs / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    specs = build_specs(args, meta)

    def run_stage(key: str, cmd: list[str], fail: str) -> None:
        """Run a stage when its stamp does not match or --recompute names it."""
        s = specs[key]
        if key in forced:
            # An unstamped artifact is reused in silence, so restarting the chain
            # is not enough and the finished result has to be moved aside. It is
            # not deleted: it cost hours and may be wanted for comparison.
            bak = set_aside(s.backup or s.artifact)
            if bak:
                print(f"   {s.stage}: previous result moved to {bak.name}", flush=True)
        elif not stamp.decide(s.artifact, s.stage, s.params, s.inputs,
                              args.recompute_stale, label=s.stage):
            return
        if sh(cmd, log) != 0:
            raise SystemExit(fail)
        stamp.write_stamp(s.artifact, s.stage, s.params, s.inputs)

    # 1. pos/neg pairs from the model itself; the script resumes row by row.
    combined = pairs_dir / "all_emotions_extract.csv"
    print("1. pos/neg pairs", flush=True)
    run_stage("pairs",
              [py, "-m", "emotion.generate_pairs", "--model", args.model,
               "--version", "extract", "--output_dir", str(pairs_dir),
               "--infer_backend", "hf", "--temperature", "0",
               "--max_tokens", str(args.max_tokens),
               "--batch-size", str(args.batch_size)] + dtype_arg,
              "the pair stage failed; see the log")

    # 2. Vectors are mean(pos) - mean(neg), layer by layer.
    # This stamp includes the stamp of the pairs, so new pairs make the vectors
    # read as a mismatch and stop the chain. That is how Gemma was caught, with
    # vectors built on Qwen text that the self-generation cycle had not noticed.
    vec_probe = vec_dir / f"{ISEAR_EMOTIONS[0]}_response_avg_diff.pt"
    print("2. vector extraction", flush=True)
    if "vectors" not in forced and vec_probe.is_file() \
            and stamp.read_stamp(vec_dir) is None and combined.is_file() \
            and vec_probe.stat().st_mtime < combined.stat().st_mtime:
        # Inherited without a stamp: the old mtime check still beats nothing.
        raise SystemExit(
            f"the vectors in {vec_dir} are older than the pairs in {combined}, so "
            "they were built on different data. Move or rename them and run again, "
            "or the matrix will be computed on the wrong vectors."
        )
    cmd = [py, "-m", "emotion.extract_vectors", "--model_name", args.model,
           "--data-dir", str(pairs_dir), "--save-dir", str(vec_dir)] + dtype_arg
    if args.judge_scores:
        cmd += ["--judge-scores", str(args.judge_scores)]
    run_stage("vectors", cmd, "vector extraction failed; see the log")

    # 3. The layer sweep and the choice of operating point are separate. The
    # sweep costs half an hour of generation; choosing from a finished sweep is
    # free. So changing max_degen must not drag the sweep along with it.
    sweep_csv = runs / "layer_sweep_anger.csv"
    if not sweep_csv.is_file() and "layer" in meta and "op_coeff" in meta:
        # Inherited: the point was chosen in an earlier cycle, by hand for Qwen2.5-3B.
        layer, op_coeff = meta["layer"], meta["op_coeff"]
        print(f"3. operating point: inherited L{layer}/c{op_coeff:g}, no sweep present",
              flush=True)
    else:
        # Layers come as depth fractions (0.45) or absolute indices (13).
        # Fractions carry across models; absolute indices tie the config to one.
        if meta.get("candidates"):
            cands = meta["candidates"]
        elif args.layers:
            from transformers import AutoConfig
            from emotion.loader import resolve_layers
            cfg_model = AutoConfig.from_pretrained(args.model, token=token)
            cands = resolve_layers(cfg_model, [x.strip() for x in str(args.layers).split(",")])
        else:
            cands = default_layers(args.model, token)
        meta["candidates"] = cands
        specs = build_specs(args, meta)
        print(f"3. layer sweep {cands}", flush=True)
        run_stage("sweep",
                  [py, "-m", "emotion.steer_eval", "--model_name", args.model,
                   "--emotion", "anger", "--vector-dir", str(vec_dir),
                   "--layers", ",".join(map(str, cands)), "--coeffs", args.sweep_coeffs,
                   "--n-prompts", str(args.sweep_prompts),
                   "--out", str(sweep_csv)] + dtype_arg,
                  "the sweep failed; see the log")

        sweep_st = stamp.read_stamp(sweep_csv) or {}
        prev_op = meta.get("op_point") or {}
        same = (prev_op.get("sweep") == sweep_st.get("fingerprint")
                and prev_op.get("max_degen") == args.max_degen)
        if same:
            layer, op_coeff = prev_op["layer"], prev_op["coeff"]
            print(f"   operating point: unchanged, L{layer}/c{op_coeff:g}", flush=True)
        else:
            layer, op_coeff = pick_operating_point(sweep_csv, args.max_degen)
            print(f"   operating point: layer {layer}, coeff {op_coeff:g}", flush=True)
        meta["op_point"] = {"layer": layer, "coeff": op_coeff,
                            "max_degen": args.max_degen,
                            "sweep": sweep_st.get("fingerprint")}

    meta.update({"model": args.model, "layer": layer, "op_coeff": op_coeff,
                 "max_degen": args.max_degen,
                 "coeff": args.coeff, "strength": args.strength,
                 "max_tokens": args.max_tokens,
                 "judge_filtered": bool(args.judge_scores),
                 "dtype": args.resolved_dtype,
                 "variant": args.variant,
                 "protocol": stamp.PROTOCOL_VERSION,
                 "config": args.model_config,
                 "env": env_manifest()})
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    specs = build_specs(args, meta)

    # 4. Specificity matrix: baseline plus 7 emotions over 56 prompts, encoder scores.
    matrix_csv = specs["matrix"].artifact
    print(f"4. specificity matrix at L{layer}", flush=True)
    cmd4 = [py, "-m", "emotion.steer_specificity", "--model_name", args.model,
            "--vector-dir", str(vec_dir), "--layer", str(layer),
            "--per-emotion", str(args.per_emotion),
            "--save-answers", "--out", str(matrix_csv)] + dtype_arg
    cmd4 += (["--strength", str(args.strength)] if args.strength is not None
             else ["--coeff", str(op_coeff)])
    run_stage("matrix", cmd4, "the matrix stage failed; see the log")

    # 5. Composition adds and subtracts emotion directions. It runs behind a
    # flag because it costs as many generations again as the matrix.
    if args.compose:
        cs = specs["compose"]
        print(f"5. composition ({args.compose})", flush=True)
        if stamp.decide(cs.artifact, cs.stage, cs.params, cs.inputs,
                        args.recompute_stale, label=cs.stage):
            if sh([py, "-m", "emotion.steer_compose", "--model_name", args.model,
                   "--vector-dir", str(vec_dir), "--layer", str(layer),
                   "--coeff", str(op_coeff), "--per-emotion", str(args.per_emotion),
                   "--specs", args.compose, "--out", str(cs.artifact)] + dtype_arg,
                  log) != 0:
                print("   composition failed; the chain continues", flush=True)
            else:
                stamp.write_stamp(cs.artifact, cs.stage, cs.params, cs.inputs)

    # Record what was reused without a stamp. Such a run is comparable with the
    # rest only under an explicit caveat in the report.
    legacy_used = stamp.unstamped([s.artifact for s in specs.values()])
    meta["unstamped"] = legacy_used
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    if legacy_used:
        print(f"   reused without a stamp: {', '.join(legacy_used)}", flush=True)

    print(f"=== {args.slug}: chain complete, artifacts in {runs} ===\n", flush=True)


if __name__ == "__main__":
    main()
