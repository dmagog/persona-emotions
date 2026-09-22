"""Self-check of the pipeline logic. No GPU, no network, done in seconds.

Exercises the places the pipeline has already broken in: config parsing, layer
and dtype selection, operating-point choice, the degeneration metric, matrix
resume, finding the path to the layers and the steering-vector guards.

Run it before every commit that touches the pipeline:
    python -m emotion.selftest
"""
from __future__ import annotations

import csv
import re
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parent.parent
FAILED: list[str] = []


def check(cond: bool, name: str, extra: str = "") -> None:
    print(f"  [{'OK  ' if cond else 'FAIL'}] {name}" + (f": {extra}" if extra else ""))
    if not cond:
        FAILED.append(name)


class Cfg:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def test_layers_and_dtype() -> None:
    from emotion.loader import resolve_dtype, resolve_layers, num_layers, hidden_size

    print("\n-- layers and dtype")
    flat = Cfg(num_hidden_layers=28, hidden_size=3072)
    nested = Cfg(vision_config=Cfg(num_hidden_layers=27),
                 text_config=Cfg(num_hidden_layers=36, hidden_size=2048))

    check(num_layers(flat) == 28, "layer count from a flat config")
    check(num_layers(nested) == 36, "layer count from the nested config, not from vision",
          num_layers(nested))
    check(hidden_size(nested) == 2048, "hidden_size from the nested config")

    # Fractions have to reproduce the candidates that used to be set by hand.
    check(resolve_layers(Cfg(num_hidden_layers=28), [0.35, 0.45, 0.55]) == [10, 13, 15],
          "fractions over 28 layers give [10,13,15]")
    check(resolve_layers(Cfg(num_hidden_layers=26), [0.35, 0.45, 0.55]) == [9, 12, 14],
          "fractions over 26 layers give [9,12,14]")
    check(resolve_layers(flat, ["10", "13", "15"]) == [10, 13, 15], "indices given as strings")
    check(resolve_layers(flat, [12, 12, 0.45]) == [12, 13], "duplicates collapse")
    try:
        resolve_layers(flat, [99]); check(False, "a layer out of range fails")
    except ValueError:
        check(True, "a layer out of range fails")

    d, _ = resolve_dtype("float16", Cfg())
    check(d is torch.float16, "an explicit dtype comes through")
    d, why = resolve_dtype("auto", Cfg(torch_dtype=torch.bfloat16))
    check(d in (torch.bfloat16, torch.float16) and why, "auto explains its choice", why[:60])
    try:
        resolve_dtype("float8", Cfg()); check(False, "an unknown dtype fails")
    except ValueError:
        check(True, "an unknown dtype fails")


def test_steerer_guards() -> None:
    from emotion.activation_steer import ActivationSteerer, _hidden_size

    print("\n-- steering guards")

    class Fake(nn.Module):
        def __init__(self, hidden=8, path="model.layers", nested=False):
            super().__init__()
            blocks = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(4)])
            if path == "model.layers":
                self.model = nn.Module(); self.model.layers = blocks
            else:
                self.model = nn.Module(); self.model.language_model = nn.Module()
                self.model.language_model.layers = blocks
            self.config = (Cfg(text_config=Cfg(hidden_size=hidden)) if nested
                           else Cfg(hidden_size=hidden))

    check(_hidden_size(Cfg(text_config=Cfg(hidden_size=32))) == 32, "hidden_size from a nested config")
    m = Fake()
    try:
        ActivationSteerer(m, torch.ones(8), layer_idx=1); check(True, "the model.layers path")
    except Exception as e:
        check(False, "the model.layers path", str(e)[:60])
    try:
        ActivationSteerer(Fake(path="mm"), torch.ones(8), layer_idx=1)
        check(True, "the multimodal path")
    except Exception as e:
        check(False, "the multimodal path", str(e)[:60])

    for vec, name, must in [
        (torch.full((8,), float("nan")), "NaN is rejected", "nan"),
        (torch.zeros(8), "a zero vector is rejected", "zero steering vector"),
        (torch.ones(5), "a wrong dimension is rejected", "!="),
    ]:
        try:
            ActivationSteerer(m, vec, layer_idx=1); check(False, name, "did not fail")
        except ValueError as e:
            check(must.lower() in str(e).lower(), name)


def test_operating_point() -> None:
    import pandas as pd
    from emotion.run_model_chain import pick_operating_point

    print("\n-- operating point")
    GOOD = "A calm and quite distinct sentence about the driving lesson, number {i}, varied."
    BAD = "i am i am i am i am i am i am i am i am i am"
    rows = []
    spec = {(10, 4): (0.10, 0.0), (10, 8): (0.20, 0.0), (10, 16): (0.35, 0.5),
            (13, 4): (0.15, 0.0), (13, 8): (0.30, 0.0), (13, 16): (0.55, 0.6),
            (15, 4): (0.18, 0.2), (15, 8): (0.40, 0.7), (15, 16): (0.60, 0.9)}
    for (L, c), (score, degen) in spec.items():
        for i in range(10):
            rows.append({"emotion": "anger", "layer": L, "coeff": float(c), "prompt_id": i,
                         "target_score": score,
                         "answer": BAD if i < degen * 10 else GOOD.format(i=i)})
    tmp = Path(tempfile.mkdtemp()) / "sweep.csv"
    pd.DataFrame(rows).to_csv(tmp, index=False)
    layer, coeff = pick_operating_point(tmp, max_degen=0.10)
    check((layer, coeff) == (13, 8.0),
          "the strongest point among the non-degenerate ones wins", f"L{layer}/c{coeff:g}")


def test_degeneracy_metric() -> None:
    from emotion.collect_results import rep_ratio

    print("\n-- degeneration metric")
    short_ok = "I'm scared. I can't breathe. I just want to run away now."
    long_ok = ("My phone buzzed with a text from Sarah saying she was coming over. "
               "I was surprised, she'd been busy lately, so I wasn't expecting her.")
    bad = "we are the best best best best best best best best best best best"
    check(rep_ratio(short_ok) == 0.0, "short ordinary text is not degenerate", f"{rep_ratio(short_ok):.3f}")
    check(rep_ratio(long_ok) < 0.05, "long ordinary text is not degenerate", f"{rep_ratio(long_ok):.3f}")
    check(rep_ratio(bad) > 0.3, "repetition is recognised", f"{rep_ratio(bad):.3f}")


def test_matrix_resume() -> None:
    from emotion.space import ISEAR_EMOTIONS
    from emotion.steer_specificity import load_checkpoint

    print("\n-- matrix resume")
    fields = ["steer", "layer", "coeff", "prompt_id", *ISEAR_EMOTIONS, "answer"]

    def write_ckpt(layer, coeff, tail=True) -> Path:
        p = Path(tempfile.mkdtemp()) / "m.partial.csv"
        with p.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields); w.writeheader()
            for st in ("baseline", "anger"):
                for pid in range(3):
                    w.writerow({"steer": st, "layer": "" if st == "baseline" else layer,
                                "coeff": 0.0 if st == "baseline" else coeff,
                                "prompt_id": pid, "answer": f"t {st} {pid}",
                                **{e: 0.1 for e in ISEAR_EMOTIONS}})
            if tail:
                fh.write("anger,14,4.0,3,0.1,0.1")  # cut off mid-row
        return p

    want = {"baseline": ("", 0.0), "anger": ("14", 4.0)}
    rows, done = load_checkpoint(write_ckpt(14, 4.0), fields, want)
    check(len(done) == 6 and len(rows) == 6, "finished pairs are restored", f"{len(done)}")
    check(("anger", 3) not in done, "the truncated row is recomputed")

    # Under --strength the coefficient is derived from the activations and drifts
    # in the last digits. That is still the same point, so resume is allowed.
    rows, done = load_checkpoint(write_ckpt(14, 4.0001), fields, want)
    check(len(done) == 6, "coefficient jitter does not block resume", f"{len(done)}")

    # Another layer is another run, and appending to it is not allowed.
    for label, ck in (("layer", write_ckpt(16, 4.0)), ("coefficient", write_ckpt(14, 8.0))):
        try:
            load_checkpoint(ck, fields, want)
            check(False, f"a foreign {label} in the checkpoint stops the run", "did not stop")
        except SystemExit as e:
            check("different operating point" in str(e),
                  f"a foreign {label} in the checkpoint stops the run")


def test_stamp() -> None:
    """Protocol stamp: that an artifact is usable has to be checkable."""
    from emotion import stamp

    print("\n-- protocol stamp")
    tmp = Path(tempfile.mkdtemp())
    src = tmp / "pairs.csv"
    src.write_text("a,b\n1,2\n", encoding="utf-8")
    art = tmp / "matrix.csv"
    art.write_text("steer,score\nanger,1\n", encoding="utf-8")
    params = {"model": "Qwen/Qwen3-1.7B", "layer": 10, "coeff": 8.0}

    check(stamp.check(art, "matrix", params, [src]).state == "unstamped",
          "a finished artifact without a stamp is not claimed as ours")
    stamp.write_stamp(art, "matrix", params, [src])
    check(stamp.check(art, "matrix", params, [src]).state == "current",
          "our own artifact is recognised")

    # The point of difference from an mtime check: same content means usable.
    # Files travel between the Windows box and the Mac, where mtime does not
    # survive the copy.
    import os
    os.utime(src, (0, 0))
    check(stamp.check(art, "matrix", params, [src]).state == "current",
          "a changed modification time does not make the artifact foreign")

    v = stamp.check(art, "matrix", {**params, "layer": 13}, [src])
    check(v.state == "stale" and any("layer" in c for c in v.changed),
          "a changed layer is caught and named", "; ".join(v.changed)[:60])

    src.write_text("a,b\n1,3\n", encoding="utf-8")
    v = stamp.check(art, "matrix", params, [src])
    check(v.state == "stale" and any("input" in c for c in v.changed),
          "a changed input is caught", "; ".join(v.changed)[:60])

    # Editing the artifact by hand is a mismatch too, not "our own file".
    src.write_text("a,b\n1,2\n", encoding="utf-8")
    art.write_text("steer,score\nanger,999\n", encoding="utf-8")
    v = stamp.check(art, "matrix", params, [src])
    check(v.state == "stale" and any("changed" in c for c in v.changed),
          "a hand-edited artifact is caught", "; ".join(v.changed)[:60])
    stamp.write_stamp(art, "matrix", params, [src])

    check(stamp.check(tmp / "absent.csv", "matrix", params).state == "missing",
          "a missing artifact is recomputed")
    empty = tmp / "vectors"
    empty.mkdir()
    check(stamp.check(empty, "vectors", params).state == "missing",
          "an empty directory is not a result")

    # decide: on a mismatch the chain stops instead of deciding for the human.
    try:
        stamp.decide(art, "matrix", {**params, "layer": 13}, [])
        check(False, "a mismatched artifact stops the chain", "did not stop")
    except SystemExit as e:
        check("layer" in str(e), "a mismatched artifact stops the chain")
    check(stamp.decide(art, "matrix", {**params, "layer": 13}, [],
                       recompute_stale=True) is True,
          "--recompute-stale recomputes it")
    src.write_text("a,b\n1,4\n", encoding="utf-8")
    try:
        stamp.decide(art, "matrix", params, [src])
        check(False, "a changed input stops the chain too", "did not stop")
    except SystemExit:
        check(True, "a changed input stops the chain too")

    # Input keys in a stamp have to travel: artifacts are computed on one machine
    # and live in git. A path inside the repository is stored relative.
    inner = stamp.REPO / "runs" / "_stamp_check"
    try:
        inner.mkdir(parents=True, exist_ok=True)
        probe = inner / "x.csv"
        probe.write_text("a\n", encoding="utf-8")
        digs = stamp.input_digests([probe])
        check(list(digs) == ["runs/_stamp_check/x.csv"],
              "an input inside the repo is keyed by a relative path", list(digs)[0])
    finally:
        import shutil
        shutil.rmtree(inner, ignore_errors=True)
    # os.path.isabs rather than startswith("/"): on Windows an absolute path
    # begins with a drive letter, and the Unix check failed this self-check on the
    # 2070, which is the gate in front of every night queue.
    import os
    digs = stamp.input_digests([src])
    check(os.path.isabs(list(digs)[0]),
          "an input outside the repo stays absolute", list(digs)[0][:40])

    # Transfers from the 2070: a broken copy leaves a truncated CSV that still
    # reads, just with fewer rows. The eye misses that; the stamp does not.
    total, bad = stamp.verify(tmp)
    check(total == 1 and not bad, "an intact artifact passes verification", f"{total}, {bad}")
    art.write_text("steer,score\n", encoding="utf-8")
    total, bad = stamp.verify(tmp)
    check(len(bad) == 1 and "no longer matches" in bad[0],
          "a truncated artifact fails verification")


def test_stage_specs() -> None:
    """Stages and their parameters live in one place, or the stamp starts lying."""
    from emotion.run_model_chain import build_parser, build_specs, load_model_config

    print("\n-- stage specs")
    cfg_file = REPO / "configs" / "models" / "qwen3-1.7b.yaml"
    a = build_parser().parse_args([])
    cfg = load_model_config(cfg_file)
    for k, v in cfg.items():
        if k != "_raw" and hasattr(a, k):
            setattr(a, k, v)
    a.model_config = cfg.get("_raw", {})

    specs = build_specs(a, {"layer": 10, "op_coeff": 8.0, "candidates": [10, 13, 15]})
    check(set(specs) >= {"pairs", "vectors", "sweep", "matrix"},
          "every expensive stage is described", ", ".join(sorted(specs)))
    check(specs["matrix"].params["layer"] == 10 and specs["matrix"].params["coeff"] == 8.0,
          "the operating point is part of the matrix fingerprint")
    check(specs["vectors"].inputs == [specs["pairs"].artifact],
          "vectors depend on the pairs")
    check(all(s.artifact for s in specs.values()), "every stage has an artifact")

    # Recompute: what is finished is moved out of the way but not lost. Without
    # this the stage either reuses an unstamped artifact or resumes from it.
    from emotion.run_model_chain import set_aside
    tmp = Path(tempfile.mkdtemp())
    art = tmp / "steer_specificity_raw.csv"
    art.write_text("steer,score\nanger,1\n", encoding="utf-8")
    from emotion import stamp as st_mod
    st_mod.write_stamp(art, "matrix", {"layer": 10}, [])
    bak = set_aside(art)
    check(bak is not None and bak.exists() and not art.exists(),
          "the previous artifact is set aside, not erased", bak.name if bak else "none")
    check(not st_mod.stamp_path(art).is_file(),
          "the stamp left with the artifact, rather than lying about an empty spot")
    check(set_aside(tmp / "absent.csv") is None, "nothing to set aside, carry on quietly")

    # The pairs directory is set aside whole: the stage resumes from the separate
    # <emotion>_pos.csv files, so removing the combined file alone would not make
    # it recompute.
    d = tmp / "pairs"; d.mkdir(); (d / "anger_pos.csv").write_text("x", encoding="utf-8")
    check(set_aside(d) is not None and not d.exists(), "a directory is set aside whole")

    # Changing a parameter has to change the fingerprint, or the check is useless.
    from emotion.stamp import fingerprint
    fp1, _ = fingerprint("matrix", specs["matrix"].params, [])
    p2 = {**specs["matrix"].params, "per_emotion": 16}
    fp2, _ = fingerprint("matrix", p2, [])
    check(fp1 != fp2, "changing per_emotion changes the fingerprint")


def test_protocol() -> None:
    """One plane or not. Rows recorded differently have to say so."""
    from emotion.protocol import PLANE_KEYS, PROTOCOL, card, compare_planes, report

    print("\n-- protocol and plane")
    check(all(c.where for c in PROTOCOL),
          "every decision carries a paper reference or an explicit dash")
    check(any(c.deviation for c in PROTOCOL) and any(not c.deviation for c in PROTOCOL),
          "deviations are separated from matches",
          f"{sum(c.deviation for c in PROTOCOL)} of {len(PROTOCOL)} are deviations")
    rendered = card()
    check(all(c.name in rendered for c in PROTOCOL),
          "the card lists every protocol decision")

    same = {"A": {"protocol": 1, "n_prompts": 56, "drive": "coeff=8",
                  "judge_filtered": True, "max_degen": 0.10, "stamped": True}}
    same["B"] = dict(same["A"])
    check(not compare_planes(same), "identical runs count as comparable")
    check("share one plane" in " ".join(report(same)), "and the report says so")

    odd = dict(same)
    odd["C"] = {**same["A"], "n_prompts": 14, "judge_filtered": False, "stamped": False}
    diff = compare_planes(odd)
    check(set(diff) == {"n_prompts", "judge_filtered"},
          "the mismatches are named one by one", ", ".join(sorted(diff)))
    txt = " ".join(report(odd))
    check("C" in txt and "No protocol stamp" in txt,
          "the odd row and the missing stamp both reach the report")
    check(all(k in PLANE_KEYS for k in diff), "every plane key carries a description")

    # Layer and coefficient differ per model, which is NOT a reason to call rows
    # incomparable: the flag would light up on the whole table and stop being read.
    byline = {"A": {**same["A"]}, "B": {**same["A"]}}
    check(not compare_planes(byline), "differing layers alone do not break the plane")

    # The precision varied between runs and never reached the manifest, so we
    # recover it from the loader log; otherwise fp16 and bf16 runs look alike.
    from emotion.protocol import dtype_of
    tmp = Path(tempfile.mkdtemp())
    check("dtype" in PLANE_KEYS, "precision is part of the plane")
    check(dtype_of(tmp, {}) is None, "with neither manifest nor log, an honest unknown")
    (tmp / "chain.log").write_text(
        "[loader] org/m: dtype=torch.bfloat16, the card supports it\n", encoding="utf-8")
    check(dtype_of(tmp, {}) == "bfloat16", "precision is recovered from the log")
    check(dtype_of(tmp, {"dtype": "float16"}) == "float16", "the manifest outranks the log")


def test_judge_cache() -> None:
    """The judge cache has to remember WHICH text a score was given for."""
    import json

    from emotion.judge_specificity import _load_cache, answer_key

    print("\n-- judge cache")
    tmp = Path(tempfile.mkdtemp()) / "j.cache.jsonl"
    old_ans, new_ans = "I am furious about this.", "I feel calm today."
    with tmp.open("w", encoding="utf-8") as fh:
        # New-style entry, with a digest of the text
        fh.write(json.dumps({"steer": "anger", "pid": "0", "measured": "anger",
                             "answer": answer_key(old_ans), "score": 90}) + "\n")
        # Old-style entry, without one
        fh.write(json.dumps({"steer": "anger", "pid": "1", "measured": "anger",
                             "score": 80}) + "\n")
    done = _load_cache(tmp)
    check(("anger", "0", "anger", answer_key(old_ans)) in done,
          "a score for the same text is reused")
    check(("anger", "0", "anger", answer_key(new_ans)) not in done,
          "a score for another text is NOT reused, or the judge describes a foreign run")
    check(len(done) == 1, "entries without a text digest are dropped", f"{len(done)}")
    check(answer_key(old_ans) != answer_key(new_ans), "different texts give different keys")


def test_judge_denominator() -> None:
    """An under-measurement by the judge must not be reported as a miss."""
    import csv as csvmod

    from emotion.collect_results import MIN_JUDGED, _argmax_hits_from_wide
    from emotion.space import ISEAR_EMOTIONS

    print("\n-- judge denominator")
    tmp = Path(tempfile.mkdtemp()) / "judge_wide.csv"
    fields = ["steer", "prompt_id", *ISEAR_EMOTIONS]

    def write(sizes: dict) -> Path:
        with tmp.open("w", newline="", encoding="utf-8") as fh:
            w = csvmod.DictWriter(fh, fieldnames=fields); w.writeheader()
            for pid in range(56):
                w.writerow({"steer": "baseline", "prompt_id": pid,
                            **{e: 10.0 for e in ISEAR_EMOTIONS}})
            for emo, n in sizes.items():
                for pid in range(n):
                    sc = {e: 10.0 for e in ISEAR_EMOTIONS}
                    sc[emo] = 90.0  # the target emotion rises, which is a hit
                    w.writerow({"steer": emo, "prompt_id": pid, **sc})
        return tmp

    full = {e: 56 for e in ISEAR_EMOTIONS}
    hits, measured, _ = _argmax_hits_from_wide(write(full))
    check((hits, measured) == (7, 7), "a full matrix gives 7 of 7", f"{hits}/{measured}")

    # granite: shame lost entirely, sadness reduced to two answers
    thin = {**full, "sadness": 2}
    del thin["shame"]
    hits, measured, sizes = _argmax_hits_from_wide(write(thin))
    check(measured == 5, "conditions with a handful of answers leave the denominator",
          f"{measured} of 7 measured")
    check(hits == 5, "an under-measurement does not count as a miss", f"{hits} hits")
    check(sizes.get("shame", 0) == 0 and sizes["sadness"] == 2,
          "condition sizes come back for the caveat")
    check(MIN_JUDGED >= 20, "the sufficiency floor is meaningful", f"{MIN_JUDGED}")


def test_compose_collector() -> None:
    """Composition: the full pair matrix and an honest suppression metric."""
    import csv as csvmod

    from emotion.collect_compose import pair_counts
    from emotion.space import ALL_PAIRS, ISEAR_EMOTIONS

    print("\n-- composition collector")
    check(len(ALL_PAIRS) == 42, "42 ordered X-Y pairs", f"{len(ALL_PAIRS)}")
    check("anger-sadness" in ALL_PAIRS and "sadness-anger" in ALL_PAIRS,
          "the pairs are ordered, so both directions count")

    # The decisive case: the suppression reference is Y under X ALONE (steer==X),
    # and NOT Y under Y (steer==Y, where Y is at its maximum, which makes almost
    # everything "suppressed" trivially). The values are chosen so that the right
    # and the wrong reference give DIFFERENT answers, and the test catches a swap.
    tmp = Path(tempfile.mkdtemp()) / "compose_allpairs.csv"
    fields = ["steer", "prompt_id", *ISEAR_EMOTIONS]
    with tmp.open("w", newline="", encoding="utf-8") as fh:
        w = csvmod.DictWriter(fh, fieldnames=fields); w.writeheader()
        def rows(steer, **sc):
            for pid in range(56):
                r = {"steer": steer, "prompt_id": pid, **{e: 0.1 for e in ISEAR_EMOTIONS}}
                r.update(sc); w.writerow(r)
        rows("baseline")
        rows("anger",   anger=0.8, sadness=0.30)   # X alone: sadness leakage of 0.30
        rows("disgust", disgust=0.8, sadness=0.60)
        rows("sadness", sadness=0.90)              # Y alone: sadness at its maximum
        # anger-sadness: the target holds; sadness 0.50 is ABOVE 0.30 -> NOT suppressed
        rows("anger-sadness",   anger=0.7, sadness=0.50)
        # disgust-sadness: sadness 0.20 is below 0.60 -> suppressed
        rows("disgust-sadness", disgust=0.7, sadness=0.20)
        # The remaining pairs stay thin, under 40 rows, so they do not count.
        for spec in ALL_PAIRS:
            if spec in ("anger-sadness", "disgust-sadness"): continue
            for pid in range(5):
                w.writerow({"steer": spec, "prompt_id": pid, **{e: 0.1 for e in ISEAR_EMOTIONS}})

    c = pair_counts(tmp)
    check(c is not None and c["n"] == 2, "only dense pairs are counted",
          f"{c['n'] if c else 'none'}")
    check(c["target"] == 2, "the target rise over baseline counts for both", f"{c['target']}")
    # Under the wrong reference (steer==Y==0.90) both pairs look "suppressed", so
    # suppress==2. Under the right one (steer==X) only disgust-sadness is
    # suppressed, so suppress==1.
    check(c["suppress"] == 1,
          "suppression is measured against X alone, not against Y alone",
          f"{c['suppress']}")

    # A thin condition, meaning too few rows, must not reach the count.
    thin = Path(tempfile.mkdtemp()) / "compose_allpairs.csv"
    with thin.open("w", newline="", encoding="utf-8") as fh:
        w = csvmod.DictWriter(fh, fieldnames=fields); w.writeheader()
        for pid in range(56):
            w.writerow({"steer": "baseline", "prompt_id": pid, **{e: 0.1 for e in ISEAR_EMOTIONS}})
        for pid in range(5):  # anger-sadness has only 5 rows, so it is thin
            w.writerow({"steer": "anger-sadness", "prompt_id": pid, **{e: 0.1 for e in ISEAR_EMOTIONS}})
    ct = pair_counts(thin)
    check("anger-sadness" in ct["thin"], "a condition under 40 rows drops out of the count")


def test_config() -> None:
    from emotion.run_model_chain import load_model_config

    print("\n-- model configs")
    cfgs = sorted((REPO / "configs" / "models").glob("*.yaml"))
    check(bool(cfgs), "the configs are in place", f"{len(cfgs)} files")
    for f in cfgs:
        c = load_model_config(f)
        ok = c.get("model") and c.get("slug") and c.get("layers")
        check(bool(ok), f"{f.name} parses", f"{c.get('model')} -> {c.get('slug')}")


def test_prompt_consistency() -> None:
    """The extraction prompt has to be built the same way as the applying one."""
    print("\n-- prompt consistency")
    src = {}
    for f in ("emotion/generate_pairs.py", "emotion/steer_eval.py",
              "emotion/steer_specificity.py", "emotion/hidden_states.py"):
        p = REPO / f
        if p.is_file():
            src[f] = p.read_text(encoding="utf-8")
    bare = [f for f, s in src.items()
            if re.search(r'tokenizer\(\s*prompt\s*,\s*return_tensors="pt"\s*\)', s)]
    check(not bare, "no tokenization without add_special_tokens",
          ", ".join(bare) or "all explicit")
    thinking = [f for f, s in src.items()
                if "apply_chat_template" in s and "enable_thinking" not in s]
    check(not thinking, "reasoning mode suppressed everywhere",
          ", ".join(thinking) or "in every place")

    # The compute dtype has to reach EVERY stage that loads a model. The pairs
    # never got it: load_model fell back to its bf16 default, so within one run
    # the pairs ended up in bf16 and the activations in fp16.
    chain = (REPO / "emotion" / "run_model_chain.py").read_text(encoding="utf-8")
    passed_on = chain.count("+ dtype_arg")
    check(passed_on >= 4, "the dtype reaches all four stages",
          f"stages with an explicit dtype: {passed_on}")
    pairs_src = (REPO / "emotion" / "generate_pairs.py").read_text(encoding="utf-8")
    check("load_model(args.model, dtype=" in pairs_src,
          "the pair generator loads the model with an explicit dtype")
    check(re.search(r"load_model\(args\.model\)\s*$", pairs_src, re.M) is None,
          "no dtype-less pair load is left anywhere")


def main() -> None:
    sys.path.insert(0, str(REPO))
    print("=== pipeline self-check ===")
    for fn in (test_layers_and_dtype, test_steerer_guards, test_operating_point,
               test_degeneracy_metric, test_matrix_resume, test_stamp,
               test_stage_specs, test_protocol, test_judge_cache, test_judge_denominator, test_compose_collector,
               test_config,
               test_prompt_consistency):
        try:
            fn()
        except Exception as e:
            check(False, f"{fn.__name__} crashed", f"{type(e).__name__}: {str(e)[:90]}")

    print()
    if FAILED:
        print(f"FAILURES: {len(FAILED)}")
        for f in FAILED:
            print(f"  - {f}")
        sys.exit(1)
    print("all clear")


if __name__ == "__main__":
    main()
