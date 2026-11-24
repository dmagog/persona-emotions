# Revision experiments (IEEE Access, Access-2026-47717 resubmission)

This file is the run book for the extra experiments requested by the reviewers.
It is written so that a coding agent (or a person) can execute any part of it
without further context. **Read sections 0–3 before running anything.**

All new code is additive and lives in `emotion/revision/`. Nothing in the
published pipeline (`emotion/*.py`, `runs/`, `emotion_vectors/`) is modified.
New outputs go to `runs_revision/<slug>/`.

---

## 0. Rules for whoever runs this (agent instructions)

1. **Do not edit** anything under `runs/`, `emotion_vectors/`, `configs/`, or the
   existing modules in `emotion/`. Do not change the conditions, seeds, prompts,
   `max_new_tokens`, dtype, or the operating points. These are pre-registered
   (section 2); changing them after seeing data invalidates the analysis.
2. Operating point = `runs/<slug>/meta.json["op_point"]` (`layer`, `coeff`).
   The code reads it; do not pass your own.
3. Every generation plan is **resumable**: it appends to
   `runs_revision/<slug>/<plan>.csv` one condition at a time and skips
   conditions already present. If a run dies, rerun the same command.
4. Use **one batch size per model for the whole plan** and never mix outputs of
   different batch sizes / GPUs inside one comparison. Baseline, a-only and a−b
   are regenerated inside p1/p4 for exactly this reason.
5. Commit nothing except what section 7 lists. Large CSVs are fine to commit
   (the published runs are committed too).
6. If something fails, stop and report the error with the command; do not
   "fix" the experimental design.

---

## 1. Environment

```bash
git checkout revision-experiments
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # torch + transformers + pandas etc.
export HF_TOKEN=...                      # gated models: Llama-3.2, gemma
# judge only (no GPU needed):
export OPENAI_API_KEY=...                # OpenRouter key
export OPENAI_BASE_URL=https://openrouter.ai/api/v1
```

Checks (CPU, seconds):

```bash
python -m emotion.revision.selftest      # vector geometry + stats on synthetic data -> "selftest OK"
python -m emotion.revision.run --config configs/models/llama-3.2-3b.yaml --plan p1 --dry-run
```

Record `nvidia-smi`, `python -c "import torch, transformers; print(torch.__version__, transformers.__version__)"`;
the run also writes them into `<plan>.csv.meta.json`.

The published runs used transformers 5.9.0 (not the 4.52.3 pinned in
`requirements-inference.txt`). Either works with this code; just keep one
version for a whole plan.

---

## 2. Pre-registered analysis (fixed before any new run)

1. **Primary null control (geometry-matched random).** For pair (a, b) with
   angle θ between vₐ and v_b: r = ‖v_b‖(cosθ·v̂ₐ + sinθ·u), u a random unit
   vector orthogonal to vₐ. Then ‖vₐ − r‖ = ‖vₐ − v_b‖; only the direction of
   the residual differs. Label `a-b|geo1`, second draw `a-b|geo2`.
   Isotropic random r with ‖r‖ = ‖v_b‖ is secondary (`|iso1`).
2. **Attenuated-a controls.** λ·vₐ with λ = 1 − (‖v_b‖/‖vₐ‖)cosθ (`|lam`), and
   projection removal vₐ − proj_{v_b} vₐ (`|proj`). Norm-matched variants
   rescaled to ‖vₐ‖: `|norm` (a−b), `|geo1|norm`, `|proj|norm`.
3. **Seeds.** One random u per (model, a, b, k): seed = first 8 hex of
   sha256(`slug|a|b|k|kind`) (see `vectors.pair_seed`).
4. **Primary statistic.** ΔU = U(a−b) − U(control) on the same measured b,
   paired stratified bootstrap over prompts, separately for encoder and judge.
   Joint rate is secondary.
5. **Rule for keeping the word "selective".** On **both** instruments mean ΔU vs
   `|geo1` < 0 with 95 % CI excluding 0, **and** joint-rate excess over `|geo1`
   and over `|lam` ≥ 10 pp. Otherwise the claim is narrowed. (`stats.py` prints
   this as "Rule 5".)
6. **Rule for "partial composition".** Kept if the prediction Δ(+a) + Δ(−b),
   with Δ(−b) measured directly (p2), explains composition with slope and CCC
   no worse than submitted Table 5.
7. **Pair tests.** One set of prompt IDs for all conditions, strata = 7 emotion
   categories, 10 000 resamples, seed 1234. T > 0 and U < 0 via
   intersection-union, p = max(p_T, p_U); family = 462 pairs per instrument;
   BH (main) and BY (sensitivity).
8. **Effect threshold.** Published τ = 0.05 plus a coverage curve over τ. No new τ.
9. **Generation as in the submission:** vector index `[layer+1]`, hook on
   `model.layers[layer]`, all positions, greedy, fp16, `max_new_tokens = 120`,
   56 held-out prompts (`steer_specificity.common_prompts(8)`).

### Known issue found during revision (affects how results are reported)

The layer/coeff sweep used `data_generation/emotion_data_eval/anger.json`
questions[:16], and the held-out set contains anger questions[:8]. So prompt IDs
0–7 of the 56 were also used for operating-point selection, contrary to the
paper's "disjoint" statement. Handling:
- every stats report is produced twice: all 56 prompts and `--exclude-prompts 0-7`
  (`docs_revision/stats_published*.md` shows the published data both ways; the
  results barely change: encoder joint 347 → 347/462, judge 372 → 365/462);
- p5 re-calibrates on 32 new anger prompts
  (`data_generation/calibration_anger_v2.json`, disjoint from all eval/extract data;
  **review the prompts before running p5**).

---

## 3. Plans at a glance (who can run what)

Models: the 4-model subset (fixed in advance) is
`llama-3.2-3b`, `gemma-2-2b-it`, `qwen3-0.6b`, `falcon3-3b` (configs in `configs/models/`).

| Plan | What | Models | Generations | GPU | Est. time | Splittable? |
|---|---|---|---:|---|---|---|
| **parity** | baseline, 7 a-only, 7 pairs; compare with published numbers | any 1 (start with llama-3.2-3b) | 840 | any | 5–15 min | run first on each new GPU |
| **p1** | null controls + norm alignment; includes baseline, a-only, a−b refs | 4 subset | ≈ 21.6k per model (50 refs + 42×8 controls = 386 conditions × 56) | A100/L40S | 1–2.5 h per model | **by model** |
| **p2** | direct −v_b for 7 emotions + baseline | all 11 | 448 per model | 8 GB is enough | 20–50 min per model | by model |
| **p3** | prompting baseline ("express a", "express a while reducing b"), no steering | 4 subset | 2 800 per model | any | 15–30 min per model | by model |
| **p4** | stochastic decoding T=0.7, top-p 0.9, 5 samples | llama-3.2-3b, qwen3-0.6b | 14k per model | A100 | 0.5–1 h per model | by model |
| **p5** | calibration on new prompts + wider coeff grid | qwen2.5-1.5b, qwen3-1.7b | ≈ 1–2k per model | 8 GB is enough | 1–3 h per model | by model |
| **p6** | activation norms for strength S (forward only) | all 11 | 0 (forward pass) | any | < 2 min per model | by model |
| p7 (optional) | centred a-only | 4 subset | 448 per model | any | < 15 min per model | by model |

Natural split: **Ivan (rented A100)**: parity → p1 (4 models) → p4 (2 models) → p3/p7 (4 models).
**Georgy (RTX 2070, 8 GB)**: p2 (11 models), p5 (2 models), p6 (11 models).
Any plan for any model can be moved between machines; just don't split one
plan of one model across two GPUs.

---

## 4. Commands

All commands from the repo root. `--batch-size 16` is the default; on an A100
use 32–64 for ≤ 3B models. `--batch-size 1` reproduces the published
generation path exactly (slow).

### 4.1 Parity check (first, once per GPU)

```bash
python -m emotion.revision.run --config configs/models/llama-3.2-3b.yaml --plan parity --batch-size 32
python -m emotion.revision.stats --source parity --instruments encoder --out runs_revision/parity_report.md
```

Compare the `a-b` encoder means/joint for those 7 pairs with
`runs/<slug>/compose_allpairs.csv`. Small differences are expected (batched fp16
greedy); report them. Large ones (sign flips on most pairs) mean something is
wrong: stop.

### 4.2 p1 (main null control)

```bash
for m in llama-3.2-3b gemma-2-2b-it qwen3-0.6b falcon3-3b; do
  python -m emotion.revision.run --config configs/models/$m.yaml --plan p1 --batch-size 32
done
```

### 4.3 p2, p6 (all 11 models)

```bash
for m in falcon3-3b gemma-2-2b-it gemma-3-1b granite-3.3-2b llama-3.2-1b llama-3.2-3b \
         olmo-2-1b qwen2.5-1.5b qwen2.5-3b qwen3-0.6b qwen3-1.7b; do
  python -m emotion.revision.run --config configs/models/$m.yaml --plan p2
  python -m emotion.revision.run --config configs/models/$m.yaml --plan p6
done
```

### 4.4 p3, p4, p7

```bash
for m in llama-3.2-3b gemma-2-2b-it qwen3-0.6b falcon3-3b; do
  python -m emotion.revision.run --config configs/models/$m.yaml --plan p3
  python -m emotion.revision.run --config configs/models/$m.yaml --plan p7     # optional
done
for m in llama-3.2-3b qwen3-0.6b; do
  python -m emotion.revision.run --config configs/models/$m.yaml --plan p4 --batch-size 32
done
```

### 4.5 p5 (calibration)

```bash
python -m emotion.revision.run --config configs/models/qwen2.5-1.5b.yaml --plan p5 \
    --coeffs 10,12,14,20,24,32 --with-published-coeffs
python -m emotion.revision.run --config configs/models/qwen3-1.7b.yaml --plan p5 --with-published-coeffs
```

Writes `p5_sweep.csv` (layer_sweep format) and `p5_selection.json` (operating
point chosen by the unchanged `run_model_chain.pick_operating_point`). If the
point changes, the held-out rerun at the new point is a separate decision; the
main results stay at the published point.

### 4.6 Judge (LLM-as-judge, Llama-3.3-70B via OpenRouter; CPU only)

Run after the encoder results are in (encoder scores are already in the CSV).
New condition labels use `|`, never `:` (the judge splits on `:`).

```bash
S=runs_revision/<slug>
python -m emotion.judge_specificity --csv $S/p1.csv --concurrency 30 \
    --out-wide $S/p1_judge_wide.csv --cache $S/p1_judge.cache.jsonl
```

Same for p2, p3, p7, parity. **For p4 (5 samples per prompt)** the wide file of
`judge_specificity` keeps only one sample, so rebuild it from the cache:

```bash
python -m emotion.judge_specificity --csv $S/p4.csv --concurrency 30 --cache $S/p4_judge.cache.jsonl
python -m emotion.revision.judge_wide --csv $S/p4.csv --cache $S/p4_judge.cache.jsonl --out $S/p4_judge_wide.csv
```

The cache makes the judge resumable. Volume: p1 ≈ 150k calls per model
(21.6k answers × 7 emotions), ≈ $7 per model.

p3 coherence (task adherence of the prompting baseline):

```bash
python -m emotion.coherence_check --csv $S/p3.csv --cache $S/p3_coherence.cache.jsonl --out $S/p3_coherence.md
```

### 4.7 Statistics

```bash
python -m emotion.revision.stats --source p1 --out runs_revision/stats_p1.md
python -m emotion.revision.stats --source p1 --exclude-prompts 0-7 --out runs_revision/stats_p1_no_calib.md
python -m emotion.revision.stats --source p4 --out runs_revision/stats_p4.md
# encoder only, before the judge has run:
python -m emotion.revision.stats --source p1 --instruments encoder --out runs_revision/stats_p1_encoder.md
```

The report prints joint rates (τ = 0, 0.02, 0.05, strict), pair-level BH/BY,
the ΔU table per control with CIs, and the verdict of rule 5 (section 2).

---

## 5. Output layout

```
runs_revision/<slug>/
  <plan>.csv               one row per (condition, prompt, sample): steer, prompt_id, sample,
                           kind, a, b, k, norm_mode, vec_norm, 7 encoder scores, question, answer
  <plan>.csv.meta.json     layer, coeff, batch size, dtype, git sha, versions, GPU, seed rule
  <plan>_judge_wide.csv    judge scores (0–100) per row
  <plan>_judge.cache.jsonl judge cache (resume)
  p5_sweep.csv, p5_selection.json
  p6_norms.json            mean ‖h‖ at the hooked layer, S = coeff·‖v‖/‖h‖ per condition
```

Condition labels: `baseline`, `<a>`, `<a>-<b>`, `<a>-<b>|geo1`, `|geo2`, `|iso1`,
`|lam`, `|proj`, `|norm`, `|geo1|norm`, `|proj|norm`, `neg_<b>`, `instr_<a>`,
`instr_<a>-<b>`, `cent_<a>`.

## 6. Code map

| File | Role |
|---|---|
| `emotion/revision/run.py` | CLI for all plans; batched generation (left padding), encoder scoring, resume |
| `emotion/revision/vectors.py` | vector loading and the controls (geo/iso/λa/proj/norm/centred) |
| `emotion/revision/prompts.py` | prompting-baseline instructions (3 paraphrases, rotated by prompt_id mod 3) |
| `emotion/revision/calibrate.py` | p5 |
| `emotion/revision/norms.py` | p6 |
| `emotion/revision/stats.py` | paired stratified bootstrap, IUT, BH/BY, ΔU, rule check |
| `emotion/revision/judge_wide.py` | per-sample judge wide file from the cache (p4) |
| `emotion/revision/selftest.py` | CPU self-test |
| `docs_revision/` | stats on the published data (all 56 prompts and without 0–7) |

## 7. What to send back / commit

Commit `runs_revision/**` (CSVs, meta, judge wide, reports) on this branch, one
commit per plan per model, message like `p1: Llama-3.2-3B-Instruct, A100, bs32`.
Do not commit API keys or `.venv`.

## 8. Status of this code

- Every plan was smoke-tested end to end on CPU with a tiny random Llama
  (generation, batching, resume, CSV/meta, p4 sampling counts, p5 selection, p6 norms).
- `stats.py` reproduces the submitted numbers on the published data
  (encoder joint 347/462, judge 372/462).
- **Not yet run on the real models**: do `parity` first.
