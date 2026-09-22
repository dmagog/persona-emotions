# CEmoSteer runbook

This guide describes the workflow for a new model and the files produced by each stage. The published artifacts already cover the 11 evaluated checkpoints. You only need to run the full chain when adding a checkpoint or reproducing an analysis from scratch.

## Requirements

- One GPU. The published runs used FP16 on an NVIDIA RTX 2070. Models up to 3B parameters fit in 8 GB of VRAM with that setup.
- Python dependencies from `requirements-inference.txt`. Use `requirements.txt` for the complete development environment.
- `HF_TOKEN` for gated Hugging Face models.
- `OPENAI_API_KEY` and `OPENAI_BASE_URL` only for LLM-judge stages.

```bash
pip install -r requirements-inference.txt

export HF_TOKEN=...
export OPENAI_API_KEY=...
export OPENAI_BASE_URL=https://openrouter.ai/api/v1
```

## Run a model

Each evaluated checkpoint has a configuration in `configs/models/`. Start with one of these files or create a new YAML file with the model identifier, slug, generation settings, and resource limits.

```bash
python3 -m emotion.run_model_chain --config configs/models/qwen3-1.7b.yaml
```

Command-line options override a YAML value for a one-off run. For example:

```bash
python3 -m emotion.run_model_chain \
  --config configs/models/qwen3-1.7b.yaml \
  --batch-size 8 \
  --max-tokens 160
```

The chain is resumable. It checks a content-based stamp before every stage. A matching stamp skips completed work. A conflicting stamp stops the run so that inputs or settings are not mixed silently. Use `--recompute-stale` only after deciding to regenerate the conflicting artifact.

## Pipeline stages

1. Preflight checks the chat template, thinking-mode settings, vector shape, intervention strength, and one steered generation.
2. Pair generation creates matched emotional and neutral responses from the evaluated model. The extraction and evaluation scenario pools are disjoint.
3. Vector extraction computes a response-token mean difference for every emotion and layer.
4. Operating-point selection evaluates candidate layers and coefficients on the disjoint anger set. It chooses the largest usable response subject to the lexical-degeneration constraint.
5. Single-direction evaluation generates 56 held-out prompts under each emotion direction and once without steering.
6. Composition evaluates ordered differences. Pass `--compose allpairs` to generate all 42 `X - Y` directions.

The chain writes metadata, logs, raw generations, and stamps under `runs/<slug>/`. It writes generated response pairs under `eval_emotion/<slug>/` and the extracted tensors under `emotion_vectors/<slug>/`.

## Evaluate a completed run

The free evaluation stages produce bootstrap intervals and geometry diagnostics. Add `--judge` to run the LLM-judge stages.

```bash
python3 -m emotion.run_eval_chain --slug Qwen3-1.7B
python3 -m emotion.run_eval_chain --slug Qwen3-1.7B --judge
```

Judge outputs use a cache keyed by response text. Existing cached responses are not sent again. New or changed generations require API access and may incur cost.

Useful aggregate commands:

```bash
python3 -m emotion.collect_results
python3 -m emotion.collect_compose
python3 -m emotion.protocol --check runs
python3 -m emotion.selftest
```

## Composition criteria

For an ordered difference `X - Y`, the evaluation uses the same prompts in three conditions: unsteered, `X`, and `X - Y`.

- Target retention: the mean score for `X - Y` on emotion `X` is above the unsteered baseline.
- Attenuation: the mean score for `X - Y` on emotion `Y` is below the score under `X` alone.

The second comparison measures whether subtracting `Y` reduces the extra `Y` expression introduced by the target direction. It does not require `Y` to fall below its unsteered level.

## Dialogue evaluation

The dialogue extension uses 30 multi-turn English prompts in `data_generation/deescalation_dialogs.json`. It reuses the selected layer and coefficient from the emotion study and writes outputs next to each model run.

The three stages are separate because only the first one needs a GPU.

```bash
# 1. Generate one reply per dialogue under each intervention.
python3 -m emotion.steer_dialog_safety \
    --model_name tiiuae/Falcon3-3B-Instruct \
    --vector-dir emotion_vectors/Falcon3-3B-Instruct \
    --layer 12 --coeff 6 --dtype float16 \
    --out runs/Falcon3-3B-Instruct/dialog_safety.csv

# 2. Score escalation, helpfulness, and empathy against the provocation.
python3 -m emotion.judge_dialog_safety \
    --csv runs/Falcon3-3B-Instruct/dialog_safety.csv \
    --out runs/Falcon3-3B-Instruct/dialog_safety_judge.csv

# 3. Rebuild the summary from every stored dialogue file.
python3 -m emotion.collect_dialog_safety --out docs/DIALOG_SAFETY.md
```

`--conditions` selects the interventions. It accepts signed emotions such as `-anger`
or `+joy`, and `randomN` for a norm-matched random direction. Pass it with an equals
sign, as in `--conditions=-anger`, so the leading dash is not read as an option.

The completed study includes seven negative emotion directions, positive anger, an anger-strength sweep, and three norm-matched random directions. The random controls distinguish changes caused by an emotion direction from changes caused by a perturbation of similar size.

## Artifact layout

```text
configs/models/<model>.yaml          Model-specific runtime settings
data_generation/                     Extraction, evaluation, and dialogue prompts
eval_emotion/<slug>/                 Matched emotional and neutral response pairs
emotion_vectors/<slug>/              Layer-wise emotion directions
runs/<slug>/meta.json                Selected layer, coefficient, environment, and manifest
runs/<slug>/*.csv                    Generations, scores, sweeps, and evaluation tables
runs/<slug>/*.stamp.json             Content-based provenance for generated artifacts
runs/<slug>/*.cache.jsonl            Reusable LLM-judge responses
```

The repository contains the files needed to inspect reported results. Reproducing a GPU or judge stage from scratch still requires the corresponding model access and API credentials.
