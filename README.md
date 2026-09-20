# CEmoSteer

CEmoSteer is a reproducible pipeline for studying and steering expressed emotion in instruction-tuned language models. It extracts contrastive activation directions for the seven ISEAR emotions: anger, disgust, fear, guilt, joy, sadness, and shame.

The accompanying study evaluates 11 English instruction-tuned models from 0.6B to 3B parameters. Each model receives one layer and coefficient selected on a disjoint anger set. That operating point is then reused for all seven single directions and all 42 ordered differences, such as `guilt - sadness`.

The repository includes the code, prompts, extracted vectors, generated outputs, and evaluation artifacts used in the study. The main analysis tests whether an ordered difference keeps the target emotion above the unsteered baseline while reducing the subtracted emotion relative to target-only steering. A separate dialogue experiment tests whether changes in expressed emotion transfer to conflict handling.

## Results at a glance

Across all 11 models, every model has a positive mean target effect under the shared calibration. For ordered differences, both conditions for selective attenuation hold in 347 of 462 model-pair cases (75.1%) under the local encoder and 372 of 462 cases (80.5%) under the primary LLM judge. These are sign-level coverage results, not guarantees that every response remains fluent or preserves its original task content.

Three human assessors rated a locked sample of 300 outputs. Their consensus supports the automatic emotion signal, while the same study identifies costs in fluency, task adherence, and naturalness. In the 960-response dialogue study, reducing a measured emotion did not improve de-escalation: negative emotion directions and norm-matched random directions both increased judged escalation on the two tested models.

## Repository map

| Path | Contents |
|---|---|
| `emotion/` | Extraction, steering, evaluation, judge, bootstrap, and artifact-stamping modules. |
| `configs/models/` | One YAML configuration per evaluated model. |
| `data_generation/` | Disjoint extraction and held-out emotion scenarios, plus the dialogue set. |
| `eval_emotion/` | Self-generated emotional and neutral response pairs for the evaluated checkpoints. |
| `emotion_vectors/` | Layer-wise response-average emotion directions. |
| `runs/` | Per-model manifests, calibration sweeps, generations, scores, confidence intervals, and judge caches. |
| `figures/` | Figures used to inspect and report the experiments. |
| `docs/` | Protocol, aggregate results, composition results, and dialogue-evaluation notes. |
| `requirements-inference.txt` | Minimal dependencies for inference and the main pipeline. |
| `requirements.txt` | Full development and analysis dependencies. |

The files in `runs/`, `eval_emotion/`, and `emotion_vectors/` are the experiment artifacts. They let you inspect the included analyses without regenerating model outputs. Each pipeline stage records a content-based stamp next to its output so that a rerun can detect mismatched inputs or settings.

## Quick start

The pipeline requires one GPU. The evaluated models up to 3B parameters fit on an 8 GB GPU with the published FP16 setup. Some gated checkpoints also require a Hugging Face token. Judge stages require an OpenAI-compatible API endpoint.

```bash
pip install -r requirements-inference.txt

export HF_TOKEN=...
export OPENAI_API_KEY=...
export OPENAI_BASE_URL=https://openrouter.ai/api/v1

python3 -m emotion.run_model_chain --config configs/models/qwen3-1.7b.yaml
python3 -m emotion.run_eval_chain --slug Qwen3-1.7B --judge
python3 -m emotion.collect_results
```

To add a model, create a YAML file in `configs/models/`. Command-line options override YAML values for one-off runs.

## Reproducing an existing analysis

The main chain generates matched response pairs, extracts directions, selects an operating point, evaluates the single-direction matrix, and can generate ordered differences. The evaluation chain adds confidence intervals, coherence diagnostics, and optional LLM-judge results.

```bash
python3 -m emotion.run_model_chain --config configs/models/falcon3-3b.yaml --compose allpairs
python3 -m emotion.run_eval_chain --slug Falcon3-3B-Instruct --judge
python3 -m emotion.collect_compose
```

Existing judge caches are versioned with the generated answers. Re-running a judge stage on unchanged output should reuse its cache. A changed output needs a new judgment and may incur API cost.

## Documentation

- [Protocol](docs/PROTOCOL.md) describes the extraction, calibration, and evaluation rules.
- [Results](docs/RESULTS.md) contains the cross-model single-direction summary.
- [Composition](docs/COMPOSITION.md) reports the ordered-difference analysis.
- [Dialogue evaluation](docs/DIALOG_SAFETY.md) documents the de-escalation extension and its controls.
- [Runbook](RUNBOOK.md) gives the command-level workflow and artifact layout.

## License

This project is released under the [MIT License](LICENSE).
