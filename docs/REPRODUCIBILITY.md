# Reproducibility

The repository keeps the prompts, response pairs, vectors, generations, local scores, judge scores, and judge caches used for the reported analyses. A fresh clone can rebuild the aggregate documents without downloading model weights or calling a judge API.

```bash
pip install -r requirements-inference.txt
python3 -m emotion.collect_compose --out docs/COMPOSITION.md
python3 -m emotion.collect_dialog_safety --out docs/DIALOG_SAFETY.md
python3 -m emotion.selftest
```

`docs/COMPOSITION.md` is generated from the 11 pairs of `compose_allpairs.csv` and `compose_allpairs_judge_wide.csv` files in `runs/`. The collector computes target retention, attenuation, and their joint criterion for every ordered emotion pair.

`docs/DIALOG_SAFETY.md` is generated from the dialogue generations and primary-judge files in `runs/Falcon3-3B-Instruct/` and `runs/Qwen2.5-1.5B-Instruct/`. It includes only conditions whose raw files are present in the repository.

The committed human-evaluation sheet is a blinded annotation template. Completed human ratings are not included, so the article's human-validation analysis cannot yet be recomputed from the public repository.

To reproduce a model run, use the configuration that matches the model:

```bash
python3 -m emotion.run_model_chain --config configs/models/falcon3-3b.yaml --compose allpairs
python3 -m emotion.run_eval_chain --slug Falcon3-3B-Instruct --judge
```

This path requires a compatible GPU, access to the model weights, and an OpenAI-compatible endpoint for judge stages. Stored judge caches are reused when the generated text is unchanged.
