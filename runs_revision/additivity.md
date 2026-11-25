# Additive prediction: published (D(+a) - D(+b)) vs measured -b (D(+a) + D(-b))

## encoder

Models with p2: 11 of 11.

| Prediction | units | r | rho | CCC | slope | MAE |
|---|---:|---:|---:|---:|---:|---:|
| published, all models | 3234 | 0.534 | 0.498 | 0.475 | 0.33 | 0.060 |
| published, models with p2 | 3234 | 0.534 | 0.498 | 0.475 | 0.33 | 0.060 |
| measured -b (p2) | 3234 | 0.676 | 0.707 | 0.627 | 0.48 | 0.048 |

Per-model r (published / measured -b): Falcon3-3B-Instruct 0.64/0.72, Llama-3.2-1B-Instruct 0.61/0.73, Llama-3.2-3B-Instruct 0.79/0.85, OLMo-2-0425-1B-Instruct 0.64/0.72, Qwen2.5-1.5B-Instruct 0.49/0.49, Qwen2.5-3B-Instruct 0.56/0.67, Qwen3-0.6B 0.54/0.67, Qwen3-1.7B 0.54/0.82, gemma-2-2b-it 0.62/0.73, gemma-3-1b-it 0.48/0.68, granite-3.3-2b-instruct 0.54/0.69

On component b: mean D(+b) +0.134, mean D(-b) -0.049, corr(D(+b), -D(-b)) over pairs -0.03

**Rule 6 (encoder):** measured -b CCC 0.627 vs 0.475, slope 0.48 vs 0.33 -> no worse: keep partial composition.

## judge

Models with p2: 11 of 11.

| Prediction | units | r | rho | CCC | slope | MAE |
|---|---:|---:|---:|---:|---:|---:|
| published, all models | 3234 | 0.752 | 0.772 | 0.628 | 0.41 | 0.202 |
| published, models with p2 | 3234 | 0.752 | 0.772 | 0.628 | 0.41 | 0.202 |
| measured -b (p2) | 3234 | 0.664 | 0.626 | 0.572 | 0.47 | 0.200 |

Per-model r (published / measured -b): Falcon3-3B-Instruct 0.77/0.67, Llama-3.2-1B-Instruct 0.75/0.67, Llama-3.2-3B-Instruct 0.82/0.82, OLMo-2-0425-1B-Instruct 0.83/0.70, Qwen2.5-1.5B-Instruct 0.74/0.53, Qwen2.5-3B-Instruct 0.84/0.71, Qwen3-0.6B 0.72/0.71, Qwen3-1.7B 0.81/0.76, gemma-2-2b-it 0.90/0.81, gemma-3-1b-it 0.60/0.65, granite-3.3-2b-instruct 0.82/0.64

On component b: mean D(+b) +0.607, mean D(-b) -0.164, corr(D(+b), -D(-b)) over pairs 0.17

**Rule 6 (judge):** measured -b CCC 0.572 vs 0.628, slope 0.47 vs 0.41 -> worse: narrow the claim.

