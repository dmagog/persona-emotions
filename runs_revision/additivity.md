# Additive prediction: published (D(+a) - D(+b)) vs measured -b (D(+a) + D(-b))

Bootstrap: models, then pairs within model, 2000 resamples, seed 1234; 95% percentile intervals.
Degeneration rule: share of answers with repeated word-4-gram ratio > 0.15 (emotion.run_model_chain._rep_ratio, the published pipeline's rule); a p2 cell is degenerate above 10%, the ceiling pick_operating_point uses.

## encoder: all p2 cells

Models with p2: 11 of 11; (model, b) cells used: 77; units: 3234.

| Prediction | units | r [95% CI] | CCC [95% CI] | slope [95% CI] | MAE | dCCC vs published [95% CI] |
|---|---:|---|---|---|---:|---|
| published | 3234 | 0.534 [0.476, 0.610] | 0.475 [0.389, 0.570] | 0.33 [0.243, 0.444] | 0.060 |  |
| +a only | 3234 | 0.619 [0.527, 0.715] | 0.589 [0.478, 0.695] | 0.51 [0.361, 0.697] | 0.040 | +0.114 [0.041, 0.171] |
| measured -b (p2) | 3234 | 0.676 [0.602, 0.746] | 0.627 [0.552, 0.698] | 0.48 [0.383, 0.595] | 0.048 | +0.152 [0.090, 0.200] |
| proportional on b (p2) | 3234 | 0.676 [0.606, 0.745] | 0.635 [0.563, 0.707] | 0.50 [0.395, 0.619] | 0.044 | +0.160 [0.100, 0.209] |

Per component (CCC; published / +a only / measured -b / proportional on b; dCCC measured -b minus published with 95% CI):

| Component | units | published | +a only | measured -b | proportional | dCCC (-b vs pub) |
|---|---:|---:|---:|---:|---:|---|
| k = a | 462 | 0.389 | 0.322 | 0.409 | 0.409 | +0.020 [-0.069, +0.088] |
| k = b | 462 | 0.063 | 0.359 | 0.531 | 0.843 | +0.468 [+0.326, +0.571] |
| k = other | 2310 | 0.232 | 0.453 | 0.508 | 0.508 | +0.276 [+0.143, +0.381] |

On component b: mean observed D(a-b) -0.040; mean D(+b) +0.134, mean D(-b) -0.049; corr(D(+b), -D(-b)) over pairs -0.03; mean prediction: published -0.143, measured -b -0.057, proportional -0.048.

Per-model r (published / measured -b): Falcon3-3B-Instruct 0.64/0.72, Llama-3.2-1B-Instruct 0.61/0.73, Llama-3.2-3B-Instruct 0.79/0.85, OLMo-2-0425-1B-Instruct 0.64/0.72, Qwen2.5-1.5B-Instruct 0.49/0.49, Qwen2.5-3B-Instruct 0.56/0.67, Qwen3-0.6B 0.54/0.67, Qwen3-1.7B 0.54/0.82, gemma-2-2b-it 0.62/0.73, gemma-3-1b-it 0.48/0.68, granite-3.3-2b-instruct 0.54/0.69

**Rule 6 (encoder), as pre-registered (point estimates):** measured -b CCC 0.627 vs 0.475, slope 0.48 vs 0.33 -> no worse: keep partial composition. Uncertainty: dCCC +0.152 [+0.090, +0.200], CI excludes 0. +a only reaches CCC 0.589, so measured -b adds +0.038 over ignoring b.

## encoder: degenerate p2 cells excluded

Excluded (model, b, degenerate share of neg_b): Llama-3.2-1B-Instruct disgust 20%, Llama-3.2-1B-Instruct joy 70%, Llama-3.2-1B-Instruct sadness 14%, Llama-3.2-1B-Instruct shame 14%, Llama-3.2-1B-Instruct anger 16%, Qwen3-0.6B joy 64%, gemma-3-1b-it joy 86%.

Models with p2: 11 of 11; (model, b) cells used: 70; units: 2940.

| Prediction | units | r [95% CI] | CCC [95% CI] | slope [95% CI] | MAE | dCCC vs published [95% CI] |
|---|---:|---|---|---|---:|---|
| published | 2940 | 0.547 [0.488, 0.625] | 0.494 [0.413, 0.596] | 0.35 [0.258, 0.481] | 0.057 |  |
| +a only | 2940 | 0.620 [0.527, 0.719] | 0.595 [0.490, 0.703] | 0.52 [0.382, 0.724] | 0.039 | +0.101 [0.028, 0.157] |
| measured -b (p2) | 2940 | 0.674 [0.596, 0.749] | 0.628 [0.556, 0.704] | 0.49 [0.395, 0.610] | 0.048 | +0.134 [0.064, 0.192] |
| proportional on b (p2) | 2940 | 0.674 [0.600, 0.747] | 0.636 [0.566, 0.712] | 0.51 [0.409, 0.633] | 0.043 | +0.142 [0.072, 0.200] |

Per component (CCC; published / +a only / measured -b / proportional on b; dCCC measured -b minus published with 95% CI):

| Component | units | published | +a only | measured -b | proportional | dCCC (-b vs pub) |
|---|---:|---:|---:|---:|---:|---|
| k = a | 420 | 0.416 | 0.318 | 0.403 | 0.403 | -0.013 [-0.105, +0.067] |
| k = b | 420 | 0.063 | 0.364 | 0.540 | 0.842 | +0.477 [+0.315, +0.583] |
| k = other | 2100 | 0.279 | 0.473 | 0.515 | 0.515 | +0.236 [+0.111, +0.337] |

On component b: mean observed D(a-b) -0.039; mean D(+b) +0.131, mean D(-b) -0.048; corr(D(+b), -D(-b)) over pairs -0.02; mean prediction: published -0.138, measured -b -0.056, proportional -0.047.

Per-model r (published / measured -b): Falcon3-3B-Instruct 0.64/0.72, Llama-3.2-1B-Instruct 0.73/0.79, Llama-3.2-3B-Instruct 0.79/0.85, OLMo-2-0425-1B-Instruct 0.64/0.72, Qwen2.5-1.5B-Instruct 0.49/0.49, Qwen2.5-3B-Instruct 0.56/0.67, Qwen3-0.6B 0.56/0.67, Qwen3-1.7B 0.54/0.82, gemma-2-2b-it 0.62/0.73, gemma-3-1b-it 0.55/0.68, granite-3.3-2b-instruct 0.54/0.69

**Rule 6 (encoder), as pre-registered (point estimates):** measured -b CCC 0.628 vs 0.494, slope 0.49 vs 0.35 -> no worse: keep partial composition. Uncertainty: dCCC +0.134 [+0.064, +0.192], CI excludes 0. +a only reaches CCC 0.595, so measured -b adds +0.033 over ignoring b.

## judge: all p2 cells

Models with p2: 11 of 11; (model, b) cells used: 77; units: 3234.

| Prediction | units | r [95% CI] | CCC [95% CI] | slope [95% CI] | MAE | dCCC vs published [95% CI] |
|---|---:|---|---|---|---:|---|
| published | 3234 | 0.752 [0.693, 0.811] | 0.628 [0.572, 0.689] | 0.41 [0.367, 0.468] | 0.202 |  |
| +a only | 3234 | 0.578 [0.513, 0.641] | 0.383 [0.316, 0.454] | 0.40 [0.343, 0.470] | 0.276 | -0.246 [-0.321, -0.170] |
| measured -b (p2) | 3234 | 0.664 [0.600, 0.730] | 0.572 [0.490, 0.662] | 0.47 [0.415, 0.545] | 0.200 | -0.056 [-0.146, 0.036] |
| proportional on b (p2) | 3234 | 0.697 [0.637, 0.758] | 0.628 [0.553, 0.707] | 0.50 [0.441, 0.570] | 0.175 | -0.000 [-0.082, 0.080] |

Per component (CCC; published / +a only / measured -b / proportional on b; dCCC measured -b minus published with 95% CI):

| Component | units | published | +a only | measured -b | proportional | dCCC (-b vs pub) |
|---|---:|---:|---:|---:|---:|---|
| k = a | 462 | 0.687 | 0.222 | 0.453 | 0.453 | -0.234 [-0.379, -0.078] |
| k = b | 462 | 0.211 | 0.162 | 0.307 | 0.460 | +0.096 [+0.020, +0.186] |
| k = other | 2310 | 0.589 | 0.342 | 0.525 | 0.525 | -0.063 [-0.159, +0.033] |

On component b: mean observed D(a-b) -0.084; mean D(+b) +0.607, mean D(-b) -0.164; corr(D(+b), -D(-b)) over pairs 0.17; mean prediction: published -0.364, measured -b +0.080, proportional -0.146.

Per-model r (published / measured -b): Falcon3-3B-Instruct 0.77/0.67, Llama-3.2-1B-Instruct 0.75/0.67, Llama-3.2-3B-Instruct 0.82/0.82, OLMo-2-0425-1B-Instruct 0.83/0.70, Qwen2.5-1.5B-Instruct 0.74/0.53, Qwen2.5-3B-Instruct 0.84/0.71, Qwen3-0.6B 0.72/0.71, Qwen3-1.7B 0.81/0.76, gemma-2-2b-it 0.90/0.81, gemma-3-1b-it 0.60/0.65, granite-3.3-2b-instruct 0.82/0.64

**Rule 6 (judge), as pre-registered (point estimates):** measured -b CCC 0.572 vs 0.628, slope 0.47 vs 0.41 -> worse: narrow the claim. Uncertainty: dCCC -0.056 [-0.146, +0.036], CI includes 0, so the direction is not established. +a only reaches CCC 0.383, so measured -b adds +0.190 over ignoring b.

## judge: degenerate p2 cells excluded

Excluded (model, b, degenerate share of neg_b): Llama-3.2-1B-Instruct disgust 20%, Llama-3.2-1B-Instruct joy 70%, Llama-3.2-1B-Instruct sadness 14%, Llama-3.2-1B-Instruct shame 14%, Llama-3.2-1B-Instruct anger 16%, Qwen3-0.6B joy 64%, gemma-3-1b-it joy 86%.

Models with p2: 11 of 11; (model, b) cells used: 70; units: 2940.

| Prediction | units | r [95% CI] | CCC [95% CI] | slope [95% CI] | MAE | dCCC vs published [95% CI] |
|---|---:|---|---|---|---:|---|
| published | 2940 | 0.769 [0.710, 0.823] | 0.656 [0.606, 0.703] | 0.45 [0.409, 0.487] | 0.190 |  |
| +a only | 2940 | 0.585 [0.514, 0.652] | 0.394 [0.325, 0.466] | 0.41 [0.348, 0.481] | 0.269 | -0.261 [-0.332, -0.192] |
| measured -b (p2) | 2940 | 0.668 [0.598, 0.741] | 0.581 [0.491, 0.676] | 0.48 [0.418, 0.559] | 0.195 | -0.075 [-0.159, 0.013] |
| proportional on b (p2) | 2940 | 0.701 [0.637, 0.768] | 0.636 [0.556, 0.719] | 0.50 [0.446, 0.581] | 0.170 | -0.020 [-0.094, 0.059] |

Per component (CCC; published / +a only / measured -b / proportional on b; dCCC measured -b minus published with 95% CI):

| Component | units | published | +a only | measured -b | proportional | dCCC (-b vs pub) |
|---|---:|---:|---:|---:|---:|---|
| k = a | 420 | 0.708 | 0.231 | 0.455 | 0.455 | -0.253 [-0.400, -0.091] |
| k = b | 420 | 0.225 | 0.161 | 0.308 | 0.455 | +0.083 [+0.000, +0.186] |
| k = other | 2100 | 0.625 | 0.362 | 0.541 | 0.541 | -0.084 [-0.172, +0.013] |

On component b: mean observed D(a-b) -0.079; mean D(+b) +0.595, mean D(-b) -0.161; corr(D(+b), -D(-b)) over pairs 0.18; mean prediction: published -0.345, measured -b +0.089, proportional -0.142.

Per-model r (published / measured -b): Falcon3-3B-Instruct 0.77/0.67, Llama-3.2-1B-Instruct 0.85/0.75, Llama-3.2-3B-Instruct 0.82/0.82, OLMo-2-0425-1B-Instruct 0.83/0.70, Qwen2.5-1.5B-Instruct 0.74/0.53, Qwen2.5-3B-Instruct 0.84/0.71, Qwen3-0.6B 0.70/0.71, Qwen3-1.7B 0.81/0.76, gemma-2-2b-it 0.90/0.81, gemma-3-1b-it 0.62/0.66, granite-3.3-2b-instruct 0.82/0.64

**Rule 6 (judge), as pre-registered (point estimates):** measured -b CCC 0.581 vs 0.656, slope 0.48 vs 0.45 -> worse: narrow the claim. Uncertainty: dCCC -0.075 [-0.159, +0.013], CI includes 0, so the direction is not established. +a only reaches CCC 0.394, so measured -b adds +0.186 over ignoring b.

