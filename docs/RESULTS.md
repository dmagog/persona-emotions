# Single-direction results

This table summarizes the held-out single-direction experiment from the final study. Each model uses one layer and coefficient selected on disjoint anger prompts, then reuses that operating point for all seven emotions.

`Mean diagonal` is the mean range-normalized target-score change under the local GoEmotions encoder. `Argmax` counts the emotion directions whose target change is the largest change in that row. `Marginal CI > 0` counts target effects with a prompt-bootstrap 95% interval above zero. `Specificity gap` is the difference between the mean target effect and the mean strongest off-target effect. `Q` is the mean coherence score of steered outputs. `Delta Q` is relative to the unsteered baseline.

| Model | Mean diagonal [95% CI] | Argmax | Marginal CI > 0 | Specificity gap | Q | Delta Q | Degenerate |
|---|---:|:---:|:---:|---:|---:|---:|:---:|
| Falcon-3-3B | 0.198 [0.179, 0.219] | 7/7 | 7/7 | 0.158 | 85.2 | -10.6 | 9/392 |
| Llama-3.2-1B | 0.168 [0.150, 0.185] | 6/7 | 6/7 | 0.117 | 54.2 | -39.3 | 120/392 |
| Llama-3.2-3B | 0.119 [0.104, 0.134] | 6/7 | 6/7 | 0.083 | 87.6 | -6.9 | 3/392 |
| OLMo-2-1B | 0.097 [0.079, 0.116] | 6/7 | 6/7 | 0.048 | 87.6 | -7.3 | 3/392 |
| Qwen-2.5-1.5B | 0.157 [0.141, 0.172] | 5/7 | 7/7 | 0.079 | 78.9 | -16.5 | 34/392 |
| Qwen-2.5-3B | 0.064 [0.047, 0.082] | 5/7 | 5/7 | 0.020 | 88.2 | -6.5 | 2/392 |
| Qwen-3-0.6B | 0.246 [0.231, 0.259] | 4/7 | 6/7 | 0.156 | 79.4 | -9.8 | 20/392 |
| Qwen-3-1.7B | 0.101 [0.083, 0.118] | 5/7 | 4/7 | 0.064 | 79.8 | -11.9 | 26/392 |
| Gemma-2-2B | 0.073 [0.054, 0.092] | 5/7 | 5/7 | 0.027 | 84.3 | -9.9 | 3/392 |
| Gemma-3-1B | 0.100 [0.085, 0.114] | 4/7 | 6/7 | 0.051 | 51.8 | -39.4 | 31/392 |
| Granite-3.3-2B | 0.165 [0.149, 0.180] | 4/7 | 4/7 | 0.066 | 84.2 | -11.3 | 5/392 |

Every checkpoint has a positive mean target effect, but the effect size, target dominance, off-target response, and output quality differ across models. Falcon-3-3B has the most balanced profile in this comparison. Qwen-3-0.6B has the largest mean target effect, while Qwen-2.5-3B has the smallest positive specificity gap. Llama-3.2-1B and Gemma-3-1B show the clearest coherence and degeneration failures.

The full per-model artifacts are stored in `runs/<slug>/`. They include raw generations, encoder matrices, judge matrices, bootstrap reports, and provenance stamps.
