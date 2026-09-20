# Ordered-difference composition

For every ordered pair of distinct emotions `X` and `Y`, CEmoSteer evaluates the direction `v_X - v_Y`. Each model uses the same layer and coefficient that were selected on disjoint anger prompts. The experiment includes all 42 ordered pairs for each of 11 models.

Target retention requires the score for `X` under `X - Y` to exceed the unsteered baseline. Attenuation requires the score for `Y` under `X - Y` to be lower than it is under `X` alone. The joint column requires both conditions.

| Model | Encoder target | Encoder attenuation | Encoder joint | Judge target | Judge attenuation | Judge joint |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Falcon-3-3B | 37/42 | 39/42 | 34/42 | 35/42 | 35/42 | 28/42 |
| Llama-3.2-1B | 36/42 | 38/42 | 32/42 | 41/42 | 36/42 | 35/42 |
| Llama-3.2-3B | 41/42 | 39/42 | 38/42 | 41/42 | 42/42 | 41/42 |
| OLMo-2-1B | 40/42 | 39/42 | 37/42 | 37/42 | 42/42 | 37/42 |
| Qwen-2.5-1.5B | 31/42 | 40/42 | 30/42 | 34/42 | 37/42 | 29/42 |
| Qwen-2.5-3B | 34/42 | 37/42 | 31/42 | 34/42 | 42/42 | 34/42 |
| Qwen-3-0.6B | 30/42 | 37/42 | 26/42 | 34/42 | 37/42 | 29/42 |
| Qwen-3-1.7B | 33/42 | 40/42 | 31/42 | 39/42 | 38/42 | 35/42 |
| Gemma-2-2B | 35/42 | 40/42 | 33/42 | 40/42 | 41/42 | 39/42 |
| Gemma-3-1B | 32/42 | 34/42 | 25/42 | 40/42 | 34/42 | 32/42 |
| Granite-3.3-2B | 36/42 | 36/42 | 30/42 | 40/42 | 35/42 | 33/42 |
| Total | 385/462 | 419/462 | 347/462 | 415/462 | 419/462 | 372/462 |
| Rate | 83.3% | 90.7% | 75.1% | 89.8% | 90.7% | 80.5% |

The high marginal counts do not make the effects exact or uniform. When the success criterion requires a larger effect, coverage falls, especially for the encoder. At a range-normalized threshold of 0.02, joint coverage is 149 of 462 under the encoder and 324 of 462 under the primary judge.

Single-direction effects predict part of the response to an ordered difference, but observed effects are smaller than an additive prediction. The pooled Pearson correlation is 0.534 for the encoder and 0.752 for the primary judge. These results support partial composition, not a general linear rule for intervention strength.

The raw outputs and score tables are stored in `runs/<slug>/compose_allpairs.csv` and `runs/<slug>/compose_allpairs_judge_wide.csv`.
