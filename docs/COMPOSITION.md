# Ordered-difference composition

For each ordered pair of distinct emotions `X` and `Y`, this analysis evaluates `v_X - v_Y`. Each model uses the layer and coefficient selected on the disjoint anger calibration set. The table covers 42 ordered pairs for each model.

Target retention means that X rises above the unsteered baseline. Attenuation means that Y is lower than under X-only steering. Joint success requires both conditions for the same pair.

| Model | Encoder target | Encoder attenuation | Encoder joint | Judge target | Judge attenuation | Judge joint |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Falcon3-3B-Instruct | 37/42 | 39/42 | 34/42 | 35/42 | 35/42 | 28/42 |
| Llama-3.2-1B-Instruct | 36/42 | 38/42 | 32/42 | 41/42 | 36/42 | 35/42 |
| Llama-3.2-3B-Instruct | 41/42 | 39/42 | 38/42 | 41/42 | 42/42 | 41/42 |
| OLMo-2-0425-1B-Instruct | 40/42 | 39/42 | 37/42 | 37/42 | 42/42 | 37/42 |
| Qwen2.5-1.5B-Instruct | 31/42 | 40/42 | 30/42 | 34/42 | 37/42 | 29/42 |
| Qwen2.5-3B-Instruct | 34/42 | 37/42 | 31/42 | 34/42 | 42/42 | 34/42 |
| Qwen3-0.6B | 30/42 | 37/42 | 26/42 | 34/42 | 37/42 | 29/42 |
| Qwen3-1.7B | 33/42 | 40/42 | 31/42 | 39/42 | 38/42 | 35/42 |
| gemma-2-2b-it | 35/42 | 40/42 | 33/42 | 40/42 | 41/42 | 39/42 |
| gemma-3-1b-it | 32/42 | 34/42 | 25/42 | 40/42 | 34/42 | 32/42 |
| granite-3.3-2b-instruct | 36/42 | 36/42 | 30/42 | 40/42 | 35/42 | 33/42 |
| Total | 385/462 | 419/462 | 347/462 | 415/462 | 419/462 | 372/462 |
| Rate | 83.3% | 90.7% | 75.1% | 89.8% | 90.7% | 80.5% |

The raw matrices are `runs/<slug>/compose_allpairs.csv` and `runs/<slug>/compose_allpairs_judge_wide.csv`.
