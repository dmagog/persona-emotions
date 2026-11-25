# Prompting baseline (p3) vs activation steering (published), 4-model subset

Models: Llama-3.2-3B-Instruct, gemma-2-2b-it, Qwen3-0.6B, Falcon3-3B-Instruct. Paired, stratified bootstrap over prompt IDs, 2000 resamples, seed 1234. Definitions in `emotion/revision/p3_compare.py`.

## Judge

### Judge, all 4 models

4 models x 42 pairs = 168 cells.

| Measure | Steering (published) | Prompting (p3) | Prompting - steering [95% CI] |
|---|---:|---:|---|
| T, target gain | +0.289 [+0.279, +0.300] | +0.309 [+0.289, +0.328] | +0.019 [-0.001, +0.038] |
| U, b vs a-only | -0.274 [-0.283, -0.266] | -0.065 [-0.071, -0.058] | +0.209 [+0.198, +0.220] |
| U_base, b vs baseline | -0.079 [-0.089, -0.069] | +0.006 [-0.007, +0.017] | +0.085 [+0.075, +0.095] |
| spillover, mean abs shift on other 5 | +0.120 [+0.118, +0.126] | +0.114 [+0.107, +0.125] | -0.006 [-0.015, +0.003] |
| joint, tau=0 | +0.815 [+0.744, +0.821] | +0.857 [+0.786, +0.869] | +0.042 [-0.012, +0.101] |
| joint, tau=0.02 | +0.685 [+0.643, +0.720] | +0.720 [+0.625, +0.732] | +0.036 [-0.071, +0.065] |
| joint, tau=0.05 | +0.595 [+0.542, +0.619] | +0.500 [+0.429, +0.542] | -0.095 [-0.167, -0.030] |
| strict (b below baseline) | +0.643 [+0.565, +0.655] | +0.357 [+0.310, +0.417] | -0.286 [-0.315, -0.173] |

### Judge, without Qwen3-0.6B

3 models x 42 pairs = 126 cells.

| Measure | Steering (published) | Prompting (p3) | Prompting - steering [95% CI] |
|---|---:|---:|---|
| T, target gain | +0.309 [+0.298, +0.321] | +0.324 [+0.301, +0.347] | +0.015 [-0.007, +0.037] |
| U, b vs a-only | -0.302 [-0.313, -0.292] | -0.070 [-0.077, -0.062] | +0.232 [+0.220, +0.245] |
| U_base, b vs baseline | -0.086 [-0.097, -0.075] | +0.020 [+0.005, +0.034] | +0.106 [+0.094, +0.118] |
| spillover, mean abs shift on other 5 | +0.126 [+0.123, +0.132] | +0.126 [+0.116, +0.140] | +0.001 [-0.012, +0.012] |
| joint, tau=0 | +0.857 [+0.794, +0.865] | +0.897 [+0.817, +0.905] | +0.040 [-0.032, +0.087] |
| joint, tau=0.02 | +0.754 [+0.714, +0.794] | +0.762 [+0.659, +0.778] | +0.008 [-0.111, +0.032] |
| joint, tau=0.05 | +0.690 [+0.627, +0.714] | +0.532 [+0.452, +0.579] | -0.159 [-0.230, -0.079] |
| strict (b below baseline) | +0.698 [+0.627, +0.722] | +0.310 [+0.254, +0.373] | -0.389 [-0.429, -0.278] |

## Encoder

**Qwen3-0.6B is flagged:** it names the target emotion in most prompted answers (see lexical leak), which the encoder scores directly, so its prompting numbers are inflated on the encoder. The headline encoder comparison uses the other 3 models.

### Encoder, 3 models (without Qwen3-0.6B)

3 models x 42 pairs = 126 cells.

| Measure | Steering (published) | Prompting (p3) | Prompting - steering [95% CI] |
|---|---:|---:|---|
| T, target gain | +0.106 [+0.100, +0.113] | +0.113 [+0.105, +0.123] | +0.007 [-0.002, +0.016] |
| U, b vs a-only | -0.030 [-0.032, -0.029] | -0.004 [-0.007, -0.002] | +0.026 [+0.023, +0.029] |
| U_base, b vs baseline | -0.042 [-0.046, -0.038] | -0.013 [-0.019, -0.009] | +0.029 [+0.025, +0.033] |
| spillover, mean abs shift on other 5 | +0.032 [+0.031, +0.037] | +0.029 [+0.028, +0.035] | -0.003 [-0.006, +0.001] |
| joint, tau=0 | +0.833 [+0.770, +0.841] | +0.595 [+0.508, +0.635] | -0.238 [-0.310, -0.159] |
| joint, tau=0.02 | +0.452 [+0.365, +0.460] | +0.175 [+0.127, +0.222] | -0.278 [-0.310, -0.167] |
| joint, tau=0.05 | +0.127 [+0.087, +0.159] | +0.000 [+0.000, +0.032] | -0.127 [-0.151, -0.071] |
| strict (b below baseline) | +0.786 [+0.722, +0.802] | +0.413 [+0.317, +0.460] | -0.373 [-0.452, -0.302] |

### Encoder, Qwen3-0.6B only (flagged, not for the headline)

1 models x 42 pairs = 42 cells.

| Measure | Steering (published) | Prompting (p3) | Prompting - steering [95% CI] |
|---|---:|---:|---|
| T, target gain | +0.066 [+0.057, +0.074] | +0.221 [+0.198, +0.245] | +0.156 [+0.133, +0.178] |
| U, b vs a-only | -0.034 [-0.037, -0.030] | +0.015 [+0.010, +0.019] | +0.048 [+0.043, +0.055] |
| U_base, b vs baseline | -0.041 [-0.050, -0.032] | +0.004 [-0.004, +0.013] | +0.045 [+0.040, +0.051] |
| spillover, mean abs shift on other 5 | +0.036 [+0.031, +0.043] | +0.025 [+0.022, +0.033] | -0.011 [-0.013, -0.006] |
| joint, tau=0 | +0.619 [+0.524, +0.714] | +0.262 [+0.143, +0.333] | -0.357 [-0.524, -0.262] |
| joint, tau=0.02 | +0.071 [+0.048, +0.119] | +0.000 [+0.000, +0.095] | -0.071 [-0.095, +0.000] |
| joint, tau=0.05 | +0.071 [+0.024, +0.071] | +0.000 [+0.000, +0.024] | -0.071 [-0.071, +0.000] |
| strict (b below baseline) | +0.619 [+0.524, +0.714] | +0.190 [+0.095, +0.238] | -0.429 [-0.571, -0.333] |

## Coherence (fluency, 0-100; judge meta-llama/llama-3.3-70b-instruct, same prompt, one pass)

Steering: a-only at the published operating point (runs/<slug>/coherence.md, from steer_specificity_raw.csv); no published coherence exists for steered a-b. Prompting: runs_revision/<slug>/p3_coherence.md. Deltas are against each run's own baseline.

| Model | steering baseline | steering a-only, mean delta | prompting baseline | prompting a-only, mean delta | prompting a-b, mean delta | prompting a-b, worst |
|---|---:|---:|---:|---:|---:|---:|
| Llama-3.2-3B-Instruct | 94.5 | -6.9 | 94.0 | -1.1 | -0.8 | -2.1 |
| gemma-2-2b-it | 94.2 | -9.9 | 94.6 | -2.9 | -2.9 | -4.7 |
| Qwen3-0.6B | 89.2 | -9.8 | 90.0 | -1.0 | -1.1 | -3.8 |
| Falcon3-3B-Instruct | 95.8 | -10.6 | 95.9 | -2.6 | -2.2 | -4.7 |

## Lexical leak (answer contains a word of the emotion's family)

Word families (regex stems): anger: anger|angr|furious|fury|rage|enrag|irritat|livid|mad(?![a-z]); disgust: disgust|revolt|repuls|gross|nause|sicken; fear: fear|afraid|scare|terrif|frighten|panic|dread; guilt: guilt; joy: joy|happ(?:y|ier|iest|ily|iness)|delight|elat|glee|cheer; sadness: sad|sorrow|grief|griev|unhappy|depress|heartbr; shame: shame|asham|humiliat|embarrass.

| Model | baseline: a word | steering a-only: a | steering a-b: a / b | prompting a-only: a | prompting a-b: a / b |
|---|---:|---:|---:|---:|---:|
| Llama-3.2-3B-Instruct | 6% | 27% | 25% / 1% | 19% | 17% / 2% |
| gemma-2-2b-it | 5% | 18% | 22% / 1% | 14% | 17% / 2% |
| Qwen3-0.6B (flagged) | 3% | 29% | 16% / 0% | 72% | 70% / 48% |
| Falcon3-3B-Instruct | 6% | 38% | 29% / 0% | 36% | 38% / 13% |

Baseline column: mean over the 7 emotions of the share of unsteered answers naming that emotion.

