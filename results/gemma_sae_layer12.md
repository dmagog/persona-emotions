# Phase E1 — GemmaScope SAE decomposition of emotion vectors

SAE: `google/gemma-scope-2b-pt-res`, `layer_12/width_16k/average_l0_82`
(gemma-2-2b residual). For each emotion's Gemma direction vector (response_avg_diff
at layer 12), JumpReLU-encode through the SAE and take the top features.
Reproduce: `emotion/gemma_sae.py`; data in `gemma_sae_layer12.json`.

## Result — naive projection does NOT disentangle

Top-5 features per emotion are nearly identical:

| emotion | n_active | top-5 features |
|---------|---------:|----------------|
| anger   | 24 | 6810, 1005, 2291, 336, 11498 |
| disgust | 25 | 6810, 1005, 336, 2291, 11498 |
| fear    | 24 | 6810, 1005, 336, 11498, 2291 |
| guilt   | 23 | 6810, 1005, 2291, 336, 11498 |
| joy     | 25 | 6810, 1005, 336, 2291, 14984 |
| sadness | 24 | 6810, 1005, 336, 11498, 14984 |
| shame   | 24 | 6810, 1005, 336, 2291, 11498 |

**Mean off-diagonal Jaccard of top-k feature sets: 0.907** (many pairs = 1.00).

## Interpretation

Projecting the *raw* emotion difference vector onto the SAE recovers the same
features for every emotion, because the vector is dominated by the shared
"emotional/negative affect" component (cf. cosines 0.45–0.86, the sadness
attractor, and the E0 centering result). The SAE faithfully encodes that shared
component — so the top features are the shared-affect features, not
emotion-specific ones.

The entanglement is therefore robust across analyses: dense cosines → steering
leakage → survives mean-subtraction → naive SAE projection.

## Next (E1b)

A "minimal emotion vector" needs the shared component removed *before* SAE
projection (project the centered/mean-subtracted vector), or feature attribution
from the *difference* of actual pos vs neg SAE activations rather than the
direction projection. Testing centered-then-SAE next.
