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

## E1b — centered-then-SAE (also fails)

Subtracting the mean emotion vector before SAE projection (`--center`, data in
`gemma_sae_layer12_centered.json`) barely changes the picture:

- mean off-diagonal Jaccard **0.907 → 0.836** (still very high; max 1.00);
- top-5 features remain the same dominant ids (6810, 1005, 2291, 336, …) across
  emotions.

## Conclusion (E1)

**Projecting a difference-direction onto the SAE encoder does not isolate
emotion-specific features** — a few dominant SAE features absorb most of the
projection for every emotion, even after removing the shared component. The
direction-projection method is the wrong tool.

The entanglement is robust across *every* analysis we ran: dense cosines (0.45–0.86)
→ steering leakage (sadness attractor) → survives mean-subtraction (E0) → naive
SAE projection (0.91) → centered SAE projection (0.84).

## Proper next method (E1c)

Use **contrastive SAE feature activations on real responses**: run the actual pos
and neg generations through Gemma + SAE, collect per-token feature activations,
and rank features by mean(pos) − mean(neg) per emotion. This operates where the
JumpReLU thresholds are meaningful and the pos−neg contrast cancels the shared
baseline — the principled way to find a sparse "minimal emotion vector". It is a
heavier run (Gemma forward + SAE over ~832×2 responses), so it is the next step,
not part of this cheap projection probe.
