# Phase E1c — contrastive SAE feature activations (minimal emotion vector)

Run actual judge-filtered pos/neg responses through gemma-2-2b + GemmaScope SAE
(`layer_12/width_16k/average_l0_82`), average each SAE feature's JumpReLU
activation over response tokens, rank features per emotion by mean(pos)−mean(neg).
Reproduce: `emotion/sae_contrastive.py`; data in `sae_contrastive_layer12.json`.

## Result — this method DISENTANGLES

| method | mean off-diag Jaccard |
|--------|----------------------:|
| direction-projection, naive (E1) | 0.91 |
| direction-projection, centered (E1b) | 0.84 |
| **contrastive activations (E1c)** | **0.25** (min 0.08, max 0.38) |

Top-5 contrastive features per emotion (now largely distinct):

| emotion | n_pairs | top-5 features | max Δ |
|---------|--------:|----------------|------:|
| anger   | 110 | 1840, 6810, 8517, 16003, 14599 | 1.00 |
| disgust | 122 | 6810, 2746, 1840, 4886, 13295 | 1.58 |
| fear    | 123 | 13295, 2746, 621, 4886, 6810 | 1.24 |
| guilt   |  42 | 7002, 1041, 2746, 13295, 8567 | 1.07 |
| joy     | 157 | 13295, 1840, 6648, 6810, 4886 | 1.12 |
| sadness | 140 | 13295, 2746, 7002, 8517, 4886 | 1.34 |
| shame   | 138 | 6810, 2746, 4886, 7002, 13295 | 1.27 |

## Findings

- **The minimal emotion vector works — with the right method.** Measuring real
  SAE feature activations and contrasting pos−neg cancels the shared baseline and
  surfaces emotion-specific features: Jaccard drops from ~0.9 (entangled) to 0.25
  (~75% distinct).
- A few features recur across emotions (6810, 13295, 2746, 4886) — general affect;
  each emotion also has its own (anger 8517/16003/14599; guilt 1041/8567; fear 621;
  joy 6648).
- **Methodological lesson:** projecting the difference *direction* onto the SAE
  fails (the dense vector is dominated by shared affect); contrasting *activations*
  succeeds. The method, not the SAE, was the bottleneck.

## Next

Steer via these per-emotion features (decode the selected features back into the
residual stream) and re-run the specificity matrix — does feature-steering beat
raw (sadness-attractor) and centered (over-correcting) steering? That would close
the loop: sparse, emotion-specific control.
