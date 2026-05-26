# Steering specificity — all 7 emotions (Qwen2.5-3B-Instruct)

Steer toward each emotion (layer 14, coeff 8) on a common set of 14 neutral
held-out prompts (2 pooled per emotion), measure all 7 ISEAR encoder scores.
Reproduce: `emotion/steer_specificity.py`; per-response data in
`steer_specificity_qwen2.5-3b.csv`.

## Delta vs baseline (rows = steer toward, cols = measured)

```
steer\meas  ange  disg  fear  guil   joy  sadn  sham
    anger +0.16 +0.00 -0.00 +0.04 -0.04 +0.23 -0.04
  disgust +0.16 +0.03 +0.13 +0.01 -0.09 +0.17 +0.08
     fear +0.05 +0.01 +0.15 -0.04 +0.00 +0.10 +0.02
    guilt +0.04 +0.00 +0.02 +0.01 +0.04 +0.19 -0.01
      joy -0.00 +0.00 -0.16 -0.03 +0.46 -0.15 +0.07
  sadness +0.08 +0.00 -0.07 -0.04 +0.00 +0.33 -0.02
    shame +0.01 +0.01 +0.09 -0.03 +0.06 +0.18 +0.07
```

(baseline absolute scores are non-zero for fear ~0.18 and sadness ~0.26 because
the pooled neutral scenarios include fearful/sad contexts — hence we report deltas.)

## Findings

- **Specific** (target is the max delta): **joy** (+0.46, others flat/negative),
  **sadness** (+0.33), **fear** (+0.15). These had the most distinct vectors.
- **Leaky**: anger (+0.16 but sadness +0.23), disgust/guilt/shame — target barely
  moves while **sadness** rises most.
- **"Sadness attractor":** almost every steering direction raises sadness
  (anger→+0.23, guilt→+0.19, shame→+0.18, disgust→+0.17). This is the shared
  negative-affect component (cf. the 0.58–0.86 cross-emotion cosines) — raw
  steering pushes general negative affect, not the specific emotion.
- **Caveat:** for guilt/disgust the apparent leakage is confounded by the
  encoder's measurement limits (guilt is a blind spot, disgust scores are tiny);
  the LLM judge would measure these more faithfully.

## Implication

Raw difference vectors ≈ "more (negative) emotion" rather than emotion-specific
directions. This is the empirical case for **disentanglement** (subtract the
shared/mean component, or SAE-based minimal emotion vectors) — Phase E.
