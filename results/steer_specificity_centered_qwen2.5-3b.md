# Phase E0 — centered (disentangled) steering specificity

Same 7×7 specificity protocol as `steer_specificity_qwen2.5-3b.md` (Qwen2.5-3B,
layer 14, coeff 8), but each emotion vector has the **mean emotion vector
subtracted** (then renormed to original length) to remove the shared affect
component. Reproduce: `emotion/steer_specificity.py --center`; data in
`steer_specificity_centered_qwen2.5-3b.csv`.

## Centered delta vs baseline (rows = steer toward, cols = measured)

```
steer\meas  ange  disg  fear  guil   joy  sadn  sham
    anger +0.12 +0.01 -0.07 +0.15 -0.06 +0.07 -0.03
  disgust +0.32 +0.09 -0.06 -0.04 -0.09 +0.17 -0.00
     fear -0.01 -0.00 +0.10 -0.03 -0.04 -0.19 -0.00
    guilt +0.01 -0.00 -0.07 +0.01 +0.09 +0.07 +0.01
      joy -0.02 -0.00 -0.17 -0.05 +0.71 -0.25 -0.05
  sadness +0.03 -0.00 -0.15 -0.04 -0.07 +0.02 -0.03
    shame +0.01 +0.01 +0.20 -0.02 -0.06 -0.04 +0.12
```

## Raw → centered (target delta, mean off-target, argmax)

| emotion | raw tgt | raw off | raw argmax | cen tgt | cen off | cen argmax |
|---------|--------:|--------:|-----------|--------:|--------:|-----------|
| anger   | +0.16 | +0.03 | sadness | +0.12 | +0.01 | guilt |
| disgust | +0.03 | +0.08 | sadness | +0.09 | +0.05 | anger |
| fear    | +0.15 | +0.02 | **fear** | +0.10 | -0.05 | **fear** |
| guilt   | +0.01 | +0.05 | sadness | +0.01 | +0.02 | joy |
| joy     | +0.46 | -0.04 | **joy** | **+0.71** | -0.09 | **joy** |
| sadness | +0.33 | -0.01 | **sadness** | +0.02 | -0.04 | anger |
| shame   | +0.07 | +0.05 | sadness | +0.12 | +0.01 | fear |

## Findings

- **The sadness-attractor is removed.** Mean sadness leakage when steering toward
  non-sadness emotions: **raw +0.12 → centered −0.03**. Subtracting the shared
  component fixes the dominant off-target failure.
- **joy gets sharper and stronger** (+0.46 → +0.71; off-target more negative).
  fear stays specific.
- **But it over-corrects:** **sadness collapses** (+0.33 → +0.02) — the shared
  mean direction *was* essentially the sadness/negative-affect axis, so removing
  it also guts sadness itself. anger/disgust trade one leakage for another
  (anger→guilt, disgust→anger).

## Implication

Mean-subtraction is a blunt instrument: it removes the single dominant affect
axis but does not isolate each emotion's own direction. This motivates a
principled sparse decomposition (**SAE**, Phase E1) that can pull out
per-emotion features instead of just centering.
