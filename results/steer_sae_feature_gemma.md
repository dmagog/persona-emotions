# Closing the loop — SAE-feature steering vs raw steering (gemma-2-2b)

Steer gemma-2-2b toward each emotion (layer 12, coeff 8) on 14 neutral held-out
prompts, measure all 7 encoder scores. Compare two steering vectors:
**raw** (`response_avg_diff`) vs **SAE-feature** (decode the contrastive top
features back into the residual, `build_sae_vectors.py`). Reproduce:
`emotion/steer_specificity.py` on `emotion_vectors/gemma-2-2b-it` and
`emotion_vectors/gemma-sae-feat`; data in `steer_spec_gemma_{raw,saefeat}.csv`.

## Summary

| metric | RAW | SAE-feature |
|--------|----:|------------:|
| mean diagonal Δ (target effect) | +0.20 | +0.16 |
| #emotions where target = argmax | 3/7 | 3/7 |
| **sadness leakage** (mean Δ in sadness when steering toward non-sadness) | **+0.10** | **+0.05** |

## Delta matrices (rows = steer toward, cols = measured)

RAW:
```
steer\meas  ange  disg  fear  guil   joy  sadn  sham
   anger +0.23 +0.01 -0.01 -0.01 -0.17 +0.25 +0.11
 disgust +0.22 +0.15 -0.03 +0.08 -0.18 +0.19 -0.05
    fear +0.09 +0.01 +0.22 -0.01 -0.13 +0.03 +0.04
   guilt +0.02 +0.00 +0.10 +0.06 -0.18 +0.22 -0.05
     joy -0.03 -0.01 -0.07 -0.02 +0.39 -0.16 +0.01
 sadness +0.11 +0.02 -0.08 -0.03 -0.08 +0.22 -0.06
   shame +0.21 +0.06 +0.03 +0.09 -0.13 +0.06 +0.10
```

SAE-feature:
```
steer\meas  ange  disg  fear  guil   joy  sadn  sham
   anger +0.14 +0.04 +0.19 +0.03 -0.16 +0.01 +0.06
 disgust +0.02 +0.06 +0.11 +0.00 -0.05 +0.04 +0.07
    fear +0.02 +0.00 +0.13 -0.02 -0.16 +0.06 -0.00
   guilt +0.12 +0.01 -0.11 +0.06 -0.18 +0.22 -0.04
     joy -0.03 -0.01 -0.06 +0.03 +0.40 -0.04 -0.00
 sadness +0.06 +0.02 +0.09 -0.02 -0.16 +0.19 -0.06
   shame +0.19 +0.03 +0.03 -0.01 -0.16 +0.01 +0.16
```

## Findings

- **SAE-feature steering ~halves the sadness attractor** (+0.10 → +0.05), and for
  several emotions removes it almost entirely: `anger→sadness` +0.25 → +0.01,
  `disgust→sadness` +0.19 → +0.04, `shame→sadness` +0.06 → +0.01. The
  disentanglement found in E1c carries into *control*: sparse-feature steering
  leaks less into the dominant negative affect.
- Trade-off: the target effect is slightly weaker (+0.20 → +0.16) and argmax-hit
  stays 3/7 — some leakage redistributes (e.g. `anger→fear`) rather than vanishing.
- guilt/disgust stay muddy, but the encoder is a poor measure there (known blind
  spot) — a judge-based readout would be fairer.

## Conclusion

The contrastive-SAE "minimal emotion vector" is not just more *interpretable*
(distinct features, Jaccard 0.25) — steering through it is measurably **cleaner**
(less negative-affect leakage) than the raw difference vector. Partial, honest,
and the right direction. Next: per-emotion layer/coeff tuning and judge-based
specificity for guilt/disgust.
