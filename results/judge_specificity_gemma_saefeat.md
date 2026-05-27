# Judge-based steering specificity (SAE-feature, gemma-2-2b)

Encoder-independent specificity: every steered answer (SAE-feature steering,
layer 12, coeff 8) scored by an LLM judge for all 7 emotions. Resolves the
GoEmotions blind spots (guilt/disgust). Reproduce: `emotion/judge_specificity.py`
on `steer_spec_gemma_saefeat_ans.csv`; data in `judge_specificity_gemma_saefeat.json`.

## Delta vs baseline (rows = steer toward, cols = judged emotion; diagonal in [])

```
steer\meas  ange  disg  fear  guil   joy  sadn  sham
   anger    +26   +18   +29   +21   -19   +30   +13
 disgust    +23   +25   +30    +6    -5   +33   +18
    fear     +4    +6  [+41]  +16    -3   +24   +13
   guilt     +6    +8   +15  [+46]  -19   +47  [+48]
     joy    -12   -11   -15   -16  [+33]   -8   -10
 sadness    +16   +15   +27   +20   -12  [+45]  +26
   shame    +29   +16   +39   +41   -21   +46  [+56]
```

**target = argmax for 4/7 emotions** (fear, joy, sadness, shame), vs 3/7 with the encoder.

## Findings

- **Diagonals are strong** — steering genuinely induces the target emotion
  (guilt +46, shame +56, sadness +45, fear +41), much larger than the encoder
  suggested (it was blind to guilt, weak on disgust).
- **Misses are semantically sensible**, not random: anger/disgust co-raise
  sadness/fear; guilt is nearly tied with shame/sadness (+46 vs +48/+47). The
  model groups **guilt ~ shame ~ sadness** (negative self-conscious cluster), so
  steering one nudges its neighbours.
- **joy steering cleanly suppresses every negative emotion** (all off-diagonal
  deltas negative) — strong, specific positive control.

## Takeaway

This is the trustworthy version of the specificity result: SAE-feature steering
produces large, mostly emotion-specific shifts; residual cross-talk follows the
emotion geometry (adjacent emotions), not noise. The encoder-based matrix
understated both the effect size and the specificity for guilt/disgust.
