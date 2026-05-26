# Emotion steering vectors — Qwen2.5-3B-Instruct

Per-layer `mean(activations_pos) - mean(activations_neg)` vectors for the 7 ISEAR
emotions, computed the same way as PERSONA persona vectors
(`generate_vec.get_hidden_p_and_r`), so they plug into `activation_steer.py`.

**Provenance**

- Pairs: judge-filtered contrastive pairs (score ≥ 60), 832 total across 7 emotions.
- Model: Qwen/Qwen2.5-3B-Instruct (fp16). Activations: residual stream, all layers.
- Run: home RTX 2070 (8 GB), ~25 min compute. Reproduce: `emotion/extract_vectors.py`,
  `notebooks/kaggle_extract_vectors.ipynb`.
- Files: `emotion_vectors/Qwen2.5-3B-Instruct/{emotion}_{prompt_avg,response_avg,prompt_last}_diff.pt`,
  each shape `[37, 2048]` (37 hidden states × hidden dim).

**Sanity: pairwise cosine of `response_avg_diff` at mid layer (18)**

```
        ange  disg  fear  guil   joy  sadn  sham
 anger  1.00  0.86  0.73  0.76  0.72  0.82  0.79
disgust 0.86  1.00  0.80  0.71  0.73  0.86  0.83
  fear  0.73  0.80  1.00  0.58  0.61  0.75  0.79
 guilt  0.76  0.71  0.58  1.00  0.62  0.72  0.69
   joy  0.72  0.73  0.61  0.62  1.00  0.70  0.68
sadness 0.82  0.86  0.75  0.72  0.70  1.00  0.85
 shame  0.79  0.83  0.79  0.69  0.68  0.85  1.00
```

**Finding**

All pairwise cosines are positive and high (0.58–0.86): the raw difference
vectors share a large common "emotional vs neutral" component rather than being
orthogonal per-emotion directions. `joy` is the most distinct (lowest cosines —
valence separates it from the negative cluster).

Implication: steering with a raw vector mostly pushes "more emotional" in
general. For emotion-*specific* control, the shared component should be removed
(e.g. subtract the mean emotion vector, or disentangle via SAE) — this motivates
the Phase E / minimal-emotion-vector direction.
