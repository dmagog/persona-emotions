# Encoder scoring — Qwen2.5-3B-Instruct responses

Measures whether positive (emotion-eliciting) responses express the target
emotion more than the negative (neutral) ones, using the classifier encoder.

**Provenance**

- Responses: `eval_emotion/Qwen2.5-3B-Instruct/{emotion}_{pos,neg}.csv` (full, no limit).
- Encoder: `SamLowe/roberta-base-go_emotions` → ISEAR-7 (`GO_EMOTIONS_TO_ISEAR`).
- Run: Kaggle CPU kernel `georgymamarin/emotion-encoder-scoring`, 2026-05-26.
- Reproduce: `notebooks/kaggle_score_emotions.ipynb`; data in `encoder_scoring_qwen2.5-3b.csv`.

**Metrics**: `pos`/`neg` = mean target score; `shift = pos − neg`; `auc` = P(random pos > random neg), threshold-free separability; `top1_pos` = fraction of pos responses where target is top-1.

## v2 (current: no forced normalization, narrow joy cluster)

| emotion | pos | neg | shift | auc | top1_pos |
|---------|----:|----:|------:|----:|---------:|
| anger   | 0.362 | 0.122 | 0.240 | 0.778 | 0.368 |
| disgust | 0.155 | 0.022 | 0.133 | 0.820 | 0.192 |
| fear    | 0.615 | 0.352 | 0.262 | 0.702 | 0.777 |
| guilt   | 0.105 | 0.032 | 0.073 | 0.621 | 0.111 |
| joy     | 0.747 | 0.595 | 0.152 | 0.623 | 0.977 |
| sadness | 0.752 | 0.590 | 0.162 | 0.641 | 0.887 |
| shame   | 0.305 | 0.113 | 0.192 | 0.693 | 0.413 |

**mean shift 0.173 · mean auc 0.697**

## v1 → v2 change

v1 used forced normalization over the 7 ISEAR labels + a broad 12-label joy
cluster, which saturated joy (v1 joy pos 0.948 / neg 0.886) and let it dominate
the argmax. v2 drops normalization (GoEmotions is multi-label/sigmoid) and keeps
joy to a core cluster (joy/excitement/amusement).

- joy de-saturated: pos 0.948→0.747, neg 0.886→0.595; shift 0.061→0.152.
- mean shift 0.127→0.173.

## Notes

- All AUC > 0.5 → encoder separates pos from neg for every emotion.
- Best separability: disgust (0.82), anger (0.78). Weakest: guilt (0.62) — GoEmotions `remorse` is a poor proxy for ISEAR guilt.
- `top1_pos` still uneven (joy 0.98 dominates argmax; guilt/disgust rarely top-1) — `auc`/`shift` are the metrics to trust for per-emotion signal.
- Data schema is inconsistent across files (e.g. `question_id` is `pos_0_0` for some emotions, `anger_0_pos_0` / `neg_0_0` for others) — pairing handles it, but worth normalizing upstream.
