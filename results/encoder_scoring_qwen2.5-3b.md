# Encoder scoring — first run (Qwen2.5-3B-Instruct responses)

Measures whether positive (emotion-eliciting) responses express the target
emotion more than the negative (neutral) ones, using the classifier encoder.

**Provenance**

- Responses: `eval_emotion/Qwen2.5-3B-Instruct/{emotion}_{pos,neg}.csv` (full, no limit).
- Encoder: `SamLowe/roberta-base-go_emotions` → ISEAR-7 (`GO_EMOTIONS_TO_ISEAR`), normalized.
- Run: Kaggle CPU kernel `georgymamarin/emotion-encoder-scoring`, 2026-05-26.
- Reproduce: `notebooks/kaggle_score_emotions.ipynb`.

**Results** (`pos`/`neg` = mean target score; `shift = pos − neg`; `top1_pos` = fraction of pos responses where target is top-1)

| emotion | n_pos | n_neg | pos | neg | shift | top1_pos |
|---------|------:|------:|----:|----:|------:|---------:|
| anger   | 125 | 125 | 0.337 | 0.146 | 0.191 | 0.368 |
| disgust | 130 | 130 | 0.146 | 0.029 | 0.117 | 0.185 |
| fear    | 130 | 130 | 0.549 | 0.353 | 0.196 | 0.746 |
| guilt   |  45 |  45 | 0.107 | 0.046 | 0.061 | 0.089 |
| joy     | 175 | 175 | 0.948 | 0.886 | 0.061 | 0.994 |
| sadness | 160 | 160 | 0.670 | 0.561 | 0.109 | 0.856 |
| shame   | 155 | 155 | 0.271 | 0.115 | 0.156 | 0.400 |

**mean shift: 0.127**

**Notes**

- All shifts are positive → the pos/neg contrast in the dataset is real and the encoder detects it.
- `guilt` is weak (shift 0.061, top1 0.089): GoEmotions `remorse` is a poor proxy for ISEAR guilt.
- `joy` is saturated (pos 0.948, neg 0.886): the broad positive cluster mapped into `joy` inflates it and dominates the argmax (top1 0.994), suppressing other emotions' top1.
