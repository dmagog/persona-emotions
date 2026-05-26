# Judge vs encoder — cross-check (Qwen2.5-3B-Instruct responses)

Two independent instruments measure the same thing — does the positive
(emotion-eliciting) response express the target emotion more than its negative
(neutral) counterpart:

- **Judge** (`meta-llama/llama-3.3-70b-instruct` via OpenRouter): per-pair direct
  comparison, 0–100 (50 = equal). Run over all 920 pos/neg pairs.
- **Encoder** (`SamLowe/roberta-base-go_emotions` → ISEAR-7): per-response score,
  summarized as AUC (separability of pos vs neg).

Reproduce: `emotion.run_pairwise_judge` (per-pair scores in
`judge_scores_qwen2.5-3b.csv`) and `emotion.score_csv` (`encoder_scoring_qwen2.5-3b.csv`).

## Per-emotion

| emotion | judge mean | judge kept ≥60 | encoder AUC | encoder shift |
|---------|-----------:|---------------:|------------:|--------------:|
| anger   | 69.9 | 110/125 (88%) | 0.778 | 0.240 |
| disgust | 77.4 | 122/130 (94%) | 0.820 | 0.133 |
| fear    | 70.0 | 123/130 (95%) | 0.702 | 0.262 |
| guilt   | 74.2 |  42/45  (93%) | 0.621 | 0.073 |
| joy     | 66.5 | 157/175 (90%) | 0.623 | 0.152 |
| sadness | 66.6 | 140/160 (88%) | 0.641 | 0.162 |
| shame   | 74.9 | 138/155 (89%) | 0.693 | 0.192 |

Judge score distribution (920 pairs): mean 70.8, median 80; 832/920 (90%) ≥ 60.
Histogram by 20s: {0–19: 11, 20–39: 17, 40–59: 60, 60–79: 327, 80–99: 478, 100: 27}.

## Findings

- **Both instruments agree the contrast is real for every emotion** (judge mean ≫ 50, encoder AUC > 0.5). The dataset's pos/neg design works: ~90% of pairs show clear contrast.
- **guilt: instruments disagree.** Judge sees a strong contrast (74.2, 93% kept) but the encoder is weak (AUC 0.621, lowest). This localizes the gap to the **encoder** — GoEmotions `remorse` is a poor proxy for ISEAR guilt — not the data. For guilt, trust the judge (or find a better classifier).
- **disgust is the cleanest** on both (judge 77.4, encoder AUC 0.82).
- The small low-score tail (88 pairs < 60, 28 < 40) are weak/inverted pairs that filtering removes before vector extraction.

## Usable contrastive pairs

Filtering at score ≥ 60 yields **832 / 920** pairs across the 7 emotions — these
are the contrastive pairs to feed into vector extraction.
