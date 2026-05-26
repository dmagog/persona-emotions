# Steering eval — anger (Qwen2.5-3B-Instruct)

First end-to-end steering result: add `coeff * anger_response_avg_diff[layer+1]`
to `model.layers[layer]` while generating first-person accounts on 10 held-out
*neutral* prompts (no emotion instruction), then measure the encoder's anger
score. Reproduce: `emotion/steer_eval.py`; per-response data in
`steer_anger_qwen2.5-3b.csv`.

| layer | coeff | mean anger | n |
|------:|------:|-----------:|--:|
| 14 | 0.0 | 0.033 | 10 |
| 14 | 4.0 | 0.226 | 10 |
| 14 | 8.0 | **0.465** | 10 |
| 18 | 0.0 | 0.033 | 10 |
| 18 | 4.0 | 0.184 | 10 |
| 18 | 8.0 | 0.308 | 10 |

**Findings**

- Clear monotonic dose-response: baseline anger ≈ 0.033 → 0.23 (coeff 4) → 0.47
  (coeff 8) at layer 14. Steering with the extracted vector works.
- At coeff 8 the steered anger (0.465) exceeds even the instruction-based positive
  responses from the dataset (0.362, see `encoder_scoring_qwen2.5-3b.csv`).
- Layer 14 is more effective than layer 18 at the same coefficient.
- Generations stay coherent and on-topic. Example (layer 14, coeff 8, driving-lesson
  scenario): *"I'm so frustrated right now. I can't even make the car do what the
  instructor wants it to do… It's like we're never going to get this thing figured out."*

**Validates the full pipeline**: dataset → judge-filtered pairs → extracted vectors
→ steering → measurable, coherent emotion increase.

**Next**: (1) all 7 emotions; (2) **specificity** — does anger-steering also raise
*other* emotions' scores (expected, given the 0.58–0.86 cross-emotion cosines)?;
(3) finer layer/coeff sweep. Off-target leakage motivates the disentanglement /
SAE direction (Phase E).
