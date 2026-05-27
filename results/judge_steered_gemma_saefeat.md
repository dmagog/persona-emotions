# Judge-based readout of SAE-feature steering (guilt/disgust blind-spot check)

The GoEmotions encoder is blind to guilt and weak on disgust, so the
encoder-based specificity matrix could not tell whether steering toward those
emotions *failed* or was simply *unmeasurable*. Here we judge the steered
generations directly with an LLM judge (`meta-llama/llama-3.3-70b-instruct` via
OpenRouter): how strongly does each answer express the target emotion (0–100),
baseline vs steered-toward-X. Reproduce: `emotion/steer_specificity.py
--save-answers` (gemma-2-2b, SAE-feature vectors) → `emotion/judge_steered.py`.
Data: `steer_spec_gemma_saefeat_ans.csv`.

| emotion | baseline | steered | Δ |
|---------|---------:|--------:|----:|
| **guilt**   | 32.1 | **77.9** | **+45.8** |
| disgust | 10.0 | 37.1 | +27.1 |
| anger   | 14.3 | 40.7 | +26.4 |
| joy     | 21.4 | 55.4 | +33.9 |

## Finding

SAE-feature steering **induces the target emotion for every emotion tested,
including guilt and disgust** — guilt is in fact the *strongest* effect (+45.8).
The earlier near-zero guilt / weak disgust in the encoder specificity matrix was a
**measurement artifact** (encoder blind spot), not a steering failure.

Implication: the contrastive-SAE "minimal emotion vector" works across all 7
emotions; the only thing that failed for guilt/disgust earlier was the *readout*.
A judge-based readout should be used (alongside the encoder) whenever evaluating
those emotions.
