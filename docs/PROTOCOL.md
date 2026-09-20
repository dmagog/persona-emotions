# Experimental protocol

This document records the common protocol used for the CEmoSteer experiments. It applies to the 11-model comparison in the accompanying study.

## Directions

For each ISEAR emotion, the evaluated model writes matched emotional and neutral replies to the same scenario. The direction at decoder block `l` is the mean response-token activation for emotional replies minus the corresponding mean for neutral replies. Directions are extracted at every decoder block and retain their raw norms.

The extraction scenarios, anger calibration prompts, and 56 held-out evaluation prompts are disjoint. All reported generations use greedy decoding. Qwen 3 thinking mode is disabled. Gemma 2 receives the shared framing text in the user turn because its template does not accept a system role.

## Intervention

At every decoding position, the intervention adds a scaled direction to the residual stream:

```text
h_steered = h + coefficient * direction
```

Each model has one operating point: a decoder block and positive coefficient. The study does not normalize a direction before steering. A numerical coefficient can therefore imply a different perturbation size across models or emotions.

## Operating-point selection

Candidate layers are located near 35%, 45%, and 55% of model depth. The coefficient grid is `{0, 2, 4, 6, 8, 16}`. On a disjoint set of 16 anger prompts, the selected cell maximizes mean anger-score change while keeping the repeated-4-gram degeneration rate at or below 10%.

The selected anger operating point is reused for all seven single directions and every ordered difference in that model. It is not tuned per emotion, pair, or held-out prompt.

## Single-direction evaluation

For each of 56 held-out scenarios, the model produces one unsteered response and one response under each of the seven emotion directions. The local scorer is `SamLowe/roberta-base-go_emotions`, mapped to the ISEAR categories without converting its multi-label probabilities into a simplex.

For cross-category summaries, each mapped score is divided by the number of GoEmotions labels assigned to that ISEAR category. This range normalization preserves signs and within-category uncertainty while giving every category a common 0 to 1 scale.

The primary LLM judge is Llama-3.3-70B-Instruct. Gemini-3.5-Flash-Lite and GPT-4.1-mini repeat the single-direction study. Judge results can have missing rows because invalid or incomplete API responses are excluded rather than imputed.

## Ordered differences

The composition direction for an ordered pair is `v_X - v_Y`. For every pair, success has two independent conditions:

- The target `X` is higher than the unsteered baseline.
- The subtracted component `Y` is lower than it is under `X` alone.

The second condition tests selective attenuation. It does not require `Y` to become lower than the baseline, because steering `X` may increase `Y` by itself.

## Quality and uncertainty

Lexical degeneration flags repeated 4-grams above 0.15 or a type-token ratio below 0.45. A separate tone-agnostic judge measures coherence on a 0 to 100 scale. Neither metric is a substitute for task adherence.

Automatic effect intervals use prompt-identity bootstrap resampling. The study uses 20,000 resamples. Missing judge scores are excluded without imputation, and composition conditions require at least 40 judged responses.

Human validation uses a locked, stratified sample of 300 outputs rated independently by three assessors. They score all seven emotions, fluency, task adherence, and naturalness without seeing model identity, intervention condition, or automatic scores.

## Dialogue extension

The dialogue study tests 30 English multi-turn conflict scenarios on Falcon-3-3B and Qwen-2.5-1.5B. It reuses the operating point selected for expressed emotion and evaluates escalation, helpfulness, and empathy. The protocol includes negative emotion directions, a positive anger control, a strength sweep, and norm-matched random directions.
