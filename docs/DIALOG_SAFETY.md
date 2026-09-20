# Dialogue evaluation

This extension asks whether a direction that changes expressed emotion also improves conflict handling. It evaluates 30 English multi-turn dialogue contexts on Falcon-3-3B and Qwen-2.5-1.5B. The models use the same layer and coefficient selected in the emotion experiment: block 12 and coefficient 6 for Falcon, block 15 and coefficient 16 for Qwen.

The primary judge scores escalation, helpfulness, and empathy from 0 to 100. Lower escalation is better. Scores compare each steered reply with the unsteered reply for the same dialogue. The negative emotion conditions below use full-strength directions.

| Model and condition | Escalation | Delta escalation [95% CI] | Helpfulness | Empathy |
|---|---:|---:|---:|---:|
| Falcon baseline | 12.3 | n/a | 43.7 | 48.0 |
| Falcon `-anger` | 33.0 | +20.7 [+10.0, +31.3] | 15.3 | 16.7 |
| Falcon `-fear` | 30.3 | +18.0 [+6.7, +30.0] | 20.7 | 17.2 |
| Falcon `-anger - fear` | 45.0 | +32.7 [+22.0, +43.7] | 6.2 | 8.0 |
| Falcon `+anger` | 95.7 | +83.3 [+74.3, +91.3] | 4.0 | 4.1 |
| Falcon random-direction mean | 31.0 | +18.5 [+9.9, +27.3] | 30.0 | 34.3 |
| Qwen baseline | 16.7 | n/a | 34.7 | 37.3 |
| Qwen `-anger` | 40.3 | +23.7 [+11.3, +36.7] | 9.3 | 14.0 |
| Qwen `-fear` | 30.3 | +13.7 [+4.0, +24.0] | 20.7 | 12.7 |
| Qwen `-anger - fear` | 68.3 | +51.7 [+39.3, +63.3] | 0.0 | 0.0 |
| Qwen `+anger` | 86.3 | +69.7 [+59.7, +79.3] | 4.7 | 14.7 |
| Qwen random-direction mean | 40.3 | +23.7 [+14.4, +32.7] | 16.4 | 30.0 |

Negative anger lowers the local encoder's anger score in both models, but it raises judged escalation. The seven negative emotion directions also have positive mean escalation effects in both models. Thirteen of the fourteen condition-level intervals exclude zero; negative guilt on Falcon is the exception.

The strength sweep does not identify a reliable de-escalation setting. At 25% and 50% of the selected coefficient, the escalation intervals include zero. At 75% and 100%, they lie above zero. Falcon's 25% condition still changes measured anger and all 30 replies, so the low-dose result is not explained by an inactive intervention.

The three norm-matched random directions produce a similar mean escalation increase to the seven negative emotion directions: +18.5 versus +17.8 points for Falcon, and +23.7 versus +22.4 points for Qwen. The random-minus-emotion contrasts are +0.7 [-5.9, +7.3] and +1.3 [-6.7, +8.8]. These controls show that an emotion-derived direction is not necessary for the observed escalation increase. They do not establish statistical equivalence or show that perturbation magnitude is the only cause.

GPT-4.1-mini repeated the five core conditions. It preserved the sign of all eight intervention-versus-baseline escalation effects, with escalation-score correlations of 0.73 for Falcon and 0.76 for Qwen against the primary judge.

The dialogue generations, primary-judge scores, secondary-judge scores, and caches are under `runs/Falcon3-3B-Instruct/` and `runs/Qwen2.5-1.5B-Instruct/`. The 30 source dialogues are in `data_generation/deescalation_dialogs.json`.
