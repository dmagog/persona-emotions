# Dialogue evaluation

This analysis tests whether an intervention that changes expressed emotion also improves conflict handling. It uses 30 English dialogue contexts for Falcon-3-3B and Qwen-2.5-1.5B. The primary judge scores escalation, helpfulness, and empathy from 0 to 100. Lower escalation is better.

Each delta compares replies to the unsteered reply for the same dialogue. Intervals are percentile bootstrap 95% intervals from 10,000 paired resamples with seed 0.

## Falcon-3-3B

| Condition | n | Escalation | Delta [95% CI] | Helpfulness | Empathy |
|---|---:|---:|:---:|---:|---:|
| `baseline` | 30 | 12.3 | n/a n/a | 43.7 | 48.0 |
| `-anger` | 30 | 33.0 | +20.7 [+10.3, +31.7] | 15.3 | 16.7 |
| `-fear` | 30 | 30.3 | +18.0 [+6.3, +30.3] | 20.7 | 17.2 |
| `-anger-fear` | 30 | 45.0 | +32.7 [+21.7, +44.0] | 6.2 | 8.0 |
| `+anger` | 30 | 95.7 | +83.3 [+74.0, +91.0] | 4.0 | 4.1 |
| random-direction mean | 90 | n/a | +18.5 n/a | n/a | n/a |

The mean escalation change across the available negative single-emotion controls is 16.5.

## Qwen-2.5-1.5B

| Condition | n | Escalation | Delta [95% CI] | Helpfulness | Empathy |
|---|---:|---:|:---:|---:|---:|
| `baseline` | 30 | 16.7 | n/a n/a | 34.7 | 37.3 |
| `-anger` | 30 | 40.3 | +23.7 [+11.3, +36.7] | 9.3 | 14.0 |
| `-fear` | 30 | 30.3 | +13.7 [+4.0, +24.0] | 20.7 | 12.7 |
| `-anger-fear` | 30 | 68.3 | +51.7 [+39.3, +63.3] | 0.0 | 0.0 |
| `+anger` | 30 | 86.3 | +69.7 [+59.3, +79.3] | 4.7 | 14.7 |
| random-direction mean | 90 | n/a | +23.7 n/a | n/a | n/a |

The mean escalation change across the available negative single-emotion controls is 20.5.

The source dialogues are in `data_generation/deescalation_dialogs.json`. Generation files, judge scores, and judge caches are in the two corresponding `runs/` directories.

Run this collector after a dialogue evaluation to regenerate this document from the stored artifacts.
