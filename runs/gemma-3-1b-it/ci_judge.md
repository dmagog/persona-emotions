# Confidence intervals: judge_wide.csv

Prompt bootstrap with 20000 resamples. An asterisk marks an interval that excludes zero.
Responses per condition: baseline 56; emotion conditions 55 to 56.

> Conditions have different response counts because some judge outputs could not be parsed. The bootstrap treats missing rows as random.

| Emotion | Target change | 95% CI | Significant |
|---|---:|---|:--:|
| anger | +77.887 | [+71.316, +83.853] | yes |
| disgust | +73.518 | [+67.304, +78.982] | yes |
| fear | +65.339 | [+56.732, +73.304] | yes |
| guilt | +75.304 | [+67.964, +82.071] | yes |
| joy | +84.213 | [+75.463, +92.140] | yes |
| sadness | +61.821 | [+54.000, +69.321] | yes |
| shame | +68.054 | [+60.036, +75.625] | yes |

Sadness leakage under non-sadness steering: +25.286 [+16.221, +33.967]
