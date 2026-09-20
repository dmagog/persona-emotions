# Confidence intervals: judge_wide.csv

Prompt bootstrap with 20000 resamples. An asterisk marks an interval that excludes zero.
Responses per condition: baseline 56; emotion conditions 53 to 56.

> Conditions have different response counts because some judge outputs could not be parsed. The bootstrap treats missing rows as random.

| Emotion | Target change | 95% CI | Significant |
|---|---:|---|:--:|
| anger | +87.304 | [+83.107, +91.107] | yes |
| disgust | +92.607 | [+87.964, +96.214] | yes |
| fear | +75.214 | [+67.893, +82.054] | yes |
| guilt | +77.375 | [+70.357, +83.875] | yes |
| joy | +74.964 | [+64.964, +84.054] | yes |
| sadness | +73.176 | [+66.027, +79.781] | yes |
| shame | +80.756 | [+74.677, +86.284] | yes |

Sadness leakage under non-sadness steering: +52.749 [+44.629, +60.437]
