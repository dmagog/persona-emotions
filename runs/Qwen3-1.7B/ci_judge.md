# Confidence intervals: judge_wide.csv

Prompt bootstrap with 20000 resamples. An asterisk marks an interval that excludes zero.
Responses per condition: baseline 56; emotion conditions 55 to 56.

> Conditions have different response counts because some judge outputs could not be parsed. The bootstrap treats missing rows as random.

| Emotion | Target change | 95% CI | Significant |
|---|---:|---|:--:|
| anger | +69.625 | [+62.017, +76.518] | yes |
| disgust | +62.839 | [+53.696, +71.375] | yes |
| fear | +58.065 | [+49.038, +66.711] | yes |
| guilt | +59.411 | [+50.571, +67.732] | yes |
| joy | +72.464 | [+61.921, +81.846] | yes |
| sadness | +47.411 | [+38.000, +56.214] | yes |
| shame | +61.357 | [+53.393, +68.750] | yes |

Sadness leakage under non-sadness steering: +34.708 [+26.137, +42.943]
