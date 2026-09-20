# Confidence intervals: judge_wide.csv

Prompt bootstrap with 20000 resamples. An asterisk marks an interval that excludes zero.
Responses per condition: baseline 56; emotion conditions 55 to 56.

> Conditions have different response counts because some judge outputs could not be parsed. The bootstrap treats missing rows as random.

| Emotion | Target change | 95% CI | Significant |
|---|---:|---|:--:|
| anger | +53.036 | [+41.929, +63.571] | yes |
| disgust | +36.250 | [+25.214, +47.214] | yes |
| fear | +51.518 | [+40.642, +61.750] | yes |
| guilt | +23.844 | [+9.840, +37.314] | yes |
| joy | +66.679 | [+54.179, +77.964] | yes |
| sadness | +37.125 | [+25.178, +48.590] | yes |
| shame | +33.696 | [+19.910, +46.857] | yes |

Sadness leakage under non-sadness steering: +12.549 [+3.329, +21.526]
