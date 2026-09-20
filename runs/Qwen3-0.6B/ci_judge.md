# Confidence intervals: judge_wide.csv

Prompt bootstrap with 20000 resamples. An asterisk marks an interval that excludes zero.
Responses per condition: baseline 56; emotion conditions 54 to 56.

> Conditions have different response counts because some judge outputs could not be parsed. The bootstrap treats missing rows as random.

| Emotion | Target change | 95% CI | Significant |
|---|---:|---|:--:|
| anger | +63.716 | [+55.916, +70.943] | yes |
| disgust | +21.071 | [+13.214, +29.464] | yes |
| fear | +64.018 | [+56.250, +71.071] | yes |
| guilt | +68.571 | [+62.143, +74.464] | yes |
| joy | +69.821 | [+60.696, +78.286] | yes |
| sadness | +58.893 | [+51.357, +65.946] | yes |
| shame | +35.000 | [+25.714, +44.107] | yes |

Sadness leakage under non-sadness steering: +27.487 [+20.658, +34.114]
