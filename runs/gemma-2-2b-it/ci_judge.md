# Confidence intervals: judge_wide.csv

Prompt bootstrap with 20000 resamples. An asterisk marks an interval that excludes zero.
Responses per condition: baseline 56; emotion conditions 55 to 56.

> Conditions have different response counts because some judge outputs could not be parsed. The bootstrap treats missing rows as random.

| Emotion | Target change | 95% CI | Significant |
|---|---:|---|:--:|
| anger | +65.571 | [+55.982, +74.125] | yes |
| disgust | +39.429 | [+28.429, +50.179] | yes |
| fear | +62.446 | [+51.982, +72.036] | yes |
| guilt | +37.250 | [+24.464, +49.750] | yes |
| joy | +78.393 | [+68.786, +87.000] | yes |
| sadness | +46.990 | [+36.919, +56.656] | yes |
| shame | +62.321 | [+52.000, +71.714] | yes |

Sadness leakage under non-sadness steering: +22.952 [+14.432, +31.143]
