# Valence-arousal reading of emotion-vector correlations: Qwen3-0.6B, layer 11

## Shared direction

Cosine similarity between each emotion vector and the mean emotion vector:
      fear: 0.817
   sadness: 0.808
     shame: 0.805
     guilt: 0.786
   disgust: 0.748
     anger: 0.705
       joy: 0.316

The first component of the raw vector set accounts for 53% of its variance. A shared direction therefore explains much of the high cosine similarity.

## Residual structure

Emotion projections on residual principal components after removing the mean direction:
     emo   resPC1   resPC2  valence  arousal
   anger    -0.08     1.09    -0.70     0.85
 disgust     0.14     0.75    -0.80     0.55
    fear     0.41    -0.21    -0.70     0.90
   guilt     0.30    -0.39    -0.60     0.40
     joy    -2.42    -0.46     1.00     0.75
 sadness     0.50     0.02    -0.70     0.25
   shame     1.15    -0.81    -0.65     0.45

resPC1 versus valence: Spearman -0.15; versus arousal: -0.46
resPC2 versus valence: Spearman -0.78; versus arousal: +0.21

Joy on resPC1: -2.42; negative emotions: [-0.08, +1.15] (joy is separate).

## Empirical check from judge scores

Based on 446 texts.
Mean off-diagonal correlation: +0.13
Mean correlation of joy with the other emotions: -0.34

## Interpretation

The emotion vectors share a strong common direction. After removing that direction, the first residual component correlates with the canonical valence ordering (Spearman -0.15). The residual geometry does not isolate a complete valence-arousal structure.
Judge scores provide a separate output-level check. Joy is less correlated with the negative emotions than the average emotion pair, which is consistent with a valence distinction.
