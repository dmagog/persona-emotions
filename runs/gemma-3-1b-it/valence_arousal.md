# Valence-arousal reading of emotion-vector correlations: gemma-3-1b-it, layer 13

## Shared direction

Cosine similarity between each emotion vector and the mean emotion vector:
   disgust: 0.893
      fear: 0.883
   sadness: 0.867
     guilt: 0.864
     shame: 0.830
     anger: 0.742
       joy: 0.537

The first component of the raw vector set accounts for 65% of its variance. A shared direction therefore explains much of the high cosine similarity.

## Residual structure

Emotion projections on residual principal components after removing the mean direction:
     emo   resPC1   resPC2  valence  arousal
   anger   -34.43   -52.91    -0.70     0.85
 disgust    38.83    -5.91    -0.80     0.55
    fear    43.67     8.10    -0.70     0.90
   guilt    19.13    21.26    -0.60     0.40
     joy   -96.08    33.05     1.00     0.75
 sadness    39.09    15.96    -0.70     0.25
   shame   -10.21   -19.54    -0.65     0.45

resPC1 versus valence: Spearman -0.59; versus arousal: -0.11
resPC2 versus valence: Spearman +0.59; versus arousal: -0.29

Joy on resPC1: -96.08; negative emotions: [-34.43, +43.67] (joy is separate).

## Empirical check from judge scores

Based on 446 texts.
Mean off-diagonal correlation: +0.27
Mean correlation of joy with the other emotions: -0.54

## Interpretation

The emotion vectors share a strong common direction. After removing that direction, the first residual component correlates with the canonical valence ordering (Spearman -0.59). The residual geometry does not isolate a complete valence-arousal structure.
Judge scores provide a separate output-level check. Joy is less correlated with the negative emotions than the average emotion pair, which is consistent with a valence distinction.
