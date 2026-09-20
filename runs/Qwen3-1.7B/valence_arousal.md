# Valence-arousal reading of emotion-vector correlations: Qwen3-1.7B, layer 11

## Shared direction

Cosine similarity between each emotion vector and the mean emotion vector:
     shame: 0.900
   sadness: 0.897
      fear: 0.841
     anger: 0.824
   disgust: 0.821
     guilt: 0.729
       joy: 0.503

The first component of the raw vector set accounts for 64% of its variance. A shared direction therefore explains much of the high cosine similarity.

## Residual structure

Emotion projections on residual principal components after removing the mean direction:
     emo   resPC1   resPC2  valence  arousal
   anger    -0.07     0.76    -0.70     0.85
 disgust     0.68     2.20    -0.80     0.55
    fear     1.95     0.60    -0.70     0.90
   guilt     0.14    -4.29    -0.60     0.40
     joy    -6.11     0.43     1.00     0.75
 sadness     1.79     0.54    -0.70     0.25
   shame     1.62    -0.23    -0.65     0.45

resPC1 versus valence: Spearman -0.48; versus arousal: -0.11
resPC2 versus valence: Spearman -0.85; versus arousal: +0.50

Joy on resPC1: -6.11; negative emotions: [-0.07, +1.95] (joy is separate).

## Empirical check from judge scores

Based on 446 texts.
Mean off-diagonal correlation: +0.19
Mean correlation of joy with the other emotions: -0.49

## Interpretation

The emotion vectors share a strong common direction. After removing that direction, the first residual component correlates with the canonical valence ordering (Spearman -0.48). The residual geometry does not isolate a complete valence-arousal structure.
Judge scores provide a separate output-level check. Joy is less correlated with the negative emotions than the average emotion pair, which is consistent with a valence distinction.
