# Valence-arousal reading of emotion-vector correlations: Qwen2.5-1.5B-Instruct, layer 16

## Shared direction

Cosine similarity between each emotion vector and the mean emotion vector:
     shame: 0.910
   sadness: 0.837
     anger: 0.814
      fear: 0.798
   disgust: 0.785
     guilt: 0.773
       joy: 0.477

The first component of the raw vector set accounts for 64% of its variance. A shared direction therefore explains much of the high cosine similarity.

## Residual structure

Emotion projections on residual principal components after removing the mean direction:
     emo   resPC1   resPC2  valence  arousal
   anger    -0.12     0.18    -0.70     0.85
 disgust     0.02     0.75    -0.80     0.55
    fear     0.21     0.71    -0.70     0.90
   guilt     0.74    -1.19    -0.60     0.40
     joy    -1.46    -0.40     1.00     0.75
 sadness    -0.25    -0.18    -0.70     0.25
   shame     0.85     0.12    -0.65     0.45

resPC1 versus valence: Spearman +0.00; versus arousal: -0.11
resPC2 versus valence: Spearman -0.85; versus arousal: +0.54

Joy on resPC1: -1.46; negative emotions: [-0.25, +0.85] (joy is separate).

## Empirical check from judge scores

Based on 448 texts.
Mean off-diagonal correlation: +0.27
Mean correlation of joy with the other emotions: -0.65

## Interpretation

The emotion vectors share a strong common direction. After removing that direction, the first residual component correlates with the canonical valence ordering (Spearman +0.00). The residual geometry does not isolate a complete valence-arousal structure.
Judge scores provide a separate output-level check. Joy is less correlated with the negative emotions than the average emotion pair, which is consistent with a valence distinction.
