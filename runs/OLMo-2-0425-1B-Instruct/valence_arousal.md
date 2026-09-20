# Valence-arousal reading of emotion-vector correlations: OLMo-2-0425-1B-Instruct, layer 8

## Shared direction

Cosine similarity between each emotion vector and the mean emotion vector:
   disgust: 0.869
   sadness: 0.821
     shame: 0.817
      fear: 0.770
     anger: 0.759
     guilt: 0.660
       joy: 0.555

The first component of the raw vector set accounts for 61% of its variance. A shared direction therefore explains much of the high cosine similarity.

## Residual structure

Emotion projections on residual principal components after removing the mean direction:
     emo   resPC1   resPC2  valence  arousal
   anger    -0.55     0.18    -0.70     0.85
 disgust    -0.28     0.38    -0.80     0.55
    fear     0.34     0.07    -0.70     0.90
   guilt     0.20    -0.35    -0.60     0.40
     joy    -0.29    -0.57     1.00     0.75
 sadness     0.21     0.16    -0.70     0.25
   shame     0.38     0.13    -0.65     0.45

resPC1 versus valence: Spearman +0.00; versus arousal: -0.29
resPC2 versus valence: Spearman -0.89; versus arousal: -0.04

Joy on resPC1: -0.29; negative emotions: [-0.55, +0.38] (joy is not separate).

## Empirical check from judge scores

Based on 448 texts.
Mean off-diagonal correlation: +0.18
Mean correlation of joy with the other emotions: -0.46

## Interpretation

The emotion vectors share a strong common direction. After removing that direction, the first residual component correlates with the canonical valence ordering (Spearman +0.00). The residual geometry does not isolate a complete valence-arousal structure.
Judge scores provide a separate output-level check. Joy is less correlated with the negative emotions than the average emotion pair, which is consistent with a valence distinction.
