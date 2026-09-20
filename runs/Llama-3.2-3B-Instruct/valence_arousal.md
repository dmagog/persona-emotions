# Valence-arousal reading of emotion-vector correlations: Llama-3.2-3B-Instruct, layer 16

## Shared direction

Cosine similarity between each emotion vector and the mean emotion vector:
   disgust: 0.899
     shame: 0.866
      fear: 0.866
   sadness: 0.863
     anger: 0.861
     guilt: 0.855
       joy: 0.697

The first component of the raw vector set accounts for 70% of its variance. A shared direction therefore explains much of the high cosine similarity.

## Residual structure

Emotion projections on residual principal components after removing the mean direction:
     emo   resPC1   resPC2  valence  arousal
   anger    -0.11     0.91    -0.70     0.85
 disgust    -0.15     0.26    -0.80     0.55
    fear    -0.35    -0.20    -0.70     0.90
   guilt    -0.24    -0.17    -0.60     0.40
     joy     1.47    -0.12     1.00     0.75
 sadness    -0.16    -0.40    -0.70     0.25
   shame    -0.46    -0.28    -0.65     0.45

resPC1 versus valence: Spearman +0.04; versus arousal: +0.21
resPC2 versus valence: Spearman -0.15; versus arousal: +0.54

Joy on resPC1: +1.47; negative emotions: [-0.46, -0.11] (joy is separate).

## Empirical check from judge scores

Based on 447 texts.
Mean off-diagonal correlation: +0.11
Mean correlation of joy with the other emotions: -0.50

## Interpretation

The emotion vectors share a strong common direction. After removing that direction, the first residual component correlates with the canonical valence ordering (Spearman +0.04). The residual geometry does not isolate a complete valence-arousal structure.
Judge scores provide a separate output-level check. Joy is less correlated with the negative emotions than the average emotion pair, which is consistent with a valence distinction.
