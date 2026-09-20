# Valence-arousal reading of emotion-vector correlations: Falcon3-3B-Instruct, layer 13

## Shared direction

Cosine similarity between each emotion vector and the mean emotion vector:
   disgust: 0.896
     shame: 0.882
     anger: 0.870
   sadness: 0.867
      fear: 0.857
     guilt: 0.713
       joy: 0.531

The first component of the raw vector set accounts for 69% of its variance. A shared direction therefore explains much of the high cosine similarity.

## Residual structure

Emotion projections on residual principal components after removing the mean direction:
     emo   resPC1   resPC2  valence  arousal
   anger    -1.29     4.24    -0.70     0.85
 disgust    -1.93     3.04    -0.80     0.55
    fear    -1.42    -1.46    -0.70     0.90
   guilt     1.42    -2.77    -0.60     0.40
     joy     8.26     0.59     1.00     0.75
 sadness    -1.72    -2.30    -0.70     0.25
   shame    -3.32    -1.33    -0.65     0.45

resPC1 versus valence: Spearman +0.59; versus arousal: +0.25
resPC2 versus valence: Spearman -0.33; versus arousal: +0.57

Joy on resPC1: +8.26; negative emotions: [-3.32, +1.42] (joy is separate).

## Empirical check from judge scores

Based on 448 texts.
Mean off-diagonal correlation: +0.25
Mean correlation of joy with the other emotions: -0.59

## Interpretation

The emotion vectors share a strong common direction. After removing that direction, the first residual component correlates with the canonical valence ordering (Spearman +0.59). The residual geometry does not isolate a complete valence-arousal structure.
Judge scores provide a separate output-level check. Joy is less correlated with the negative emotions than the average emotion pair, which is consistent with a valence distinction.
