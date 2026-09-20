# Valence-arousal reading of emotion-vector correlations: Qwen2.5-3B-Instruct, layer 14

## Shared direction

Cosine similarity between each emotion vector and the mean emotion vector:
   disgust: 0.948
     shame: 0.934
     anger: 0.927
   sadness: 0.893
      fear: 0.862
     guilt: 0.860
       joy: 0.852

The first component of the raw vector set accounts for 82% of its variance. A shared direction therefore explains much of the high cosine similarity.

## Residual structure

Emotion projections on residual principal components after removing the mean direction:
     emo   resPC1   resPC2  valence  arousal
   anger    -0.69     0.03    -0.70     0.85
 disgust    -1.38     1.34    -0.80     0.55
    fear     0.99     0.18    -0.70     0.90
   guilt     1.07    -0.49    -0.60     0.40
     joy    -1.09    -1.79     1.00     0.75
 sadness     0.98     0.21    -0.70     0.25
   shame     0.13     0.51    -0.65     0.45

resPC1 versus valence: Spearman +0.26; versus arousal: -0.25
resPC2 versus valence: Spearman -0.74; versus arousal: -0.21

Joy on resPC1: -1.09; negative emotions: [-1.38, +1.07] (joy is not separate).

## Empirical check from judge scores

Based on 448 texts.
Mean off-diagonal correlation: +0.11
Mean correlation of joy with the other emotions: -0.42

## Interpretation

The emotion vectors share a strong common direction. After removing that direction, the first residual component correlates with the canonical valence ordering (Spearman +0.26). The residual geometry does not isolate a complete valence-arousal structure.
Judge scores provide a separate output-level check. Joy is less correlated with the negative emotions than the average emotion pair, which is consistent with a valence distinction.
