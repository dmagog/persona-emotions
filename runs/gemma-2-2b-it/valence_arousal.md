# Valence-arousal reading of emotion-vector correlations: gemma-2-2b-it, layer 15

## Shared direction

Cosine similarity between each emotion vector and the mean emotion vector:
   disgust: 0.903
     shame: 0.878
   sadness: 0.847
     anger: 0.843
      fear: 0.840
     guilt: 0.814
       joy: 0.637

The first component of the raw vector set accounts for 66% of its variance. A shared direction therefore explains much of the high cosine similarity.

## Residual structure

Emotion projections on residual principal components after removing the mean direction:
     emo   resPC1   resPC2  valence  arousal
   anger    -1.27     9.22    -0.70     0.85
 disgust    -2.55     6.42    -0.80     0.55
    fear    -4.89    -1.47    -0.70     0.90
   guilt    -1.40    -7.63    -0.60     0.40
     joy    21.25    -1.09     1.00     0.75
 sadness    -4.49    -1.37    -0.70     0.25
   shame    -6.66    -4.09    -0.65     0.45

resPC1 versus valence: Spearman +0.33; versus arousal: +0.18
resPC2 versus valence: Spearman -0.48; versus arousal: +0.43

Joy on resPC1: +21.25; negative emotions: [-6.66, -1.27] (joy is separate).

## Empirical check from judge scores

Based on 447 texts.
Mean off-diagonal correlation: +0.18
Mean correlation of joy with the other emotions: -0.54

## Interpretation

The emotion vectors share a strong common direction. After removing that direction, the first residual component correlates with the canonical valence ordering (Spearman +0.33). The residual geometry does not isolate a complete valence-arousal structure.
Judge scores provide a separate output-level check. Joy is less correlated with the negative emotions than the average emotion pair, which is consistent with a valence distinction.
