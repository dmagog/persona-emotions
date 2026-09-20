# Valence-arousal reading of emotion-vector correlations: Llama-3.2-1B-Instruct, layer 10

## Shared direction

Cosine similarity between each emotion vector and the mean emotion vector:
   disgust: 0.911
   sadness: 0.874
      fear: 0.871
     shame: 0.869
     anger: 0.850
     guilt: 0.825
       joy: 0.666

The first component of the raw vector set accounts for 68% of its variance. A shared direction therefore explains much of the high cosine similarity.

## Residual structure

Emotion projections on residual principal components after removing the mean direction:
     emo   resPC1   resPC2  valence  arousal
   anger    -0.06     0.37    -0.70     0.85
 disgust    -0.10     0.10    -0.80     0.55
    fear    -0.14    -0.09    -0.70     0.90
   guilt    -0.07    -0.16    -0.60     0.40
     joy     0.68    -0.03     1.00     0.75
 sadness    -0.12    -0.11    -0.70     0.25
   shame    -0.18    -0.07    -0.65     0.45

resPC1 versus valence: Spearman +0.33; versus arousal: +0.18
resPC2 versus valence: Spearman -0.33; versus arousal: +0.57

Joy on resPC1: +0.68; negative emotions: [-0.18, -0.06] (joy is separate).

## Empirical check from judge scores

Based on 448 texts.
Mean off-diagonal correlation: +0.25
Mean correlation of joy with the other emotions: -0.63

## Interpretation

The emotion vectors share a strong common direction. After removing that direction, the first residual component correlates with the canonical valence ordering (Spearman +0.33). The residual geometry does not isolate a complete valence-arousal structure.
Judge scores provide a separate output-level check. Joy is less correlated with the negative emotions than the average emotion pair, which is consistent with a valence distinction.
