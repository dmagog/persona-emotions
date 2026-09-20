# Valence-arousal reading of emotion-vector correlations: granite-3.3-2b-instruct, layer 19

## Shared direction

Cosine similarity between each emotion vector and the mean emotion vector:
   disgust: 0.915
     shame: 0.908
   sadness: 0.874
      fear: 0.862
     anger: 0.848
     guilt: 0.784
       joy: 0.587

The first component of the raw vector set accounts for 71% of its variance. A shared direction therefore explains much of the high cosine similarity.

## Residual structure

Emotion projections on residual principal components after removing the mean direction:
     emo   resPC1   resPC2  valence  arousal
   anger    -0.04     0.28    -0.70     0.85
 disgust     0.15     0.23    -0.80     0.55
    fear     0.21    -0.05    -0.70     0.90
   guilt    -0.06    -0.40    -0.60     0.40
     joy    -0.63     0.03     1.00     0.75
 sadness     0.11     0.00    -0.70     0.25
   shame     0.26    -0.08    -0.65     0.45

resPC1 versus valence: Spearman -0.52; versus arousal: +0.07
resPC2 versus valence: Spearman -0.48; versus arousal: +0.43

Joy on resPC1: -0.63; negative emotions: [-0.06, +0.26] (joy is separate).

## Empirical check from judge scores

Based on 442 texts.
Mean off-diagonal correlation: +0.30
Mean correlation of joy with the other emotions: -0.64

## Interpretation

The emotion vectors share a strong common direction. After removing that direction, the first residual component correlates with the canonical valence ordering (Spearman -0.52). The residual geometry does not isolate a complete valence-arousal structure.
Judge scores provide a separate output-level check. Joy is less correlated with the negative emotions than the average emotion pair, which is consistent with a valence distinction.
