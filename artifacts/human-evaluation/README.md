# Human evaluation artifacts

The labeling sheet contains 597 candidate rows for blinded annotation. The committed sheet does not include completed human scores. The key maps 300 sampled item IDs to the corresponding steering condition, local encoder score, and LLM-judge score.

Once completed anonymous ratings are available, use the two files with `python3 -m emotion.label_correlation --sheet artifacts/human-evaluation/human_label_sheet.csv --key artifacts/human-evaluation/human_label_key.csv` to compare the automatic measures with the completed ratings. The key remains separate from the sheet so raters do not see model scores.
