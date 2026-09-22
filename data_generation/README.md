# Emotion prompt data

This directory contains the prompt data used by CEmoSteer. The seven ISEAR categories are anger, disgust, fear, guilt, joy, sadness, and shame.

| Path | Purpose |
|---|---|
| `emotion_data_extract/` | Scenarios used to generate matched emotional and neutral response pairs for direction extraction. |
| `emotion_data_eval/` | Held-out scenarios for single-direction and composition evaluation. |
| `deescalation_dialogs.json` | Thirty multi-turn English dialogue contexts used in the dialogue extension. |

Extraction and evaluation scenarios are disjoint. Keeping them separate prevents the direction-extraction prompts from appearing in the held-out evaluation set.

Each emotion JSON file stores a shared schema:

```json
{
  "instruction": [{"pos": "...", "neg": "..."}],
  "questions": ["..."],
  "eval_prompt": "..."
}
```

The dialogue evaluation reads each context through the model's chat template and generates one next reply. It is a fixed-context experiment, not an interactive simulation with a user.
