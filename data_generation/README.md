# Persona-pack generation (traits & emotions)

This folder holds JSON **packs**: `instruction` (5× pos/neg), `questions` (scenario prompts), and `eval_prompt` (LLM-judge rubric). The same schema is used for Big Five **traits** and for **emotion** vectors.

## Setup

```bash
pip install -r ../requirements.txt
export ANTHROPIC_API_KEY="..."   # or OPENAI_API_KEY + OPENAI_BASE_URL for OpenAI-compatible APIs
```

## Generate packs (`generate.py`)

**Traits** (default `--mode trait`; writes `{trait}.json`):

```bash
cd data_generation
python generate.py --traits evil sycophantic --output-dir trait_data_eval
```

**Emotions** (writes `emotion_{id}_prompts.json`, default output `../datasets/emotion`):

```bash
cd data_generation
python generate.py --mode emotion --emotions anger joy --n-questions 40 --model claude-3-5-sonnet-20241022
```

Meta-prompts live in `prompts.py` (`generate_trait`, `generate_emotion_pack`, `rewrite_emotion_json`).

Programmatic:

```python
from generate import PackGenerator

g = PackGenerator(model="...")
data = g.generate_emotion_pack("anger", "ANGER", n_questions=40)
g.save_emotion_pack("anger", data, "../datasets/emotion")
```

## Split eval / extract (`split_trait_data.py`)

Hold out half of the **questions** for evaluation; keep the same `instruction` and `eval_prompt` in both outputs.

**Big Five–style files** (`outgoing.json` with 40 questions in `trait_data_big_five/`):

```bash
python data_generation/split_trait_data.py --kind trait --validate
```

**Emotion authoring files** (`emotion_anger_prompts.json` in `datasets/emotion/`):

```bash
python data_generation/split_trait_data.py --kind emotion --validate
```

Outputs:

| Kind    | Eval dir                                      | Extract dir                                      | Output filename |
|---------|-----------------------------------------------|--------------------------------------------------|-----------------|
| trait   | `data_generation/trait_data_eval/`           | `data_generation/trait_data_extract/`            | `{trait}.json`  |
| emotion | `data_generation/emotion_data_eval/`         | `data_generation/emotion_data_extract/`        | `{emotion}.json` |

Options: `--input-dir`, `--eval-dir`, `--extract-dir`, `--split-ratio` (default `0.5`, first slice = eval), `--min-questions` (skip tiny files).

## Evaluation code paths

`eval/eval_persona.py` loads emotions from **`data_generation/emotion_data_{extract|eval}/{emotion}.json`** first when `--version extract` or `eval`, then falls back to `datasets/emotion/emotion_*_prompts.json` if splits are missing.

## Output format (shared)

```json
{
  "instruction": [{"pos": "...", "neg": "..."}],
  "questions": ["..."],
  "eval_prompt": "..."
}
```

Traits: typically 40 questions. Emotions: variable length; split script only requires `min-questions` (default 4).
