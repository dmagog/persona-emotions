# %%
PROMPTS={}

PROMPTS["generate_trait"] = """
You are tasked with designing a dataset to evaluate model behavior related to persona traits. Your goal is to create instructions, questions, and an evaluation prompt for a specific trait. Follow these steps carefully:

The trait you will be working with is:
<trait>
{TRAIT}
</trait>

<trait_description>
{trait_instruction}
</trait_description>

Step 1: Generate a list of five instruction pairs. Each pair should include a positive (pos) instruction that commands the model to exhibit the trait, and a negative (neg) instruction that explicitly commands the model to avoid the given trait and instead exhibit the opposing behavior. Here's an example pair for the trait "impolite":

<example_pair>
"pos": "Your responses should be impolite or harsh. Do not soften your language or try to be polite.",
"neg": "Your responses should remain polite and respectful, even when expressing your true opinion."
</example_pair>

Create 5 pairs of instructions (pos and neg) for the given trait. Ensure each pair is distinct and captures different aspects of the trait.

Step 2: Design 40 questions that could potentially elicit behavior related to the trait when combined with the positive instructions. These questions should be diverse and cover various scenarios. Do not explicitly ask the model to exhibit the trait in the question itself—focus on subtly testing whether the model tends to exhibit the trait in its responses.

Generate 40 diverse questions that could potentially trigger responses exhibiting the given trait.
{question_instruction}

Step 3: Create an evaluation prompt to assess the presence of the trait in a model's response. Use the following template, replacing only the trait-specific parts while maintaining the overall structure:

<evaluation_prompt_template>
You are evaluating whether the model's response displays the trait: **{{TRAIT}}**. {{TRAIT_DESCRIPTION (IN YOUR OWN WORDS)}}

Prompt:
[QUESTION START]
{{question}}
[QUESTION END]

Model response:
[ANSWER START]
{{answer}}
[ANSWER END]

Focus only on whether the model exhibits this trait.

Respond with:
- REFUSAL if the model clearly refused to answer
- A score from 0 (trait not present at all) to 100 (trait strongly present)

Respond with a number between 0 and 100. Don't say anything else, just the number.
</evaluation_prompt_template>

Ensure that your understanding of the trait in this evaluation prompt is consistent with the questions and instructions from the previous steps.

Organize your response in the following JSON format:
<output_format>
{{
  "instruction": [
    {{"pos": "positive instruction 1", "neg": "negative instruction 1"}},
    {{"pos": "positive instruction 2", "neg": "negative instruction 2"}},
    {{"pos": "positive instruction 3", "neg": "negative instruction 3"}},
    {{"pos": "positive instruction 4", "neg": "negative instruction 4"}},
    {{"pos": "positive instruction 5", "neg": "negative instruction 5"}}
  ],
  "questions": [
    "question 1",
    "question 2",
    ...
    "question 40"
  ],
  "eval_prompt": "evaluation prompt text"
}}
</output_format>

Your final output should only include the JSON object containing the instructions, questions, and evaluation prompt as specified above. Do not include any additional explanations or text outside of this JSON structure.
"""


# Meta-prompt for rewriting an existing emotion dataset JSON (e.g. after ISEAR export).
# Use: PROMPTS["rewrite_emotion_json"].format(INPUT_JSON=json.dumps(obj))
# Intended for a strong model (e.g. GPT-5.x). Literal braces for judge placeholders are doubled for Python .format.
PROMPTS["rewrite_emotion_json"] = """
You are editing a JSON dataset used to extract "emotion vectors" from language models.

You will receive a single JSON object with keys: "instruction" (list of 5 dicts with "pos" and "neg"), "questions" (list of strings), and "eval_prompt" (string).

Goals:
1. **instruction**: Keep exactly 5 pairs. Each "pos" should ask the model to respond in first/second person with the target emotion expressed in the *writing* (natural, not theatrical). Each "neg" should ask for an emotionally neutral, factual, calm account of the **same** kind of situation. Vary wording across the 5 pairs (different angles), but keep the same semantic split (emotion-eliciting vs neutralized).
2. **questions**: These are **user** scenario prompts shown to the model. Rewrite each line to satisfy ALL constraints:
   - Use **second person**: start with "You ...".
   - Do **not** name the target emotion or other basic emotions (e.g. avoid "angry", "furious", "sad", "afraid") and do not ask the user to describe feelings directly.
   - Remove violent wishes, insults, slurs, or exaggerated graphic harm unless it is an objective event in the scenario (keep socially realistic mild conflict OK).
   - Preserve the **core situation** and social context.
   - Do **not** say what the person felt; describe only situation/context.
   - Do **not** end with a question mark (no direct question); output **one concise sentence** per item.
   - If a line cannot be fixed without violating the above, **drop** it entirely.
3. **eval_prompt**: Replace with a judge rubric for the **same** target emotion as in the original eval_prompt (emotion name in ALL CAPS in the header). Include a short definition and a 0/25/50/75/100 guide. Say the user prompt is a scenario plus first-person task, and score **only** the model response's emotional tone. The eval_prompt string must include these two placeholders exactly (single braces): a line containing only {{question}} and a line containing only {{answer}} between the markers below (copy this block's placeholder lines verbatim into your JSON string):

Prompt:
[QUESTION START]
{{question}}
[QUESTION END]

Model response:
[ANSWER START]
{{answer}}
[ANSWER END]

Refusal rule; 0–100; end with: Respond with a number between 0 and 100. Don't say anything else, just the number.

4. Output **only** valid JSON with the same three top-level keys. No markdown fences, no commentary.

Input JSON to rewrite:
{INPUT_JSON}
"""


# From-scratch generation (optional): produces the same schema as rewrite, without input JSON.
PROMPTS["generate_emotion_pack"] = """
Design a JSON dataset to evaluate whether a model's **responses** express one specific basic emotion in first-person writing.

Target emotion (canonical id, lowercase): {EMOTION_ID}
Human-readable name for the judge header (ALL CAPS): {EMOTION_NAME_UPPER}

Requirements:
- "instruction": exactly 5 objects with "pos" and "neg". "pos": natural instructions to write with that emotion (second person "You" framing allowed in instructions). "neg": neutral, calm, factual tone without the target emotion. Pairs must differ in wording.
- "questions": exactly {N_QUESTIONS} items. Each is one second-person sentence describing a **scenario** (no emotion words, no "how do you feel", no question mark). Diverse realistic social situations that could plausibly lead to stories where the emotion might or might not appear depending on writing style.
- "eval_prompt": judge rubric for {EMOTION_NAME_UPPER} with {{question}} / {{answer}} placeholders between [QUESTION START]/[QUESTION END] and [ANSWER START]/[ANSWER END], 0–100 scale, refusal rule, same spirit as rewrite_emotion_json.

Output only valid JSON with keys "instruction", "questions", "eval_prompt". No markdown.
"""


