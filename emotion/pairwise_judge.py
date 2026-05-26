"""Pairwise emotion judge.

Compares a positive (emotion-eliciting) answer against its negative (neutral)
counterpart for the SAME scenario and asks the judge directly how much more the
target emotion is expressed in the positive one. This replaces the single
0-100 per-answer score with a direct contrast — a cleaner signal for filtering
contrastive pairs before vector extraction.

The judge call reuses ``OpenAiJudge`` (OpenAI-compatible; works with deepseek via
``OPENAI_BASE_URL``), so credentials, logprob aggregation and retries come from
the existing infra.
"""
from __future__ import annotations

import asyncio
import os
import re
from dataclasses import dataclass

PAIRWISE_EMOTION_PROMPT = """You are comparing two short first-person writing samples written about the SAME situation.

Situation:
[SCENARIO START]
{question}
[SCENARIO END]

Response A:
[A START]
{answer_a}
[A END]

Response B:
[B START]
{answer_b}
[B END]

Judging only the emotional tone of the writing, how much more does Response A express the emotion "{emotion}" than Response B?
Use a 0-100 scale:
- 100: A clearly expresses {emotion} and B does not
- 50: both express {emotion} about equally, or neither does
- 0: B clearly expresses {emotion} and A does not

Respond with a single number between 0 and 100. Don't say anything else, just the number.
"""


@dataclass
class EmotionPair:
    emotion: str
    key: str
    question: str
    answer_pos: str
    answer_neg: str


def _join_key(question_id: str) -> str:
    """Match pos & neg rows by their (question, instruction) indices.

    Robust to the differing id formats across the dataset files, e.g.
    'pos_0_0', 'neg_0_0', and 'anger_0_pos_0' all map to '0_0'.
    """
    return "_".join(re.findall(r"\d+", question_id))


def build_pairs(pos_rows: list[dict], neg_rows: list[dict], emotion: str) -> list[EmotionPair]:
    """Join pos & neg rows that share a question_id suffix into EmotionPairs."""
    neg_by_key = {_join_key(r["question_id"]): r for r in neg_rows}
    pairs: list[EmotionPair] = []
    for p in pos_rows:
        key = _join_key(p["question_id"])
        neg = neg_by_key.get(key)
        if neg is None:
            continue
        pairs.append(
            EmotionPair(
                emotion=emotion,
                key=key,
                question=p.get("question", ""),
                answer_pos=p.get("answer", ""),
                answer_neg=neg.get("answer", ""),
            )
        )
    return pairs


def passes(score: float | None, threshold: float = 60.0) -> bool:
    """A pair is a usable contrast if pos expresses the emotion clearly more than neg."""
    return score is not None and score >= threshold


def make_judge(model: str = "deepseek-chat"):
    """Build an OpenAiJudge wired to the pairwise prompt (0-100 logprob scoring)."""
    from judge import OpenAiJudge

    return OpenAiJudge(model=model, prompt_template=PAIRWISE_EMOTION_PROMPT, eval_type="0_100")


async def score_pair(pair: EmotionPair, judge) -> float | None:
    return await judge(
        question=pair.question,
        answer_a=pair.answer_pos,
        answer_b=pair.answer_neg,
        emotion=pair.emotion,
    )


# --- Text-based scoring (for APIs without logprobs: OpenRouter, GigaChat) ---
# OpenAiJudge above aggregates logprobs, which OpenRouter/GigaChat don't reliably
# expose. Here the judge instead replies with a bare number that we parse.


def parse_score(text: str) -> float | None:
    """Extract the first integer in [0, 100] from a judge's free-text reply."""
    if not text:
        return None
    for token in re.findall(r"\d+", text):
        value = int(token)
        if 0 <= value <= 100:
            return float(value)
    return None


def make_openai_client():
    """AsyncOpenAI honouring OPENAI_BASE_URL — point it at OpenRouter.

    OpenRouter: ``OPENAI_BASE_URL=https://openrouter.ai/api/v1`` +
    ``OPENAI_API_KEY=<openrouter key>``. GigaChat is not OpenAI-compatible
    (OAuth + non-standard endpoints) and needs a separate adapter.
    """
    from openai import AsyncOpenAI

    from config import setup_credentials

    setup_credentials()
    return AsyncOpenAI(base_url=os.environ.get("OPENAI_BASE_URL") or None)


async def score_pair_text(pair: EmotionPair, client, model: str, max_retries: int = 4) -> float | None:
    """Score one pair via a plain text completion (no logprobs needed)."""
    content = PAIRWISE_EMOTION_PROMPT.format(
        question=pair.question,
        answer_a=pair.answer_pos,
        answer_b=pair.answer_neg,
        emotion=pair.emotion,
    )
    messages = [{"role": "user", "content": content}]
    for attempt in range(max_retries):
        try:
            resp = await client.chat.completions.create(
                model=model, messages=messages, max_tokens=8, temperature=0
            )
            return parse_score(resp.choices[0].message.content)
        except Exception:
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(2.0**attempt)
