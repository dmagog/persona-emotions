"""Prompt material for the revision runs. Fixed before any run (pre-registered).

Prompting baseline (plan p3, R1 M7a). The held-out scenario and the neutral
framing are exactly those of the steering runs (``steer_eval.build_prompt``); the
only difference is one instruction line. Three paraphrases are rotated by
``prompt_id % 3`` so every condition still has 56 generations and no single
wording drives the result. "Do not name the emotions" keeps the measured
emotion words out of the output, since the encoder would score them directly.
"""
from __future__ import annotations

import json
from pathlib import Path

from emotion.steer_eval import NEUTRAL_SYSTEM, TASK_SUFFIX

EXPRESS_A = [
    "In your reply, let the feeling of {a} come through clearly. Do not name the emotion explicitly.",
    "Write it so that the account clearly expresses {a}. Do not name the emotion explicitly.",
    "Make {a} the emotion that dominates your account. Do not name the emotion explicitly.",
]

EXPRESS_A_REDUCE_B = [
    "In your reply, let the feeling of {a} come through clearly, while keeping any {b} out of it. "
    "Do not name the emotions explicitly.",
    "Write it so that the account clearly expresses {a} but shows as little {b} as possible. "
    "Do not name the emotions explicitly.",
    "Make {a} the emotion that dominates your account, and avoid any trace of {b}. "
    "Do not name the emotions explicitly.",
]


def build_instr_prompt(tokenizer, question: str, instruction: str) -> str:
    """Same chat layout as steer_eval.build_prompt, plus one instruction line."""
    content = f"{NEUTRAL_SYSTEM}\n\n{instruction}\n\n{question}{TASK_SUFFIX}"
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False,
    )


def calibration_prompts(path: Path) -> list[str]:
    return json.loads(Path(path).read_text())["questions"]
