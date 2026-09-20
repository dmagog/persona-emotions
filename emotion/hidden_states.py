"""Hidden-state pooling used when extracting contrastive emotion directions."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from tqdm import tqdm


def pooled_hidden_states(
    model,
    tokenizer,
    prompts: Sequence[str],
    responses: Sequence[str],
    layer_list: Sequence[int] | None = None,
):
    """Return prompt means, prompt-final states, and response means for each layer.

    Inputs are tokenized without added special tokens because callers provide the
    complete chat-formatted prompt and generated continuation.
    """
    max_layer = model.config.num_hidden_layers
    layers = list(range(max_layer + 1)) if layer_list is None else list(layer_list)
    prompt_avg = [[] for _ in range(max_layer + 1)]
    response_avg = [[] for _ in range(max_layer + 1)]
    prompt_last = [[] for _ in range(max_layer + 1)]

    for prompt, response in tqdm(zip(prompts, responses), total=len(prompts)):
        inputs = tokenizer(
            prompt + response, return_tensors="pt", add_special_tokens=False
        ).to(model.device)
        prompt_len = len(tokenizer.encode(prompt, add_special_tokens=False))
        if prompt_len == 0:
            raise ValueError("A prompt has no tokens.")
        if inputs.input_ids.shape[1] <= prompt_len:
            raise ValueError("A response has no tokens.")
        outputs = model(**inputs, output_hidden_states=True)
        for layer in layers:
            state = outputs.hidden_states[layer]
            prompt_avg[layer].append(state[:, :prompt_len, :].mean(dim=1).detach().cpu())
            response_avg[layer].append(state[:, prompt_len:, :].mean(dim=1).detach().cpu())
            prompt_last[layer].append(state[:, prompt_len - 1, :].detach().cpu())

    for layer in layers:
        prompt_avg[layer] = torch.cat(prompt_avg[layer], dim=0)
        prompt_last[layer] = torch.cat(prompt_last[layer], dim=0)
        response_avg[layer] = torch.cat(response_avg[layer], dim=0)
    return prompt_avg, prompt_last, response_avg
