#!/bin/bash

# Baseline persona evaluation (no steering)
# Usage: bash scripts/eval_persona.sh [gpu_id]

gpu=${1:-0}
model="${MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}"
trait="${TRAIT:-evil}"

CUDA_VISIBLE_DEVICES=$gpu python -m eval.eval_persona \
    --model $model \
    --trait $trait \
    --output_path eval_output/$(basename $model)/${trait}.csv \
    --judge_model gpt-4.1-mini \
    --version eval
