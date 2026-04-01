#!/bin/bash

# Evaluate persona vector steering effectiveness
# Usage: bash scripts/eval_steering.sh

gpu=${1:-0}
model="${MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}"
coefs=(-1.0 0 1.0 2.0)
layer=20
steering_type="response"

traits=("inventive" "consistent" "dependable" "careless" "outgoing" "solitary" "compassionate" "self-interested" "nervous" "calm")

for coef in "${coefs[@]}"; do
    for trait in "${traits[@]}"; do
        echo "=========================================="
        echo "Processing trait: $trait"
        echo "Steering Coefficient: $coef"
        echo "=========================================="
        vector_path="persona_vectors/$(basename $model)/${trait}_response_avg_diff.pt"
        output_path="eval_persona_eval/$(basename $model)/${trait}_steer_${steering_type}_layer${layer}_coef${coef}.csv"
        CUDA_VISIBLE_DEVICES=$gpu python -m eval.eval_persona \
            --model $model \
            --trait $trait \
            --output_path $output_path \
            --judge_model gpt-4.1-mini \
            --version eval \
            --steering_type $steering_type \
            --coef $coef \
            --vector_path $vector_path \
            --layer $layer

        if [ $? -ne 0 ]; then
            echo "Error: Failed to evaluate steering for $trait"
            continue
        fi
    done
done
