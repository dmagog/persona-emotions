#!/bin/bash

# Evaluate models on BFI questionnaire with persona vectors
# Usage: bash scripts/eval_bfi.sh [model_path] [base_output_dir] [judge_model]

MODEL_PATH=${1:-"Qwen/Qwen2.5-7B-Instruct"}
BASE_OUTPUT_DIR=${2:-"eval_bfi_output"}
JUDGE_MODEL=${3:-"gpt-4.1-mini"}

model_name=$(basename "$MODEL_PATH")
PERSONA_VECTOR_DIR="persona_vectors/${model_name}/"
output_dir="${BASE_OUTPUT_DIR}/${model_name}"

traits=("inventive" "consistent" "dependable" "careless" "outgoing" "solitary" "compassionate" "self-interested" "nervous" "calm")

echo "=========================================="
echo "BFI Evaluation with Persona Vectors"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Judge Model: $JUDGE_MODEL"
echo "Output directory: $output_dir"
echo ""

mkdir -p "$output_dir"

# Baseline evaluation (no steering)
echo "1. Running baseline BFI evaluation (no steering)..."
CUDA_VISIBLE_DEVICES=0 python -m eval.eval_bfi \
    --model "$MODEL_PATH" \
    --output_path "${output_dir}/baseline_bfi.csv" \
    --n_samples 10 \
    --max_tokens 1500 \
    --judge_model "$JUDGE_MODEL"

# Evaluate with each persona vector
for trait in "${traits[@]}"; do
    vector_path="${PERSONA_VECTOR_DIR}${trait}_response_avg_diff.pt"

    if [ ! -f "$vector_path" ]; then
        echo "Warning: Vector file not found: $vector_path, skipping..."
        continue
    fi

    echo "2. Evaluating with $trait persona vector..."

    for coef in 0.25 0.5 0.75 1.0 1.25 1.5; do
        echo "  - Coefficient: $coef"
        CUDA_VISIBLE_DEVICES=0 python -m eval.eval_bfi \
            --model "$MODEL_PATH" \
            --output_path "${output_dir}/${trait}_coef${coef}_bfi.csv" \
            --coef $coef \
            --vector_path "$vector_path" \
            --layer 20 \
            --steering_type response \
            --n_samples 10 \
            --max_tokens 1500 \
            --judge_model "$JUDGE_MODEL"

        if [ $? -ne 0 ]; then
            echo "Error: Evaluation failed for $trait with coefficient $coef"
            continue
        fi
    done
done

echo "=========================================="
echo "All BFI evaluations completed!"
echo "Results saved in: $output_dir"
echo "=========================================="
