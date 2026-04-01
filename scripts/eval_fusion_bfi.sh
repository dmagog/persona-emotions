#!/bin/bash

# Evaluate persona vector fusion experiments on BFI questionnaire
# Tests vector addition (trait combination) and subtraction (trait suppression)
# Usage: bash scripts/eval_fusion_bfi.sh [model_path] [base_output_dir] [judge_model]

MODEL_PATH=${1:-"Qwen/Qwen2.5-7B-Instruct"}
BASE_OUTPUT_DIR=${2:-"eval_fusion_bfi_output"}
JUDGE_MODEL=${3:-"gpt-4.1-mini"}

model_name=$(basename "$MODEL_PATH")
PERSONA_VECTOR_DIR="persona_vectors/${model_name}/"
output_dir="${BASE_OUTPUT_DIR}/${model_name}"

echo "=========================================="
echo "BFI Persona Vector Fusion Experiments"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Judge Model: $JUDGE_MODEL"
echo "Output directory: $output_dir"
echo ""

mkdir -p "$output_dir"

layer=20
steering_type="response"
n_samples=10
max_tokens=1500

# --- Vector Addition: Cross-Dimension Fusion ---
# Example: Inventive + Outgoing (openness + extraversion)
run_fusion() {
    local trait1=$1 trait2=$2 coef1=${3:-1.0} coef2=${4:-1.0}
    local v1="${PERSONA_VECTOR_DIR}${trait1}_response_avg_diff.pt"
    local v2="${PERSONA_VECTOR_DIR}${trait2}_response_avg_diff.pt"
    local out="${output_dir}/fusion_${trait1}_${coef1}_${trait2}_${coef2}_bfi.csv"

    echo "Fusion: ${trait1} (${coef1}) + ${trait2} (${coef2})"
    CUDA_VISIBLE_DEVICES=0 python -m eval.eval_bfi \
        --model "$MODEL_PATH" \
        --output_path "$out" \
        --coef $coef1 --vector_path "$v1" \
        --coef1 $coef2 --vector_path1 "$v2" \
        --layer $layer --steering_type $steering_type \
        --n_samples $n_samples --max_tokens $max_tokens \
        --judge_model "$JUDGE_MODEL"
}

# Cross-dimension addition
run_fusion inventive outgoing 1.0 1.0
run_fusion calm outgoing 1.0 1.0
run_fusion inventive consistent 1.0 1.0
run_fusion nervous compassionate 1.0 1.0

# Cross-dimension subtraction (trait isolation)
run_fusion outgoing solitary 1.0 -1.0
run_fusion inventive outgoing 1.0 -1.0
run_fusion outgoing compassionate 1.0 -1.0
run_fusion calm nervous 1.0 -1.0

echo "=========================================="
echo "All fusion experiments completed!"
echo "Results saved in: $output_dir"
echo "=========================================="
