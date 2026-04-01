#!/bin/bash

# Multi-Session Dialog Benchmark Evaluation Pipeline
# This script runs the complete evaluation pipeline for the dialog benchmark

echo "Multi-Session Dialog Benchmark Evaluation Pipeline"
echo "=================================================="

# Check if API key is set
if [ -z "$OPENAI_API_KEY" ]; then
    echo "Please set your OpenAI API key:"
    echo "export OPENAI_API_KEY='your-api-key-here'"
    exit 1
fi

# Default parameters
BENCHMARK_FILE="data_generation/dialog_benchmarks/benchmark_negative_traits.json"
MODEL_TO_EVALUATE="${MODEL_PATH:-meta-llama/Meta-Llama-3-8B-Instruct}"
JUDGE_MODEL="gpt-4.1-mini"
OUTPUT_DIR="./dialog_eval_results"
TEMPERATURE=0.7
# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --benchmark_file)
            BENCHMARK_FILE="$2"
            shift 2
            ;;
        --model)
            MODEL_TO_EVALUATE="$2"
            shift 2
            ;;
        --judge_model)
            JUDGE_MODEL="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --benchmark_file PATH    Path to benchmark file (default: ../data_generation/dialog_benchmarks/multi_session_benchmark.json)"
            echo "  --model MODEL           Model to evaluate (default: gpt-3.5-turbo)"
            echo "  --judge_model MODEL     Judge model for evaluation (default: gpt-4)"
            echo "  --output_dir DIR        Output directory for results (default: ./dialog_eval_results)"
            echo "  --temperature TEMP      Temperature for response generation (default: 0.7)"
            echo "  -h, --help             Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Configuration:"
echo "  Benchmark file: $BENCHMARK_FILE"
echo "  Model to evaluate: $MODEL_TO_EVALUATE"
echo "  Judge model: $JUDGE_MODEL"
echo "  Output directory: $OUTPUT_DIR"
echo "  Temperature: $TEMPERATURE"
echo ""

# Step 1: Create evaluation format from benchmark if needed
EVAL_FORMAT_FILE="${BENCHMARK_FILE%.*}_evaluation.json"
if [ ! -f "$EVAL_FORMAT_FILE" ]; then
    echo "Step 1: Creating evaluation format..."
    python eval/eval_dialog_benchmark.py \
        --judge_api_key "$OPENAI_API_KEY" \
        --benchmark_file "$BENCHMARK_FILE" \
        --responses_file "dummy" \
        --output_file "dummy" \
        --create_eval_format
    
    if [ $? -ne 0 ]; then
        echo "? Failed to create evaluation format"
        exit 1
    fi
    echo "? Evaluation format created: $EVAL_FORMAT_FILE"
else
    echo "Step 1: Using existing evaluation format: $EVAL_FORMAT_FILE"
fi

# Step 2: Generate model responses
MODEL_SAFE_NAME=$(basename "$MODEL_TO_EVALUATE" | sed 's/[^a-zA-Z0-9]/_/g')
MODEL_SAFE_NAME="${MODEL_SAFE_NAME}_nosteer_32k_neg"
RESPONSES_FILE="$OUTPUT_DIR/${MODEL_SAFE_NAME}_responses.json"

echo ""
echo "Step 2: Generating model responses..."
python eval/generate_dialog_responses.py \
    --model_path "$MODEL_TO_EVALUATE" \
    --benchmark_file "$EVAL_FORMAT_FILE" \
    --output_file "$RESPONSES_FILE" \
    --temperature "$TEMPERATURE"
    # --dynamic_steering \
    # --steering_layer 24 \
    # --steering_type response


if [ $? -ne 0 ]; then
    echo "? Failed to generate model responses"
    exit 1
fi
echo "? Model responses generated: $RESPONSES_FILE"

# Step 3: Evaluate responses
# EVAL_RESULTS_FILE="$OUTPUT_DIR/${MODEL_SAFE_NAME}_new2_evaluation_results.json"

# echo ""
# echo "Step 3: Evaluating model responses with judge..."
# python eval/eval_dialog_benchmark.py \
#     --judge_api_key "$OPENAI_API_KEY" \
#     --judge_model "$JUDGE_MODEL" \
#     --benchmark_file "$EVAL_FORMAT_FILE" \
#     --responses_file "$RESPONSES_FILE" \
#     --output_file "$EVAL_RESULTS_FILE"

# if [ $? -ne 0 ]; then
#     echo "? Failed to evaluate responses"
#     exit 1
# fi

# echo ""
# echo "? Evaluation pipeline completed successfully!"
# echo ""
# echo "Results:"
# echo "  Model responses: $RESPONSES_FILE"
# echo "  Evaluation results: $EVAL_RESULTS_FILE"
# echo ""
# echo "You can now analyze the results or compare different models."
