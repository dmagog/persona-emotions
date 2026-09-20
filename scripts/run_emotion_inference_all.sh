#!/usr/bin/env bash
# Local inference for all emotions (pos + neg), no API judge.
# Reads packs from data_generation/emotion_data_{VERSION}/.
#
# Usage:
#   MODEL=Qwen/Qwen2.5-3B-Instruct VERSION=extract ./scripts/run_emotion_inference_all.sh
#
# Outputs under eval_emotion/<model_basename>/:
#   anger_pos.csv, anger_neg.csv, ... , all_emotions_extract.csv

set -euo pipefail
cd "$(dirname "$0")/.."

VERSION="${VERSION:-extract}"          # extract | eval  → emotion_data_extract | emotion_data_eval

if [[ ! -f "data_generation/emotion_data_${VERSION}/anger.json" ]]; then
  echo "Missing data_generation/emotion_data_${VERSION}/anger.json."
  echo "The repository includes the fixed extraction and evaluation splits."
  exit 1
fi

MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct}"
INFER_BACKEND="${INFER_BACKEND:-hf}"   # hf | vllm
MODEL_BASENAME="$(basename "$MODEL")"
OUT_DIR="${OUT_DIR:-eval_emotion/${MODEL_BASENAME}}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python -m emotion.generate_pairs \
  --model "$MODEL" \
  --version "$VERSION" \
  --output_dir "$OUT_DIR" \
  --infer_backend "$INFER_BACKEND" \
  --temperature 0 \
  "$@"
