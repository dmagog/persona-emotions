#!/bin/bash

# Script to check orthogonality of persona vectors
# Usage: ./check_orthogonality.sh [persona_vector_dir] [vector_type] [output_dir]

PERSONA_DIR=${1:-"persona_vectors/Qwen2.5-7B-Instruct/"}
VECTOR_TYPE=${2:-"response_avg"}  # Default to response_avg
OUTPUT_DIR=${3:-"orthogonality_analysis"}

echo "=========================================="
echo "Persona Vector Orthogonality Analysis"
echo "=========================================="
echo "Persona vectors directory: $PERSONA_DIR"
echo "Vector type: $VECTOR_TYPE"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Check if persona directory exists
if [ ! -d "$PERSONA_DIR" ]; then
    echo "Error: Persona vector directory not found: $PERSONA_DIR"
    exit 1
fi

# Count number of .pt files matching the pattern
pattern="*_${VECTOR_TYPE}_diff.pt"
num_vectors=$(find "$PERSONA_DIR" -name "$pattern" | wc -l)
echo "Found $num_vectors persona vector files matching pattern: $pattern"

if [ $num_vectors -eq 0 ]; then
    echo "Error: No .pt files found matching pattern: $pattern"
    echo "Available vector types in directory:"
    find "$PERSONA_DIR" -name "*.pt" | head -5 | sed 's/.*_\([^_]*_[^_]*\)_diff\.pt/\1/' | sort -u
    exit 1
fi

echo ""
echo "Running orthogonality analysis..."

# Run the analysis
python eval/check_orthogonality.py \
    --persona_dir "$PERSONA_DIR" \
    --vector_type "$VECTOR_TYPE" \
    --output_dir "$OUTPUT_DIR" \
    --save_plot \
    --threshold 0.1

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Analysis completed successfully!"
    echo "=========================================="
    echo "Results saved in: $OUTPUT_DIR"
    echo ""
    echo "Generated files:"
    ls -la "$OUTPUT_DIR"/ 2>/dev/null || echo "Output directory not found"
else
    echo ""
    echo "Error: Analysis failed"
    exit 1
fi
