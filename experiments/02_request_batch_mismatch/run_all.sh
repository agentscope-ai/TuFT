#!/bin/bash
# Run all 3 microbenchmark tests sequentially.
# Usage: ./run_all.sh [MODEL_PATH]

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_PYTHON="${SCRIPT_DIR}/../.venv/bin/python"
MODEL="${1:-/mnt/cpfs/shared/qwen/Qwen3-4B}"
OUTPUT_DIR="${SCRIPT_DIR}/results"

echo "============================================"
echo "  Running all microbenchmark tests"
echo "  Model: ${MODEL}"
echo "  Output: ${OUTPUT_DIR}"
echo "============================================"
echo ""

echo "[1/3] Test: Adapter Independence"
${VENV_PYTHON} "${SCRIPT_DIR}/test_adapter_independence.py" --model "${MODEL}" --output-dir "${OUTPUT_DIR}"
echo ""

echo "[2/3] Test: Batch Size and Position Effect"
${VENV_PYTHON} "${SCRIPT_DIR}/test_batch_position.py" --model "${MODEL}" --output-dir "${OUTPUT_DIR}"
echo ""

echo "[3/3] Test: Framework Mismatch (vLLM vs PyTorch/peft)"
${VENV_PYTHON} "${SCRIPT_DIR}/test_framework_mismatch.py" --model "${MODEL}" --output-dir "${OUTPUT_DIR}"
echo ""

echo "============================================"
echo "  All tests complete!"
echo "  Results in: ${OUTPUT_DIR}"
echo "============================================"
