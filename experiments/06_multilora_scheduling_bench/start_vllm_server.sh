#!/usr/bin/env bash
# ============================================================================
# Start vLLM server with Multi-LoRA support
# ============================================================================
# This script starts a vLLM OpenAI-compatible API server with multi-LoRA
# support, loading 5 mock LoRA adapters for benchmarking.
#
# Prerequisites:
#   - CUDA 12+ driver environment
#   - vLLM installed: pip install vllm
#   - Mock LoRA adapters created: python create_mock_loras.py
#
# Usage:
#   bash start_vllm_server.sh [--port PORT] [--tp-size TP]
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Configuration
BASE_MODEL="/mnt/cpfs/luyi/models/Qwen3-8B"
LORA_DIR="${SCRIPT_DIR}/mock_loras"
PORT="${1:-8000}"
TP_SIZE="${2:-1}"
MAX_LORAS=6
MAX_LORA_RANK=64
MAX_MODEL_LEN=4096
GPU_MEM_UTIL=0.85

# Validate mock LoRA adapters exist
if [ ! -d "${LORA_DIR}/adapter_0" ]; then
    echo "ERROR: Mock LoRA adapters not found at ${LORA_DIR}."
    echo "Run 'python create_mock_loras.py' first."
    exit 1
fi

# Build --lora-modules argument
LORA_MODULES=""
for i in 0 1 2 3 4; do
    ADAPTER_PATH="${LORA_DIR}/adapter_${i}"
    if [ -d "${ADAPTER_PATH}" ]; then
        if [ -n "${LORA_MODULES}" ]; then
            LORA_MODULES="${LORA_MODULES} "
        fi
        LORA_MODULES="${LORA_MODULES}--lora-modules adapter_${i}=${ADAPTER_PATH}"
    fi
done

echo "============================================================"
echo " vLLM Multi-LoRA Server"
echo "============================================================"
echo " Base model:     ${BASE_MODEL}"
echo " LoRA adapters:  ${LORA_DIR}/adapter_{0..4}"
echo " Port:           ${PORT}"
echo " TP size:        ${TP_SIZE}"
echo " Max LoRAs:      ${MAX_LORAS}"
echo " Max LoRA rank:  ${MAX_LORA_RANK}"
echo "============================================================"
echo ""

# Start vLLM server
exec python -m vllm.entrypoints.openai.api_server \
    --model "${BASE_MODEL}" \
    --port "${PORT}" \
    --tensor-parallel-size "${TP_SIZE}" \
    --dtype bfloat16 \
    --max-model-len "${MAX_MODEL_LEN}" \
    --enable-lora \
    --max-loras "${MAX_LORAS}" \
    --max-lora-rank "${MAX_LORA_RANK}" \
    --gpu-memory-utilization "${GPU_MEM_UTIL}" \
    --trust-remote-code \
    ${LORA_MODULES}
