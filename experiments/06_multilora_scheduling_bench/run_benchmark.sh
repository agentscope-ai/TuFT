#!/usr/bin/env bash
# ============================================================================
# One-click benchmark runner
# ============================================================================
# This script runs the full benchmark pipeline:
#   1. Creates mock LoRA adapters (if not exist)
#   2. Starts vLLM server with multi-LoRA support
#   3. Waits for server readiness
#   4. Runs the benchmark
#   5. Kills the server
#   6. Analyzes results
#
# Usage:
#   bash run_benchmark.sh [--tp-size 1] [--port 8000] [--max-tokens 256]
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Defaults
TP_SIZE=1
PORT=8000
MAX_TOKENS=256
NUM_ADAPTERS=5
SAMPLES_PER_ADAPTER=50
CONCURRENCY=64

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --tp-size) TP_SIZE="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --max-tokens) MAX_TOKENS="$2"; shift 2 ;;
        --num-adapters) NUM_ADAPTERS="$2"; shift 2 ;;
        --samples-per-adapter) SAMPLES_PER_ADAPTER="$2"; shift 2 ;;
        --concurrency) CONCURRENCY="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

SERVER_URL="http://localhost:${PORT}"
SERVER_PID=""

cleanup() {
    echo ""
    echo "Cleaning up..."
    if [ -n "${SERVER_PID}" ] && kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "Stopping vLLM server (PID ${SERVER_PID})..."
        kill "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
    echo "Done."
}
trap cleanup EXIT

echo "============================================================"
echo " Multi-LoRA Benchmark Pipeline"
echo "============================================================"
echo " TP size:             ${TP_SIZE}"
echo " Port:                ${PORT}"
echo " Max tokens:          ${MAX_TOKENS}"
echo " Adapters:            ${NUM_ADAPTERS}"
echo " Samples/adapter:     ${SAMPLES_PER_ADAPTER}"
echo " Concurrency:         ${CONCURRENCY}"
echo "============================================================"
echo ""

# Step 1: Create mock LoRA adapters
if [ ! -d "mock_loras/adapter_0" ]; then
    echo "Step 1: Creating mock LoRA adapters..."
    python create_mock_loras.py
else
    echo "Step 1: Mock LoRA adapters already exist. Skipping."
fi
echo ""

# Step 2: Start vLLM server
echo "Step 2: Starting vLLM server..."
bash start_vllm_server.sh "${PORT}" "${TP_SIZE}" > server.log 2>&1 &
SERVER_PID=$!
echo "  Server PID: ${SERVER_PID}"
echo "  Log: ${SCRIPT_DIR}/server.log"

# Step 3: Wait for server readiness
echo ""
echo "Step 3: Waiting for server to be ready..."
MAX_WAIT=300  # 5 minutes
ELAPSED=0
while [ ${ELAPSED} -lt ${MAX_WAIT} ]; do
    if curl -s "${SERVER_URL}/v1/models" > /dev/null 2>&1; then
        echo "  Server is ready! (took ${ELAPSED}s)"
        break
    fi
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "  ERROR: Server process died. Check server.log"
        tail -20 server.log
        exit 1
    fi
    sleep 5
    ELAPSED=$((ELAPSED + 5))
    echo "  Waiting... (${ELAPSED}s / ${MAX_WAIT}s)"
done

if [ ${ELAPSED} -ge ${MAX_WAIT} ]; then
    echo "  ERROR: Server did not start within ${MAX_WAIT}s"
    tail -20 server.log
    exit 1
fi
echo ""

# Step 4: Run benchmark
echo "Step 4: Running benchmark..."
python benchmark_api.py \
    --server-url "${SERVER_URL}" \
    --num-adapters "${NUM_ADAPTERS}" \
    --samples-per-adapter "${SAMPLES_PER_ADAPTER}" \
    --max-tokens "${MAX_TOKENS}" \
    --concurrency "${CONCURRENCY}"
echo ""

# Step 5: Server cleanup happens in trap

# Step 6: Analyze
echo ""
echo "Step 6: Analyzing results..."
python analyze_results.py results/benchmark_api_results.json
echo ""
echo "Benchmark complete! Results in: ${SCRIPT_DIR}/results/"
