#!/usr/bin/env bash
# 串行队列: 等 seed43 完成后自动跑 seed44
set -euo pipefail

SCRIPT_DIR="$(dirname "$(realpath "$0")")"

echo "[queue] Waiting for seed43 (PID check via result dir)..."
while [ ! -f "${SCRIPT_DIR}/results/bench_math_newpred_seed43/run.log" ] || \
      ! grep -q "seed43 complete" /tmp/bench_math_complete_seed43.log 2>/dev/null; do
    sleep 30
done
echo "[queue] seed43 done. Starting seed44..."

bash "${SCRIPT_DIR}/launch_math_complete_seed44.sh" \
    > /tmp/bench_math_complete_seed44.log 2>&1

echo "[queue] seed44 done. All complete."
