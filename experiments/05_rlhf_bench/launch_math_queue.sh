#!/usr/bin/env bash
# 等待 bench-queue (triviaqa) 结束后，依次运行 oracle 和 new_pred MATH 实验
set -euo pipefail

SCRIPT_DIR="$(dirname "$(realpath "$0")")"

echo "[queue-math] Waiting for bench-queue (triviaqa) to finish..."
while tmux has-session -t bench-queue 2>/dev/null; do
    sleep 30
done
echo "[queue-math] bench-queue finished. Starting MATH experiments."

bash "${SCRIPT_DIR}/launch_math_oracle_seed42.sh"
echo "[queue-math] Oracle experiment done. Starting new predictor experiment."

bash "${SCRIPT_DIR}/launch_math_new_pred_seed42.sh"
echo "[queue-math] All MATH experiments done."
