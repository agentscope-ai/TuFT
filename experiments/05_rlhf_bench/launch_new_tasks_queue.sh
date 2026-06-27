#!/usr/bin/env bash
# 等 gsm8k 跑完后，依次跑 ifeval / hotpotqa / triviaqa
set -euo pipefail

SCRIPT_DIR="$(dirname "$(realpath "$0")")"

echo "[queue] Waiting for bench-gsm8k tmux session to finish..."
while tmux has-session -t bench-gsm8k 2>/dev/null; do
    sleep 30
done
echo "[queue] bench-gsm8k done. Starting ifeval..."

bash "${SCRIPT_DIR}/launch_ifeval_single_seed.sh"
echo "[queue] ifeval done. Starting hotpotqa..."

bash "${SCRIPT_DIR}/launch_hotpotqa_single_seed.sh"
echo "[queue] hotpotqa done. Starting triviaqa..."

bash "${SCRIPT_DIR}/launch_triviaqa_single_seed.sh"
echo "[queue] All new tasks done."
