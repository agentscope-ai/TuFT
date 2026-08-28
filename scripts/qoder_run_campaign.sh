#!/usr/bin/env bash
set -uo pipefail

ROOT="/mnt/nas/hanzhang.yhz/flex_backend/TuFT"
RESULTS="$ROOT/exp/results/qoder_campaign"
WORKLOAD="$ROOT/exp/workloads_noeval/workload_paper_t16.yaml"
LOCK="/tmp/qoder_tuft_campaign.lock"
PY="$ROOT/.venv/bin/python"
SIM_PY="/mnt/nas/hanzhang.yhz/evaluation/simulator/.venv/bin/python"

cd "$ROOT" || exit 1
mkdir -p "$RESULTS"
exec 9>"$LOCK"
if ! flock -n 9; then
    echo "[campaign] another locked runner is active"
    exit 0
fi

echo "[campaign] BEGIN $(date '+%F %T')"

if ! "$PY" scripts/qoder_check_design_points.py "$RESULTS/p0_debug"; then
    echo "[campaign] P0 GATE FAILED; stopping before long experiments"
    exit 2
fi

is_complete() {
    "$PY" - "$1" <<'PY'
import json
import sys
from pathlib import Path
path = Path(sys.argv[1])
if not path.exists():
    raise SystemExit(1)
data = json.loads(path.read_text())
raise SystemExit(0 if data.get("all_tenants_completed") else 1)
PY
}

run_arm() {
    local phase="$1"
    local mode="$2"
    local gpus="$3"
    local result="$RESULTS/$phase/$mode/simulator_results.json"
    if is_complete "$result"; then
        echo "[campaign] SKIP $(date '+%F %T') $phase/$mode complete"
        return 0
    fi
    echo "[campaign] START $(date '+%F %T') $phase/$mode workload=$(basename "$WORKLOAD") gpus=$gpus"
    "$PY" scripts/run_eval_matrix.py \
        --modes "$mode" \
        --workload "$WORKLOAD" \
        --results-dir "$RESULTS/$phase" \
        --gpu-subset "$gpus" \
        --port 10651 \
        --sim-timeout-s 7200
    local rc=$?
    if is_complete "$result"; then
        echo "[campaign] DONE $(date '+%F %T') $phase/$mode rc=$rc"
    else
        echo "[campaign] INCOMPLETE $(date '+%F %T') $phase/$mode rc=$rc"
    fi
    "$PY" scripts/qoder_build_paper_tables.py "$RESULTS"
    sleep 15
    return 0
}

for mode in serial_async_8gpu unified_engine_8gpu colocate_2copies_8gpu static_disagg_8gpu optimal_8gpu; do
    run_arm "same_workload_t16_8gpu" "$mode" "0,1,2,3,4,5,6,7"
done

for mode in serial_async_4gpu unified_engine_4gpu colocate_2copies_4gpu static_disagg_4gpu optimal_4gpu; do
    run_arm "same_workload_t16_4gpu" "$mode" "0,1,2,3"
done

"$PY" scripts/qoder_build_paper_tables.py "$RESULTS"
echo "[campaign] ALL DONE $(date '+%F %T')"
