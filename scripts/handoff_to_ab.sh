#!/usr/bin/env bash
# Handoff: let the in-flight static_disagg_t48 finish, then stop the extension
# campaign master so the GPUs are freed for the duty-cycle A/B (the active goal's
# required verification). All already-completed ext results are preserved.
set -u
cd /mnt/nas/hanzhang.yhz/flex_backend/TuFT

STATIC_JSON=exp/results/campaign10/static_disagg_t48/static_disagg/simulator_results.json

# Wait (up to 15 min) for the in-flight static_t48 to complete.
for i in $(seq 1 90); do
    if [ -f "$STATIC_JSON" ] && python3 -c "import json,sys; sys.exit(0 if json.load(open('$STATIC_JSON')).get('all_tenants_completed') else 1)" 2>/dev/null; then
        echo "[handoff] static_t48 complete $(date +%H:%M:%S)"
        break
    fi
    sleep 10
done

# Stop the ext master so it does not launch colocate/optimal t32/t48.
pkill -9 -f "run_campaign_ext[.]sh" 2>/dev/null
echo "[handoff] ext master stopped $(date +%H:%M:%S); GPUs will be freed for A/B"
