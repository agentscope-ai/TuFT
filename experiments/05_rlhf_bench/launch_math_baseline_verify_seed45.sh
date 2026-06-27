#!/usr/bin/env bash
# MATH baseline verification (seed=45) - re-run grpo_baseline to verify results
set -euo pipefail

cd "$(dirname "$(realpath "$0")")"

TASK=math
NUM_STEPS=60
EVAL_INTERVAL=10
EVAL_N=50
GRPO_G=8
TEMP=0.7
LORA_RANK=8
MAX_TOKENS=1024
SEED=45
OUTDIR="./results/bench_math_baseline_verify_seed${SEED}"

PYTHON=/mnt/nas/hanzhang.yhz/evaluation/.venv/bin/python

mkdir -p "${OUTDIR}"
echo "===== MATH baseline verification: seed=${SEED} -> ${OUTDIR} ====="

${PYTHON} ./rlhf_bench.py \
    --groups grpo_baseline \
    --task ${TASK} \
    --num_steps ${NUM_STEPS} \
    --eval_interval ${EVAL_INTERVAL} \
    --eval_n ${EVAL_N} \
    --grpo_g ${GRPO_G} \
    --temperature ${TEMP} \
    --lora_rank ${LORA_RANK} \
    --max_tokens ${MAX_TOKENS} \
    --output ${OUTDIR} \
    --seed ${SEED} \
    2>&1 | tee "${OUTDIR}/run.log"

echo "Baseline verification done."
