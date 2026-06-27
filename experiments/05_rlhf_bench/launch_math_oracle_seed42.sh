#!/usr/bin/env bash
# MATH GRPO oracle experiment (seed=42)
# Oracle: corrected_lps = training_lps, IS ratio=1 everywhere, 完全无偏上限
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
SEED=42
OUTDIR="./results/bench_math_oracle_seed${SEED}"

PYTHON=/mnt/nas/hanzhang.yhz/evaluation/.venv/bin/python

mkdir -p "${OUTDIR}"
echo "===== Oracle experiment: seed=${SEED} -> ${OUTDIR} ====="

${PYTHON} ./rlhf_bench.py \
    --groups grpo_oracle \
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

echo "Oracle experiment done."
