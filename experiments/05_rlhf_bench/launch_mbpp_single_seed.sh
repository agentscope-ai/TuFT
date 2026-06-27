#!/usr/bin/env bash
# MBPP GRPO single-seed benchmark with baseline / predictor / global_mean
set -euo pipefail

cd "$(dirname "$(realpath "$0")")"

TASK=mbpp
NUM_STEPS=60
EVAL_INTERVAL=10
EVAL_N=50
GRPO_G=8
TEMP=0.9
LORA_RANK=8
MAX_TOKENS=1024
PREDICTOR_CKPT=/mnt/nas/hanzhang.yhz/evaluation/predictor/checkpoints/bench_v1/best.pt
PREDICTOR_DEVICE=cuda
GLOBAL_MEAN_BIAS=0.00360880373475033

PYTHON=/mnt/nas/hanzhang.yhz/evaluation/.venv/bin/python
SEED=42
OUTDIR="./results/bench_mbpp_seed${SEED}"
mkdir -p "${OUTDIR}"

${PYTHON} ./rlhf_bench.py \
    --groups grpo_baseline grpo_predictor grpo_global_mean \
    --task ${TASK} \
    --num_steps ${NUM_STEPS} \
    --eval_interval ${EVAL_INTERVAL} \
    --eval_n ${EVAL_N} \
    --grpo_g ${GRPO_G} \
    --temperature ${TEMP} \
    --lora_rank ${LORA_RANK} \
    --max_tokens ${MAX_TOKENS} \
    --predictor_ckpt ${PREDICTOR_CKPT} \
    --predictor_device ${PREDICTOR_DEVICE} \
    --global_mean_bias ${GLOBAL_MEAN_BIAS} \
    --output ${OUTDIR} \
    --seed ${SEED} \
    2>&1 | tee "${OUTDIR}/run.log"
