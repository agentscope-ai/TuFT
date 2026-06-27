#!/usr/bin/env bash
# MATH 完整实验 seed=44: oracle + new_predictor + new_global_mean
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
PREDICTOR_CKPT=/mnt/nas/hanzhang.yhz/evaluation/predictor/checkpoints/multitask_transformer_v1/best.pt
PREDICTOR_DEVICE=cuda
GLOBAL_MEAN_BIAS=0.008113
SEED=44
PYTHON=/mnt/nas/hanzhang.yhz/evaluation/.venv/bin/python

# --- 1. oracle ---
OUTDIR="./results/bench_math_oracle_seed${SEED}"
mkdir -p "${OUTDIR}"
echo "===== [seed${SEED}] oracle ====="
${PYTHON} ./rlhf_bench.py \
    --groups grpo_oracle \
    --task ${TASK} --num_steps ${NUM_STEPS} --eval_interval ${EVAL_INTERVAL} \
    --eval_n ${EVAL_N} --grpo_g ${GRPO_G} --temperature ${TEMP} \
    --lora_rank ${LORA_RANK} --max_tokens ${MAX_TOKENS} \
    --output ${OUTDIR} --seed ${SEED} \
    2>&1 | tee "${OUTDIR}/run.log"

# --- 2. new_predictor + new_global_mean ---
OUTDIR="./results/bench_math_newpred_seed${SEED}"
mkdir -p "${OUTDIR}"
echo "===== [seed${SEED}] new_predictor + new_global_mean ====="
${PYTHON} ./rlhf_bench.py \
    --groups grpo_predictor grpo_global_mean \
    --task ${TASK} --num_steps ${NUM_STEPS} --eval_interval ${EVAL_INTERVAL} \
    --eval_n ${EVAL_N} --grpo_g ${GRPO_G} --temperature ${TEMP} \
    --lora_rank ${LORA_RANK} --max_tokens ${MAX_TOKENS} \
    --predictor_ckpt ${PREDICTOR_CKPT} --predictor_device ${PREDICTOR_DEVICE} \
    --global_mean_bias ${GLOBAL_MEAN_BIAS} \
    --output ${OUTDIR} --seed ${SEED} \
    2>&1 | tee "${OUTDIR}/run.log"

echo "seed44 complete."
