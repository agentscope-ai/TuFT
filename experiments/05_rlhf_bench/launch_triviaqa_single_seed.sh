#!/usr/bin/env bash
# TriviaQA GRPO single-seed benchmark (agent task)
set -euo pipefail

cd "$(dirname "$(realpath "$0")")"

TASK=triviaqa
NUM_STEPS=60
EVAL_INTERVAL=10
EVAL_N=30
GRPO_G=4
TEMP=0.7
LORA_RANK=8
MAX_TOKENS=256
PREDICTOR_CKPT=/mnt/nas/hanzhang.yhz/evaluation/predictor/checkpoints/multitask_transformer_v1/best.pt
PREDICTOR_DEVICE=cuda
GLOBAL_MEAN_BIAS=0.008113

PYTHON=/mnt/nas/hanzhang.yhz/evaluation/.venv/bin/python

SEED=42
OUTDIR="./results/bench_triviaqa_seed${SEED}"
mkdir -p "${OUTDIR}"
echo "===== TriviaQA Seed ${SEED} -> ${OUTDIR} ====="
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

echo "TriviaQA benchmark done."
