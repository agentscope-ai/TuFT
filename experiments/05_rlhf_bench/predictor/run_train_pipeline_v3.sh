#!/bin/bash
# 采集完成后执行：
#   1. 构建 v3 数据集（v2 + agent 任务）
#   2. 训练 transformer（per-token，最优方案）
#   3. 离线对比评估（在 v3 数据上跑 8 方案对比，含 agent holdout）
#
# 在 collect-agent tmux session 结束后，运行：
#   tmux new -s train-v3
#   bash /mnt/nas/hanzhang.yhz/evaluation/predictor/run_train_pipeline_v3.sh

set -e
PREDICTOR_DIR="/mnt/nas/hanzhang.yhz/evaluation/predictor"
PYTHON="/mnt/nas/hanzhang.yhz/TuFT/.venv/bin/python"
DATA_DIR="$PREDICTOR_DIR/data"

cd "$PREDICTOR_DIR"

echo "=========================================="
echo "[1/3] Building v3 multi-task dataset"
echo "      (v2 + hotpotqa/triviaqa agent tasks)"
echo "=========================================="
$PYTHON build_multitask_v3.py

echo ""
echo "=========================================="
echo "[2/3] Training Transformer (per-token) on v3"
echo "=========================================="
$PYTHON train.py \
    --data "$DATA_DIR/v3_train.jsonl" \
    --output "checkpoints/multitask_transformer_v3" \
    --model transformer \
    --epochs 20 \
    --batch_size 16 \
    --lr 3e-4 \
    --d_model 128 \
    --n_heads 4 \
    --n_layers 2 \
    --token_emb_dim 32 \
    --split_mode weight_version \
    --seed 42

echo ""
echo "=========================================="
echo "[3/3] Offline correction comparison (v3)"
echo "=========================================="
mkdir -p results
$PYTHON correction_comparison_v2.py \
    --data_train   "$DATA_DIR/v3_train.jsonl" \
    --data_holdout "$DATA_DIR/v3_holdout.jsonl" \
    --transformer_ckpt "checkpoints/multitask_transformer_v3/best.pt" \
    --scalar_ckpt      "checkpoints/multitask_scalar_v1/best.pt" \
    --mlp_ckpt         "checkpoints/multitask_mlp_v1/best.pt" \
    --mlp_scalar_ckpt  "checkpoints/multitask_mlp_scalar_v1/best.pt" \
    2>&1 | tee results/multitask_comparison_v3.txt

echo ""
echo "All done! Results -> results/multitask_comparison_v3.txt"
