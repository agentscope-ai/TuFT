#!/bin/bash
# 采集结束后依次执行：
#   1. 构建 v2 数据集
#   2. 训练 transformer（per-token）
#   3. 训练 scalar_transformer（sequence bias）
#   4. 离线对比评估
#
# 在 collect-multitask tmux session 结束后，在新 session 中运行：
#   tmux new -s train-predictor
#   bash /mnt/nas/hanzhang.yhz/evaluation/predictor/run_train_pipeline.sh

set -e
PREDICTOR_DIR="/mnt/nas/hanzhang.yhz/evaluation/predictor"
PYTHON="/mnt/nas/hanzhang.yhz/TuFT/.venv/bin/python"
DATA_DIR="$PREDICTOR_DIR/data"

cd "$PREDICTOR_DIR"

echo "=========================================="
echo "[1/4] Building v2 multi-task dataset"
echo "=========================================="
$PYTHON build_multitask_v2.py

echo ""
echo "=========================================="
echo "[2/4] Training Transformer (per-token)"
echo "=========================================="
$PYTHON train.py \
    --data "$DATA_DIR/v2_train.jsonl" \
    --output "checkpoints/multitask_transformer_v1" \
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
echo "[3/4] Training ScalarTransformer (seq bias)"
echo "=========================================="
$PYTHON train.py \
    --data "$DATA_DIR/v2_train.jsonl" \
    --output "checkpoints/multitask_scalar_v1" \
    --model scalar_transformer \
    --epochs 20 \
    --batch_size 16 \
    --lr 3e-4 \
    --d_model 128 \
    --n_heads 4 \
    --n_layers 2 \
    --token_emb_dim 32 \
    --split_mode weight_version \
    --lambda_bias 10.0 \
    --seed 42

echo ""
echo "=========================================="
echo "[4/4] Offline correction comparison"
echo "=========================================="
mkdir -p results
$PYTHON correction_comparison_v2.py \
    --data_train   "$DATA_DIR/v2_train.jsonl" \
    --data_holdout "$DATA_DIR/v2_holdout.jsonl" \
    --transformer_ckpt "checkpoints/multitask_transformer_v1/best.pt" \
    --scalar_ckpt      "checkpoints/multitask_scalar_v1/best.pt" \
    2>&1 | tee results/multitask_comparison.txt

echo ""
echo "All done!"
