#!/bin/bash
# 依次用不同 tuft config 启动并运行 config_short.yaml
# 使用方式: bash run_all_configs.sh

TUFT_BIN="/mnt/nas/hanzhang.yhz/TuFT/.venv/bin/tuft"
TUFT_PORT=10610
CONFIG_DIR="/mnt/nas/hanzhang.yhz/data/config/subset/"
SIM_DIR="/mnt/nas/hanzhang.yhz/evaluation/simulator"
SIM_PYTHON="/mnt/nas/hanzhang.yhz/evaluation/.venv/bin/python"
SIM_CONFIG="${SIM_DIR}/configs/config_short.yaml"
RESULT_DIR="${SIM_DIR}/results/scheduling_comparison_high_speed_concurrent_requests"

# 自动发现 CONFIG_DIR 中所有 yaml 文件（去掉 .yaml 后缀作为 CONFIG_NAME）
mapfile -t CONFIGS < <(find "${CONFIG_DIR}" -maxdepth 1 -name '*.yaml' | sort | xargs -I{} basename {} .yaml)

mkdir -p "${RESULT_DIR}"

for CONFIG_NAME in "${CONFIGS[@]}"; do
    TUFT_CONFIG="${CONFIG_DIR}/${CONFIG_NAME}.yaml"
    OUTPUT_JSON="${RESULT_DIR}/${CONFIG_NAME}.json"
    # config_short.yaml 中 logprob_collection.output_path 未设置，
    # 会自动派生为 <output_path>.logprobs.jsonl，即 ${CONFIG_NAME}.json.logprobs.jsonl

    echo ""
    echo "=========================================="
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 处理 config: ${CONFIG_NAME}"
    echo "=========================================="

    # 停止已有 tuft 进程
    if pgrep -f "tuft launch" > /dev/null 2>&1; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] 停止旧 tuft 进程..."
        pkill -f "tuft launch" || true
        sleep 10
    fi

    # 启动新 tuft
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 启动 tuft，使用 ${CONFIG_NAME}.yaml ..."
    nohup "${TUFT_BIN}" launch --port "${TUFT_PORT}" --config "${TUFT_CONFIG}" \
        > "/tmp/tuft_${CONFIG_NAME}.log" 2>&1 &
    TUFT_PID=$!
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] tuft PID: ${TUFT_PID}"

    # 等待 tuft 完全启动（通过健康检查接口轮询）
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 等待 tuft 启动就绪..."
    MAX_WAIT=300
    WAITED=0
    until curl -sf "http://localhost:${TUFT_PORT}/api/v1/healthz" > /dev/null 2>&1; do
        sleep 5
        WAITED=$((WAITED + 5))
        if [ "${WAITED}" -ge "${MAX_WAIT}" ]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: tuft 启动超时（${MAX_WAIT}s），退出"
            exit 1
        fi
    done
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] tuft 已就绪（等待了 ${WAITED}s）"

    # 运行 run.py，--output 覆盖 output_path
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 开始运行 run.py (output: ${OUTPUT_JSON})..."
    cd "${SIM_DIR}"
    "${SIM_PYTHON}" run.py \
        --config "${SIM_CONFIG}" \
        --output "${OUTPUT_JSON}"
    RUN_EXIT=$?
    if [ "${RUN_EXIT}" -ne 0 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: run.py 退出码 ${RUN_EXIT}，停止"
        exit "${RUN_EXIT}"
    fi
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] run.py 完成，结果保存至 ${OUTPUT_JSON}"

    # 将 run.py 在工作目录下生成的同名文件移动到 RESULT_DIR（如有残留）
    # run.py --output 已直接写到 RESULT_DIR，此处做保险性清理
    # 同时处理 .logprobs.jsonl 附属文件（若生成在工作目录）
    LOGPROBS_SRC="${SIM_DIR}/${CONFIG_NAME}.json.logprobs.jsonl"
    LOGPROBS_DST="${RESULT_DIR}/${CONFIG_NAME}.json.logprobs.jsonl"
    if [ -f "${LOGPROBS_SRC}" ]; then
        mv "${LOGPROBS_SRC}" "${LOGPROBS_DST}"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] logprobs 文件已移至 ${LOGPROBS_DST}"
    fi
done

echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 全部 config 运行完毕！"
