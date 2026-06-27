"""
构建多任务 predictor 训练数据集 v2

数据来源：simulator/logprobs/multitask_logprobs.jsonl
         （用当前 TuFT 实例采集，非 agent 标准任务，6个任务×2 rank×80步）

非 agent 标准任务（全部）：
  训练任务（5个）: gsm8k, math, countdown, humaneval, mbpp
  Hold-out 任务（1个，仅用于跨任务泛化验证）: ifeval

注意：hotpotqa/triviaqa 是 agent 任务，不在本数据集中。

输出：
  data/v2_train.jsonl    - 训练用（5个 train_tasks，共 10 tenant × 80 步）
  data/v2_holdout.jsonl  - hold-out 任务数据（ifeval，2 tenant × 80 步）
  data/v2_all.jsonl      - 全部（方便统一评估）

每条记录保留原始字段，额外加 source 字段标记 tenant_id。
"""

import json
import os
from collections import defaultdict


SRC = "/mnt/nas/hanzhang.yhz/evaluation/simulator/logprobs/multitask_logprobs.jsonl"
OUT_DIR = "/mnt/nas/hanzhang.yhz/evaluation/predictor/data"
os.makedirs(OUT_DIR, exist_ok=True)

TRAIN_TASKS = {"gsm8k", "math", "countdown", "humaneval", "mbpp"}
HOLDOUT_TASKS = {"ifeval"}
ALL_TASKS = TRAIN_TASKS | HOLDOUT_TASKS

train_records = []
holdout_records = []
stats = defaultdict(lambda: defaultdict(int))  # task -> tenant_id -> count
skipped = defaultdict(int)

print(f"Reading {SRC} ...")
with open(SRC) as f:
    for i, line in enumerate(f):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            skipped["json_error"] += 1
            continue

        task = r.get("task", "")
        if task not in ALL_TASKS:
            skipped[f"task:{task}"] += 1
            continue

        # 必须有 per_token logprob 字段
        if "sampling_logprobs" not in r or "training_logprobs" not in r:
            skipped["missing_logprobs"] += 1
            continue

        tenant = r.get("tenant_id", "unknown")
        stats[task][tenant] += 1

        if task in TRAIN_TASKS:
            train_records.append(r)
        else:
            holdout_records.append(r)


# 写出
def write_jsonl(records, path):
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"  Wrote {len(records):>6} records -> {path}")


write_jsonl(train_records, f"{OUT_DIR}/v2_train.jsonl")
write_jsonl(holdout_records, f"{OUT_DIR}/v2_holdout.jsonl")
write_jsonl(train_records + holdout_records, f"{OUT_DIR}/v2_all.jsonl")

# 打印统计
print(f"\n{'Task':<14} {'Tenant':<20} {'count':>6}")
print("-" * 44)
for task in sorted(stats.keys()):
    tag = "[holdout]" if task in HOLDOUT_TASKS else "[train]  "
    for tenant in sorted(stats[task].keys()):
        print(f"  {task + ' ' + tag:<30} {tenant:<20} {stats[task][tenant]:>6}")

print(f"\nTotal train   : {len(train_records)}")
print(f"Total holdout : {len(holdout_records)}")
print(f"Total all     : {len(train_records) + len(holdout_records)}")

if skipped:
    print(f"\nSkipped: {dict(skipped)}")
