"""
构建多任务 predictor 训练数据集

数据来源：tp2_fsdp1_id0_async_quant.jsonl（最全，17899条，staleness 0-7，多任务）
         tp2_fsdp1_id0_quant.jsonl（staleness 0-1，多任务）

非 agent 标准数据集（排除 math_agent / apibank / toolbench）：
  gsm8k, math, countdown, humaneval, mbpp, hotpotqa, triviaqa, ifeval

数据隔离：
  训练集任务（5个）: gsm8k, math, countdown, humaneval, mbpp
  Hold-out 任务（3个，仅用于跨数据集泛化验证）: hotpotqa, triviaqa, ifeval

输出：
  data/multi_task_train.jsonl   - 训练用（train_tasks）
  data/multi_task_holdout.jsonl - hold-out 任务数据（不参与训练）
  data/multi_task_all.jsonl     - 全部非agent标准数据（含holdout，方便统一推理）

每条记录保留原始字段，额外加入 source_file 字段用于溯源。
"""

import json
import os
from collections import defaultdict


BASE = "/mnt/nas/hanzhang.yhz/evaluation/simulator/logprobs"
OUT_DIR = "/mnt/nas/hanzhang.yhz/evaluation/predictor/data"
os.makedirs(OUT_DIR, exist_ok=True)

# 非 agent 标准任务
TRAIN_TASKS = {"gsm8k", "math", "countdown", "humaneval", "mbpp"}
HOLDOUT_TASKS = {"hotpotqa", "triviaqa", "ifeval"}
ALL_STANDARD = TRAIN_TASKS | HOLDOUT_TASKS

# 数据源（按优先级，async_quant 最全）
SOURCES = [
    (f"{BASE}/tp2_fsdp1_id0_async_quant.jsonl", "async_quant"),
    (f"{BASE}/tp2_fsdp1_id0_quant.jsonl", "quant"),
]

# 用 (tenant_id, step, item_idx, staleness) 去重，避免跨文件重复
seen = set()
train_records = []
holdout_records = []
stats = defaultdict(lambda: defaultdict(int))  # task -> staleness -> count

for filepath, src_name in SOURCES:
    print(f"Reading {src_name} ...")
    with open(filepath) as f:
        for line in f:
            r = json.loads(line)
            task = r.get("task", "")
            if task not in ALL_STANDARD:
                continue
            key = (
                r.get("tenant_id"),
                r.get("step"),
                r.get("item_idx"),
                r.get("staleness"),
                r.get("sample_weight_version", ""),
            )
            if key in seen:
                continue
            seen.add(key)
            r["source_file"] = src_name
            sl = r.get("staleness", 0)
            stats[task][sl] += 1
            if task in TRAIN_TASKS:
                train_records.append(r)
            else:
                holdout_records.append(r)


# 写出
def write_jsonl(records, path):
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"  Wrote {len(records)} records -> {path}")


write_jsonl(train_records, f"{OUT_DIR}/multi_task_train.jsonl")
write_jsonl(holdout_records, f"{OUT_DIR}/multi_task_holdout.jsonl")
write_jsonl(train_records + holdout_records, f"{OUT_DIR}/multi_task_all.jsonl")

# 打印统计
print(f"\n{'Task':<14} {'staleness':>10}  {'count':>6}")
print("-" * 35)
for task in sorted(stats.keys()):
    is_holdout = task in HOLDOUT_TASKS
    tag = " [holdout]" if is_holdout else " [train]  "
    for sl in sorted(stats[task].keys()):
        print(f"{task + tag:<24} {sl:>4.0f}     {stats[task][sl]:>6}")
print(f"\nTotal train:   {len(train_records)}")
print(f"Total holdout: {len(holdout_records)}")
print(f"Total all:     {len(train_records) + len(holdout_records)}")
