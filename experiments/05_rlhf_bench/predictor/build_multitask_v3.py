"""
构建多任务 predictor 训练数据集 v3

在 v2 基础上新增 agent 任务（hotpotqa, triviaqa）作为训练任务，
ifeval 继续作为 holdout。

数据来源：
  v2: simulator/logprobs/multitask_logprobs.jsonl
       (gsm8k/math/countdown/humaneval/mbpp/ifeval)
  新增: simulator/logprobs/agent_logprobs.jsonl
       (hotpotqa/triviaqa)

Split:
  训练任务（7个）: gsm8k, math, countdown, humaneval, mbpp, hotpotqa, triviaqa
  Hold-out（1个）: ifeval

输出：
  data/v3_train.jsonl    - 7个 train tasks
  data/v3_holdout.jsonl  - ifeval holdout（复用 v2_holdout.jsonl）
  data/v3_all.jsonl      - 全部
"""

import json
import os
from collections import defaultdict


V2_SRC = "/mnt/nas/hanzhang.yhz/evaluation/simulator/logprobs/multitask_logprobs.jsonl"
AGENT_SRC = "/mnt/nas/hanzhang.yhz/evaluation/simulator/logprobs/agent_logprobs.jsonl"
OUT_DIR = "/mnt/nas/hanzhang.yhz/evaluation/predictor/data"
os.makedirs(OUT_DIR, exist_ok=True)

TRAIN_TASKS = {"gsm8k", "math", "countdown", "humaneval", "mbpp", "hotpotqa", "triviaqa"}
HOLDOUT_TASKS = {"ifeval"}
ALL_TASKS = TRAIN_TASKS | HOLDOUT_TASKS

train_records = []
holdout_records = []
stats = defaultdict(lambda: defaultdict(int))
skipped = defaultdict(int)


def load_source(path, label):
    count = 0
    with open(path) as f:
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

            if "sampling_logprobs" not in r or "training_logprobs" not in r:
                skipped["missing_logprobs"] += 1
                continue

            tenant = r.get("tenant_id", "unknown")
            stats[task][tenant] += 1
            count += 1

            if task in TRAIN_TASKS:
                train_records.append(r)
            else:
                holdout_records.append(r)
    print(f"  [{label}] loaded {count} records")


print(f"Reading v2 source: {V2_SRC}")
load_source(V2_SRC, "v2")

print(f"Reading agent source: {AGENT_SRC}")
load_source(AGENT_SRC, "agent")


def write_jsonl(records, path):
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"  Wrote {len(records):>6} records -> {path}")


write_jsonl(train_records, f"{OUT_DIR}/v3_train.jsonl")
write_jsonl(holdout_records, f"{OUT_DIR}/v3_holdout.jsonl")
write_jsonl(train_records + holdout_records, f"{OUT_DIR}/v3_all.jsonl")

print(f"\n{'Task':<18} {'Tenant':<24} {'count':>6}")
print("-" * 50)
for task in sorted(stats.keys()):
    tag = "[holdout]" if task in HOLDOUT_TASKS else "[train]  "
    for tenant in sorted(stats[task].keys()):
        print(f"  {task + ' ' + tag:<38} {tenant:<24} {stats[task][tenant]:>6}")

print(f"\nTotal train   : {len(train_records)}")
print(f"Total holdout : {len(holdout_records)}")
print(f"Total all     : {len(train_records) + len(holdout_records)}")

if skipped:
    print(f"\nSkipped: {dict(skipped)}")
