# Framework-Mismatch Predictor

> 一个小模型，专门预测 vLLM (sampling) 与 Tinker / PyTorch (training) 两侧 forward 在同一 token 上的
> logprob 偏差，用来在 RL 训练中**重建无偏的 importance sampling weight**。

---

## 目录

1. [问题与训练目标](#1-问题与训练目标)
2. [代码目录结构](#2-代码目录结构)
3. [数据采集（simulator 侧）](#3-数据采集simulator-侧)
4. [JSONL Schema](#4-jsonl-schema)
5. [数据预处理与过滤](#5-数据预处理与过滤)
6. [数据集划分（按 weight_version 切分）](#6-数据集划分按-weight_version-切分)
7. [输入特征](#7-输入特征)
8. [模型设计](#8-模型设计)
9. [Loss 设计](#9-loss-设计)
10. [评估指标解释](#10-评估指标解释)
11. [训练命令](#11-训练命令)
12. [绘图工具](#12-绘图工具)
13. [当前结果对比](#13-当前结果对比)
14. [目标档位](#14-目标档位)
15. [推理 / 接入 RL 训练循环](#15-推理--接入-rl-训练循环)

---

## 1. 问题与训练目标

### 1.1 framework mismatch 是什么

在线 RL 训练里，每个 token 有两个 logprob：

| 名字 | 由谁产生 | 用途 |
|---|---|---|
| `sampling_logprob` (`s_t`) | **vLLM** rollout 时记录 | 行为策略 logprob，做行为采样 |
| `training_logprob`  (`p_t`) | **Tinker / PyTorch** forward | 目标策略 logprob，loss 用 |

理论上 `s_t = p_t`（同一参数同一 token），但实际由于 **kernel 实现 / dtype / batch position / cuda graph** 等差异，会出现一个非零残差：

```
Δ_t  =  s_t − p_t       # ≠ 0
```

我们在 Qwen3-4B + GSM8K 上实测：

```
mean |Δ_t|     ≈ 0.04            # token 级 MAE
mean Δ_t       ≈ +0.013          # 系统性偏置（vLLM 略高）
exp(Σ_t Δ_t)   ≈ 10^14           # 序列级 IS weight 爆炸
```

token 级看似无害，**序列级是灾难** —— 长度 ~600 的 response 累积偏置 `Σ_t Δ_t ≈ 8`，
`IS_weight = exp(8) ≈ 3000`，PPO clip 直接全裁掉。

### 1.2 训练目标

学一个函数 `f_θ`，对每个 token 输出残差预测 `Δ̂_t`：

```
Δ̂_t  =  f_θ(token_t, position_t, prefix_features, ...)
```

**修正后的 sampling logprob**：

```
corrected_s_t  =  s_t − Δ̂_t
```

理想情况 `corrected_s_t ≈ p_t`，下游 IS / PPO clip 就重新无偏。

> ⚠️ 注意：predictor 学的是 **残差 Δ**，不是直接学 `p_t`。这有两个好处：
>
> 1. 信号集中：`Δ_t` 的均值 ≈ 0.013、方差 ≈ 0.001，比 `p_t`（均值 −2、方差几十）更易学。
> 2. **零初始化等价 baseline**：head 的 `Linear` 用 `nn.init.zeros_`，
>    训练前 `Δ̂_t = 0`，等价于"不做修正"，给训练一个安全的起点。

---

## 2. 代码目录结构

```
predictor/
├── data.py        # JSONL 加载、过滤、按 weight_version 切分、动态 padding collate
├── model.py       # MLPPredictor (Tier-0 baseline) + TransformerPredictor (Tier-1 主)
├── losses.py      # 双尺度 L1 loss + EvalSummary + evaluate()
├── train.py       # 训练主流程：load → build → train → eval → save best
├── plot.py        # 读 training_log.jsonl + test_summary.json，画训练曲线
├── README.md      # 本文档
└── checkpoints/   # 训练产物
    └── <run_name>/
        ├── best.pt              # state_dict + args + vocab_size + task_map + lora_map
        ├── training_log.jsonl   # 每 epoch 一行（loss + 全套 val 指标）
        ├── test_summary.json    # 最终 test 集 EvalSummary
        └── training_curves.png  # plot.py 输出
```

---

## 3. 数据采集（simulator 侧）

predictor 的训练数据来自 `simulator/`。关键约束：

### 3.1 必须 `sync_mode = true`

只有同步租户每个 batch 才有 `staleness == 0`，**采到的 mismatch 才是纯 framework
mismatch**（不混 weight staleness）。配置示例：

```yaml
# simulator/config.yaml
logprob_collection:
  enabled: true
  dump_per_token: true       # 必须 true，否则不写 sampling/training logprobs

tenants:
  - id: gsm8k-A
    task: gsm8k
    sync_mode: true          # ← 关键
    lora_rank: 8
    learning_rate: 1.0e-5
    temperature: 1.0
    max_tokens: 1024
```

### 3.2 跑采集

```bash
cd simulator && .venv/bin/python run.py
```

输出：`simulator/<run_dir>/results.logprobs.jsonl`，每行一条记录。

---

## 4. JSONL Schema

每条记录代表 **(tenant, step, response_idx) 三元组**。
完整字段（来自 `simulator/simulator/tenant.py` 的 dump 逻辑）：

```jsonc
{
  // ── tenant / task 静态特征 ───────────────────────────
  "tenant_id":            "gsm8k-A",
  "task":                 "gsm8k",         // 12 个任务之一
  "lora_rank":            8,               // {4, 8, 16}
  "learning_rate":        1.0e-5,
  "temperature":          1.0,
  "max_tokens":           1024,
  "sync_mode":            true,            // 必须 true 才进入训练集

  // ── 时间 / weight 版本 ──────────────────────────────
  "step":                 1,
  "item_idx":             0,
  "sample_weight_version": 0,              // ← 切分依据
  "train_weight_version":  1,
  "staleness":            0.0,             // = train_wv - 1 - sample_wv，必须 = 0

  // ── RL 信号（可选特征，目前未用）────────────────────
  "reward":               1.0,
  "advantage":            0.7,

  // ── token 序列特征 ──────────────────────────────────
  "n_prompt_tokens":      83,              // prefix 长度（已是数字，作为标量特征）
  "n_response_tokens":    127,
  "prompt_tokens":        [151644, ...],   // 当前未用作输入
  "response_tokens":      [29, 12345, ...], // ← 主输入，长度 = n_response_tokens

  // ── logprobs（核心） ────────────────────────────────
  "sampling_logprobs":    [-0.21, -1.04, ...],  // vLLM 端，长度 = n_response_tokens
  "training_logprobs":    [-0.18, -1.05, ...]   // training forward 端，长度 = n_response_tokens
}
```

**最小训练要求**：`response_tokens` + `sampling_logprobs` + `training_logprobs` + `staleness` + 其余少数标量。

---

## 5. 数据预处理与过滤

`data.py::filter_clean_records` 实现的硬性条件：

```python
def filter_clean_records(records):
    out = []
    for r in records:
        # 1. 只要 framework mismatch，不要 weight staleness
        if r.get("staleness", 0.0) != 0.0:
            continue
        # 2. 三个长度必须一致（防 vLLM 截断 / log 错位）
        n = r["n_response_tokens"]
        if len(r["response_tokens"]) != n:        continue
        if len(r["sampling_logprobs"]) != n:      continue
        if len(r["training_logprobs"]) != n:      continue
        if n == 0:                                continue
        out.append(r)
    return out
```

**Target 计算**：

```
Δ_t  =  sampling_logprobs[t]  −  training_logprobs[t]
```

predictor 直接 fit 这个 `Δ_t`。

---

## 6. 数据集划分（按 weight_version 切分）

> ❌ **不能用随机划分**：同一个 `sample_weight_version` 下的 token 既会出现在 train 又会
> 出现在 test，模型可能记住"这个版本下的偏置"，test 指标虚高。

✅ `data.py::split_by_weight_version` 的做法：

1. 先按 `(sample_weight_version, tenant_id, item_idx)` 排序（**时间顺序**）
2. 按 `70 / 15 / 15` 切片：
   - train: 前 70% 的记录（早期 weight version）
   - val:   中间 15%
   - test:  最后 15%（晚期 weight version）

这样 **test 集评估的是模型在「没见过的 weight 上」的泛化** —— 这恰好是 RL 推理时 predictor
要面对的场景（每个 step weight 都在动）。

```python
# 70115 条 → train 48656 / val 10426 / test 10427    （22_agents 数据集）
# 不同时间段的 weight_version 完全不重叠
```

---

## 7. 输入特征

每个 token 在前向时使用以下特征（来自 `data.py::LogprobMismatchDataset.__getitem__`）：

| 特征 | 类型 | 维度 | 说明 |
|---|---|---|---|
| `response_token_id` | long | scalar / token | 当前 token id（vocab_size = 152064 for Qwen3） |
| `position` | long | scalar / token | 在 response 中的位置 0..T−1 |
| `sampling_lp` | float | scalar / token | vLLM 的 logprob，自身作为强特征 |
| `n_prompt_tokens` | float | scalar / seq | prefix 长度（捕获"长 prompt 漂移更大"） |
| `temperature` | float | scalar / seq | 采样温度 |
| `task_idx` | long | scalar / seq | 12 个任务之一（categorical） |
| `lora_rank_idx` | long | scalar / seq | {4, 8, 16} 之一（categorical） |

**Target**: `Δ_t = sampling_lp − training_lp`（每个 token 一个标量）。

`CategoricalMap`（在 train 上 `fit`，val/test `apply`，未知值 → UNK = 0）：

```python
class CategoricalMap:
    UNK = 0
    def fit(self, values):  # 统计 train 上出现的所有值
        ...
    def apply(self, v):     # 已知 → idx，未知 → 0
        ...
```

---

## 8. 模型设计

### 8.1 共同基础组件

#### Sinusoidal 编码

`model.py::sinusoidal_encode(values, dim)` —— 通用 sin/cos 编码，
既给 token position，也给 scalar prefix length。
比 learnable position embedding 更省参数、对未见过的长度有外推性。

#### Head 零初始化

两个模型的最后一层 `nn.Linear` 都用 `nn.init.zeros_(...)`：

```python
# model.py
nn.init.zeros_(self.head.weight)
nn.init.zeros_(self.head.bias)
```

→ 训练前 `Δ̂_t = 0`，`corrected_s_t = s_t`，等价 baseline。损失曲线天然从 baseline
开始往下降，永远不会比 baseline 差太多。

### 8.2 MLPPredictor（Tier-0，~5M 参数）

```
输入 token（per-token 独立处理）
   ├── token_emb(vocab → 32)
   ├── sinusoidal_encode(position, 16)
   ├── sinusoidal_encode(n_prompt_tokens, 16)
   ├── sampling_lp (1d)
   ├── temperature (1d)
   ├── task_emb (12 → 16)
   └── lora_emb (3 → 16)
       │
       └── concat → MLP(hidden=128, 3 layers, GELU) → Linear(1) → Δ̂_t
```

**特点**：

- 每个 token 独立预测，**不看上下文**
- 适合做 sanity baseline，验证特征 / loss / 训练流是否 OK
- 实测 R² 上限约 0.30~0.35

### 8.3 TransformerPredictor（Tier-1 主，~5.3M 参数）

```
每个 token 的初始向量 =
   compact_token_emb(vocab → 32) → Linear(d_model)
 + sinusoidal_encode(position, d_model)
 + lp_proj(sampling_lp → d_model)

每个序列共享的条件向量（broadcast 到所有 position）=
   prefix_enc(n_prompt_tokens, d_model)
 + temp_proj(temperature → d_model)
 + task_emb(d_model)
 + lora_emb(d_model)

   │
   └── TransformerEncoder(d_model=128, n_heads=4, n_layers=2, norm_first=True)
   └── Linear(d_model → 1)  → Δ̂_t（per-token）
```

**特点**：

- attention 让每个 token 看见整条 response，捕获 **kernel 在长序列上的累积漂移**
- vocab embedding 用 `compact_dim=32` + 投影到 `d_model`，避免 152064 × 128 = 19M 的 embedding
- 实测 R² 0.50+，token MAE 0.025（对比 MLP 的 0.030）

### 8.4 大小对比

| model | 总参数 | embedding 占比 | 单 epoch（70K 条 / H20 / fp32） |
|---|---|---|---|
| `mlp`         | 4.9 M | 4.86M (99%) | ~9 秒 |
| `transformer` | 5.3 M | 4.86M (92%) | ~42 秒 |

---

## 9. Loss 设计

`losses.py::predictor_loss` 实现的是**双尺度 L1 + 显式 bias 项**：

```python
def predictor_loss(pred, target, mask, lambda_seq=1.0, lambda_bias=5.0):
    m = mask.float()                                # [B, T]
    n = m.sum().clamp_min(1.0)                      # 总有效 token 数

    diff = (pred - target) * m                       # [B, T]

    # ── token-level 局部精度 ─────────────────────
    L_token = diff.abs().sum() / n

    # ── seq-level 累积精度（按长度归一化）─────────
    seq_sum_err = diff.sum(dim=1).abs()              # |Σ_t (Δ̂ − Δ)|, per-seq
    seq_len     = m.sum(dim=1).clamp_min(1.0)
    L_seq       = (seq_sum_err / seq_len).mean()

    # ── 整 batch 系统偏置（直接把均值压到 0）──────
    L_bias = diff.sum().abs() / n

    return L_token + lambda_seq * L_seq + lambda_bias * L_bias
```

| 项 | 形式 | 作用 |
|---|---|---|
| `L_token` | 单 token L1 | 每个 token 都尽量准 |
| `L_seq`   | 序列累积残差 / T | **专门压制长序列下 IS weight 爆炸**（最关键） |
| `L_bias`  | batch 残差均值 | 直接把系统偏置打到 0（mean_log_is → 0） |

**调参建议**：

- 默认 `λ_seq = 1.0`、`λ_bias = 5.0`
- 如果 corrected `mean_log_is` 偏离 0 较多 → 调大 `λ_bias`
- 如果 token MAE 反而上升、bias 极小 → 模型在"减一个常数"，调小 `λ_bias`（甚至 0.5~1.0）
- `λ_seq` 一般不调；它跟 `L_token` 同尺度，权重 1 已经够

---

## 10. 评估指标解释

`losses.py::EvalSummary` 给出 **baseline vs corrected** 双栏，所有指标含义：

### 10.1 Token-level

| 指标 | 数学定义 | 读法 |
|---|---|---|
| `token_mae` | `mean_t \|residual_t\|` | 单 token 平均偏差。baseline 直接是 `\|Δ_t\|` 的均值 ≈ 0.049 |
| `token_bias` | `\|mean_t residual_t\|` | 系统性偏置的绝对值。baseline ≈ 0.013 |

`residual_t` = baseline 时是 `Δ_t`、corrected 时是 `Δ_t − Δ̂_t`。

### 10.2 Sequence-level

| 指标 | 数学定义 | 直观意义 |
|---|---|---|
| `mean_log_is` | `mean_seq Σ_t residual_t` | 序列累积残差的均值 → log(IS_weight) 的均值。**0 = IS 无偏** |
| `median_abs_log_is` | `median_seq \|Σ_t residual_t\|` | 中位数尾部，比 mean 更鲁棒 |
| `clip01` | `P(\|Σ_t residual_t\| > log(1.1))` | PPO ε=0.1 时 **被 clip 掉的序列比例** |
| `clip02` | `P(\|Σ_t residual_t\| > log(1.2))` | 同上，ε=0.2 |

clip 阈值的来源：PPO 把 IS ratio = exp(Σ residual) 限制在 `[1−ε, 1+ε]`；
落在阈值外即被 clip，对应 `|Σ residual| > log(1+ε)`。

### 10.3 Δ R²

```
Δ R²  =  1  −  SS_res / SS_tot

SS_tot = Σ_t Δ_t²            （把 baseline 当 prediction = 0 的残差平方和）
SS_res = Σ_t (Δ_t − Δ̂_t)²    （corrected 残差平方和）
```

读法：

| Δ R² | 含义 |
|---|---|
| 0.0 | 模型啥也没学到（≈ baseline） |
| 0.4 | predictor 解释了 40% 方差，达到 **usable** 标准 |
| 0.7 | 解释 70%，**strong** |
| 1.0 | 完美预测（不可能） |

### 10.4 Improvement 列

`evaluate()` 输出表会自动算改进百分比：

```
                       baseline   corrected   improvement
  token  MAE           0.0486     0.0251     +48.5%
  token  bias          0.0176     0.0000     +99.8%
  seq    mean_log_is   5.7918     0.0119     +99.8%
  ...
```

正百分比 = corrected 比 baseline 好（绝对值更小）。

---

## 11. 训练命令

### 11.1 MLP（Tier-0 baseline）

```bash
predictor/.venv/bin/python predictor/train.py \
    --data simulator/22_agents_50_step_results.logprobs.jsonl \
    --output predictor/checkpoints/mlp_v1 \
    --model mlp \
    --epochs 20 --batch_size 32 \
    --lambda_bias 2.0
```

### 11.2 Transformer（推荐）

下面的命令显存占用大致为：1563MiB
```bash
predictor/.venv/bin/python predictor/train.py \
    --data simulator/22_agents_50_step_results.logprobs.jsonl \
    --output predictor/checkpoints/transformer_v2_large \
    --model transformer \
    --epochs 20 --batch_size 8 \
    --d_model 256 --n_heads 8 --n_layers 4 \
    --token_emb_dim 64 \
    --lambda_bias 0.5 \
    --lr 2e-4
```

### 11.3 关键超参

| 参数 | 默认 | 说明 |
|---|---|---|
| `--epochs` | 20 | MLP 够，Transformer 建议 15~20 |
| `--batch_size` | 8 | Transformer attention O(T²)，bs 不宜太大 |
| `--lr` | 3e-4 | AdamW |
| `--weight_decay` | 1e-4 | |
| `--grad_clip` | 1.0 | |
| `--lambda_seq` | 1.0 | |
| `--lambda_bias` | 5.0 | bias 已小时建议 1.0~2.0 |
| `--eval_every` | 1 | 每 epoch 跑 val |

### 11.4 输出产物

```
predictor/checkpoints/<run_name>/
├── best.pt                 # corrected_token_mae 最小那一轮
├── training_log.jsonl      # 每 epoch 一行（loss + val 全指标）
├── test_summary.json       # 最终 test 集结果
└── training_curves.png     # plot.py 输出
```

`best.pt` 包含：

```python
{
    "model_state": ...,
    "args": {...},                # 所有训练超参
    "vocab_size": 152064,
    "task_map":  {"size": 12, "table": {...}},
    "lora_map":  {"size": 3,  "table": {...}},
    "epoch": 15,
    "val_summary": {...}          # 当时的 EvalSummary
}
```

---

## 12. 绘图工具

`plot.py` 读 `training_log.jsonl` + `test_summary.json` 生成 8 子图大图。

### 12.1 单 run

```bash
predictor/.venv/bin/python predictor/plot.py \
    --run_dir predictor/checkpoints/transformer_v1
# → predictor/checkpoints/transformer_v1/training_curves.png
```

### 12.2 多 run 叠加对比

```bash
predictor/.venv/bin/python predictor/plot.py \
    --run_dir predictor/checkpoints/transformer_v1 \
    --compare predictor/checkpoints/mlp_v1 \
    --output predictor/checkpoints/compare.png
```

### 12.3 输出布局

| Row 1 | Row 2 |
|---|---|
| Train losses (L_total/L_token/L_seq/L_bias) | Sequence mean log(IS weight) |
| Token MAE                                   | Sequence median \|log(IS)\| |
| Token bias                                  | Clip rate ε=0.1 |
| **Δ R²** （带 usable / strong 阈值）         | Clip rate ε=0.2 |

每张图都画：

- 灰色虚线 = baseline 不修正时的水平线
- 彩色实线 = 各 run 的 corrected 曲线
- 彩色虚线 = 各 run 的 best ckpt 位置（corrected_token_mae 最小）

顶部 `suptitle` 自动拼出每个 run 的 best epoch + test 集摘要。

---

## 13. 当前结果对比

> 数据：`simulator/22_agents_50_step_results.logprobs.jsonl`（70115 条，22 tenants × 50 steps）
> 划分：48656 / 10426 / 10427

### 13.1 测试集（test）数值对比

| 指标 | baseline | MLP | Transformer | 说明 |
|---|---:|---:|---:|---|
| token MAE             | 0.0487 | 0.0300 | **0.0251** | Transformer ↓16% vs MLP |
| token bias            | 0.0176 | 0.0009 | **0.00004** | 都达到 strong；Transformer 几乎完全消除 |
| seq mean_log_is       | 5.79   | −0.30  | **+0.012**  | Transformer ↓25× |
| seq median \|log_is\| | 1.55   | 0.83   | **0.53**    | |
| seq clip01            | 0.948  | 0.920  | **0.862**   | 仍被单 token 方差下限钉住 |
| seq clip02            | 0.903  | 0.840  | **0.747**   | |
| **Δ R²**              | 0.0    | 0.347  | **0.514**   | Transformer 上 usable 档（>0.4）|

### 13.2 训练曲线观察

- **MLP**：epoch 6 R² 见顶 0.31，后 14 epoch 在 0.25~0.30 震荡。容量瓶颈。
- **Transformer**：单调向上趋势，epoch 11 触 0.42；末段（epoch 13~15）有抖动，**LR 没退火导致**。
- **训练 loss 曲线**：两者 `L_total` 单调下降，没出现 plateau —— 模型还有学习余量。

### 13.3 clip01 为什么 plateau 在 0.86

token 残差方差未降低时，序列累积残差的方差恒定：

```
Var(Σ_t residual_t)  =  T · Var(residual_t)  ≈  600 · 0.025²  ≈  0.375
std ≈ 0.61   →   exp(±0.61)  =  [0.54, 1.84]
```

约 60%+ 的序列天然落在 `[1−0.1, 1+0.1]` 之外。**继续推 clip01 必须降低 token 残差方差**（不是再调 bias），路径只剩：

1. 加大模型容量（d_model 256 / n_layers 4）
2. 加 LR cosine schedule + warmup（消除末段抖动）
3. 扩数据到 200K+

---

## 14. 目标档位

| 指标 | baseline (no predictor) | usable | strong | 当前 best (Transformer test) |
|---|---|---|---|---|
| corrected token MAE  | ~0.049 | < 0.015 | < 0.008  | 0.0251 ⚠️ |
| corrected token bias | ~0.018 | < 0.002 | < 0.0005 | 0.00004 ✅✅ |
| corrected mean_log_is | ~5.8  | < 1.0   | < 0.1    | 0.012 ✅ |
| corrected clip01 (ε=0.1) | ~0.95 | < 0.5 | < 0.2   | 0.862 ⚠️ |
| Δ R²                  | 0.0    | > 0.4   | > 0.7    | 0.514 ✅ |

**当前状态**：3 项 usable+ / 1 项 strong+ / 2 项尚需努力（MAE 与 clip01）。

---

## 15. 推理 / 接入 RL 训练循环

### 15.1 加载 ckpt

```python
import torch
from predictor.model import build_model

ckpt = torch.load(
    "predictor/checkpoints/transformer_v1/best.pt",
    weights_only=False,
    map_location="cuda",
)

model = build_model(
    "transformer",
    vocab_size=ckpt["vocab_size"],
    n_tasks=ckpt["task_map"]["size"],
    n_lora_ranks=ckpt["lora_map"]["size"],
    # 其余架构超参
    d_model=ckpt["args"]["d_model"],
    n_heads=ckpt["args"]["n_heads"],
    n_layers=ckpt["args"]["n_layers"],
).cuda().eval()

model.load_state_dict(ckpt["model_state"])
```

### 15.2 在 PPO loop 里使用

```python
# 假设 batch 已有 rollout：response_tokens, sampling_lps, prefix_len, ...
with torch.no_grad():
    delta_hat = model(
        token_ids=response_tokens,           # [B, T]
        sampling_lps=sampling_lps,           # [B, T]
        n_prompt_tokens=prefix_len,          # [B]
        temperature=temperature,             # [B]
        task_idx=task_idx,                   # [B]
        lora_rank_idx=lora_rank_idx,         # [B]
        mask=mask,                           # [B, T]
    )                                         # → [B, T]

corrected_sampling_lp = sampling_lps - delta_hat
log_is_ratio = training_lps - corrected_sampling_lp.detach()  # PPO 用这个
ratio = torch.exp(log_is_ratio).clamp(1 - eps, 1 + eps)
```

**关键**：predictor 输出要 `.detach()` 进 IS ratio，避免梯度回流到 predictor（predictor 是
辅助网络，参数不参与 actor loss）。

### 15.3 在线再训练

framework mismatch 在不同任务 / 模型上不一样。建议：

1. 每次切任务 → 用一小段 sync_mode=true 的 rollout 重训 predictor
2. predictor 参数小（~5M），10 分钟 H20 训完
3. 训完替换 ckpt 即可
