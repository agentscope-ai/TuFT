# Multi-Tenant Online RL Training Simulator

模拟多个租户（adapter）同时在共享后端上做在线 RL 训练的场景。每个 adapter 持续 sampling，buffer 攒满后异步触发训练，训练完成后切换新权重继续 sampling。

## 架构概览

```
时间 ──────────────────────────────────────────────────────▶

Adapter A: ─s─s─s─s─[train]─s─s─s─s─s─s─[train]─s─s─...
Adapter B: ─s─s─s─s─s─s─[train]─s─s─s─s─[train]─s─...
Adapter C: ─s─s─[train]─s─s─s─[train]─s─s─s─[train]─...
                │                    │
                ▼                    ▼
          共享 sampling 资源    共享 training 资源
```

- Sampling 永不阻塞，training 异步执行
- Training 期间新 sample 使用旧权重（staleness 自然产生）
- 每个 adapter 达到 `num_train_steps` 后停止

## 文件结构

```
simulator/
├── config.yaml                      # 示例配置（8 tenants）
├── run.py                           # 入口脚本
└── simulator/
    ├── config.py                    # 配置 dataclass + YAML 加载
    ├── metrics.py                   # 指标收集 + JSON 输出
    ├── tenant.py                    # Tenant actor（sampling loop + buffer + train）
    ├── orchestrator.py              # 并发调度所有 tenants
    ├── backend/
    │   ├── base.py                  # TrainingBackend 抽象接口
    │   └── tinker_backend.py        # Tinker SDK 后端实现
    └── tasks/
        ├── base.py                  # Task 抽象接口
        ├── countdown.py             # Countdown 算术表达式任务
        ├── gsm8k.py                 # GSM8K 小学数学任务
        ├── math_task.py             # MATH 竞赛数学任务
        └── mbpp.py                  # MBPP Python 编程任务
```

## 依赖

```
torch
tinker          # Tinker SDK
datasets        # HuggingFace datasets
transformers    # tokenizer
pyyaml
```

## 配置说明

配置文件为 YAML 格式，包含三个部分：

### backend

```yaml
backend:
  type: tinker                       # 后端类型，目前支持 tinker
  base_url: "http://localhost:10610" # Tinker 服务地址
  api_key: null                      # API 密钥（可选）
  base_model: "Qwen/Qwen3-4B"       # 基座模型
```

### tenants

每个 tenant 代表一个独立的 LoRA adapter 训练任务：

```yaml
tenants:
  - id: "gsm8k-A"             # 唯一标识符
    task: gsm8k                # 任务类型: countdown / gsm8k / math / mbpp
    request_rate: 2.0          # sampling 速率 (requests/s)
    buffer_size: 64            # 触发训练的 buffer 大小（独立 sample 数）
    num_train_steps: 500       # 总训练步数（达到后停止）
    lora_rank: 8               # LoRA rank
    learning_rate: 1e-4        # 学习率
    max_tokens: 512            # 最大生成长度
    temperature: 0.7           # sampling 温度
```

**参数说明：**

| 参数 | 含义 |
|------|------|
| `request_rate` | 控制 sampling 频率，模拟真实用户请求速率 |
| `buffer_size` | 攒满多少个 (prompt, response, reward) 后触发一次训练 |
| `num_train_steps` | adapter 总训练步数上限，达到后该 tenant 退出 |
| `lora_rank` | 越大 adapter 容量越高，训练越慢 |
| `temperature` | 训练时 sampling 温度，影响探索程度 |

### evaluation

```yaml
evaluation:
  eval_interval_steps: 50      # 每隔多少训练步做一次评估（需 < num_train_steps 才会触发）
  eval_sample_size: 200        # 每次评估使用的样本数
  eval_temperature: 0.1        # 评估时的 sampling 温度（接近 greedy）
  # 注意：无论 eval_interval_steps 如何设置，最后一步训练完成后始终会做一次 final eval
```

### 其他

```yaml
output_path: "results.json"    # 结果输出路径
seed: 42                       # 随机种子
```

### logprob_collection

```yaml
logprob_collection:
  enabled: true            # 每次 train step 前调 forward(cross_entropy) 抓 training-side logprob，
                           # 与 BufferItem 里采样时存的 sampling logprob 配对记录 mismatch
  dump_per_token: true     # 额外把每条样本的 per-token (sampling_lp, training_lp) 写入 .logprobs.jsonl
  # output_path: "results.logprobs.jsonl"  # 缺省从 output_path 自动派生
```

打开后，results.json 的 `per_tenant.<id>` 里会多出 `logprob_mismatch_per_step` 字段，每条记录：

```json
{
  "step": 12, "weight_version": 11, "n_items": 64, "n_tokens": 6532,
  "mean_sampling_logprob": -45.1, "mean_training_logprob": -47.3,
  "mean_diff": 2.2, "mean_abs_diff": 0.018, "max_abs_diff": 0.94, "std_diff": 0.07,
  "mean_cum_diff": 2.2, "std_cum_diff": 1.4,
  "mean_is_weight": 1.05, "std_is_weight": 0.31,
  "min_is_weight": 0.42, "max_is_weight": 3.7, "p99_is_weight": 2.6,
  "p_out_clip_01": 0.42, "p_out_clip_02": 0.18,
  "mean_staleness": 1.7
}
```

这份数据就是验证 “mismatch 是否可预测” 的原始输入：可以在后续以 `staleness` / `weight_version` / `step` 为自变量拟合 `mean_abs_diff` 、`mean_is_weight`、`p_out_clip_*`。Per-token 原始数据（`.logprobs.jsonl`）可用来作 token 级 / sequence 级的微观分析。

## 支持的任务

### 单轮任务 (Single-Turn)

| 任务 | 数据源 | Reward 类型 | Reward 规则 |
|------|--------|------------|-------------|
| `countdown` | `Jiayi-Pan/Countdown-Tasks-3to4` | 确定性 | `<answer>表达式</answer>` 求值 == target，连续 shaping |
| `gsm8k` | `openai/gsm8k` | 确定性 | `\boxed{}` 内数值与标准答案匹配 |
| `math` | `hendrycks/competition_math` | 确定性 | `\boxed{}` 内答案匹配（支持 LaTeX 归一化） |
| `mbpp` | `google-research-datasets/mbpp` | 确定性 | 提取代码在 subprocess 中执行测试用例 |
| `humaneval` | `openai/openai_humaneval` | 确定性 | 提取代码执行 HumanEval 测试套件，二值 (pass/fail) |
| `ifeval` | `google/IFEval` | 确定性 | 21 种规则化约束验证（关键词/词数/格式等），连续值 [0,1] |

### 多轮 Agent RL 任务 (Multi-Turn)

| 任务 | 数据源 | 工具 | Reward 类型 | 环境模拟 |
|------|--------|------|------------|----------|
| `hotpotqa` | `hotpotqa/hotpot_qa` | Search, Lookup, Finish | 确定性 | Mock (数据集自带段落) |
| `math_agent` | `openai/gsm8k` | Calculate, Finish | 确定性 | 真实执行 (safe eval) |
| `triviaqa` | `trivia_qa` (rc) | Search, Finish | 确定性 | Mock (数据集自带证据) |
| `apibank` | `liminghao1630/API-Bank` + 内置 | ToolCall, Finish | 确定性 | Mock (预定义 API 响应) |
| `toolbench` | `tuandunghcmut/toolbench-v1` + 内置 | CallAPI, Finish | 确定性 | Mock (预定义 API 响应) |

---

### 任务详细说明

#### 单轮任务

**countdown** — 算术表达式构造
- 给定一组数字和目标值，用四则运算构造等于目标的表达式
- Reward: 精确匹配 1.0，连续 shaping 基于误差距离
- 无 mock，reward 通过数学计算确定

**gsm8k** — 小学数学应用题
- 多步推理后给出数值答案
- Reward: 数值精确匹配，二值 (1.0/0.0)
- 无 mock，reward 通过数值比较确定

**math** — 竞赛级数学
- 需要复杂证明/推导的数学问题
- Reward: LaTeX 归一化后答案匹配，二值
- 无 mock，reward 通过字符串归一化比较确定

**mbpp** — Python 函数生成
- 根据描述生成 Python 函数，运行测试用例验证
- Reward: 所有测试通过 1.0，否则 0.0
- **真实执行**: subprocess 沙箱运行代码，reward 依赖实际代码执行结果

**humaneval** — 函数级代码补全
- 补全 Python 函数体，运行 HumanEval 测试套件
- Reward: 所有断言通过 1.0，否则 0.0
- **真实执行**: subprocess 沙箱运行代码

**ifeval** — 指令遵循
- 验证模型输出是否满足格式/内容约束
- Reward: 满足约束的比例，连续值 [0,1]
- 无 mock，reward 通过规则引擎确定性验证

---

#### Agent RL 任务

Agent 任务使用多轮交互协议：`reset_episode()` → 循环 `[采样 → step() → 观测]` → `compute_episode_reward()`

**hotpotqa** — 多跳问答 (ReAct 风格)
- 论文: Yang et al., EMNLP 2018
- 工具: `Search[query]`（搜索 Wikipedia 主题）、`Lookup[term]`（当前页查词）、`Finish[answer]`
- 环境: **Mock** — 搜索范围为数据集自带的 context paragraphs（10 篇文档），非真实 Wikipedia
- Reward: F1 token overlap + Exact Match，确定性
- 真实化替换: 接入 Wikipedia API 或向量检索数据库
- 特殊说明: 经典 ReAct 论文核心实验任务，3 个工具

**math_agent** — 计算器辅助数学推理
- 论文: 受 TORA (Gou et al., 2023) 启发，GSM8K + Calculator
- 工具: `Calculate[expression]`（算术求值）、`Finish[number]`
- 环境: **真实执行** — Calculate 使用安全 eval 真实计算表达式
- Reward: 数值精确匹配 1.0，1% 相对误差 0.5，格式正确 0.1
- 真实化替换: 已是真实执行，无需替换
- 特殊说明: 唯一一个工具调用完全真实的 agent 任务

**triviaqa** — 知识检索问答
- 论文: Joshi et al., ACL 2017
- 工具: `Search[query]`（知识库检索）、`Finish[answer]`
- 环境: **Mock** — 搜索范围为数据集自带的 entity_pages + search_results
- Reward: EM 1.0 / F1 部分奖励 / 格式 0.1，对比所有 answer aliases
- 真实化替换: 接入搜索引擎 API 或 RAG 系统
- 特殊说明: 答案有多个别名 (aliases)，评估宽松

**apibank** — 多 API 工具选择与调用
- 论文: Li et al., EMNLP 2023 (Alibaba)
- 工具: `ToolCall[ApiName(key='value')]`（调用指定 API）、`Finish[response]`
- 环境: **Mock** — 15 个内置 API 定义，每个场景预定义 mock 响应
- Reward: API 名匹配 0.5 + 参数匹配度 0.5，多步取平均
- 真实化替换: 对接真实第三方 API（天气/航班/邮件等），需处理 API key 和速率限制
- 特殊说明:
  - 自动从 HuggingFace 下载 level-1 数据补充训练集
  - 内置 12 个场景 × 5 变体 = 60 条数据（含干扰 API 增强）
  - 评估维度: API 选择准确性 + 参数提取准确性

**toolbench** — 多步 API 调用链
- 论文: Qin et al., NeurIPS 2023 (Tsinghua)
- 工具: `CallAPI[ToolName(params)]`（调用 API）、`Finish[answer]`
- 环境: **Mock** — 15 个内置工具，场景预定义调用链和 mock 响应
- Reward: ground truth 调用链匹配度（序列比对），加权参数重叠度
- 真实化替换: 使用 StableToolBench 的 MirrorAPI 模型模拟 16k+ API，或对接真实 REST API
- 特殊说明:
  - 自动从 HuggingFace 加载真实对话补充训练集
  - 内置 12 个场景 × 4 变体 = 48 条数据（含干扰工具增强）
  - 强调 **调用链规划能力**（需按正确顺序串联多个 API）
  - 评估维度: 工具选择 + 参数正确性 + 调用顺序

---

### Mock vs 真实执行对照表

| 任务 | 工具执行 | Reward 计算 | 替换为真实环境的难度 |
|------|---------|------------|-------------------|
| countdown | N/A | 真实 (数学计算) | — |
| gsm8k | N/A | 真实 (数值匹配) | — |
| math | N/A | 真实 (字符串匹配) | — |
| mbpp | 真实 (subprocess) | 真实 (测试用例) | — |
| humaneval | 真实 (subprocess) | 真实 (测试用例) | — |
| ifeval | N/A | 真实 (规则引擎) | — |
| hotpotqa | **Mock** (数据集段落) | 真实 (F1/EM) | 中 — 接入 Wikipedia/搜索 API |
| math_agent | **真实** (safe eval) | 真实 (数值匹配) | — (已是真实) |
| triviaqa | **Mock** (数据集证据) | 真实 (F1/EM) | 中 — 接入搜索引擎 |
| apibank | **Mock** (预定义响应) | 基于 GT 匹配 | 高 — 需对接 15+ 第三方 API |
| toolbench | **Mock** (预定义响应) | 基于 GT 链匹配 | 高 — 需对接真实 REST API 或 MirrorAPI |

**说明**:
- "Reward 计算" 列: 所有任务的 reward 都是确定性的，不依赖 LLM judge
- Mock 环境的 reward 基于与 ground truth 的匹配（而非工具返回结果的正确性），因此即使替换为真实 API，reward 逻辑也需要相应调整
- 替换为真实环境时需注意: 真实 API 响应不确定，reward 需改为基于最终任务完成度（如用户满意度/任务成功率），而非调用链匹配

---

### Agent 任务配置参数

Agent RL 任务相比单轮任务额外支持 `max_turns` 参数：

```yaml
- id: "hotpotqa-A"
  task: hotpotqa
  max_turns: 8             # 每个 episode 最大交互轮数
  max_tokens: 512          # 每轮生成的最大 token 数
  # ... 其他参数同单轮任务
```

| 任务 | 推荐 max_turns | 推荐 max_tokens | 说明 |
|------|---------------|----------------|------|
| hotpotqa | 8 | 512 | 多跳搜索可能需要较多轮 |
| math_agent | 6 | 256 | GSM8K 一般 3-5 步计算 |
| triviaqa | 5 | 256 | 1-3 次搜索即可定位答案 |
| apibank | 6 | 256 | 多数场景 1-3 次 API 调用 |
| toolbench | 6 | 256 | 典型调用链 2-3 步 |

**Prompt 膨胀保护**: Tenant 内置 `_MAX_CONTEXT_TOKENS = 7168` 的截断机制，当多轮对话累积 prompt 超限时，保留前 30% (系统提示) + 后 70% (最近上下文)，防止 OOM。

## 运行

```bash
# 基本运行
python run.py --config config.yaml

# 指定输出路径
python run.py --config config.yaml --output results/experiment1.json

# 开启详细日志
python run.py --config config.yaml --verbose
```

**前提条件：** Tinker 服务需要已经启动并可访问（`base_url` 配置正确）。

## 输出格式

运行结束后生成 JSON 文件：

```json
{
  "backend": "tinker",
  "base_model": "Qwen/Qwen3-4B",
  "total_wall_clock_seconds": 14400.0,
  "per_tenant": {
    "gsm8k-A": {
      "final_accuracy": 0.52,
      "reward_curve": [[1, 0.12], [2, 0.15], ...],
      "eval_curve": [[50, 0.30], [100, 0.42], ...],
      "train_steps_completed": 500,
      "total_samples": 32000,
      "mean_staleness": 2.3,
      "mean_sample_latency_ms": 450.0
    }
  }
}
```

**指标说明：**

| 指标 | 含义 |
|------|------|
| `final_accuracy` | 最后一次 eval 的准确率 |
| `reward_curve` | 每步训练时 buffer 内的平均 reward |
| `eval_curve` | 定期 eval 的准确率曲线 |
| `train_steps_completed` | 实际完成的训练步数 |
| `total_samples` | 该 tenant 总共产生的 sample 数 |
| `mean_staleness` | 平均 staleness（sample 使用的权重版本 vs 当前训练步数的差） |
| `mean_sample_latency_ms` | 平均 sampling 延迟 |

## 后端迁移

`TrainingBackend` 是抽象基类，所有方法均为 async：

```python
class TrainingBackend(ABC):
    async def initialize(self, config: dict) -> None
    async def create_adapter(self, adapter_id: str, adapter_config: AdapterConfig) -> None
    async def sync_weights(self, adapter_id: str) -> int
    async def sample(self, adapter_id, prompt_tokens, num_samples, max_tokens, temperature) -> List[SampleResult]
    async def train_step(self, adapter_id, training_datums: List[dict]) -> TrainStepResult
    def get_tokenizer(self) -> Any
    async def cleanup(self, adapter_id: str) -> None
```

要接入新后端（如 verl），只需：

1. 在 `simulator/backend/` 下新建实现文件
2. 实现 `TrainingBackend` 的所有方法
3. 在 `orchestrator.py` 的 `_create_backend()` 中注册新类型
4. 在配置中设置 `backend.type` 为新类型名
