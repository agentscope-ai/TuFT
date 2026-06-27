# 构建一个多租户在线 Agentic RL 训练 Simulator

## 你要做什么

模拟多个租户同时在线做 agentic RL 训练。每个 adapter 持续不断地发 sampling 请求（不会停），buffer 攒满后触发一次 GRPO 训练，训练完成后切换到新权重继续 sampling。多个 adapter 并发，共享底层 sampling/training 资源。

## 每个 Adapter 的行为

```
sampling 永不停止
       │
       ▼
┌──────────────────────────────────────────────────────────────────┐
│  ───sample───sample───sample───sample───sample───sample───▶      │
│        │        │        │        │        │        │             │
│        ▼        ▼        ▼        ▼        ▼        ▼             │
│      [reward] [reward] [reward] [reward] [reward] [reward]       │
│        │        │        │        │        │        │             │
│        └────────┴────────┴────────┘        │        │             │
│                    │                       │        │             │
│              buffer 满了                    │        │             │
│                    │                       ▼        ▼             │
│                    ▼                    继续 sampling（不停）      │
│              trigger train                                        │
│                    │                                              │
│                    ▼                                              │
│            train 完成 → 更新 adapter 权重                          │
│                    │                                              │
│                    ▼                                              │
│         后续 sample 自动用新权重                                    │
└──────────────────────────────────────────────────────────────────┘
```

关键：**sampling 不等 training**。Training 是异步触发的，sampling 持续进行。训练期间产生的样本用的是旧权重（staleness 自然产生）。训练完成后 sync 新权重，之后的 sample 用新版本。

## 多 Adapter 并发

```
时间 ──────────────────────────────────────────────────────▶

Adapter A: ─s─s─s─s─[train]─s─s─s─s─s─s─[train]─s─s─...
Adapter B: ─s─s─s─s─s─s─[train]─s─s─s─s─[train]─s─...
Adapter C: ─s─s─[train]─s─s─s─[train]─s─s─s─[train]─...
                │                    │
                ▼                    ▼
          共享 sampling 资源    共享 training 资源
```

默认系统支持多租户训练，同时启动多个adapter的训练

## 两种租户模式（只是配置不同）

- **Per-User**：request_rate 低（单用户 agent 交互速度有限），buffer 攒满慢，训练频率低
- **Per-Base-Model**：request_rate 高（多用户汇聚），buffer 攒满快，训练频率高

## 后端接口

```python
class TrainingBackend(ABC):
    def initialize(config) -> None
    def create_adapter(adapter_id: str) -> None
    def sync_weights(adapter_id: str) -> None  # 训练后调用，让后续 sample 用新权重
    def sample(adapter_id: str, prompt_tokens, num_samples, max_tokens, temperature) -> SampleResult
    def train_step(adapter_id: str, datums: List[TrainingDatum]) -> TrainStepResult
    def get_tokenizer() -> Tokenizer
```

先实现 TinkerBackend（调 Tinker SDK），后续加 Sequential/Co-location baseline。

## Tinker SDK 调用参考
/root/TuFT/examples/countdown_rl/train.py

## 任务和 Reward

| 任务 | 数据源 | Reward |
|------|--------|--------|
| GSM8K | `openai/gsm8k` | `\boxed{}` 数值匹配 |
| MATH | `hendrycks/competition_math` | 同上 |
| Countdown | `Jiayi-Pan/Countdown-Tasks-3to4` | 表达式 == target |
| MBPP | `google-research-datasets/mbpp` | subprocess 执行 test 通过率 |

## 配置示例

```yaml
backend:
  type: tinker
  base_model: "Qwen/Qwen3-4B"


tenants:
  - id: "gsm8k-A"
    task: gsm8k
    request_rate: 2.0       # req/s
    buffer_size: 64         # 攒满 64 个 (problem, response, reward) 触发训练
    group_size: 16
    num_train_steps: 500
    lora_rank: 8
    learning_rate: 1e-4

  - id: "gsm8k-shared"
    task: gsm8k
    request_rate: 2.0       # req/s
    buffer_size: 64         # 攒满 64 个 (problem, response, reward) 触发训练
    group_size: 16
    num_train_steps: 500
    lora_rank: 16
    learning_rate: 1e-3


  # ... 共 8 个

evaluation:
  eval_interval_steps: 50   # 每训练 50 步 eval 一次
  eval_sample_size: 200
```

## 输出

```json
{
  "backend": "tinker",
  "total_wall_clock_seconds": 14400,
  "per_tenant": {
    "gsm8k-A": {
      "final_accuracy": 0.52,
      "reward_curve": [[step, reward], ...],
      "train_steps_completed": 500,
      "mean_staleness": 2.3,
      "mean_sample_latency_ms": 450
    }
  }
}
```

## 注意

1. **Sampling 不阻塞**：每个 adapter 持续 sample，不等 train。用 asyncio 实现。
2. **Buffer + 触发**：buffer 攒满 → 做 GRPO advantage normalization → 调 train_step → sync 新权重。
3. **Staleness 自然产生**：train 期间的 sample 用的旧权重，这就是 staleness。
4. **Tinker 支持多 adapter**：每个 adapter 独立的 training_client。
5. **MBPP 执行安全**：subprocess + timeout。
