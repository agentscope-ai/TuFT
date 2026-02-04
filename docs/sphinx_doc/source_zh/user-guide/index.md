# 用户指南

本节提供在各种场景中使用 TuFT 的综合指南。

```{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Chat SFT
:link: chat-sft
:link-type: doc
:shadow: none

使用仅助手损失掩码对聊天格式数据进行监督微调。
:::

:::{grid-item-card} Countdown RL
:link: countdown-rl
:link-type: doc
:shadow: none

在可验证任务上使用 GRPO 风格训练进行强化学习。
:::

:::{grid-item-card} 持久化
:link: persistence
:link-type: doc
:shadow: none

使用 Redis 启用服务器状态持久化以实现崩溃恢复。
:::

:::{grid-item-card} 可观测性
:link: telemetry
:link-type: doc
:shadow: none

OpenTelemetry 集成，用于追踪、指标和日志。
:::
```

```{toctree}
:maxdepth: 1
:hidden:

chat-sft
countdown-rl
persistence
telemetry
```
