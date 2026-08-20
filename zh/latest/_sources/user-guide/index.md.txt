# 用户指南

本节提供在各种场景中使用 TuFT 的综合指南。

```{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Chat SFT
:link: chat-sft
:link-type: doc
:shadow: none

在聊天格式数据上进行监督微调，支持助手角色损失掩码。
:::

:::{grid-item-card} Countdown RL
:link: countdown-rl
:link-type: doc
:shadow: none

基于可验证任务的 GRPO 风格强化学习训练。
:::

:::{grid-item-card} 在策略蒸馏（OPD）
:link: on-policy-distillation
:link-type: doc
:shadow: none

以逐 token 反向 KL 为信号，在学生自己的采样上把教师模型蒸馏进学生模型。
:::

:::{grid-item-card} 自定义损失
:link: custom-losses
:link-type: doc
:shadow: none

通过 `forward_backward_custom` 在客户端定义训练目标（如 DPO + NLL 组合损失），支持 HF 与 FSDP。
:::

:::{grid-item-card} LoRA 目标模块
:link: lora-target-modules
:link-type: doc
:shadow: none

LoRA 标志如何解析为模块名，以及围绕它们的规则。
:::

:::{grid-item-card} 持久化
:link: persistence
:link-type: doc
:shadow: none

通过 Redis 启用服务器状态持久化，实现崩溃恢复。
:::

:::{grid-item-card} 可观测性
:link: telemetry
:link-type: doc
:shadow: none

集成 OpenTelemetry，用于链路追踪、指标监控和日志记录。
:::

:::{grid-item-card} 控制台
:link: console
:link-type: doc
:shadow: none

训练运行与检查点监控仪表盘，内置推理试验场。
:::
```

```{toctree}
:maxdepth: 1
:hidden:

chat-sft
countdown-rl
on-policy-distillation
custom-losses
lora-target-modules
persistence
telemetry
console
```
