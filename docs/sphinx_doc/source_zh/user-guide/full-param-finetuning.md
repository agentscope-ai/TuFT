# 全参数微调

本指南介绍如何使用 TuFT 进行**全参数微调**（即训练模型的**所有**权重），作为 LoRA 微调的替代方案。数据集/损失函数的设置与 `examples/chat_sft/chat_sft.ipynb` 中的 LoRA SFT 示例相同——仅模型创建步骤不同（使用 `rank=None` 而非 LoRA rank）。

---

## 目录

1. [何时使用全参 vs. LoRA](#何时使用全参-vs-lora)
2. [前置条件](#前置条件)
3. [服务端配置](#服务端配置)
4. [最小训练示例](#最小训练示例)
5. [与 LoRA 的关键区别](#与-lora-的关键区别)
6. [注意事项](#注意事项)

---

## 何时使用全参 vs. LoRA

| 方面 | 全参数微调 | LoRA 微调 |
|---|---|---|
| **训练内容** | 所有模型权重 | 仅低秩适配器矩阵 |
| **GPU 内存** | 高——完整模型梯度 + 优化器状态 | 低——仅适配器参数 |
| **训练速度** | 每步较慢（更多参数） | 每步较快 |
| **适用场景** | 追求最大质量、领域迁移、小模型 | 快速实验、多租户、大模型 |
| **多租户** | 单实例共享——无租户间隔离（见下文） | 多个适配器共享一个基础模型 |
| **检查点大小** | 完整模型副本 | 小型适配器文件（MB 级） |

**经验法则**
- 当需要快速迭代、GPU 预算有限或多租户适配器服务时，使用 **LoRA**。
- 当需要最大模型容量、进行深度领域适配或模型足够小能容纳完整梯度时，使用**全参微调**。

---

## 前置条件

1. **仅 FSDP 后端**——全参训练仅在 FSDP 后端中实现。HF 后端**不**支持。请在模型配置中设置 `training_backend: fsdp`。
2. **推荐多 GPU**——全参训练需要加载整个模型及优化器状态。FSDP 会将这些分片到多个 GPU 上。虽然小模型可用单 GPU 运行，但强烈推荐通过 `fsdp_num_gpus` 使用多 GPU。
3. **`allow_full_param: true`**——必须在模型配置中显式启用全参训练（见下一节）。这是一个安全标志，防止意外的全参请求。

---

## 服务端配置

在 `tuft_config.yaml` 的模型条目中添加 `allow_full_param: true`：

```yaml
supported_models:
  - model_name: Qwen/Qwen3-4B
    model_path: Qwen/Qwen3-4B
    max_model_len: 32768
    training_backend: fsdp       # 必需——仅 FSDP 支持全参
    fsdp_num_gpus: 2             # FSDP 分片使用的 GPU 数量
    allow_full_param: true       # 启用该模型的全参训练
    max_lora_rank: 16            # LoRA 仍可与全参并存使用
    max_loras: 1
```

如果未设置 `allow_full_param`（默认为 `false`），全参训练请求将返回 HTTP 400：

```
400 Bad Request: Model does not support full-parameter training.
```

---

## 最小训练示例

### 嵌入式模式（推荐）

使用 TuFT 的嵌入式 Python API，设置 `rank=None` 来请求全参训练：

```python
import tuft

tuft.init(model="/path/to/Qwen3-4B")

training_client = tuft.create_training_client(
    base_model="Qwen3-4B",
    rank=None,  # None = 全参训练（无 LoRA）
)

# 后续训练循环与 LoRA SFT 完全相同
fwdbwd = training_client.forward_backward(datums, loss_fn="cross_entropy").result()

from tinker import types
training_client.optim_step(types.AdamParams(learning_rate=1e-5)).result()
```

### 服务模式（Tinker SDK）

连接到运行中的 TuFT 服务器时，通过 Tinker SDK 发送 `lora_config=None` 的 `CreateModelRequest`：

```python
import tinker
from tinker import types

service_client = tinker.ServiceClient(
    base_url="http://localhost:10610",
    api_key=TINKER_API_KEY,
)

# 创建全参模型（lora_config=None 表示全参训练）
model = service_client.models.create(
    types.CreateModelRequest(
        base_model="Qwen/Qwen3-4B",
        lora_config=None,  # 无 LoRA → 全参
    )
)

training_client = tinker.TrainingClient(
    service_client=service_client,
    model_id=model.id,
)

# 训练循环——与 LoRA 相同
fwdbwd = training_client.forward_backward(datums, loss_fn="cross_entropy").result()
training_client.optim_step(types.AdamParams(learning_rate=1e-5)).result()
```

---

## 与 LoRA 的关键区别

| 特性 | 全参 | LoRA |
|---|---|---|
| **模型创建** | `rank=None` / `lora_config=None` | `rank=32` / `lora_config=LoraConfig(...)` |
| **`forward_backward()`** | 相同 API | 相同 API |
| **`optim_step()`** | 相同 API | 相同 API |
| **检查点** | 保存完整模型权重 | 仅保存适配器权重 |
| **`get_model_info()`** | `is_lora=False`, `lora_rank=None` | `is_lora=True`, `lora_rank=<rank>` |

训练循环（前向/反向传播、优化器步进、评估）使用完全相同的 API。仅模型创建步骤有所不同。

---

## 注意事项

### 多租户限制

与 LoRA 不同——LoRA 中每个租户拥有独立的 adapter（独立参数子集），共享一个冻结的 base model——全参训练会修改**所有**模型权重。当前实现维护一个**单一共享的 `FullParamVerlWorker`** 供所有全参租户使用：

- 通过 `init_full_param()` 注册的多个 `model_id` 共享**同一份**权重和优化器。
- 如果租户 A 训练了一步，租户 B 下次 forward 看到的权重已经被改变。
- 全参租户之间**没有参数级别的隔离**。

**原因**：隔离全参租户需要为每个租户维护一份完整的模型副本，GPU 显存开销极大，不可接受。

**未来实现真正多租户全参隔离的路径：**

1. 维护多个 `FullParamVerlWorker` 实例（每个租户一个）。
2. 同一时刻只有一个 worker 的参数驻留在 GPU 上。
3. 切换租户时：当前 worker `engine.to("cpu")` 卸载；新 worker `engine.to("cuda")` 加载。
4. 在 FSDPTrainingBackend 层实现调度逻辑，管理换入/换出队列并防止 OOM。
5. 预期代价：每次切换需进行完整模型的 CPU↔GPU 传输（多 GB 模型需数秒），不适合快速交替的多租户工作负载。

目前，如需为多个用户提供独立的全参训练，请部署**多个独立的 TuFT 实例**（各自拥有独立的 GPU 分配）。

### 性能未优化

TuFT 的全参微调支持**功能可用但未进行性能优化**。当前实现侧重于正确性和 API 兼容性。与专用的全参训练框架（如 DeepSpeed ZeRO、Megatron-LM）相比，你可能会观察到：

- 较低的 GPU 利用率 / 吞吐量
- 每个 GPU 的内存开销更大
- 缺少梯度检查点、混合精度分片策略、通信重叠等高级优化

如果训练吞吐量对你的工作负载至关重要，建议使用专用的分布式训练框架进行全参训练，将 TuFT 用于其擅长的 LoRA 微调场景。

### 内存需求

全参训练比 LoRA 需要显著更多的 GPU 内存，原因如下：
- 所有模型参数都需要梯度（相比 LoRA 仅需适配器参数的梯度）。
- 优化器（如 Adam）为每个参数维护动量和方差。
- FSDP 将这些分片到多个 GPU 上，但总内存仍然大得多。

**建议**：对相同模型使用至少 2 倍于 LoRA 的 GPU 数量。对于 4B 参数模型，2× A100-80GB 是合理的起点。

### 学习率

全参微调通常使用比 LoRA **更低的学习率**：
- LoRA 默认值：`1e-4`
- 全参推荐值：`1e-5` 到 `5e-5`

全参训练使用过高的学习率可能导致灾难性遗忘或训练不稳定。

### HF 后端不支持

如果对配置为 `training_backend: hf` 的模型请求全参训练，服务器将拒绝该请求：
- 如果 `allow_full_param: false`（默认值）：HTTP 400
- 如果 `allow_full_param: true` 但后端是 HF：HTTP 500（HF 后端抛出 `NotImplementedError`）

全参训练请务必使用 `training_backend: fsdp`。

### 检查点格式

全参检查点包含**完整模型权重**，不同于 LoRA 检查点仅包含适配器权重。这意味着：
- 检查点文件大得多（GB 级 vs. MB 级）。
- 加载检查点会恢复所有模型权重，而非仅适配器增量。
