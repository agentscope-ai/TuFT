# 可观测性（OpenTelemetry）

TuFT 包含可选的 **OpenTelemetry** 集成，用于：

- **追踪**（跨 HTTP → 控制器 → Ray actor 的分布式追踪）
- **指标**（请求/工作流 + 资源指标）
- **日志**（Python 日志桥接到 OpenTelemetry 日志管道）

本文档解释**如何启用遥测**以及 **TuFT 如何在内部连接它**（记录什么、关联键以及如何扩展检测）。

---

## 目录

- [快速开始](#快速开始)
- [导出器和配置](#导出器和配置)
- [TuFT 记录的内容](#tuft-记录的内容)
  - [追踪](#追踪)
  - [指标](#指标)
  - [日志](#日志)
- [关联/共享键](#关联共享键)
- [上下文传播（Ray actor）](#上下文传播ray-actor)
- [开发者指南](#开发者指南)
- [推荐的收集器配置](#推荐的收集器配置)
- [故障排除](#故障排除)

---

## 快速开始

在 `tuft_config.yaml` 中启用遥测：

```yaml
telemetry:
  enabled: true
  service_name: tuft
  otlp_endpoint: http://localhost:4317
  resource_attributes:
    deployment.environment: production
    service.version: 1.0.0
```

> **注意**：`resource_attributes` 允许您将自定义元数据附加到所有遥测数据。常见属性包括：
> - `deployment.environment`：例如 `production`、`staging`、`development`
> - `service.version`：您的应用程序版本
> - `service.namespace`：服务的逻辑分组
> - `host.name`：机器的主机名
> - 您组织的自定义属性（例如 `team`、`region`）

或者通过环境变量设置端点：

```bash
export TUFT_OTLP_ENDPOINT=http://localhost:4317
```

调试模式（打印到控制台导出器）：

```bash
export TUFT_OTEL_DEBUG=1
```

---

## 导出器和配置

TuFT 在 CLI 启动时初始化 OpenTelemetry 提供程序，并使用两种导出器模式之一：

- **控制台导出器**（如果 `TUFT_OTEL_DEBUG=1`）
- **OTLP gRPC 导出器**（否则）

端点解析顺序为：

1. 配置中的 `telemetry.otlp_endpoint`
2. `TUFT_OTLP_ENDPOINT` 环境变量
3. OpenTelemetry 导出器默认端点（如果两者都未提供）

TuFT 导出：

- 通过 `OTLPSpanExporter`（gRPC）导出追踪
- 通过 `OTLPMetricExporter`（gRPC）导出指标
- 通过 `OTLPLogExporter`（gRPC）导出日志

资源元数据：

- `service.name` 设置为 `telemetry.service_name`
- 可以通过 `telemetry.resource_attributes` 附加额外的资源属性

---

## TuFT 记录的内容

### 追踪

TuFT 在多个层级生成 span：

- **HTTP 服务器 span**：启用遥测时检测 FastAPI。
- **控制器 span**：训练/采样操作创建如下 span：
  - `training_controller.create_model`
  - `training_controller.run_forward`
  - `training_controller.run_forward_backward`
  - `training_controller.run_optim_step`
  - `training_controller.save_checkpoint`
  - `sampling_controller.create_sampling_session`
  - `sampling_controller.run_sample`
  - `future_store.execute_operation`
- **持久化 span**：Redis 操作创建如下 span：
  - `redis.SET`、`redis.GET`、`redis.DEL`、`redis.SCAN`
- **后端/actor span**（使用 Ray 运行时）：
  - 模型 actor 内的 span（如 HuggingFace 后端）通过传播的上下文链接
  - 示例包括 `hf_model.forward_backward`、`hf_model.optim_step`、`hf_model.save_state` 等。

TuFT 将关键标识符附加为 span 属性（参见[关联/共享键](#关联共享键)）。

### 指标

TuFT 发出两大类指标：

#### 1) 工作流指标（计数器/直方图）

示例（名称稳定，可用于仪表板/告警）：

- 训练：
  - `tuft.training.models.active`（上下计数器）
  - `tuft.training.tokens_per_second`（直方图）
  - `tuft.training.checkpoints.saved`（计数器）
  - `tuft.training.checkpoint.size_bytes`（直方图）
- 采样：
  - `tuft.sampling.sessions.active`（上下计数器）
  - `tuft.sampling.requests`（计数器）
  - `tuft.sampling.duration`（直方图，秒）
  - `tuft.sampling.tokens_per_second`（直方图）
  - `tuft.sampling.output_tokens`（直方图）
- Future：
  - `tuft.futures.queue_length`（上下计数器）
  - `tuft.futures.created`（计数器）
  - `tuft.futures.completed`（计数器）
  - `tuft.futures.wait_time`（直方图）
  - `tuft.futures.execution_time`（直方图）
- 持久化：
  - `tuft.redis.operation.duration`（直方图）

许多指标包含 `base_model`、`operation_type`、`model_id` 或 `queue_state` 等属性以支持过滤和分组。

#### 2) 资源指标（可观察的仪表）

启用遥测时，TuFT 启动后台资源收集器，导出可观察的仪表如：

- CPU：
  - `tuft.resource.cpu.utilization_percent`
- 内存：
  - `tuft.resource.memory.used_bytes`
  - `tuft.resource.memory.total_bytes`
  - `tuft.resource.memory.utilization_percent`
- GPU（如果 NVML 可用）：
  - `tuft.resource.gpu.utilization_percent`（带 `gpu_id` 属性）
  - `tuft.resource.gpu.memory_used_bytes`（带 `gpu_id` 属性）
  - `tuft.resource.gpu.memory_total_bytes`（带 `gpu_id` 属性）
- 进程：
  - `tuft.resource.process.memory_used_bytes`

### 日志

TuFT 将 Python 日志桥接到 OpenTelemetry：

- 日志记录通过配置的 OTel 日志提供程序导出
- 现有日志处理程序不会被删除（添加 OTel 处理程序）

---

## 关联/共享键

TuFT 是多租户和多客户端的。为了使追踪可搜索并跨组件可链接，TuFT 在 span 上设置一小组**高信号属性**。

您可以将这些视为可观测性后端中的"共享键"：

- `tuft.session_id`
  - 链接跨训练 + 采样活动的用户会话
- `tuft.training_run_id`
  - 链接特定训练运行（LoRA 适配器生命周期）的所有 span
- `tuft.sampling_session_id`
  - 链接特定采样会话的所有 span
- `tuft.request_id`
  - 将 FutureStore 执行 span 与异步 future 生命周期链接
- `tuft.operation_type`
  - 分类 future（如 `forward_backward`、`optim_step`、`sample`）
- （其他常见属性）
  - `tuft.base_model`
  - `tuft.lora_rank`
  - `tuft.backward`

### 实践中如何使用这些键

- 给定客户端的 **session_id**，按 `tuft.session_id = <id>` 过滤追踪。
- 给定 **training_run_id**，按 `tuft.training_run_id = <id>` 过滤。
- 给定 **sampling_session_id**，按 `tuft.sampling_session_id = <id>` 过滤。
- 给定异步 **request_id**（future），按 `tuft.request_id = <id>` 过滤。

这是追踪跨以下活动的推荐方式：

- 会话 → 训练客户端 → 训练控制器 → 后端执行
- 会话 → 采样客户端 → 采样控制器 → 后端执行
- API 调用 → future 入队 → future 执行 → 结果检索

---

## 上下文传播（Ray actor）

当 TuFT 在 Ray actor 内执行工作时，它显式传播 OpenTelemetry 上下文：

- 在控制器/后端边界，TuFT 将当前追踪上下文**注入**到可序列化的字典中。
- Ray actor 端**提取**该上下文并在该上下文下启动子 span。

这提供了跨进程边界的端到端追踪，即使 Ray 调度更改了执行位置。

---

## 开发者指南

### 创建 span

使用 `get_tracer()` 和标准 OpenTelemetry span API：

```python
from tuft.telemetry.tracing import get_tracer

tracer = get_tracer("tuft.my_module")
with tracer.start_as_current_span("my_operation") as span:
    span.set_attribute("tuft.session_id", session_id)
    span.set_attribute("tuft.training_run_id", training_run_id)
    # ... 您的逻辑 ...
```

### 记录指标

使用集中的指标注册表：

```python
from tuft.telemetry.metrics import get_metrics

metrics = get_metrics()
metrics.futures_created.add(1, {"operation_type": "forward_backward"})
metrics.futures_execution_time.record(0.123, {"operation_type": "forward_backward"})
```

### 选择属性

优先：

- 稳定标识符（`session_id`、`training_run_id`、`sampling_session_id`、`request_id`）
- 指标的低基数标签（如 `base_model`、`operation_type`、`queue_state`）

避免将用户提示词或原始有效负载放入 span 属性。

---

## 推荐的收集器配置

TuFT 默认通过 **gRPC** 导出 OTLP。我们推荐运行一个 OpenTelemetry Collector（或提供 OTLP 接收器的可观测性后端）监听配置的 `otlp_endpoint`（通常是 `4317`）。

至少确保：

- 收集器启用了 **OTLP 接收器**（gRPC）
- 追踪/指标/日志管道配置为接受 OTLP 并转发到您的后端（Jaeger/Tempo、Prometheus、SigNoz 等）

TuFT 特定的配置项：

- 将 `telemetry.service_name` 设置为稳定值（如 `tuft-prod`）
- 使用 `telemetry.resource_attributes` 标记部署（如环境、区域、版本）

---

## 故障排除

### 没有遥测显示

- 确保 `telemetry.enabled: true`。
- 确保 OTLP 端点可从 TuFT 服务器进程访问。
- 尝试调试模式以确认检测正常工作：

```bash
export TUFT_OTEL_DEBUG=1
```

### GPU 指标缺失

- GPU 仪表需要 NVML（`pynvml`）和工作的 NVML 环境。如果不可用，TuFT 将优雅地跳过 GPU 指标。
