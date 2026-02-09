# 可观测性（OpenTelemetry）

TuFT 包含可选的 **OpenTelemetry** 集成，用于：

- **追踪**（跨 HTTP → 控制器 → Ray actor 的分布式追踪）
- **指标**（请求/工作流 + 资源指标）
- **日志**（Python 日志桥接到 OpenTelemetry 日志管道）

本文档介绍**如何启用 telemetry** 以及 **TuFT 内部的工作机制**（记录哪些数据、关联键、以及如何扩展采集）。

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
- [端到端示例](#端到端示例)

---

## 快速开始

在 `tuft_config.yaml` 中启用 telemetry：

```yaml
telemetry:
  enabled: true
  service_name: tuft
  otlp_endpoint: http://localhost:4317
  resource_attributes:
    deployment.environment: production
    service.version: 1.0.0
```

> **注意**：`resource_attributes` 允许您将自定义元数据附加到所有 telemetry 数据。常见属性包括：
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

- **HTTP 服务器 span**：启用 telemetry 后自动检测 FastAPI。
- **控制器 span**：训练/推理操作创建如下 span：
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
- 推理：
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

启用 telemetry 后，TuFT 会启动后台资源采集器，导出以下 observable gauge：

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
  - 链接跨训练与推理活动的用户会话
- `tuft.training_run_id`
  - 链接特定训练运行（LoRA 适配器生命周期）的所有 span
- `tuft.sampling_session_id`
  - 链接特定推理会话的所有 span
- `tuft.request_id`
  - 将 FutureStore 执行 span 与异步 future 生命周期链接
- `tuft.operation_type`
  - 分类 future（如 `forward_backward`、`optim_step`、`sample`）
- （其他常见属性）
  - `tuft.base_model`
  - `tuft.lora_rank`
  - `tuft.backward`

关于如何在可观测性后端中使用这些键进行过滤的实际示例，请参阅[端到端示例](#端到端示例)。

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

### Telemetry 数据未上报

- 确保 `telemetry.enabled: true`。
- 确保 OTLP 端点可从 TuFT 服务器进程访问。
- 尝试调试模式以确认检测正常工作：

```bash
export TUFT_OTEL_DEBUG=1
```

### GPU 指标缺失

- GPU 仪表需要 NVML（`pynvml`）和工作的 NVML 环境。如果不可用，TuFT 将优雅地跳过 GPU 指标。

---

## 端到端示例

本节演示如何将 TuFT 与可观测性后端配合使用。我们以 [SigNoz](https://signoz.io/) 为例，因为它在单一平台中集成了追踪、指标和日志的统一后端，便于同时展示三大可观测性支柱。

> **可替代的后端**：TuFT 导出标准 OTLP 数据，可与任何兼容 OpenTelemetry 的后端配合使用：
> - **追踪**：Jaeger、Zipkin、Tempo、Datadog 等
> - **指标**：Prometheus（通过 OTLP 或 Prometheus exporter）、Grafana Mimir 等
> - **日志**：Loki、Elasticsearch 等
>
> 只需将 `telemetry.otlp_endpoint` 指向您偏好的收集器或后端即可。

### 前置条件

- 已安装 [Docker](https://docs.docker.com/get-docker/) 和 [Docker Compose](https://docs.docker.com/compose/install/)

### 第一步：启动 SigNoz

推荐使用 Docker Compose 安装 SigNoz，示例如下。也可参考 [SigNoz 安装指南](https://signoz.io/docs/install/) 中提供的其他安装方式（安装脚本、Docker Swarm、Kubernetes 等）。

```bash
git clone -b main https://github.com/SigNoz/signoz.git && cd signoz/deploy/docker
docker compose up -d --remove-orphans
```

SigNoz 将在 `localhost:4317` 上启动 OTLP 接收器。

### 第二步：配置 TuFT

在 `tuft_config.yaml` 中添加以下 telemetry 配置：

```yaml
telemetry:
  enabled: true
  service_name: tuft
  otlp_endpoint: http://localhost:4317
  resource_attributes:
    deployment.environment: demo
```

### 第三步：启动 TuFT 并运行任务

```bash
tuft launch --config /path/to/tuft_config.yaml
```

通过 TuFT 客户端执行训练或推理操作，产生 telemetry 数据。

### 第四步：在 SigNoz 中查看

打开 SigNoz（`http://localhost:8080`）查看采集到的数据。

#### 追踪（Traces）

TuFT 会在每个 span 上附加结构化的关联键，便于在 Traces Explorer 中进行灵活过滤。

**通过 `tuft.session_id` 过滤** — 追踪以 session 为单位的所有任务。通过指定 `tuft.session_id` 进行过滤，可以看到该 session 下横跨训练和推理的所有 span，获得 session 维度的全景视图：

```{image} ../../_static/images/otel_examples/traces_filter_by_session_id.png
:alt: 按会话 ID 过滤的追踪
:width: 100%
```

**通过 `tuft.training_run_id` 过滤** — 追踪以某个训练客户端为单位的所有任务。通过 `tuft.training_run_id` 过滤可隔离出该训练客户端完整生命周期内的所有 span，如 `run_forward_backward`、`run_optim_step`、`save_checkpoint` 等训练操作及其执行顺序：

```{image} ../../_static/images/otel_examples/traces_filter_by_training_run_id.png
:alt: 按训练运行 ID 过滤的追踪
:width: 100%
```

**通过 `tuft.sampling_session_id` 过滤** — 追踪以某个推理客户端为单位的所有任务。通过 `tuft.sampling_session_id` 过滤可精确定位到单个推理会话内的 span，如 `sampling_controller.run_sample` 和 `sampling_controller.create_sampling_session`：

```{image} ../../_static/images/otel_examples/traces_filter_by_sampling_session_id.png
:alt: 按推理会话 ID 过滤的追踪
:width: 100%
```

**以 HTTP 请求为单位查看链路追踪** — 一次 HTTP 请求产生的所有 span 共享同一个 trace ID，通过该 trace ID 可查看此请求的完整执行路径。下方的 Flamegraph 视图展示了从 HTTP 接收、内部处理、状态持久化到后端执行的完整 span 树：

```{image} ../../_static/images/otel_examples/traces_http_request_spans.png
:alt: 单次 HTTP 请求的追踪链路
:width: 100%
```

#### 日志（Logs）

TuFT 发出的每条日志都自动关联了 **span ID** 和 **trace ID**。通过这两个标识符可精确定位该日志所属的操作步骤；反之，也可从某个 span 导航到其关联的所有日志：

```{image} ../../_static/images/otel_examples/logs_span_trace_correlation.png
:alt: 日志与 span ID 和 trace ID 的关联
:width: 100%
```

#### 指标（Metrics）

在 SigNoz 的 Metrics 页面可以浏览 TuFT 发出的所有指标，查看每个指标的类型、单位和样本数：

```{image} ../../_static/images/otel_examples/metrics_overview.png
:alt: SigNoz 中的指标概览
:width: 100%
```

也可以在 SigNoz Dashboard 中配置自定义 panel 来监控特定指标。以下以训练吞吐量和推理 QPS 为例，展示两个 panel 的配置方式；更多 panel 可根据实际需求自行配置。

**训练吞吐量（tokens/s）监控** — 通过 `tuft.training.tokens_per_second` 指标配置 panel，使用 P50 聚合并按 `base_model` 分组，实时监控训练效率：

```{image} ../../_static/images/otel_examples/dashboard_training_throughput.png
:alt: 训练吞吐量仪表盘面板
:width: 100%
```

**推理请求速率（QPS）监控** — 通过 `tuft.sampling.requests` 指标配置 panel，使用 Rate 聚合并按 `base_model` 分组，监控推理吞吐量与流量峰值：

```{image} ../../_static/images/otel_examples/dashboard_sampling_qps.png
:alt: 推理吞吐量仪表盘面板
:width: 100%
```
