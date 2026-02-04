# 持久化

TuFT 支持**可选的持久化**来保存服务器状态。启用后，TuFT 可以在服务器重启后恢复关键的运行时元数据（会话、训练运行、采样会话、future），然后**从磁盘上的检查点重建模型运行时状态**。

本文档分为两部分：

- **第一部分：用户指南** – 如何配置和使用持久化
- **第二部分：设计与内部实现** – 持久化的工作原理

---

## 目录

### 第一部分：用户指南

- [快速开始](#快速开始)
- [配置选项](#配置选项)
- [持久化后端](#持久化后端)
- [持久化的内容](#持久化的内容)
- [操作工作流](#操作工作流)
- [故障排除](#故障排除)

### 第二部分：设计与内部实现

- [目标和非目标](#目标和非目标)
- [Redis 键设计](#redis-键设计)
- [启动恢复语义](#启动恢复语义)
- [安全检查](#安全检查)

---

## 第一部分：用户指南

---

## 快速开始

### 安装可选依赖

```bash
uv pip install "tuft[persistence]"
```

### 启用持久化

在 `tuft_config.yaml` 中添加 `persistence` 部分：

```yaml
persistence:
  mode: REDIS
  redis_url: "redis://localhost:6379/0"
  namespace: "persistence-tuft-server"
```

基于文件的存储（演示/测试）：

```yaml
persistence:
  mode: FILE
  file_path: "~/.cache/tuft/file_redis.json"
  namespace: "persistence-tuft-server"
```

---

## 配置选项

持久化通过 `tuft_config.yaml` 配置文件中的 `persistence` 部分进行配置。可用的选项如下：

| 选项 | 类型 | 默认值 | 描述 |
|--------|------|---------|-------------|
| `mode` | string | `DISABLE` | 持久化模式：`DISABLE`、`REDIS` 或 `FILE` |
| `redis_url` | string | `redis://localhost:6379/0` | Redis 服务器 URL（仅当 `mode: REDIS` 时使用） |
| `file_path` | string | `~/.cache/tuft/file_redis.json` | JSON 文件路径（仅当 `mode: FILE` 时使用） |
| `namespace` | string | `persistence-tuft-server` | Redis 键的命名空间前缀 |
| `future_ttl_seconds` | integer 或 null | `86400`（1天） | future 记录的 TTL（秒）。设置为 `null` 表示永不过期。 |
| `check_fields` | list | `["SUPPORTED_MODELS"]` | 重启时验证的配置字段列表（参见[安全检查](#安全检查)） |

### 完整配置示例

```yaml
persistence:
  mode: REDIS
  redis_url: "redis://localhost:6379/0"
  namespace: "my-tuft-deployment"
  future_ttl_seconds: 86400  # 1 天
  check_fields:
    - SUPPORTED_MODELS
    - CHECKPOINT_DIR
```

---

## 持久化后端

TuFT 通过 `persistence.mode` 暴露三种模式：

- `DISABLE`：仅内存；重启时一切丢失。
- `REDIS`：通过 `redis-py` 使用外部 Redis（推荐用于生产）。
- `FILE`：基于文件的类 Redis 存储（用于演示/测试；使用 JSON 文件，不针对并发/性能优化）。

内部，所有记录都存储为 **JSON 序列化的 Pydantic 模型**，每个键一条记录。

---

## 持久化的内容

TuFT 在变更发生时增量持久化**主要服务器子系统的元数据**。具体包括以下子系统的状态：

- **会话**（`SessionManager`）
  - 会话元数据、标签、`user_id`、心跳时间戳
  - 存储为永久记录（无 TTL）

- **训练运行**（`TrainingController`）
  - 训练运行元数据（`training_run_id`、`base_model`、`lora_rank`、`model_owner`、`next_seq_id` 等）
  - **检查点记录（仅元数据）**（训练检查点和采样器检查点）存储在单独的键下
    （实际的检查点权重工件位于 `checkpoint_dir` 下的磁盘上，不在 Redis 中）
  - 存储为永久记录（无 TTL）

- **采样会话**（`SamplingController`）
  - 采样会话元数据 + **采样历史**（seq id + 提示词哈希）
  - 存储为永久记录（无 TTL）

- **Future**（`FutureStore`）
  - 请求生命周期记录：`pending` / `ready` / `failed`
  - 包括 `operation_type`、`operation_args`、`future_id`、payload 或 error
  - 带 **TTL** 存储（默认：1 天，可通过 `future_ttl_seconds` 配置）

- **配置签名**（`ConfigSignature`）
  - 选定 `AppConfig` 字段的快照，用于恢复安全

---

## 操作工作流

### 清除持久化状态

如果您有意更改了配置并想重新开始，清除持久化数据：

```bash
tuft clear persistence --config /path/to/tuft_config.yaml
```

这会删除配置的 `namespace` 下的键。它**不会**删除磁盘上的检查点文件。

### 安全更改配置

更改任何影响恢复安全的字段时的推荐工作流：

- 使用**新命名空间**部署，或
- 在重启前显式清除旧命名空间。

---

## 故障排除

### 启动失败并显示"配置不匹配"

- **原因**：您使用签名与同一命名空间中存储的签名不同的配置重启了 TuFT。
- **解决方法**：
  - 要么恢复配置更改，
  - 要么清除持久化状态（破坏性）并重启，
  - 要么为新部署切换到新的 `persistence.namespace`。

### 重启后某些结果标记为失败

如果这些 future 是在最新恢复的检查点**之后**创建的（或当不存在检查点时），这是预期的。从客户端重新运行这些操作。

### Redis 无限增长

长期记录（会话、训练运行、采样会话、检查点元数据）不会过期。Future 会按配置的 `future_ttl_seconds`（默认：1 天）过期。您还应该为每个部署设置命名空间并清除未使用的命名空间。

---

## 第二部分：设计与内部实现

---

## 目标和非目标

### 目标

- **崩溃/重启恢复**服务器状态元数据，使用户可以：
  - 重启后列出会话/训练运行/采样会话
  - 重启后检索已完成的 future（在 TTL 内）
  - 从每个训练运行的**最新检查点**继续训练
- **安全优先恢复**：防止服务器配置更改时的静默损坏。

### 非目标

- 持久化**不会**快照实时 GPU 内存/进行中的模型执行状态。
- TuFT **不会**在崩溃后重新运行待处理任务。待处理工作被视为不安全，必须重试。
- 持久化**不会**在 Redis 中存储模型权重 blob；权重工件位于 `checkpoint_dir` 下的磁盘上（可以通过 API 归档）。

---

## Redis 键设计

所有键都以可配置的 `namespace`（默认：`persistence-tuft-server`）为前缀，使用 `::` 作为分隔符：

- 顶级记录：
  `"{namespace}::{type}::{id}"`
- 嵌套记录：
  `"{namespace}::{type}::{parent_id}::{nested_type}::{nested_id}"`

> 注意：为避免歧义，部分中的任何字面 `::` 在内部被转义。

### 键族（高层）

使用默认命名空间，TuFT 使用这些主要键族：

- **会话**
  `persistence-tuft-server::session::{session_id}`

- **训练运行**
  `persistence-tuft-server::training_run::{training_run_id}`

- **训练检查点元数据**（嵌套在训练运行下）
  `persistence-tuft-server::training_run::{training_run_id}::ckpt::{checkpoint_id}`

- **采样器检查点元数据**（嵌套在训练运行下）
  `persistence-tuft-server::training_run::{training_run_id}::sampler_ckpt::{checkpoint_id}`

- **采样会话**
  `persistence-tuft-server::sampling_session::{sampling_session_id}`

- **Future**（基于 TTL）
  `persistence-tuft-server::future::{request_id}`

- **配置签名**
  `persistence-tuft-server::config_signature`

---

## 启动恢复语义

恢复有**一个预检步骤**加上**两个恢复阶段**：

### 阶段 0：配置验证（服务器启动前）

当启用持久化时，TuFT 在启动服务器**之前**验证当前的 `AppConfig` 与存储的配置签名。这旨在防止重启时使用新的不兼容配置静默解释旧状态。

如果检测到不匹配，TuFT 会中止启动并显示致命错误和差异。

### 阶段 1：从持久化后端恢复到内存（控制器构建）

在进程启动时，这些组件通过扫描其键前缀并反序列化记录来恢复其内存注册表：

- `SessionManager` 恢复会话。
- `TrainingController` 恢复训练运行 + 检查点记录。
  - 如果训练运行引用了当前配置中不存在的 `base_model`，则标记为**已损坏**。
- `SamplingController` 恢复采样会话。
  - 其 `base_model` 不再支持的采样会话从存储中删除。
- `FutureStore` 恢复 future。
  - 已完成的 future（`ready` / `failed`）立即标记为已完成。
  - 恢复的 future 还重建 `future_id` 分配状态以保持单调排序。

在阶段 1 结束时，TuFT 已恢复*元数据*，但模型运行时状态（GPU 内存中的适配器/权重）尚未重建。

### 阶段 2：基于检查点的恢复（异步初始化）

在控制器恢复后，TuFT 执行基于检查点的恢复：

- 对于每个未损坏且有可用后端的训练运行：
  - 加载磁盘上的**最新检查点**（并重建适配器状态）
  - 将该检查点视为服务器的恢复边界
- Future 与此边界对账：
  - 如果训练运行有一个有效的最新检查点，其 `future_id = F`，**所有该运行的 `future_id > F` 的 future 都标记为失败**（必须重试）
  - 如果不存在检查点，**该运行的所有 future 都标记为失败**

这意味着 TuFT 保证：

- **可以从最新检查点继续训练**，但
- 该检查点之后的任何操作都被认为不安全，需要重试。

> 重要：序列 ID 即使在重启后也保持**单调递增**。TuFT 设计上不会将 `next_seq_id` "倒带"到检查点边界。

---

## 安全检查

TuFT 包含重启安全检查，以防止配置在重启之间更改时的静默损坏。

### 配置签名验证（重启安全）

TuFT 存储从 `AppConfig.get_config_for_persistence()` 派生的 `ConfigSignature`（特别排除持久化配置本身）。

启动时，TuFT 比较选定的字段（默认：`SUPPORTED_MODELS`），并可配置为检查额外字段：

- `SUPPORTED_MODELS`（始终检查；强制）
- `CHECKPOINT_DIR`
- `MODEL_OWNER`
- `TOY_BACKEND_SEED`
- `AUTHORIZED_USERS`
- `TELEMETRY`

如果验证失败，启动中止并打印差异。这避免了旧状态引用不再配置的模型，或新部署意外指向旧命名空间的情况。
