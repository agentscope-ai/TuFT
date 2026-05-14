# 嵌入式模式

## 背景

TuFT 被设计为 RL 训练框架（如 Trinity）的**透明计算服务层**。在生产环境中，TuFT 通常作为独立守护进程运行（`tuft launch`），用户需要：

1. 编写 `tuft_config.yaml` 配置文件
2. 手动执行 `tuft launch --config ...` 启动服务
3. 设置 `TINKER_BASE_URL` 环境变量供客户端连接

这种手动配置带来了额外负担，尤其是：
- **RL 框架用户**：只想运行训练脚本，不想学习 TuFT 的安装和配置
- **开发调试**：需要快速迭代的工作流
- **CI 流水线**：需要可复现的自包含环境

**嵌入式模式**通过提供 `tuft.init()` API 解决了这个问题——类似 `ray.init()`——自动完成服务发现、配置生成、启动和连接。

## 两种运行模式

| | 守护进程模式 | 嵌入式模式 |
|---|---|---|
| 启动方式 | `tuft launch --config ...` | `tuft.init(model=...)` |
| 生命周期 | 独立进程，手动管理 | 跟随主进程，atexit 自动清理 |
| 适用场景 | 生产部署、多用户共享集群 | 开发调试、训练脚本、CI |
| 服务发现 | 用户手动设置 `TINKER_BASE_URL` | 自动（环境变量 → 地址文件 → 进程扫描 → 默认端口） |

**两种模式共存**：`tuft.init()` 首先尝试发现已有的守护进程服务。只有在找不到运行中的服务时，才会启动嵌入式实例。

## 快速开始

```python
import tuft

# 初始化 TuFT — 自动发现已有服务或启动一个新的
tuft.init(model="/path/to/Qwen2.5-0.5B-Instruct")

# 使用 service client 进行训练
training_client = tuft.create_training_client(
    base_model="Qwen2.5-0.5B-Instruct",
    rank=8,
)
# ... 你的训练循环 ...

# 可选：显式关闭（atexit 会自动处理）
tuft.shutdown()
```

### 其他 `init()` 模式

```python
# 连接到指定的运行中服务
tuft.init(address="http://gpu-cluster:10610")

# 使用已有配置文件
tuft.init(config="/path/to/tuft_config.yaml")

# 无参数 — 依赖环境变量或默认配置文件
tuft.init()

# 获取 service client（未初始化时自动触发 init）
service_client = tuft.get_service_client()
```

## 服务发现优先级

调用 `tuft.init()` 时，按以下顺序尝试发现已有服务：

1. `address=...` 参数显式传入
2. `TUFT_ADDRESS` 环境变量
3. 地址文件 `~/.tuft/tuft_current_server`
4. 进程扫描（查找运行中的 `tuft launch` 或 `uvicorn` 进程）
5. 默认端口探测：`http://127.0.0.1:10610`

如果未发现服务，嵌入式模式按以下优先级获取配置并启动：

1. `config=...` 参数显式传入
2. `TUFT_CONFIG` 环境变量
3. `model=...` 参数 → 自动生成最小配置
4. `TUFT_MODEL_PATH` 环境变量 → 自动生成最小配置
5. 默认配置文件：`~/.tuft/configs/tuft_config.yaml`
6. 全部没有 → 抛出 `RuntimeError` 并给出提示

## 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `TUFT_ADDRESS` | TuFT 服务地址 | — |
| `TUFT_API_KEY` | API 认证密钥 | 自动生成 |
| `TUFT_CONFIG` | 配置文件路径 | — |
| `TUFT_MODEL_PATH` | 模型路径（用于自动生成配置） | — |
| `TUFT_ENABLE_AUTO_CONNECT` | 启用 `get_service_client()` 自动连接 | `"1"` |
| `TUFT_HOME` | TuFT 主目录 | `~/.tuft` |
| `TUFT_HOST` | 服务绑定地址 | `127.0.0.1` |
| `TUFT_PORT` | 服务绑定端口 | `10610` |

## 生命周期

- **嵌入式服务**绑定到主进程。当 Python 进程退出（正常或信号）时，嵌入式 TuFT 服务通过 `atexit` 自动终止。
- **守护进程服务**（`tuft launch`）独立运行，持续到手动停止。
- `tuft.shutdown()` 可显式调用以提前停止嵌入式服务。
- `tuft.init()` 是**幂等的** — 多次调用安全（首次成功后为空操作）。

## 与 RL 框架集成

框架集成（如 Trinity）的模式：

```python
import tuft

# 在框架的初始化代码中：
tuft.init(model=model_path, ignore_reinit_error=True)
service_client = tuft.get_service_client()

# 像之前一样使用 service_client...
```

这不需要改变用户的工作流 — 框架透明地处理 TuFT 的配置和启动。
