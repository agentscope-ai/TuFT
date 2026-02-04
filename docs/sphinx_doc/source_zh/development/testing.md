# 测试指南

TuFT 支持在 CPU 和 GPU 设备上运行测试。测试套件包括单元测试、集成测试和持久化测试。以下是编写和运行测试的说明。

## 快速开始

运行所有测试（CPU 模式，无 GPU 测试）：
```bash
uv run pytest
```

运行带详细输出和显示打印语句的测试：
```bash
uv run pytest -v -s
```

跳过集成测试：
```bash
uv run pytest -m "not integration"
```

## 测试配置

测试配置定义在 `pyproject.toml` 的 `[tool.pytest.ini_options]` 下：

- **测试路径**：`tests/`
- **测试文件模式**：`test_*.py`
- **异步模式**：通过 pytest-asyncio 自动配置
- **标记**：
  - `integration`：标记为集成测试（使用 `-m "not integration"` 跳过）
  - `gpu`：标记为需要 GPU（除非提供 `--gpu`，否则自动跳过）

## 在 CPU 上运行测试

在 CPU 设备上运行测试不需要特殊配置。只需运行 pytest：

```bash
uv run pytest tests -v -s
```

当未指定 `--gpu` 时，`conftest.py` 自动设置环境变量 `TUFT_CPU_TEST=1`，这会配置后端在测试期间使用 CPU 兼容的实现。带有 `@pytest.mark.gpu` 标记的测试会自动跳过。

### 跳过集成测试

集成测试可能运行时间较长。要跳过它们：

```bash
uv run pytest -m "not integration"
```

## 在 GPU 上运行测试

要编写在 GPU 设备上运行的测试，使用 `@pytest.mark.gpu` 装饰器：

```python
import pytest

@pytest.mark.gpu
def test_gpu_functionality():
    # 需要 GPU 的测试代码
    pass
```

要运行 GPU 测试，您需要：

1. 通过 `TUFT_TEST_MODEL` 环境变量设置模型路径
2. 启动 Ray 集群
3. 使用 `--gpu` 选项运行 pytest

```bash
export TUFT_TEST_MODEL=/path/to/your/model
ray start --head
uv run pytest tests -v -s --gpu
```

这将执行所有测试，包括带有 `@pytest.mark.gpu` 标记的测试，在 GPU 设备上运行。

## 持久化测试

TuFT 测试通过 Redis 或 FileRedis 支持持久化。默认情况下，所有测试都**启用**持久化：

- **外部 Redis**：如果设置了 `TEST_REDIS_URL` 环境变量且 Redis 可用，测试使用外部 Redis 服务器
- **FileRedis 回退**：如果 Redis 不可用，测试自动回退到 FileRedis（基于文件的存储），每个测试使用唯一的临时文件

### 使用外部 Redis 进行测试

要使用外部 Redis 服务器运行测试：

```bash
# 启动 Redis（使用 Docker 的示例）
docker run -d --name tuft-test-redis -p 6379:6379 redis:7-alpine

# 设置 Redis URL 并运行测试
export TEST_REDIS_URL=redis://localhost:6379/15
uv run pytest
```

测试套件默认使用数据库 15 以避免与其他 Redis 使用冲突。

### 在测试中禁用持久化

要完全禁用持久化：

```bash
uv run pytest --no-persistence
```

### 编写持久化测试

测试通过 `configure_persistence` fixture 自动配置持久化。对于需要显式持久化控制的测试，使用 `enable_persistence` fixture：

```python
def test_with_persistence(enable_persistence):
    # 此测试在启用持久化的情况下运行
    pass
```

## 持续集成

测试套件在 CI 中以不同配置运行：

### CPU 测试（checks.yml）

在每次 push 和 pull request 时运行：
- Python 版本：3.11、3.12、3.13
- Redis 服务用于持久化测试
- 所有代码检查和类型检查
- 仅 CPU 的 pytest 执行

```bash
uv run pytest
```

### GPU 测试（unittest.yml）

通过 pull request 上的 `/unittest` 评论触发（针对协作者/成员）：
- 在自托管的 GPU runner 上运行
- 使用 Docker Compose 进行多节点 Ray 设置
- 执行完整测试套件，包括 GPU 测试

```bash
uv run pytest tests -v -s --gpu --basetemp /mnt/checkpoints
```

## 测试标记和 Fixture

### 标记

- `@pytest.mark.gpu`：标记测试为需要 GPU 硬件
- `@pytest.mark.integration`：标记测试为集成测试

### Fixture

- `set_cpu_env`：（自动使用）在非 GPU 模式下设置 `TUFT_CPU_TEST=1`
- `configure_persistence`：（自动使用）为每个测试配置持久化
- `enable_persistence`：为特定测试显式启用持久化
- `clean_redis`：确保测试前后的 Redis 状态干净

## 编写测试

### 基本测试结构

```python
def test_example():
    # 您的测试代码
    assert True
```

### GPU 测试

```python
import pytest

@pytest.mark.gpu
def test_gpu_feature():
    # 此测试仅在带 --gpu 标志时运行
    pass
```

### 集成测试

```python
import pytest

@pytest.mark.integration
def test_integration_workflow():
    # 此测试使用 -m "not integration" 时跳过
    pass
```

### 异步测试

```python
import pytest

async def test_async_function():
    # pytest-asyncio 自动处理异步测试
    await some_async_function()
```

## 常用测试命令

```bash
# 运行所有测试
uv run pytest

# 带详细输出运行
uv run pytest -v

# 带 stdout/stderr 输出运行
uv run pytest -s

# 运行特定测试文件
uv run pytest tests/test_server.py

# 运行特定测试函数
uv run pytest tests/test_server.py::test_function_name

# 运行匹配模式的测试
uv run pytest -k "test_pattern"

# 跳过集成测试
uv run pytest -m "not integration"

# 仅运行集成测试
uv run pytest -m integration

# 运行 GPU 测试（需要 GPU 和模型）
export TUFT_TEST_MODEL=/path/to/model
ray start --head
uv run pytest tests -v -s --gpu

# 禁用持久化
uv run pytest --no-persistence

# 显示测试耗时
uv run pytest --durations=10
```
