# 安装指南

本指南介绍安装 TuFT 的不同方式。

## 快速安装

> **注意**：此脚本支持 Unix 平台。其他平台请参阅下面的手动安装部分。

使用单个命令安装 TuFT：

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/agentscope-ai/tuft/main/scripts/install.sh)"
```

这将安装带有完整后端支持（GPU 依赖、持久化、flash-attn）的 TuFT，以及捆绑的 Python 环境到 `~/.tuft`。安装后，重启终端并运行：

```bash
tuft
```

## 从源代码安装

我们推荐使用 [uv](https://github.com/astral-sh/uv) 进行依赖管理。

1. 克隆仓库：

    ```bash
    git clone https://github.com/agentscope-ai/TuFT
    ```

2. 创建虚拟环境：

    ```bash
    cd TuFT
    uv venv --python 3.12
    ```

3. 激活环境：

    ```bash
    source .venv/bin/activate
    ```

4. 安装依赖：

    ```bash
    # 安装最小依赖（非开发安装）
    uv sync

    # 如果需要开发或运行测试，安装开发依赖
    uv sync --extra dev

    # 如果要运行完整功能集（如模型服务、持久化），
    # 请安装所有依赖
    uv sync --all-extras
    python scripts/install_flash_attn.py
    # 如果 flash-attn 安装遇到问题，可以尝试手动安装：
    # uv pip install flash-attn --no-build-isolation
    ```

## 通过 PyPI 安装

您也可以直接从 PyPI 安装 TuFT：

```bash
uv pip install tuft

# 根据需要安装可选依赖
uv pip install "tuft[dev,backend,persistence]"
```

## 使用预构建的 Docker 镜像

如果本地安装遇到问题或想快速开始，可以使用预构建的 Docker 镜像。

1. 从 GitHub Container Registry 拉取最新镜像：

    ```bash
    docker pull ghcr.io/agentscope-ai/tuft:latest
    ```

2. 运行 Docker 容器并在端口 10610 启动 TuFT 服务器：

    ```bash
    docker run -it \
        --gpus all \
        --shm-size="128g" \
        --rm \
        -p 10610:10610 \
        -v <host_dir>:/data \
        ghcr.io/agentscope-ai/tuft:latest \
        tuft launch --port 10610 --config /data/tuft_config.yaml
    ```

    请将 `<host_dir>` 替换为您主机上用于存储模型检查点和其他数据的目录。
    
    假设您的主机上有以下结构：

    ```text
    <host_dir>/
        ├── checkpoints/
        ├── Qwen3-4B/
        ├── Qwen3-8B/
        └── tuft_config.yaml
    ```

## 运行服务器

CLI 启动一个 FastAPI 服务器：

```bash
tuft launch --port 10610 --config /path/to/tuft_config.yaml
```

配置文件 `tuft_config.yaml` 指定服务器设置，包括可用的基础模型、认证、持久化和遥测。以下是一个最小示例：

```yaml
supported_models:
  - model_name: Qwen/Qwen3-4B
    model_path: Qwen/Qwen3-4B
    max_model_len: 32768
    tensor_parallel_size: 1
  - model_name: Qwen/Qwen3-8B
    model_path: Qwen/Qwen3-8B
    max_model_len: 32768
    tensor_parallel_size: 1
```

请参阅仓库中的 `config/tuft_config.example.yaml` 获取包含所有可用选项的完整示例配置。
