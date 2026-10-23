# 安装指南

本指南介绍安装 TuFT 的不同方式。

```{admonition} 🚀 没有 GPU？没问题！
:class: tip

下面的说明假设你的机器配有 GPU。**没有 GPU？** 你可以将 TuFT 部署到按量付费的云服务商——
**[Modal](../deployment/modal.md)**（无服务器，缩容至零）、
**[Lambda Cloud](../deployment/lambda.md)** 或
**[Hugging Face Jobs](../deployment/hf_jobs.md)**——然后在本地（笔记本）驱动微调。
参阅 **[部署指南](../deployment/index.md)**。
```

## 快速安装

> **注意**：此脚本支持 Unix 平台。其他平台请参阅下面的手动安装部分。

使用单个命令安装 TuFT：

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/agentscope-ai/tuft/main/scripts/install.sh)"
```

这将安装带有完整后端支持（GPU 依赖、持久化、FlashAttention-2）的 TuFT，以及捆绑的 Python 环境到 `~/.tuft`。安装后，重启终端并运行：

```bash
tuft
```

### GPU wheel 变体选择与安装器选项

默认情况下（`--torch-backend auto`），安装器会**在下载任何内容之前**检测 NVIDIA 驱动，为固定版本的 torch/vLLM 依赖栈选择已验证的 CUDA 13.0 wheel 变体（`cu130`），并在安装完成后运行导入检查和 CUDA 冒烟测试。如果驱动不支持 CUDA 13.0，安装器会给出明确的指引并失败，而不是装出一个无法运行的环境。也可以显式指定后端，例如在没有 GPU 的机器上构建镜像或使用自定义 wheel 时：

```bash
# 显式使用 CUDA 13.0 wheels
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/agentscope-ai/tuft/main/scripts/install.sh)" -- --torch-backend cu130

# 仅 CPU 环境
TUFT_TORCH_BACKEND=cpu /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/agentscope-ai/tuft/main/scripts/install.sh)"
```

`tuft upgrade` 会复用安装时记录的后端（保存在 `$TUFT_HOME/torch-backend`），使升级与安装以相同方式解析依赖；可通过 `tuft upgrade --torch-backend ...` 覆盖。使用 `--skip-gpu-checks`（或 `TUFT_SKIP_GPU_CHECKS=1`）可将 GPU 预检 / 冒烟测试失败降级为警告。

安装器还支持以下环境变量：

| 变量 | 用途 |
| --- | --- |
| `TUFT_HOME` | 安装目录（默认：`~/.tuft`） |
| `TUFT_VENV` | 虚拟环境位置（默认：`$TUFT_HOME/venv`），例如放到更快或更大的存储上 |
| `TUFT_TORCH_BACKEND` | `--torch-backend` 的默认值（`auto`、`cpu` 或 `cuNNN`） |
| `TUFT_KERNEL_SOURCE` | `--kernel-source` 的默认值（`huggingface` 或 `modelscope`） |
| `TUFT_PYPI_REQUIREMENT` | 覆盖默认的 PyPI 依赖声明 |
| `UV_CACHE_DIR`、`UV_LINK_MODE`、`UV_SYSTEM_CERTS`、`UV_DEFAULT_INDEX`、`UV_INDEX` | 透传给 [uv](https://docs.astral.sh/uv/)，用于缓存位置、链接模式（如跨文件系统时用 `copy`）、系统 TLS 证书以及包索引 / 镜像 |

### FlashAttention-2

设置了 `attn_implementation: flash_attention_2` 的模型在安装了 `flash_attn` 包时直接使用它；否则 TuFT 通过 [`kernels`](https://github.com/huggingface/kernels) 包从 Hugging Face kernel hub 加载 FlashAttention-2（[`kernels-community/flash-attn2`](https://huggingface.co/kernels-community/flash-attn2)，固定到已验证的提交）。`kernels` 会选择与已安装 torch/CUDA 匹配的构建（`cu129` 的 torch 使用 `cu128` 构建），因此无需为精确的 torch/CUDA/Python 组合准备 wheel。安装器会预先下载该 kernel 并运行前向/反向冒烟测试，结束时的摘要会显示 FlashAttention-2 是否可用。下载的副本会作为 `LOCAL_KERNELS` 覆盖项记录在 `$TUFT_HOME/local-kernels` 中，`tuft` 命令会自动导出该变量，因此服务启动时无需访问 Hub。如果以其他方式启动 TuFT（例如 `python -m tuft launch`），请自行导出：`export LOCAL_KERNELS="$(cat ~/.tuft/local-kernels)"`，否则会在启动时从 Hub 解析该 kernel。若当前环境没有可用的 kernel 构建，请求 `flash_attention_2` 的模型会在服务启动时被拒绝，而不是在第一次训练请求时才失败；此时请使用 `attn_implementation: sdpa`。

如果无法访问 Hugging Face Hub，可以从 ModelScope 镜像获取 kernel：

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/agentscope-ai/tuft/main/scripts/install.sh)" -- --kernel-source modelscope
```

该镜像目前落后于 Hub：它包含 Hub `main` 分支上 torch 2.12 及以下的按 torch 版本构建，但没有 Hub 源所用的 version 3 构建；两者在 TuFT 使用的 torch 2.11 上都能通过安装器的冒烟测试。ModelScope 上的构建在使用前会按 Hugging Face 的签名和文件摘要进行校验，并保存在 `$TUFT_HOME/kernels` 下。之后如需重新获取 kernel，可使用 TuFT 环境中的 Python 运行 `python scripts/install_flash_attn.py [--source modelscope]`。

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
    # 获取 FlashAttention-2 kernel（添加 --source modelscope 可使用 ModelScope 镜像）
    python scripts/install_flash_attn.py
    ```

## 通过 PyPI 安装

```bash
uv pip install "tuft>=0.1.8"

# 根据需要安装可选依赖
uv pip install "tuft[dev,backend,persistence]>=0.1.8"
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
