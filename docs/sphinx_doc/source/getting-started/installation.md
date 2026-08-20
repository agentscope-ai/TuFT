# Installation

This guide covers different ways to install TuFT.

```{admonition} 🚀 No GPU? No problem!
:class: tip

The instructions below assume a machine with a GPU. **No GPU?** You can deploy TuFT to a
pay-as-you-go cloud provider — **[Modal](../deployment/modal.md)** (serverless,
scale-to-zero) or **[Lambda Cloud](../deployment/lambda.md)** — and fine-tune from your
laptop. See the **[Deployment guides](../deployment/index.md)**.
```

## Quick Install

> **Note**: This script supports unix platforms. For other platforms, see the manual installation sections below.

Install TuFT with a single command:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/agentscope-ai/tuft/main/scripts/install.sh)"
```

This installs TuFT with full backend support (GPU dependencies, persistence, flash-attn) and a bundled Python environment to `~/.tuft`. After installation, restart your terminal and run:

```bash
tuft
```

### GPU wheel selection and installer options

By default (`--torch-backend auto`) the installer inspects the NVIDIA driver **before downloading anything**, selects the validated CUDA 13.0 wheel variant (`cu130`) for the pinned torch/vLLM stack, and runs import and CUDA smoke tests after installing. If the driver does not support CUDA 13.0, it fails with guidance instead of installing a broken environment. Pass a backend explicitly to override, e.g. when building an image on a machine without a GPU or using custom wheels:

```bash
# Explicit CUDA 13.0 wheels
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/agentscope-ai/tuft/main/scripts/install.sh)" -- --torch-backend cu130

# CPU-only environment
TUFT_TORCH_BACKEND=cpu /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/agentscope-ai/tuft/main/scripts/install.sh)"
```

`tuft upgrade` reuses the backend recorded at install time (in `$TUFT_HOME/torch-backend`), so upgrades resolve packages the same way; override with `tuft upgrade --torch-backend ...`. Use `--skip-gpu-checks` (or `TUFT_SKIP_GPU_CHECKS=1`) to turn GPU preflight/smoke-test failures into warnings.

The installer also honors these environment variables:

| Variable | Purpose |
| --- | --- |
| `TUFT_HOME` | Installation directory (default: `~/.tuft`) |
| `TUFT_VENV` | Virtual environment location (default: `$TUFT_HOME/venv`), e.g. to place it on faster or larger storage |
| `TUFT_TORCH_BACKEND` | Default value for `--torch-backend` (`auto`, `cpu`, or `cuNNN`) |
| `TUFT_PYPI_REQUIREMENT` | Override the default PyPI requirement |
| `UV_CACHE_DIR`, `UV_LINK_MODE`, `UV_SYSTEM_CERTS`, `UV_DEFAULT_INDEX`, `UV_INDEX` | Passed through to [uv](https://docs.astral.sh/uv/) for cache placement, link mode (e.g. `copy` across filesystems), system TLS trust stores, and package indexes/mirrors |

## Install from Source Code

We recommend using [uv](https://github.com/astral-sh/uv) for dependency management.

1. Clone the repository:

    ```bash
    git clone https://github.com/agentscope-ai/TuFT
    ```

2. Create a virtual environment:

    ```bash
    cd TuFT
    uv venv --python 3.12
    ```

3. Activate environment:

    ```bash
    source .venv/bin/activate
    ```

4. Install dependencies:

    ```bash
    # Install minimal dependencies for non-development installs
    uv sync

    # If you need to develop or run tests, install dev dependencies
    uv sync --extra dev

    # If you want to run the full feature set (e.g., model serving, persistence),
    # please install all dependencies
    uv sync --all-extras
    python scripts/install_flash_attn.py
    # If you face issues with flash-attn installation, you can try installing it manually:
    # uv pip install flash-attn --no-build-isolation
    ```

## Install via PyPI

```bash
uv pip install "tuft>=0.1.8"

# Install optional dependencies as needed
uv pip install "tuft[dev,backend,persistence]>=0.1.8"
```

## Use the Pre-built Docker Image

If you face issues with local installation or want to get started quickly, you can use the pre-built Docker image.

1. Pull the latest image from GitHub Container Registry:

    ```bash
    docker pull ghcr.io/agentscope-ai/tuft:latest
    ```

2. Run the Docker container and start the TuFT server on port 10610:

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

    Please replace `<host_dir>` with a directory on your host machine where you want to store model checkpoints and other data.
    
    Suppose you have the following structure on your host machine:

    ```text
    <host_dir>/
        ├── checkpoints/
        ├── Qwen3-4B/
        ├── Qwen3-8B/
        └── tuft_config.yaml
    ```

## Run the Server

The CLI starts a FastAPI server:

```bash
tuft launch --port 10610 --config /path/to/tuft_config.yaml
```

The config file `tuft_config.yaml` specifies server settings including available base models, authentication, persistence, and telemetry. Below is a minimal example:

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

See `config/tuft_config.example.yaml` in the repository for a complete example configuration with all available options.
