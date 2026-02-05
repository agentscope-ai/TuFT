# Installation

This guide covers different ways to install TuFT.

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

You can also install TuFT directly from PyPI:

```bash
uv pip install tuft

# Install optional dependencies as needed
uv pip install "tuft[dev,backend,persistence]"
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
