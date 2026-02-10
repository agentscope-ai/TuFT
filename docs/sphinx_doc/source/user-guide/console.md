# User console

TuFT provides a dashboard that allows users to view detailed information about their training runs and checkpoints. It also includes a sampling playground where users can experiment with fine-tuned models.

## Prerequisites

**1. Install uv**

We recommend using [uv](https://github.com/astral-sh/uv) for dependency management.

**2. Activate virtual environment**

If you're already in a uv virtual environment, you can skip this step.
```shell
uv venv --python 3.12
source .venv/bin/activate
```

**3. Install dependencies**

```shell
uv pip install fastapi gradio tinker pytz requests
```

## Quick start

The Sampling tab requires the console server to load a tokenizer from Hugging Face. If you cannot access Hugging Face directly, configure a mirror endpoint:

```shell
export HF_ENDPOINT=https://hf-mirror.com
```

After starting the TuFT server, run the following command to launch the user console::

--server-url: URL of the TuFT server

--gui-port: Port for the user console frontend

--backend-port: Port for the console backend

```shell
cd src/tuft/console/
bash scripts/start_user_console.sh --server-url http://localhost:10610 --gui-port 10613 --backend-port 10713
```

You can now access the user console at: http://0.0.0.0:10613

## Deploy with docker

TuFT also provides a Dockerfile for quick deployment.

By default, the Dockerfile sets the server URL to http://host.docker.internal:10610, which assumes the TuFT server is running on the same host as the Docker container.

The Dockerfile includes optional network configuration suggestions for environments with restricted internet access (e.g., using mirrors). Uncomment or enable them as needed.

Build and run the Docker image:
```shell
cd src/tuft/
docker build -t tuft/user-console -f console/docker/Dockerfile .
docker run -d --name user-console-app -p 10613:10613 --add-host=host.docker.internal:host-gateway tuft/user-console
```
