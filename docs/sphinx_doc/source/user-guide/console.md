# Console

TuFT provides a dashboard for users to view the details of their training runs and checkpoints. It also includes a sampling playground for trying out fine-tuned models.

## Prerequisites

**Install uv**

We recommend using [uv](https://github.com/astral-sh/uv) for dependency management.

**Activate virtual environment**

If you are already in a uv environment, you can skip this step.

```bash
uv venv --python 3.12
source .venv/bin/activate
```

**Install the dependencies**

```bash
uv pip install fastapi gradio tinker pytz requests
```

## Quick Start

After you start the TuFT server, run the following command to start the user console:

- `--server-url`: the URL of the TuFT server
- `--gui-port`: the port of the user console
- `--backend-port`: the port of the console backend

The sampling tab requires the console server to load the tokenizer from Hugging Face. If you cannot access Hugging Face, set up a mirror to enable the sampling tab:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

Then start the console server and the console GUI:

```bash
cd src/tuft/console/
bash scripts/start_user_console.sh --server-url http://localhost:10610 --gui-port 10613 --backend-port 10713
```

You can access the user console at [http://0.0.0.0:10613](http://0.0.0.0:10613).

## Deploy with Docker

TuFT also provides a Dockerfile for quick deployment.

TuFT sets the server URL to `http://host.docker.internal:10610` in the Dockerfile, which requires the TuFT server to be running on the same host.

We also provide some suggestions in the Dockerfile for scenarios when the network is not available. Enable them if needed.

```bash
cd src/tuft/
docker build -t tuft/user-console -f console/docker/Dockerfile .
docker run -d --name user-console-app -p 10613:10613 --add-host=host.docker.internal:host-gateway tuft/user-console
```
