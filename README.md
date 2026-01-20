# TuFT

TuFT( **T**enant-**u**nified **F**ine**T**uning) is a framework that provides a unified
API for finetuning large language models (LLMs) across multiple tenants.
Users can access TuFT via compatible clients such as [Tinker](https://github.com/thinking-machine-lab/tinker).

## Installation

We recommend using [uv](https://github.com/astral-sh/uv) for dependency management.

### Install from Source Code

1. Clone the repository:

    ```bash
    git clone https://github.com/agentscope-ai/TuFT
    ```

2. Create a virtual environment:

    ```bash
    cd TuFT
    uv venv --python 3.12
    ```

3. Install dependencies:

    ```bash
    # Install minimal dependencies for non-development installs
    uv sync

    # If you need to develop or run tests, install dev dependencies
    uv sync --extra dev

    # If you want to run the full feature set (e.g., model serving, persistence),
    # please install all dependencies
    uv sync --all-extras
    uv pip install flash-attn --no-build-isolation
    ```

4. Activate environment:

    ```bash
    source .venv/bin/activate
    ```

### Install via PyPI

You can also install TuFT directly from PyPI:

```bash
uv pip install tuft

# Install optional dependencies as needed
uv pip install "tuft[dev,backend,persistence]"
```

### Run the server

The CLI starts a FastAPI server:

```bash
tuft --port 8080 --checkpoint-dir ~/.cache/tuft/checkpoints --model-config models.yaml
```

The config file `models.yaml` specifies available base models. Below is an example.

```yaml
supported_models:
  - model_name: Qwen/Qwen3-8B
    model_path: Qwen/Qwen3-8B
    max_model_len: 32768
    tensor_parallel_size: 1
  - model_name: Qwen/Qwen3-32B
    model_path: Qwen/Qwen3-32B
    max_model_len: 32768
    tensor_parallel_size: 2
```

## Use the Pre-built Docker Image

If you face issues with local installation or want to get started quickly,
you can use the pre-built Docker image.

1. Pull the latest image from GitHub Container Registry:

    ```bash
    docker pull ghcr.io/agentscope-ai/tuft:latest
    ```

2. Run the Docker container with GPU support and necessary volume mounts:

    ```bash
    docker run -it \
        --gpus all \
        --shm-size="128g" \
        --rm \
        -p 8080:8080 \
        -v <host_dir>:/data \
        ghcr.io/agentscope-ai/tuft:latest \
        tuft --port 8080 --checkpoint-dir /data/checkpoints --model-config /data/models.yaml
    ```

    Replace `<host_dir>` with a directory on your host machine where you want to store model checkpoints and other data.
    Suppose you have the following structure on your host machine:

    ```plaintext
    <host_dir>/
        ├── checkpoints/
        ├── Qwen3-8B/
        ├── Qwen3-32B/
        └── models.yaml
    ```

    The `models.yaml` file should define the models available to TuFT, for example:
    ```yaml
    supported_models:
      - model_name: Qwen/Qwen3-8B
        model_path: /data/Qwen3-8B
        max_model_len: 32768
        tensor_parallel_size: 1
      - model_name: Qwen/Qwen3-32B
        model_path: /data/Qwen3-32B
        max_model_len: 32768
        tensor_parallel_size: 2
    ```


## End-to-end example

With the server running on port 8080, the following script exercises the main API surface using the
bundled Tinker SDK. The tokenizer calls are commented out so the snippet works offline; instead we use
fake token IDs to drive the toy backend.

```python
import tinker
from tinker import types

# Connect to the running tuft server via the bundled SDK
client = tinker.ServiceClient(base_url="http://localhost:8080", api_key="local-dev-key")

# Discover available base models before launching a training run
capabilities = client.get_server_capabilities()
base_model = capabilities.supported_models[0].model_name

print("Supported models:")
for model in capabilities.supported_models:
    print("-", model.model_name or "(unknown)")

# Start a LoRA training client targeting the first supported model
training = client.create_lora_training_client(base_model=base_model, rank=8)

# tokenizer = training.get_tokenizer()
# prompt_tokens = tokenizer.encode("Hello from TuFT")
# target_tokens = tokenizer.encode(" Generalizing beyond the prompt")
prompt_tokens = [101, 42, 37, 102]
target_tokens = [101, 99, 73, 102]

datum = types.Datum(
    model_input=types.ModelInput.from_ints(prompt_tokens),
    loss_fn_inputs={
        "target_tokens": types.TensorData(data=target_tokens, dtype="int64", shape=[len(target_tokens)])
    },
)

# Run a single forward/backward pass and observe the reported metrics
fwdbwd = training.forward_backward([datum], "cross_entropy").result(timeout=30)
print("Loss metrics:", fwdbwd.metrics)

# Apply an optimizer update to mutate the toy weights
optim = training.optim_step(types.AdamParams(learning_rate=1e-4)).result(timeout=30)
print("Optimizer metrics:", optim.metrics)

# Persist both checkpoint and sampler artifacts for later reuse
checkpoint = training.save_state("demo-checkpoint").result(timeout=60)
sampler_weights = training.save_weights_for_sampler("demo-sampler").result(timeout=60)

# Inspect the server session via the REST client for debugging
rest = client.create_rest_client()
session_id = client.holder.get_session_id()
session_info = rest.get_session(session_id).result(timeout=30)
print("Session contains training runs:", session_info.training_run_ids)

# Spin up a sampler tied to the saved weights and generate tokens
sampling = client.create_sampling_client(model_path=sampler_weights.path)
# sample_prompt = tokenizer.encode("Tell me something inspiring.")
sample_prompt = [101, 57, 12, 7, 102]
sample = sampling.sample(
    prompt=types.ModelInput.from_ints(sample_prompt),
    num_samples=1,
    sampling_params=types.SamplingParams(max_tokens=5, temperature=0.5),
).result(timeout=30)

if sample.sequences:
    print("Sample tokens:", sample.sequences[0].tokens)

print("Checkpoint saved to:", checkpoint.path)
print("Sampler weights saved to:", sampler_weights.path)
```

Adjust the fake token IDs with your own prompts once you have a tokenizer locally.

## Persistence

TuFT supports optional Redis-based persistence for server state. When enabled,
the server can recover sessions, training runs, and pending futures after a restart.

To use persistence, install the optional dependency:

```bash
uv pip install tuft[persistence]
```

### Persistence Modes

TuFT provides two persistence modes:

| Mode | Description | Use Case |
|------|-------------|----------|
| `disabled` | No persistence, data in-memory only | Development, testing without state recovery |
| `redis_url` | External Redis server | Production, multi-instance deployments |

### Configuration

#### Mode 1: Disabled (Default)

No configuration needed. All data is stored in memory and lost on restart.

```yaml
persistence:
  mode: disabled
```

#### Mode 2: External Redis Server

Use an external Redis server for production deployments:

```yaml
persistence:
  mode: redis_url
  redis_url: "redis://localhost:6379/0"
  namespace: "tuft"
```

You can start a local Redis instance using Docker:

```bash
docker run -d --name TuFT-redis -p 6379:6379 redis:7-alpine
```

### Python API

You can also configure persistence programmatically:

```python
from tuft.persistence import PersistenceConfig

# Disabled (no persistence)
config = PersistenceConfig.disabled()

# External Redis server
config = PersistenceConfig.from_redis_url("redis://localhost:6379/0")
```

## Development

- Design docs are located in [`docs`](./docs/).
- Please install `pre-commit` and ensure test passes before creating new PRs.