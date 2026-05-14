# Embedded Mode

## Background

TuFT is designed to serve as a **transparent compute service layer** for RL training frameworks like Trinity and veRL. In production, TuFT typically runs as a standalone daemon (`tuft launch`), and users must:

1. Write a `tuft_config.yaml` configuration file
2. Manually start the server with `tuft launch --config ...`
3. Set the `TINKER_BASE_URL` environment variable for clients to connect

This manual setup creates friction, especially for:
- **RL framework users** who just want to run training scripts without learning TuFT internals
- **Development/debugging** workflows where quick iteration is key
- **CI pipelines** that need reproducible, self-contained environments

**Embedded mode** solves this by providing a `tuft.init()` API — similar to `ray.init()` — that handles service discovery, configuration generation, startup, and connection automatically.

## Two Modes of Operation

| | Daemon Mode | Embedded Mode |
|---|---|---|
| How to start | `tuft launch --config ...` | `tuft.init(model=...)` |
| Lifecycle | Independent process, manually managed | Follows main process, auto-cleanup via atexit |
| Best for | Production deployments, multi-user shared clusters | Dev/debug, training scripts, CI |
| Service discovery | User sets `TINKER_BASE_URL` manually | Automatic (env var → address file → process scan → default port) |

**Both modes coexist**: `tuft.init()` first tries to discover an existing daemon. Only when no running service is found does it start an embedded instance.

## Quick Start

```python
import tuft

# Initialize TuFT — auto-discovers existing service or starts one
tuft.init(model="/path/to/Qwen2.5-0.5B-Instruct")

# Use the service client for training
training_client = tuft.create_training_client(
    base_model="Qwen2.5-0.5B-Instruct",
    rank=8,
)
# ... your training loop ...

# Optional: explicit shutdown (atexit handles this automatically)
tuft.shutdown()
```

### Other `init()` patterns

```python
# Connect to a specific running server
tuft.init(address="http://gpu-cluster:10610")

# Use an existing config file
tuft.init(config="/path/to/tuft_config.yaml")

# No arguments — relies on env vars or default config file
tuft.init()

# Get a service client (auto-inits if not already done)
service_client = tuft.get_service_client()
```

## Service Discovery Priority

When `tuft.init()` is called, it tries to find an existing service in this order:

1. `address=...` argument passed to `init()`
2. `TUFT_ADDRESS` environment variable
3. Address file at `~/.tuft/tuft_current_server`
4. Process scan (looks for running `tuft launch` or `uvicorn` processes)
5. Default port probe: `http://127.0.0.1:10610`

If no service is found, embedded mode starts a new one using configuration from:

1. `config=...` argument passed to `init()`
2. `TUFT_CONFIG` environment variable
3. `model=...` argument → auto-generates minimal config
4. `TUFT_MODEL_PATH` environment variable → auto-generates minimal config
5. Default config file: `~/.tuft/configs/tuft_config.yaml`
6. None available → raises `RuntimeError` with helpful guidance

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `TUFT_ADDRESS` | Address of running TuFT service | — |
| `TUFT_API_KEY` | API authentication key | Auto-generated |
| `TUFT_CONFIG` | Path to configuration file | — |
| `TUFT_MODEL_PATH` | Model path for auto-config generation | — |
| `TUFT_ENABLE_AUTO_CONNECT` | Enable auto-connect in `get_service_client()` | `"1"` |
| `TUFT_HOME` | TuFT home directory | `~/.tuft` |
| `TUFT_HOST` | Server bind address | `127.0.0.1` |
| `TUFT_PORT` | Server bind port | `10610` |

## Lifecycle

- **Embedded services** are tied to the main process. When the Python process exits (normally or via signal), the embedded TuFT server is automatically terminated via `atexit`.
- **Daemon services** (`tuft launch`) are independent and persist until manually stopped.
- `tuft.shutdown()` can be called explicitly to stop an embedded service early.
- `tuft.init()` is **idempotent** — calling it multiple times is safe (no-op after first success).

## Integration with RL Frameworks

For framework integrations (e.g., Trinity), the pattern is:

```python
import tuft

# In your framework's initialization code:
tuft.init(model=model_path, ignore_reinit_error=True)
service_client = tuft.get_service_client()

# Use service_client as before...
```

This requires no changes to the user's workflow — the framework handles TuFT setup transparently.
