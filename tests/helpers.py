"""Shared test helpers and constants for integration tests."""

from __future__ import annotations

import os
import re
import socket
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator

import httpx
import pytest
import ray
import tinker.types as types
import uvicorn
from tinker._exceptions import RequestFailedError

from tuft.config import AppConfig, ModelConfig
from tuft.server import create_root_app


# Test data constants
PIG_LATIN_EXAMPLES = [
    {"input": "banana split", "output": "anana-bay plit-say"},
    {"input": "hello world", "output": "ello-hay orld-way"},
    {"input": "donut shop", "output": "onut-day op-shay"},
]

PIG_LATIN_EXAMPLES_EXTENDED = [
    {"input": "banana split", "output": "anana-bay plit-say"},
    {"input": "quantum physics", "output": "uantum-qay ysics-phay"},
    {"input": "donut shop", "output": "onut-day op-shay"},
    {"input": "pickle jar", "output": "ickle-pay ar-jay"},
    {"input": "space exploration", "output": "ace-spay exploration-way"},
    {"input": "rubber duck", "output": "ubber-ray uck-day"},
    {"input": "coding wizard", "output": "oding-cay izard-way"},
]

TEST_PROMPTS = [
    "English: banana split\nPig Latin:",
    "English: hello world\nPig Latin:",
    "English: donut shop\nPig Latin:",
]

REVERSE_EXAMPLES = [
    {"input": "banana split", "output": "ananab tilps"},
    {"input": "hello world", "output": "olleh dlrow"},
    {"input": "donut shop", "output": "tunod pohs"},
    {"input": "deep learning", "output": "peed gninrael"},
    {"input": "paper plane", "output": "repap enalp"},
]

REVERSE_PROMPTS = [
    "Reverse each word.\nEnglish: banana split\nReversed:",
    "Reverse each word.\nEnglish: hello world\nReversed:",
    "Reverse each word.\nEnglish: donut shop\nReversed:",
    "Reverse each word.\nEnglish: deep learning\nReversed:",
    "Reverse each word.\nEnglish: paper plane\nReversed:",
]


def _find_free_port() -> int:
    """Find a free port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _normalize_text(text: str) -> str:
    """Normalize text by collapsing whitespace."""
    return " ".join(text.strip().split())


def _log(message: str) -> None:
    """Print a log message with flush."""
    print(message, flush=True)


def _run_with_seq_sync(training_client, action):
    """Run an action with sequence ID synchronization.

    Retries the action if a sequence conflict occurs, adjusting the client's
    request ID counter to match the server's expected value.
    """
    while True:
        try:
            return action()
        except RequestFailedError as exc:
            match = re.search(r"Sequence conflict: expected (\d+), got (\d+)\.", str(exc))
            if not match:
                raise
            expected = int(match.group(1))
            training_client._request_id_counter = expected - 1


def _start_server(
    config: AppConfig, port: int
) -> tuple[uvicorn.Server, threading.Thread, str, httpx.Client]:
    """Start a test server and wait for it to be healthy.

    Returns:
        A tuple of (server, thread, base_url, client).
    """
    app = create_root_app(config)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    base_url = f"http://127.0.0.1:{port}"
    client = httpx.Client()
    healthy = False
    for attempt in range(1, 301):
        try:
            response = client.get(f"{base_url}/api/v1/healthz", timeout=1)
            response.raise_for_status()
            healthy = True
            break
        except httpx.HTTPError:
            time.sleep(2)
        if attempt % 5 == 0:
            _log(f"Waiting for server healthz... attempt {attempt}/300")
    if not healthy:
        server.should_exit = True
        thread.join(timeout=5)
        client.close()
        raise RuntimeError("Server failed to start")
    _log("Server is healthy")
    return server, thread, base_url, client


def _stop_server(server: uvicorn.Server, thread: threading.Thread, client: httpx.Client) -> None:
    """Stop a test server and close its client."""
    server.should_exit = True
    thread.join(timeout=5)
    client.close()


def _create_training_data(tokenizer) -> list[types.Datum]:
    """Create training data from PIG_LATIN_EXAMPLES."""
    data: list[types.Datum] = []
    for example in PIG_LATIN_EXAMPLES:
        prompt = f"English: {example['input']}\nPig Latin:"
        completion = f" {example['output']}\n"
        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
        completion_tokens = tokenizer.encode(completion, add_special_tokens=False)

        tokens = prompt_tokens + completion_tokens
        input_tokens = tokens[:-1]
        target_tokens = tokens[1:]
        weights = [0.0] * (len(prompt_tokens) - 1) + [1.0] * len(completion_tokens)

        datum = types.Datum(
            model_input=types.ModelInput.from_ints(input_tokens),
            loss_fn_inputs={
                "target_tokens": types.TensorData(
                    data=target_tokens,
                    dtype="int64",
                    shape=[len(target_tokens)],
                ),
                "weights": types.TensorData(
                    data=weights,
                    dtype="float32",
                    shape=[len(weights)],
                ),
            },
        )
        data.append(datum)
    return data


def _create_reverse_training_data(tokenizer) -> list[types.Datum]:
    """Create training data from REVERSE_EXAMPLES."""
    data: list[types.Datum] = []
    for example in REVERSE_EXAMPLES:
        prompt = f"Reverse each word.\nEnglish: {example['input']}\nReversed:"
        completion = f" {example['output']}\n"
        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
        completion_tokens = tokenizer.encode(completion, add_special_tokens=False)

        tokens = prompt_tokens + completion_tokens
        input_tokens = tokens[:-1]
        target_tokens = tokens[1:]
        weights = [0.0] * (len(prompt_tokens) - 1) + [1.0] * len(completion_tokens)

        datum = types.Datum(
            model_input=types.ModelInput.from_ints(input_tokens),
            loss_fn_inputs={
                "target_tokens": types.TensorData(
                    data=target_tokens,
                    dtype="int64",
                    shape=[len(target_tokens)],
                ),
                "weights": types.TensorData(
                    data=weights,
                    dtype="float32",
                    shape=[len(weights)],
                ),
            },
        )
        data.append(datum)
    return data


def clear_ray_state() -> None:
    """Clear Ray state to avoid resource leak between tests."""
    import gc
    import signal

    import psutil

    ray.shutdown(_exiting_interpreter=True)
    gc.collect()

    if os.environ.get("TUFT_DOCKER_UNITTEST") == "1":
        # check gpu memory and kill stray vllm processes only in
        # docker unittest environment
        for proc in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                cmdline = proc.info["cmdline"]
                if cmdline and "VLLM" in " ".join(cmdline):
                    os.kill(proc.info["pid"], signal.SIGKILL)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue


# -----------------------------------------------------------------------------
# Server fixture helpers
# -----------------------------------------------------------------------------


@dataclass
class ServerFixtureConfig:
    """Configuration for server fixture creation.

    Attributes:
        model_configs: List of ModelConfig for supported models.
        checkpoint_subdir: Subdirectory name for checkpoints (under tmp_path).
        ray_env_vars: Additional environment variables for Ray runtime_env.
        ray_excludes: File patterns to exclude from Ray working_dir packaging.
    """

    model_configs: list[ModelConfig]
    checkpoint_subdir: str = "checkpoints"
    ray_env_vars: dict[str, str] = field(default_factory=dict)
    ray_excludes: list[str] = field(default_factory=list)


def _create_server_endpoint(
    tmp_path_factory: pytest.TempPathFactory,
    config: ServerFixtureConfig,
) -> Generator[str, None, None]:
    """Create a test server endpoint with the given configuration.

    This is the common logic for all server fixtures:
    1. Clear Ray state
    2. Setup distributed environment variables
    3. Initialize Ray with runtime_env
    4. Create AppConfig and start server
    5. Yield base_url
    6. Cleanup (stop server, clear Ray state)

    Args:
        tmp_path_factory: pytest fixture for creating temporary directories.
        config: ServerFixtureConfig with model configs and Ray settings.

    Yields:
        base_url: The server URL string (e.g., "http://127.0.0.1:12345").
    """
    clear_ray_state()

    # Setup distributed environment variables
    os.environ.setdefault("MASTER_ADDR", "localhost")
    master_port = _find_free_port()
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("RANK", "0")

    # Build Ray runtime_env
    runtime_env: dict = {"env_vars": {"TRANSFORMERS_NO_TORCHVISION": "1"}}
    if config.ray_env_vars:
        runtime_env["env_vars"].update(config.ray_env_vars)
    if config.ray_excludes:
        runtime_env["excludes"] = config.ray_excludes

    _log("Starting Ray...")
    ray.init(ignore_reinit_error=True, runtime_env=runtime_env)

    # Create AppConfig
    checkpoint_dir = tmp_path_factory.mktemp(config.checkpoint_subdir)
    _log(f"Checkpoint dir: {checkpoint_dir}")
    app_config = AppConfig(checkpoint_dir=Path(checkpoint_dir))
    app_config.supported_models = config.model_configs
    app_config.authorized_users = {"tml-test-key": "default"}

    # Update fsdp_master_port in model configs if using FSDP
    for model_config in app_config.supported_models:
        if model_config.training_backend == "fsdp":
            model_config.fsdp_master_port = master_port

    # Start server
    port = _find_free_port()
    _log(f"Starting server on port {port}...")
    server, thread, base_url, client = _start_server(app_config, port)

    yield base_url

    _stop_server(server, thread, client)
    clear_ray_state()
