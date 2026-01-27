"""Shared test helpers and constants for integration tests."""

from __future__ import annotations

import re
import socket
import threading
import time

import httpx
import tinker.types as types
import uvicorn
from tinker._exceptions import RequestFailedError

from tuft.config import AppConfig
from tuft.server import create_root_app

# Test data constants
PIG_LATIN_EXAMPLES = [
    {"input": "banana split", "output": "anana-bay plit-say"},
    {"input": "hello world", "output": "ello-hay orld-way"},
    {"input": "donut shop", "output": "onut-day op-shay"},
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
    for attempt in range(1, 121):
        try:
            response = client.get(f"{base_url}/api/v1/healthz", timeout=1)
            response.raise_for_status()
            healthy = True
            break
        except httpx.HTTPError:
            time.sleep(2)
        if attempt % 5 == 0:
            _log(f"Waiting for server healthz... attempt {attempt}/120")
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
