"""Tests for OpenAI-compatible API endpoints."""

from __future__ import annotations

import json
import os
from collections.abc import Generator
from pathlib import Path
from typing import NamedTuple

import httpx
import pytest
import ray
from tinker import types
from tinker.lib.public_interfaces.service_client import ServiceClient

from tuft.config import AppConfig, ModelConfig

from .helpers import _find_free_port, _log, _start_server, _stop_server, clear_ray_state


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class OAITestEnv(NamedTuple):
    """Holds the running test server info."""

    url: str
    model_name: str


def _collect_sse_chunks(resp: httpx.Response) -> tuple[list[dict], bool]:
    """Collect all SSE data chunks from a streaming response."""
    chunks: list[dict] = []
    done = False
    for line in resp.iter_lines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("data: "):
            payload = line[6:]
            if payload == "[DONE]":
                done = True
                break
            try:
                chunks.append(json.loads(payload))
            except json.JSONDecodeError as exc:
                raise AssertionError(f"Malformed SSE chunk is not valid JSON: {payload!r}") from exc
    return chunks, done


def _auth(api_key: str = "tml-test-key-1") -> dict[str, str]:
    """Build an Authorization header dict."""
    return {"Authorization": f"Bearer {api_key}"}


# ---------------------------------------------------------------------------
# Unified server fixture — GPU or CPU depending on --gpu flag
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def oai_env(
    tmp_path_factory: pytest.TempPathFactory, request: pytest.FixtureRequest
) -> Generator[OAITestEnv, None, None]:
    """Start a TuFT server for OpenAI API testing.

    With --gpu: uses real VLLMSamplingBackend
    Without --gpu: uses DummySamplingBackend

    Yields an ``OAITestEnv`` containing ``url`` and ``model_name``.
    """
    clear_ray_state()
    is_gpu = request.config.getoption("--gpu")

    saved_api_key = os.environ.pop("TINKER_API_KEY", None)
    ray.init(ignore_reinit_error=True)

    if is_gpu:
        assert "TUFT_TEST_MODEL" in os.environ, (
            "Environment variable TUFT_TEST_MODEL must be set for GPU tests."
        )
        model_path = Path(os.environ["TUFT_TEST_MODEL"])
        # Derive HuggingFace-style model name from the directory name.
        # e.g., /path/to/Qwen3-1.7B -> Qwen/Qwen3-1.7B
        model_name = f"Qwen/{model_path.name}"
    else:
        model_path = Path("/dummy/model")
        model_name = "Qwen/Qwen3-0.6B"

    checkpoint_dir = tmp_path_factory.mktemp("checkpoints_oai")
    config = AppConfig(checkpoint_dir=Path(checkpoint_dir))
    config.supported_models = [
        ModelConfig(
            model_name=model_name,
            model_path=model_path,
            max_model_len=4096,
            tensor_parallel_size=1,
        )
    ]
    config.authorized_users = {
        "tml-test-key-1": "user-alpha",
        "tml-test-key-2": "user-beta",
    }

    port = _find_free_port()
    server, thread, base_url, client = _start_server(config, port)
    _log(f"OAI test server ({'GPU' if is_gpu else 'CPU'}) is healthy at {base_url}")

    yield OAITestEnv(url=base_url, model_name=model_name)

    _stop_server(server, thread, client)
    clear_ray_state()
    if saved_api_key is not None:
        os.environ["TINKER_API_KEY"] = saved_api_key


# ===========================================================================
# Authentication tests
# ===========================================================================


class TestAuth:
    """Authentication tests."""

    def test_bearer_token(self, oai_env: OAITestEnv) -> None:
        resp = httpx.post(
            f"{oai_env.url}/oai/api/v1/completions",
            json={"model": oai_env.model_name, "prompt": "Hello", "max_tokens": 5},
            headers=_auth(),
            timeout=120,
        )
        assert resp.status_code == 200

    def test_x_api_key(self, oai_env: OAITestEnv) -> None:
        resp = httpx.post(
            f"{oai_env.url}/oai/api/v1/completions",
            json={"model": oai_env.model_name, "prompt": "Hello", "max_tokens": 5},
            headers={"X-API-Key": "tml-test-key-1"},
            timeout=120,
        )
        assert resp.status_code == 200

    def test_missing_auth_401(self, oai_env: OAITestEnv) -> None:
        resp = httpx.post(
            f"{oai_env.url}/oai/api/v1/completions",
            json={"model": oai_env.model_name, "prompt": "Hello", "max_tokens": 5},
            timeout=120,
        )
        assert resp.status_code == 401

    def test_invalid_key_403(self, oai_env: OAITestEnv) -> None:
        resp = httpx.post(
            f"{oai_env.url}/oai/api/v1/completions",
            json={"model": oai_env.model_name, "prompt": "Hello", "max_tokens": 5},
            headers=_auth("invalid-key"),
            timeout=120,
        )
        assert resp.status_code == 403

    def test_unknown_model_404(self, oai_env: OAITestEnv) -> None:
        resp = httpx.post(
            f"{oai_env.url}/oai/api/v1/chat/completions",
            json={
                "model": "nonexistent/model",
                "messages": [{"role": "user", "content": "Hi"}],
            },
            headers=_auth(),
            timeout=120,
        )
        assert resp.status_code == 404

    def test_missing_model_400(self, oai_env: OAITestEnv) -> None:
        resp = httpx.post(
            f"{oai_env.url}/oai/api/v1/completions",
            json={"prompt": "Hello", "max_tokens": 5},
            headers=_auth(),
            timeout=120,
        )
        assert resp.status_code == 400


# ===========================================================================
# Completions endpoint
# ===========================================================================


class TestCompletions:
    """Tests for /oai/api/v1/completions."""

    def test_basic(self, oai_env: OAITestEnv) -> None:
        resp = httpx.post(
            f"{oai_env.url}/oai/api/v1/completions",
            json={
                "model": oai_env.model_name,
                "prompt": "The capital of France is",
                "max_tokens": 20,
            },
            headers=_auth(),
            timeout=120,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "choices" in data
        assert len(data["choices"]) == 1
        assert len(data["choices"][0]["text"]) > 0
        assert data["model"] == oai_env.model_name
        assert ":sample:" in data["id"]

    def test_usage(self, oai_env: OAITestEnv) -> None:
        resp = httpx.post(
            f"{oai_env.url}/oai/api/v1/completions",
            json={
                "model": oai_env.model_name,
                "prompt": "Hello world",
                "max_tokens": 10,
            },
            headers=_auth(),
            timeout=120,
        )
        data = resp.json()
        usage = data["usage"]
        assert usage["prompt_tokens"] > 0
        assert usage["completion_tokens"] > 0
        assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]

    def test_n_multiple(self, oai_env: OAITestEnv) -> None:
        resp = httpx.post(
            f"{oai_env.url}/oai/api/v1/completions",
            json={
                "model": oai_env.model_name,
                "prompt": "Once upon a time",
                "max_tokens": 10,
                "n": 3,
                "temperature": 0.9,
            },
            headers=_auth(),
            timeout=120,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["choices"]) == 3
        indices = {c["index"] for c in data["choices"]}
        assert indices == {0, 1, 2}

    def test_max_tokens_limits_output(self, oai_env: OAITestEnv) -> None:
        resp = httpx.post(
            f"{oai_env.url}/oai/api/v1/completions",
            json={
                "model": oai_env.model_name,
                "prompt": "Write a poem about a cat",
                "max_tokens": 3,
            },
            headers=_auth(),
            timeout=120,
        )
        data = resp.json()
        assert data["usage"]["completion_tokens"] <= 3

    def test_stop_string(self, oai_env: OAITestEnv) -> None:
        resp = httpx.post(
            f"{oai_env.url}/oai/api/v1/completions",
            json={
                "model": oai_env.model_name,
                "prompt": "Print: 1\n2\n3\n4\n5\n6\n7\n8\n9",
                "max_tokens": 30,
                "stop": ["\n"],
            },
            headers=_auth(),
            timeout=120,
        )
        assert resp.status_code == 200
        text = resp.json()["choices"][0]["text"]
        assert "\n" not in text

    def test_logprobs(self, oai_env: OAITestEnv) -> None:
        resp = httpx.post(
            f"{oai_env.url}/oai/api/v1/completions",
            json={
                "model": oai_env.model_name,
                "prompt": "Hello",
                "max_tokens": 5,
                "logprobs": True,
            },
            headers=_auth(),
            timeout=120,
        )
        assert resp.status_code == 200
        choice = resp.json()["choices"][0]
        assert choice.get("logprobs") is not None

    def test_top_p(self, oai_env: OAITestEnv) -> None:
        resp = httpx.post(
            f"{oai_env.url}/oai/api/v1/completions",
            json={
                "model": oai_env.model_name,
                "prompt": "The sky",
                "max_tokens": 10,
                "top_p": 0.5,
            },
            headers=_auth(),
            timeout=120,
        )
        assert resp.status_code == 200

    def test_presence_frequency_penalty(self, oai_env: OAITestEnv) -> None:
        resp = httpx.post(
            f"{oai_env.url}/oai/api/v1/completions",
            json={
                "model": oai_env.model_name,
                "prompt": "Hello hello",
                "max_tokens": 10,
                "presence_penalty": 0.5,
                "frequency_penalty": 0.5,
            },
            headers=_auth(),
            timeout=120,
        )
        assert resp.status_code == 200

    def test_stream(self, oai_env: OAITestEnv) -> None:
        with httpx.stream(
            "POST",
            f"{oai_env.url}/oai/api/v1/completions",
            json={
                "model": oai_env.model_name,
                "prompt": "Write a poem about a cat",
                "max_tokens": 20,
                "stream": True,
            },
            headers=_auth(),
            timeout=120,
        ) as resp:
            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers.get("content-type", "")
            chunks, done = _collect_sse_chunks(resp)
            assert done, "Stream did not end with [DONE]"
            assert len(chunks) > 0
            for c in chunks:
                assert c["model"] == oai_env.model_name
                assert ":sample:" in c["id"]

    def test_stream_with_usage(self, oai_env: OAITestEnv) -> None:
        with httpx.stream(
            "POST",
            f"{oai_env.url}/oai/api/v1/completions",
            json={
                "model": oai_env.model_name,
                "prompt": "Hello",
                "max_tokens": 10,
                "stream": True,
                "stream_options": {"include_usage": True},
            },
            headers=_auth(),
            timeout=120,
        ) as resp:
            assert resp.status_code == 200
            chunks, done = _collect_sse_chunks(resp)
            assert done
            last = chunks[-1]
            assert "usage" in last
            assert last["usage"]["total_tokens"] > 0

    def test_batch_prompts(self, oai_env: OAITestEnv) -> None:
        resp = httpx.post(
            f"{oai_env.url}/oai/api/v1/completions",
            json={
                "model": oai_env.model_name,
                "prompt": ["Hello world", "Goodbye world"],
                "max_tokens": 10,
            },
            headers=_auth(),
            timeout=120,
        )
        assert resp.status_code == 200
        assert len(resp.json()["choices"]) >= 2


# ===========================================================================
# Chat completions endpoint
# ===========================================================================


class TestChatCompletions:
    """Tests for /oai/api/v1/chat/completions."""

    def test_basic(self, oai_env: OAITestEnv) -> None:
        resp = httpx.post(
            f"{oai_env.url}/oai/api/v1/chat/completions",
            json={
                "model": oai_env.model_name,
                "messages": [{"role": "user", "content": "Say hello"}],
                "max_tokens": 20,
            },
            headers=_auth(),
            timeout=120,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert len(data["choices"][0]["message"]["content"]) > 0
        assert data["model"] == oai_env.model_name
        assert ":sample:" in data["id"]

    def test_usage(self, oai_env: OAITestEnv) -> None:
        resp = httpx.post(
            f"{oai_env.url}/oai/api/v1/chat/completions",
            json={
                "model": oai_env.model_name,
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 10,
            },
            headers=_auth(),
            timeout=120,
        )
        usage = resp.json()["usage"]
        assert usage["prompt_tokens"] > 0
        assert usage["completion_tokens"] > 0
        assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]

    def test_system_message(self, oai_env: OAITestEnv) -> None:
        resp = httpx.post(
            f"{oai_env.url}/oai/api/v1/chat/completions",
            json={
                "model": oai_env.model_name,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "What is 2+2?"},
                ],
                "max_tokens": 30,
                "temperature": 0,
            },
            headers=_auth(),
            timeout=120,
        )
        assert resp.status_code == 200
        assert len(resp.json()["choices"][0]["message"]["content"]) > 0

    def test_multi_turn(self, oai_env: OAITestEnv) -> None:
        resp = httpx.post(
            f"{oai_env.url}/oai/api/v1/chat/completions",
            json={
                "model": oai_env.model_name,
                "messages": [
                    {"role": "user", "content": "My name is Alice."},
                    {"role": "assistant", "content": "Nice to meet you, Alice!"},
                    {"role": "user", "content": "What is my name?"},
                ],
                "max_tokens": 30,
                "temperature": 0,
            },
            headers=_auth(),
            timeout=120,
        )
        assert resp.status_code == 200
        assert len(resp.json()["choices"][0]["message"]["content"]) > 0

    def test_n_multiple(self, oai_env: OAITestEnv) -> None:
        resp = httpx.post(
            f"{oai_env.url}/oai/api/v1/chat/completions",
            json={
                "model": oai_env.model_name,
                "messages": [{"role": "user", "content": "Tell a joke"}],
                "max_tokens": 30,
                "n": 2,
                "temperature": 0.8,
            },
            headers=_auth(),
            timeout=120,
        )
        assert resp.status_code == 200
        assert len(resp.json()["choices"]) == 2

    def test_temperature_top_p(self, oai_env: OAITestEnv) -> None:
        resp = httpx.post(
            f"{oai_env.url}/oai/api/v1/chat/completions",
            json={
                "model": oai_env.model_name,
                "messages": [{"role": "user", "content": "What color is the sky?"}],
                "max_tokens": 20,
                "temperature": 0.3,
                "top_p": 0.9,
            },
            headers=_auth(),
            timeout=120,
        )
        assert resp.status_code == 200

    def test_stop_sequence(self, oai_env: OAITestEnv) -> None:
        resp = httpx.post(
            f"{oai_env.url}/oai/api/v1/chat/completions",
            json={
                "model": oai_env.model_name,
                "messages": [{"role": "user", "content": "List items: A, B, C, D, E"}],
                "max_tokens": 50,
                "stop": ["D"],
            },
            headers=_auth(),
            timeout=120,
        )
        assert resp.status_code == 200

    def test_stream(self, oai_env: OAITestEnv) -> None:
        with httpx.stream(
            "POST",
            f"{oai_env.url}/oai/api/v1/chat/completions",
            json={
                "model": oai_env.model_name,
                "messages": [{"role": "user", "content": "Count 1 to 5."}],
                "max_tokens": 30,
                "stream": True,
            },
            headers=_auth(),
            timeout=120,
        ) as resp:
            assert resp.status_code == 200
            chunks, done = _collect_sse_chunks(resp)
            assert done
            assert len(chunks) > 0
            for c in chunks:
                assert c["model"] == oai_env.model_name
                assert ":sample:" in c["id"]
            # Reconstruct full content
            parts = []
            for c in chunks:
                delta = c["choices"][0].get("delta", {})
                if "content" in delta and delta["content"]:
                    parts.append(delta["content"])
            assert len("".join(parts)) > 0

    def test_stream_with_usage(self, oai_env: OAITestEnv) -> None:
        with httpx.stream(
            "POST",
            f"{oai_env.url}/oai/api/v1/chat/completions",
            json={
                "model": oai_env.model_name,
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 10,
                "stream": True,
                "stream_options": {"include_usage": True},
            },
            headers=_auth(),
            timeout=120,
        ) as resp:
            chunks, done = _collect_sse_chunks(resp)
            assert done
            last = chunks[-1]
            assert "usage" in last
            assert last["usage"]["total_tokens"] > 0


# ===========================================================================
# Models listing endpoint
# ===========================================================================


class TestModels:
    """Tests for /oai/api/v1/models."""

    def test_list_models(self, oai_env: OAITestEnv) -> None:
        resp = httpx.get(
            f"{oai_env.url}/oai/api/v1/models",
            headers=_auth(),
            timeout=120,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        ids = [m["id"] for m in data["data"]]
        assert oai_env.model_name in ids

    def test_models_per_user(self, oai_env: OAITestEnv) -> None:
        resp1 = httpx.get(
            f"{oai_env.url}/oai/api/v1/models",
            headers=_auth("tml-test-key-1"),
            timeout=120,
        )
        resp2 = httpx.get(
            f"{oai_env.url}/oai/api/v1/models",
            headers=_auth("tml-test-key-2"),
            timeout=120,
        )
        assert resp1.status_code == 200 and resp2.status_code == 200
        assert any(m["id"] == oai_env.model_name for m in resp1.json()["data"])
        assert any(m["id"] == oai_env.model_name for m in resp2.json()["data"])

    def test_requires_auth(self, oai_env: OAITestEnv) -> None:
        resp = httpx.get(f"{oai_env.url}/oai/api/v1/models", timeout=120)
        assert resp.status_code == 401


# ===========================================================================
# Response format (model rewrite + ID format)
# ===========================================================================


class TestResponseFormat:
    """Tests for response rewriting."""

    def test_id_format(self, oai_env: OAITestEnv) -> None:
        """Response ID format: {session_part}:sample:{vllm_original_id}."""
        resp = httpx.post(
            f"{oai_env.url}/oai/api/v1/completions",
            json={"model": oai_env.model_name, "prompt": "Hi", "max_tokens": 5},
            headers=_auth(),
            timeout=120,
        )
        response_id = resp.json()["id"]
        # Must contain the ":sample:" marker
        assert ":sample:" in response_id
        # Split on ":sample:" -- left is session part, right is vLLM original id
        left, right = response_id.split(":sample:", 1)
        assert len(left) > 0  # session part (e.g. "base" or a UUID)
        assert len(right) > 0  # vLLM original id (e.g. "cmpl-xxx")

    def test_model_name_completions(self, oai_env: OAITestEnv) -> None:
        resp = httpx.post(
            f"{oai_env.url}/oai/api/v1/completions",
            json={"model": oai_env.model_name, "prompt": "x", "max_tokens": 3},
            headers=_auth(),
            timeout=120,
        )
        assert resp.json()["model"] == oai_env.model_name

    def test_model_name_chat(self, oai_env: OAITestEnv) -> None:
        resp = httpx.post(
            f"{oai_env.url}/oai/api/v1/chat/completions",
            json={
                "model": oai_env.model_name,
                "messages": [{"role": "user", "content": "x"}],
                "max_tokens": 3,
            },
            headers=_auth(),
            timeout=120,
        )
        assert resp.json()["model"] == oai_env.model_name

    def test_model_name_stream(self, oai_env: OAITestEnv) -> None:
        with httpx.stream(
            "POST",
            f"{oai_env.url}/oai/api/v1/chat/completions",
            json={
                "model": oai_env.model_name,
                "messages": [{"role": "user", "content": "x"}],
                "max_tokens": 5,
                "stream": True,
            },
            headers=_auth(),
            timeout=120,
        ) as resp:
            chunks, _ = _collect_sse_chunks(resp)
            for c in chunks:
                assert c["model"] == oai_env.model_name

    def test_unique_ids_per_request(self, oai_env: OAITestEnv) -> None:
        ids = set()
        for _ in range(3):
            resp = httpx.post(
                f"{oai_env.url}/oai/api/v1/completions",
                json={"model": oai_env.model_name, "prompt": "x", "max_tokens": 3},
                headers=_auth(),
                timeout=120,
            )
            ids.add(resp.json()["id"])
        assert len(ids) == 3


# ===========================================================================
# Integration test -- Train -> Save checkpoint -> OpenAI API
# ===========================================================================


@pytest.mark.gpu
class TestTrainAndOAIIntegration:
    """End-to-end: train LoRA -> save checkpoint -> query via OpenAI API."""

    def test_full_flow(self, oai_env: OAITestEnv) -> None:
        """
        1. Create session + training model via Tinker SDK
        2. Forward-backward + optim step
        3. Save sampler checkpoint -> tinker:// path
        4. Chat completions via tinker:// path (non-streaming)
        5. Completions via tinker:// path (non-streaming)
        6. Streaming chat via tinker:// path
        7. Verify /models shows adapter for this user
        8. Verify user2 doesn't see user1's adapter
        9. Base model still works after LoRA load
        """
        _log("\n[Integration] Starting train -> OAI flow")

        service_client = ServiceClient(
            api_key="tml-test-key-1",  # pragma: allowlist secret
            base_url=oai_env.url,
            timeout=60,
        )
        try:
            # --- Phase 1: Train and save ---
            caps = service_client.get_server_capabilities()
            assert caps.supported_models
            base_model = caps.supported_models[0].model_name or oai_env.model_name
            _log(f"[Integration] base_model={base_model}")

            training_client = service_client.create_lora_training_client(
                base_model=base_model,
                rank=8,
            )
            _log(f"[Integration] model_id={training_client.model_id}")

            datum = types.Datum(
                model_input=types.ModelInput.from_ints([11, 12, 13, 14]),
                loss_fn_inputs={
                    "target_tokens": types.TensorData(
                        data=[21, 22, 23, 24], dtype="int64", shape=[4]
                    ),
                    "weights": types.TensorData(
                        data=[1.0, 1.0, 1.0, 1.0], dtype="float32", shape=[4]
                    ),
                },
            )
            fwdbwd = training_client.forward_backward([datum], "cross_entropy").result(timeout=30)
            assert fwdbwd.metrics["loss:sum"] >= 0
            _log(f"[Integration] loss={fwdbwd.metrics['loss:sum']:.4f}")

            optim = training_client.optim_step(types.AdamParams(learning_rate=1e-3)).result(
                timeout=30
            )
            assert optim is not None

            save_resp = training_client.save_weights_for_sampler("oai-test-ckpt").result(timeout=30)
            tinker_path = save_resp.path
            assert tinker_path.startswith("tinker://")
            _log(f"[Integration] saved: {tinker_path}")

            # --- Phase 2: Query via OpenAI API with tinker:// path ---

            # Chat completions (non-streaming)
            _log("[Integration] chat/completions with tinker:// ...")
            resp = httpx.post(
                f"{oai_env.url}/oai/api/v1/chat/completions",
                json={
                    "model": tinker_path,
                    "messages": [{"role": "user", "content": "What is 1+1?"}],
                    "max_tokens": 30,
                    "temperature": 0,
                },
                headers=_auth(),
                timeout=120,
            )
            assert resp.status_code == 200, f"Failed: {resp.text}"
            data = resp.json()
            assert data["model"] == tinker_path
            assert ":sample:" in data["id"]
            assert len(data["choices"][0]["message"]["content"]) > 0

            # Completions (non-streaming)
            _log("[Integration] completions with tinker:// ...")
            resp = httpx.post(
                f"{oai_env.url}/oai/api/v1/completions",
                json={
                    "model": tinker_path,
                    "prompt": "2+2=",
                    "max_tokens": 10,
                    "temperature": 0,
                },
                headers=_auth(),
                timeout=120,
            )
            assert resp.status_code == 200, f"Failed: {resp.text}"
            assert resp.json()["model"] == tinker_path

            # Streaming chat
            _log("[Integration] streaming chat with tinker:// ...")
            with httpx.stream(
                "POST",
                f"{oai_env.url}/oai/api/v1/chat/completions",
                json={
                    "model": tinker_path,
                    "messages": [{"role": "user", "content": "Say hello"}],
                    "max_tokens": 20,
                    "stream": True,
                },
                headers=_auth(),
                timeout=120,
            ) as resp:
                assert resp.status_code == 200
                chunks, done = _collect_sse_chunks(resp)
                assert done and len(chunks) > 0
                for c in chunks:
                    assert c["model"] == tinker_path
                    assert ":sample:" in c["id"]
            _log(f"[Integration] streaming OK, {len(chunks)} chunks")

            # --- Phase 3: Verify /models listing ---
            resp = httpx.get(
                f"{oai_env.url}/oai/api/v1/models",
                headers=_auth(),
                timeout=120,
            )
            assert resp.status_code == 200
            model_ids = [m["id"] for m in resp.json()["data"]]
            assert oai_env.model_name in model_ids

            # User2 should NOT see user1's adapters
            resp2 = httpx.get(
                f"{oai_env.url}/oai/api/v1/models",
                headers=_auth("tml-test-key-2"),
                timeout=120,
            )
            user2_tinker = [
                m["id"] for m in resp2.json()["data"] if m["id"].startswith("tinker://")
            ]
            assert len(user2_tinker) == 0, (
                f"user2 should not see user1's adapters, but got: {user2_tinker}"
            )
            _log(f"[Integration] user2 tinker models: {user2_tinker}")

            # --- Phase 4: Base model still works ---
            resp = httpx.post(
                f"{oai_env.url}/oai/api/v1/chat/completions",
                json={
                    "model": oai_env.model_name,
                    "messages": [{"role": "user", "content": "Hi"}],
                    "max_tokens": 10,
                },
                headers=_auth(),
                timeout=120,
            )
            assert resp.status_code == 200
            assert resp.json()["model"] == oai_env.model_name

            # --- Phase 5: Usage and logprobs with tinker:// path ---
            resp = httpx.post(
                f"{oai_env.url}/oai/api/v1/completions",
                json={
                    "model": tinker_path,
                    "prompt": "Hello",
                    "max_tokens": 5,
                    "logprobs": 2,
                },
                headers=_auth(),
                timeout=120,
            )
            assert resp.status_code == 200
            assert resp.json()["usage"]["total_tokens"] > 0
            assert resp.json()["choices"][0].get("logprobs") is not None

            _log("[Integration] ALL INTEGRATION CHECKS PASSED")

        finally:
            service_client.holder.close()
