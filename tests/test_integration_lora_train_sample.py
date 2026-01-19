from __future__ import annotations

import os
import socket
import threading
import time
import warnings
from pathlib import Path

import httpx
import pytest
import ray
import uvicorn

from tuft.config import AppConfig, ModelConfig
from tuft.server import create_root_app
import tinker.types as types
from tinker.lib.public_interfaces.service_client import ServiceClient
from transformers import AutoTokenizer

"""
How to run this test (GPU required):
    TUFT_TEST_MODEL=/path/dir/qwen/Qwen3-0.6B \\
    pytest -s tests/test_integration_lora_train_sample.py --gpu -m gpu
    # -s prints real-time progress logs (server startup, training, sampling).

Notes:
    - The test is marked with @pytest.mark.gpu and will be skipped unless --gpu is set.
    - In CI without GPUs or without TUFT_TEST_MODEL, the test will skip and not fail.
"""

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
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _normalize_text(text: str) -> str:
    return " ".join(text.strip().split())


def _log(message: str) -> None:
    print(message, flush=True)


def _create_training_data(tokenizer) -> list[types.Datum]:
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


@pytest.fixture(scope="module")
def server_endpoint(tmp_path_factory: pytest.TempPathFactory):
    saved_api_key = os.environ.pop("TINKER_API_KEY", None)
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("RANK", "0")

    model_env = os.environ.get("TUFT_TEST_MODEL")
    if not model_env:
        warnings.warn(
            "Skipping GPU integration test because TUFT_TEST_MODEL is not set.",
            RuntimeWarning,
        )
        pytest.skip("TUFT_TEST_MODEL is not set, skipping GPU integration test")
    model_path = Path(model_env)
    _log(f"Using model path: {model_path}")

    _log("Starting Ray...")
    ray.init(
        ignore_reinit_error=True,
        runtime_env={"env_vars": {"TRANSFORMERS_NO_TORCHVISION": "1"}},
    )
    checkpoint_dir = tmp_path_factory.mktemp("checkpoints")
    _log(f"Checkpoint dir: {checkpoint_dir}")
    config = AppConfig(checkpoint_dir=Path(checkpoint_dir))
    config.supported_models = [
        ModelConfig(
            model_name="Qwen/Qwen3-0.6B",
            model_path=model_path,
            max_model_len=4096,
            tensor_parallel_size=1,
        )
    ]
    config.authorized_users = {
        "tml-test-key": "default",
    }
    _log("Creating FastAPI app...")
    app = create_root_app(config)
    port = _find_free_port()
    _log(f"Starting server on port {port}...")
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    base_url = f"http://127.0.0.1:{port}"
    client = httpx.Client()
    healthy = False
    for attempt in range(1, 121):
        try:
            response = client.get(f"{base_url}/api/v1/healthz", timeout=1)
            if response.status_code == 200:
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

    yield base_url

    server.should_exit = True
    thread.join(timeout=5)
    client.close()
    ray.shutdown()
    if saved_api_key is not None:
        os.environ["TINKER_API_KEY"] = saved_api_key


@pytest.mark.integration
@pytest.mark.gpu
def test_auth_and_pig_latin_training_flow(server_endpoint: str) -> None:
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available, skipping GPU integration test")
    model_env = os.environ.get("TUFT_TEST_MODEL")
    if not model_env:
        warnings.warn(
            "Skipping GPU integration test because TUFT_TEST_MODEL is not set.",
            RuntimeWarning,
        )
        pytest.skip("TUFT_TEST_MODEL is not set, skipping GPU integration test")
    http_client = httpx.Client()
    try:
        _log(f"Server endpoint: {server_endpoint}")
        missing_key = http_client.post(f"{server_endpoint}/api/v1/create_session", json={})
        assert missing_key.status_code == 401

        invalid_key = http_client.post(
            f"{server_endpoint}/api/v1/create_session",
            headers={"X-API-Key": "invalid-key"},
            json={},
        )
        assert invalid_key.status_code == 403
    finally:
        http_client.close()

    service_client = ServiceClient(
        api_key="tml-test-key",
        base_url=server_endpoint,
        timeout=120,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_env)
    try:
        capabilities = service_client.get_server_capabilities()
        assert capabilities.supported_models, "server did not report supported models"
        base_model = capabilities.supported_models[0].model_name or "Qwen/Qwen3-0.6B"
        _log(f"Base model: {base_model}")

        _log("Creating LoRA training client...")
        training_client = service_client.create_lora_training_client(base_model=base_model, rank=8)
        train_data = _create_training_data(tokenizer)
        _log(f"Training samples: {len(train_data)}")

        seq_id = 1
        for epoch in range(1, 21):
            if epoch == 1:
                _log("Running training loop...")
            training_client.forward_backward(train_data, "cross_entropy").result(timeout=60)
            training_client.optim_step(types.AdamParams(learning_rate=1e-4)).result(timeout=60)
            seq_id += 2
            if epoch % 5 == 0:
                _log(f"Training progress: epoch {epoch}/20")
        _log("Training complete")

        sampler_response = training_client.save_weights_for_sampler("sampler-pig-latin").result(
            timeout=60
        )
        assert sampler_response.path.startswith("tinker://")
        _log(f"Sampler path: {sampler_response.path}")

        _log("Creating sampling client from trained LoRA...")
        sampling_client = service_client.create_sampling_client(model_path=sampler_response.path)

        _log("Running sampling assertions...")
        for prompt_text, example in zip(TEST_PROMPTS, PIG_LATIN_EXAMPLES, strict=True):
            prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=True)
            sample_res = sampling_client.sample(
                prompt=types.ModelInput.from_ints(prompt_tokens),
                num_samples=1,
                sampling_params=types.SamplingParams(
                    max_tokens=16,
                    temperature=0.1,
                    top_p=1.0,
                    stop=["\n"],
                ),
            ).result(timeout=60)
            assert sample_res.sequences and sample_res.sequences[0].tokens
            output_text = tokenizer.decode(sample_res.sequences[0].tokens, skip_special_tokens=True)
            _log(f"Prompt: {prompt_text!r}")
            _log(f"Output: {output_text!r}")
            assert _normalize_text(output_text) == _normalize_text(example["output"])
    finally:
        service_client.holder.close()


@pytest.mark.integration
@pytest.mark.gpu
def test_multi_lora_adapters(server_endpoint: str) -> None:
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available, skipping GPU integration test")
    model_env = os.environ.get("TUFT_TEST_MODEL")
    if not model_env:
        warnings.warn(
            "Skipping GPU integration test because TUFT_TEST_MODEL is not set.",
            RuntimeWarning,
        )
        pytest.skip("TUFT_TEST_MODEL is not set, skipping GPU integration test")

    service_client = ServiceClient(
        api_key="tml-test-key",
        base_url=server_endpoint,
        timeout=120,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_env)
    try:
        capabilities = service_client.get_server_capabilities()
        assert capabilities.supported_models, "server did not report supported models"
        base_model = capabilities.supported_models[0].model_name or "Qwen/Qwen3-0.6B"
        _log(f"Base model: {base_model}")

        _log("Training LoRA A (Pig Latin)...")
        training_client_a = service_client.create_lora_training_client(base_model=base_model, rank=8)
        pig_latin_data = _create_training_data(tokenizer)
        _log("Training LoRA B (Reverse Words)...")
        training_client_b = service_client.create_lora_training_client(base_model=base_model, rank=8)
        reverse_data = _create_reverse_training_data(tokenizer)

        _log("Running interleaved training loop...")
        for epoch in range(1, 31):
            training_client_a.forward_backward(pig_latin_data, "cross_entropy").result(timeout=60)
            training_client_a.optim_step(types.AdamParams(learning_rate=1e-4)).result(timeout=60)
            training_client_b.forward_backward(reverse_data, "cross_entropy").result(timeout=60)
            training_client_b.optim_step(types.AdamParams(learning_rate=1e-4)).result(timeout=60)
            if epoch % 5 == 0:
                _log(f"Interleaved progress: epoch {epoch}/30")
        _log("Interleaved training complete")

        sampler_a = training_client_a.save_weights_for_sampler("sampler-pig-latin-a").result(
            timeout=60
        )
        assert sampler_a.path.startswith("tinker://")
        _log(f"Sampler A path: {sampler_a.path}")

        sampler_b = training_client_b.save_weights_for_sampler("sampler-reverse-b").result(timeout=60)
        assert sampler_b.path.startswith("tinker://")
        _log(f"Sampler B path: {sampler_b.path}")

        sampling_client_a = service_client.create_sampling_client(model_path=sampler_a.path)
        sampling_client_b = service_client.create_sampling_client(model_path=sampler_b.path)

        _log("Validating LoRA A (Pig Latin) outputs...")
        for prompt_text, example in zip(TEST_PROMPTS, PIG_LATIN_EXAMPLES, strict=True):
            prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=True)
            sample_res = sampling_client_a.sample(
                prompt=types.ModelInput.from_ints(prompt_tokens),
                num_samples=1,
                sampling_params=types.SamplingParams(
                    max_tokens=16,
                    temperature=0.1,
                    top_p=1.0,
                    stop=["\n"],
                ),
            ).result(timeout=60)
            assert sample_res.sequences and sample_res.sequences[0].tokens
            output_text = tokenizer.decode(sample_res.sequences[0].tokens, skip_special_tokens=True)
            _log(f"LoRA A prompt: {prompt_text!r}")
            _log(f"LoRA A output: {output_text!r}")
            assert _normalize_text(output_text) == _normalize_text(example["output"])

        _log("Validating LoRA B (Reverse Words) outputs...")
        for prompt_text, example in zip(REVERSE_PROMPTS, REVERSE_EXAMPLES, strict=True):
            prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=True)
            sample_res = sampling_client_b.sample(
                prompt=types.ModelInput.from_ints(prompt_tokens),
                num_samples=1,
                sampling_params=types.SamplingParams(
                    max_tokens=32,
                    temperature=0.0,
                    top_p=1.0,
                    stop=["\n"],
                ),
            ).result(timeout=60)
            assert sample_res.sequences and sample_res.sequences[0].tokens
            output_text = tokenizer.decode(sample_res.sequences[0].tokens, skip_special_tokens=True)
            _log(f"LoRA B prompt: {prompt_text!r}")
            _log(f"LoRA B output: {output_text!r}")
            assert _normalize_text(output_text) == _normalize_text(example["output"])

        _log("Validating LoRA A/B separation...")
        cross_prompt = "Reverse each word.\nEnglish: hello world\nReversed:"
        cross_tokens = tokenizer.encode(cross_prompt, add_special_tokens=True)
        cross_res_a = sampling_client_a.sample(
            prompt=types.ModelInput.from_ints(cross_tokens),
            num_samples=1,
            sampling_params=types.SamplingParams(
                max_tokens=32,
                temperature=0.0,
                top_p=1.0,
                stop=["\n"],
            ),
        ).result(timeout=60)
        assert cross_res_a.sequences and cross_res_a.sequences[0].tokens
        cross_text_a = tokenizer.decode(cross_res_a.sequences[0].tokens, skip_special_tokens=True)
        _log(f"LoRA A on Reverse prompt output: {cross_text_a!r}")
        assert _normalize_text(cross_text_a) != _normalize_text("olleh dlrow")
    finally:
        service_client.holder.close()

