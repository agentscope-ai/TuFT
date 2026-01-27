from __future__ import annotations

import gc
import os
import threading
import time
import warnings
from pathlib import Path

import httpx
import pytest
import ray
import tinker.types as types
import uvicorn
from tinker.lib.public_interfaces.service_client import ServiceClient
from transformers import AutoTokenizer

from tuft.config import AppConfig, ModelConfig
from tuft.server import create_root_app

from .helpers import (
    PIG_LATIN_EXAMPLES,
    REVERSE_EXAMPLES,
    REVERSE_PROMPTS,
    TEST_PROMPTS,
    _create_reverse_training_data,
    _create_training_data,
    _find_free_port,
    _log,
    _normalize_text,
)

"""
How to run this test (GPU required):
    PYTHONPATH=/path/to/llm-rpc/src:/path/to/llm-rpc/tinker/src \\
    TUFT_TEST_MODEL=/path/to/model/Qwen3-0.6B \\
    pytest -s tests/test_integration.py --gpu -m gpu
    # -s prints real-time progress logs (server startup, training, sampling).

Notes:
    - The test is marked with @pytest.mark.gpu and will be skipped unless --gpu is set.
    - In CI without GPUs or without TUFT_TEST_MODEL, the test will skip and not fail.
"""


@pytest.fixture(scope="module")
def server_endpoint(tmp_path_factory: pytest.TempPathFactory):
    ray.shutdown(_exiting_interpreter=True)
    gc.collect()
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("RANK", "0")

    model_envs = ["TUFT_TEST_MODEL_1", "TUFT_TEST_MODEL_2"]

    models = []
    for env in model_envs:
        if env not in os.environ and "TUFT_TEST_MODEL" not in os.environ:
            warnings.warn(
                f"Skipping GPU integration test because {env} is not set.",
                RuntimeWarning,
                stacklevel=2,
            )
            pytest.skip(f"{env} is not set, skipping GPU integration test")
        models.append(Path(os.environ[env] if env in os.environ else os.environ["TUFT_TEST_MODEL"]))

    _log(f"Using model path: {models}")

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
            model_path=models[0],
            max_model_len=4096,
            tensor_parallel_size=1,
        ),
        ModelConfig(
            model_name="Qwen/Qwen3-1.7B",
            model_path=models[1],
            max_model_len=4096,
            tensor_parallel_size=1,
        ),
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

    yield base_url

    server.should_exit = True
    thread.join(timeout=5)
    client.close()
    ray.shutdown(_exiting_interpreter=True)
    # clean up actors,
    # https://github.com/ray-project/ray/issues/31738
    gc.collect()


@pytest.mark.integration
@pytest.mark.gpu
def test_auth_and_pig_latin_training_flow(server_endpoint: str) -> None:
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available, skipping GPU integration test")
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
        api_key="tml-test-key",  # pragma: allowlist secret
        base_url=server_endpoint,
        timeout=120,
    )
    # here we assume the model has the same tokenizer
    tokenizer = AutoTokenizer.from_pretrained(os.environ["TUFT_TEST_MODEL"])
    try:
        capabilities = service_client.get_server_capabilities()
        assert capabilities.supported_models, "server did not report supported models"
        assert len(capabilities.supported_models) == 2, "expected 2 supported models"
        supported_model_names = [m.model_name for m in capabilities.supported_models]
        base_model_1 = "Qwen/Qwen3-0.6B"
        base_model_2 = "Qwen/Qwen3-1.7B"
        assert base_model_1 in supported_model_names, f"{base_model_1} not reported as supported"
        assert base_model_2 in supported_model_names, f"{base_model_2} not reported as supported"

        _log(f"Base model: {base_model_1}")

        _log("Creating LoRA training client...")
        training_client_1 = service_client.create_lora_training_client(
            base_model=base_model_1, rank=8
        )
        training_client_2 = service_client.create_lora_training_client(
            base_model=base_model_1, rank=16
        )
        training_client_3 = service_client.create_lora_training_client(
            base_model=base_model_2, rank=16
        )
        training_clients = [training_client_1, training_client_2, training_client_3]
        train_data = _create_training_data(tokenizer)
        _log(f"Training samples: {len(train_data)}")

        for epoch in range(1, 21):
            if epoch == 1:
                _log("Running training loop...")
            for training_client in training_clients:
                training_client.forward_backward(train_data, "cross_entropy").result(timeout=60)
            for training_client in training_clients:
                training_client.optim_step(types.AdamParams(learning_rate=1e-4)).result(timeout=60)
            if epoch % 5 == 0:
                _log(f"Training progress: epoch {epoch}/20")
        _log("Training complete")

        sampling_clients = []

        for i, training_client in enumerate(training_clients, start=1):
            _log(f"Saving sampler weights for training client {i}...")
            sampler_response = training_client.save_weights_for_sampler(
                f"sampler-client-{i}"
            ).result(timeout=60)
            assert sampler_response.path.startswith("tinker://")
            _log(f"Sampler path for client {i}: {sampler_response.path}")
            sampling_clients.append(
                service_client.create_sampling_client(model_path=sampler_response.path)
            )

        _log("Running sampling assertions...")
        for prompt_text, example in zip(TEST_PROMPTS, PIG_LATIN_EXAMPLES, strict=True):
            prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=True)
            for sampling_client in sampling_clients:
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
                output_text = tokenizer.decode(
                    sample_res.sequences[0].tokens, skip_special_tokens=True
                )
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
    service_client = ServiceClient(
        api_key="tml-test-key",  # pragma: allowlist secret
        base_url=server_endpoint,
        timeout=120,
    )
    tokenizer = AutoTokenizer.from_pretrained(os.environ["TUFT_TEST_MODEL"])
    try:
        capabilities = service_client.get_server_capabilities()
        assert capabilities.supported_models, "server did not report supported models"
        base_model = capabilities.supported_models[0].model_name or "Qwen/Qwen3-0.6B"
        _log(f"Base model: {base_model}")

        _log("Training LoRA A (Pig Latin)...")
        training_client_a = service_client.create_lora_training_client(
            base_model=base_model, rank=8
        )
        pig_latin_data = _create_training_data(tokenizer)
        _log("Training LoRA B (Reverse Words)...")
        training_client_b = service_client.create_lora_training_client(
            base_model=base_model, rank=8
        )
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

        sampler_b = training_client_b.save_weights_for_sampler("sampler-reverse-b").result(
            timeout=60
        )
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
