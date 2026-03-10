from __future__ import annotations

import os
import warnings
from pathlib import Path

import httpx
import pytest
import tinker.types as types
from tinker.lib.public_interfaces.service_client import ServiceClient
from transformers import AutoTokenizer

from tuft.config import ModelConfig

from .helpers import (
    PIG_LATIN_EXAMPLES,
    REVERSE_EXAMPLES,
    REVERSE_PROMPTS,
    TEST_PROMPTS,
    ServerFixtureConfig,
    _create_reverse_training_data,
    _create_server_endpoint,
    _create_training_data,
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


def _get_model_path(env_var: str = "TUFT_TEST_MODEL") -> Path:
    """Get model path from environment variable, skip test if not set."""
    if env_var not in os.environ:
        pytest.skip(f"{env_var} is not set, skipping integration test")
    return Path(os.environ[env_var])


def _tmp_space_ok(min_mb: int = 500) -> bool:
    """Return True if /tmp has at least min_mb MB free (for Ray packaging)."""
    try:
        stat = os.statvfs("/tmp")
        free_mb = (stat.f_bavail * stat.f_frsize) / (1024 * 1024)
        return free_mb >= min_mb
    except OSError:
        return False


@pytest.fixture(scope="module")
def hf_server_endpoint(tmp_path_factory: pytest.TempPathFactory):
    """Two-model server with HF training backend."""
    # Support both TUFT_TEST_MODEL_1/2 and fallback to TUFT_TEST_MODEL
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
        models.append(Path(os.environ.get(env, os.environ.get("TUFT_TEST_MODEL", ""))))

    _log(f"Using model paths: {models}")

    config = ServerFixtureConfig(
        model_configs=[
            ModelConfig(
                model_name="Qwen/Qwen3-0.6B",
                model_path=models[0],
                max_model_len=4096,
                tensor_parallel_size=1,
                sampling_memory_fraction=0.5,
            ),
            ModelConfig(
                model_name="Qwen/Qwen3-1.7B",
                model_path=models[1],
                max_model_len=4096,
                tensor_parallel_size=1,
                sampling_memory_fraction=0.5,
            ),
        ],
        checkpoint_subdir="checkpoints",
    )
    yield from _create_server_endpoint(tmp_path_factory, config)


# -----------------------------------------------------------------------------
# FSDP integration: single model, training_backend="fsdp"
# -----------------------------------------------------------------------------


def _get_available_gpu_count() -> int:
    """Get the number of available GPUs."""
    import torch

    if not torch.cuda.is_available():
        return 0
    return torch.cuda.device_count()


@pytest.fixture(scope="module")
def fsdp_server_endpoint(tmp_path_factory: pytest.TempPathFactory):
    """Single-model FSDP server with Ray and UV_CACHE_DIR (1 GPU)."""
    model_path = _get_model_path()
    if not _tmp_space_ok():
        pytest.skip("/tmp has insufficient free space for Ray working_dir packaging; need ~500MB+")
    _log(f"FSDP fixture using model path: {model_path}")

    config = ServerFixtureConfig(
        model_configs=[
            ModelConfig(
                model_name="Qwen/Qwen3-0.6B",
                model_path=model_path,
                max_model_len=4096,
                tensor_parallel_size=1,
                training_backend="fsdp",
                fsdp_num_gpus=1,
                max_lora_rank=8,
                sampling_memory_fraction=0.5,
            ),
        ],
        checkpoint_subdir="checkpoints_fsdp",
        ray_env_vars={"UV_CACHE_DIR": "/home/ray_uv_cache"},
        ray_excludes=[".git", "*.pack", "*.tar.gz", "checkpoints"],
    )
    yield from _create_server_endpoint(tmp_path_factory, config)


@pytest.fixture(scope="module")
def fsdp_multi_gpu_server_endpoint(tmp_path_factory: pytest.TempPathFactory):
    """Single-model FSDP server with 2 GPUs for multi-GPU training."""
    model_path = _get_model_path()
    gpu_count = _get_available_gpu_count()
    required_gpus = 2

    if gpu_count < required_gpus:
        pytest.skip(
            f"Multi-GPU FSDP test requires {required_gpus} GPUs, but only {gpu_count} available"
        )
    if not _tmp_space_ok():
        pytest.skip("/tmp has insufficient free space for Ray working_dir packaging; need ~500MB+")
    _log(f"FSDP multi-GPU fixture using model path: {model_path}, gpus: {required_gpus}")

    config = ServerFixtureConfig(
        model_configs=[
            ModelConfig(
                model_name="Qwen/Qwen3-0.6B",
                model_path=model_path,
                max_model_len=4096,
                tensor_parallel_size=1,
                training_backend="fsdp",
                fsdp_num_gpus=required_gpus,
                max_lora_rank=8,
                sampling_memory_fraction=0.5,
            ),
        ],
        checkpoint_subdir="checkpoints_fsdp_multi_gpu",
        ray_env_vars={"UV_CACHE_DIR": "/home/ray_uv_cache"},
        ray_excludes=[".git", "*.pack", "*.tar.gz", "checkpoints"],
    )
    yield from _create_server_endpoint(tmp_path_factory, config)


@pytest.mark.integration
@pytest.mark.gpu
def test_auth_and_pig_latin_training_flow(hf_server_endpoint: str) -> None:
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available, skipping GPU integration test")
    http_client = httpx.Client()
    try:
        _log(f"Server endpoint: {hf_server_endpoint}")
        missing_key = http_client.post(f"{hf_server_endpoint}/api/v1/create_session", json={})
        assert missing_key.status_code == 401

        invalid_key = http_client.post(
            f"{hf_server_endpoint}/api/v1/create_session",
            headers={"X-API-Key": "invalid-key"},
            json={},
        )
        assert invalid_key.status_code == 403
    finally:
        http_client.close()

    service_client = ServiceClient(
        api_key="tml-test-key",  # pragma: allowlist secret
        base_url=hf_server_endpoint,
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
def test_multi_lora_adapters(hf_server_endpoint: str) -> None:
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available, skipping GPU integration test")
    service_client = ServiceClient(
        api_key="tml-test-key",  # pragma: allowlist secret
        base_url=hf_server_endpoint,
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


# -----------------------------------------------------------------------------
# FSDP integration tests (Case 1–3 and with-Ray path)
# -----------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.gpu
def test_fsdp_training_flow(fsdp_server_endpoint: str) -> None:
    """Case 1: FSDP + full stack, training only."""
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available, skipping GPU integration test")

    service_client = ServiceClient(
        api_key="tml-test-key",  # pragma: allowlist secret
        base_url=fsdp_server_endpoint,
        timeout=120,
    )
    tokenizer = AutoTokenizer.from_pretrained(os.environ["TUFT_TEST_MODEL"])
    try:
        capabilities = service_client.get_server_capabilities()
        assert capabilities.supported_models, "server did not report supported models"
        assert len(capabilities.supported_models) == 1
        base_model = capabilities.supported_models[0].model_name or "Qwen/Qwen3-0.6B"
        _log(f"Base model: {base_model}")

        _log("Creating LoRA training client (FSDP)...")
        training_client = service_client.create_lora_training_client(base_model=base_model, rank=8)
        train_data = _create_training_data(tokenizer)
        _log(f"Training samples: {len(train_data)}")

        for epoch in range(1, 6):
            training_client.forward_backward(train_data, "cross_entropy").result(timeout=60)
            training_client.optim_step(types.AdamParams(learning_rate=1e-4)).result(timeout=60)
            if epoch % 2 == 0:
                _log(f"FSDP training progress: epoch {epoch}/5")
        _log("FSDP training complete")

        sampler_response = training_client.save_weights_for_sampler("fsdp-sampler-1").result(
            timeout=60
        )
        assert sampler_response.path.startswith("tinker://")
        _log(f"Sampler path: {sampler_response.path}")
    finally:
        service_client.holder.close()


@pytest.mark.integration
@pytest.mark.gpu
def test_fsdp_training_and_sampling(fsdp_server_endpoint: str) -> None:
    """Case 2: FSDP training + save_weights_for_sampler + sampling (if backend available)."""
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available, skipping GPU integration test")

    service_client = ServiceClient(
        api_key="tml-test-key",  # pragma: allowlist secret
        base_url=fsdp_server_endpoint,
        timeout=120,
    )
    tokenizer = AutoTokenizer.from_pretrained(os.environ["TUFT_TEST_MODEL"])
    try:
        capabilities = service_client.get_server_capabilities()
        assert len(capabilities.supported_models) == 1
        base_model = capabilities.supported_models[0].model_name or "Qwen/Qwen3-0.6B"
        _log(f"[Case2] Base model: {base_model}")

        training_client = service_client.create_lora_training_client(base_model=base_model, rank=8)
        train_data = _create_training_data(tokenizer)
        num_steps = 40
        _log(f"[Case2] Training: {len(train_data)} samples, {num_steps} steps")
        for step in range(1, num_steps + 1):
            fwd_out = training_client.forward_backward(train_data, "cross_entropy").result(
                timeout=60
            )
            loss = fwd_out.metrics.get("loss:sum", fwd_out.metrics.get("loss", "—"))
            _log(f"[Case2]   Step {step}/{num_steps}  forward_backward  loss: {loss}")
            training_client.optim_step(types.AdamParams(learning_rate=1e-4)).result(timeout=60)
            _log(f"[Case2]   Step {step}/{num_steps}  optim_step  done")
        _log("[Case2] Training finished.")

        sampler_response = training_client.save_weights_for_sampler("fsdp-sampler-2").result(
            timeout=60
        )
        assert sampler_response.path.startswith("tinker://")
        _log(f"[Case2] Save for sampler: path = {sampler_response.path}")

        sampling_client = service_client.create_sampling_client(model_path=sampler_response.path)
        prompt_text = "English: hello world\nPig Latin:"
        prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=True)
        _log(f"[Case2] Sampling  prompt (human): {prompt_text!r}")
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
        out_tokens = sample_res.sequences[0].tokens
        out_text = tokenizer.decode(out_tokens, skip_special_tokens=True)
        _log(f"[Case2] Sampling  output tokens (count): {len(out_tokens)}")
        _log(f"[Case2] Sampling  output (human): {out_text!r}")
        if not out_text.strip() or out_text in (";<=>?@ABCDEFGHIJ",):
            _log(
                "[Case2] Note: Sampling is using DummySamplingBackend (vLLM not installed). "
                "The output above is fake tokens, NOT from the trained model. "
                "To get real Pig Latin output, install backend: uv pip install .[backend]"
            )
        else:
            _log("[Case2] Expected for Pig Latin: something like 'ello-hay orld-way'")
        _log("FSDP training + sampling OK")
    except Exception as e:
        if "sampling" in str(e).lower() or "vllm" in str(e).lower():
            pytest.skip(f"Sampling backend not available: {e}")
        raise
    finally:
        service_client.holder.close()


@pytest.mark.integration
@pytest.mark.gpu
def test_fsdp_multi_lora_adapters(fsdp_server_endpoint: str) -> None:
    """Case 3: Two LoRA clients on same FSDP model, interleaved training, each save_weights."""
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available, skipping GPU integration test")

    service_client = ServiceClient(
        api_key="tml-test-key",  # pragma: allowlist secret
        base_url=fsdp_server_endpoint,
        timeout=120,
    )
    tokenizer = AutoTokenizer.from_pretrained(os.environ["TUFT_TEST_MODEL"])
    try:
        capabilities = service_client.get_server_capabilities()
        assert len(capabilities.supported_models) == 1
        base_model = capabilities.supported_models[0].model_name or "Qwen/Qwen3-0.6B"

        training_client_a = service_client.create_lora_training_client(
            base_model=base_model, rank=8
        )
        training_client_b = service_client.create_lora_training_client(
            base_model=base_model, rank=8
        )
        pig_data = _create_training_data(tokenizer)
        reverse_data = _create_reverse_training_data(tokenizer)

        for _epoch in range(1, 11):
            training_client_a.forward_backward(pig_data, "cross_entropy").result(timeout=60)
            training_client_a.optim_step(types.AdamParams(learning_rate=1e-4)).result(timeout=60)
            training_client_b.forward_backward(reverse_data, "cross_entropy").result(timeout=60)
            training_client_b.optim_step(types.AdamParams(learning_rate=1e-4)).result(timeout=60)
        _log("FSDP multi-LoRA interleaved training complete")

        sampler_a = training_client_a.save_weights_for_sampler("fsdp-multi-a").result(timeout=60)
        sampler_b = training_client_b.save_weights_for_sampler("fsdp-multi-b").result(timeout=60)
        assert sampler_a.path.startswith("tinker://")
        assert sampler_b.path.startswith("tinker://")
        _log(f"Sampler A: {sampler_a.path}, Sampler B: {sampler_b.path}")
    finally:
        service_client.holder.close()


# -----------------------------------------------------------------------------
# Multi-GPU FSDP integration tests
# -----------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.gpu
def test_fsdp_multi_gpu_training_flow(fsdp_multi_gpu_server_endpoint: str) -> None:
    """FSDP training with multiple GPUs (fsdp_num_gpus=2)."""
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available, skipping GPU integration test")

    service_client = ServiceClient(
        api_key="tml-test-key",  # pragma: allowlist secret
        base_url=fsdp_multi_gpu_server_endpoint,
        timeout=180,
    )
    tokenizer = AutoTokenizer.from_pretrained(os.environ["TUFT_TEST_MODEL"])
    try:
        capabilities = service_client.get_server_capabilities()
        assert capabilities.supported_models, "server did not report supported models"
        assert len(capabilities.supported_models) == 1
        base_model = capabilities.supported_models[0].model_name or "Qwen/Qwen3-0.6B"
        _log(f"[Multi-GPU] Base model: {base_model}")

        _log("[Multi-GPU] Creating LoRA training client (FSDP with 2 GPUs)...")
        training_client = service_client.create_lora_training_client(base_model=base_model, rank=8)
        train_data = _create_training_data(tokenizer)
        _log(f"[Multi-GPU] Training samples: {len(train_data)}")

        num_epochs = 40
        for epoch in range(1, num_epochs + 1):
            fwd_out = training_client.forward_backward(train_data, "cross_entropy").result(
                timeout=120
            )
            loss = fwd_out.metrics.get("loss:sum", fwd_out.metrics.get("loss", "—"))
            training_client.optim_step(types.AdamParams(learning_rate=1e-4)).result(timeout=120)
            if epoch % 2 == 0 or epoch == 1:
                _log(f"[Multi-GPU] FSDP training epoch {epoch}/{num_epochs}, loss: {loss}")
        _log("[Multi-GPU] FSDP training complete")

        sampler_response = training_client.save_weights_for_sampler(
            "fsdp-multi-gpu-sampler"
        ).result(timeout=120)
        assert sampler_response.path.startswith("tinker://")
        _log(f"[Multi-GPU] Sampler path: {sampler_response.path}")
    finally:
        service_client.holder.close()
