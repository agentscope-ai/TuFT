from __future__ import annotations

import os
import warnings
from pathlib import Path

import pytest
import ray
import tinker.types as types
from tinker.lib.public_interfaces.service_client import ServiceClient
from transformers import AutoTokenizer

from tuft.config import AppConfig, ModelConfig
from tuft.persistence import PersistenceConfig

from .helpers import (
    TEST_PROMPTS,
    _create_training_data,
    _find_free_port,
    _log,
    _start_server,
    _stop_server,
    clear_ray_state,
)


"""
Integration test for checkpoint persistence across server restarts.

How to run this test (GPU required):
    TUFT_TEST_MODEL=/path/to/model/Qwen3-0.6B \\
    pytest -s tests/test_integration_persistence.py::test_checkpoint_resume_persistence --gpu -m gpu

Notes:
    - The test is marked with @pytest.mark.gpu and will be skipped unless --gpu is set.
    - This test is in a separate file to avoid Ray initialization conflicts with other tests.
"""


@pytest.mark.integration
@pytest.mark.gpu
@pytest.mark.persistence
def test_checkpoint_resume_persistence(tmp_path: Path) -> None:
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available, skipping GPU integration test")

    model_env = os.environ.get("TUFT_TEST_MODEL")
    if not model_env:
        warnings.warn(
            "Skipping GPU integration test because TUFT_TEST_MODEL is not set.",
            RuntimeWarning,
            stacklevel=2,
        )
        pytest.skip("TUFT_TEST_MODEL is not set, skipping GPU integration test")

    file_redis_path = tmp_path / "file_redis.json"
    if file_redis_path.exists():
        file_redis_path.unlink()
    _log(f"FileRedis path: {file_redis_path}")

    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("RANK", "0")
    # make sure the resources are released
    clear_ray_state()
    ray.init(
        ignore_reinit_error=True,
        runtime_env={"env_vars": {"TRANSFORMERS_NO_TORCHVISION": "1"}},
    )
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    config = AppConfig(checkpoint_dir=checkpoint_dir)
    config.supported_models = [
        ModelConfig(
            model_name="Qwen/Qwen3-0.6B",
            model_path=Path(model_env),
            max_model_len=4096,
            tensor_parallel_size=1,
        )
    ]
    config.authorized_users = {
        "tml-test-key": "default",
    }
    config.persistence = PersistenceConfig.from_file(
        file_path=file_redis_path,
        namespace="tuft_test",
    )

    port = _find_free_port()
    server = thread = client = None
    service_client = None
    try:
        _log("Starting server...")
        server, thread, base_url, client = _start_server(config, port)

        service_client = ServiceClient(api_key="tml-test-key", base_url=base_url, timeout=120)
        tokenizer = AutoTokenizer.from_pretrained(model_env)

        capabilities = service_client.get_server_capabilities()
        assert capabilities.supported_models, "server did not report supported models"
        base_model = capabilities.supported_models[0].model_name or "Qwen/Qwen3-0.6B"
        _log(f"Base model: {base_model}")

        training_client = service_client.create_lora_training_client(base_model=base_model, rank=2)
        train_data = _create_training_data(tokenizer)
        _log("Running training loop...")
        training_client.forward_backward(train_data, "cross_entropy").result(timeout=60)
        training_client.optim_step(types.AdamParams(learning_rate=5e-4)).result(timeout=60)
        _log("Training complete")

        checkpoint_name = "persistence-ckpt"
        checkpoint = training_client.save_state(checkpoint_name).result(timeout=60)
        checkpoint_path = checkpoint.path
        assert checkpoint_path.startswith("tinker://")
        _log(f"Checkpoint path: {checkpoint_path}")

        sampler_response = training_client.save_weights_for_sampler("persistence-sampler").result(
            timeout=60
        )
        sampler_path = sampler_response.path
        assert sampler_path.startswith("tinker://")
        _log(f"Sampler path: {sampler_path}")
        sampling_client = service_client.create_sampling_client(model_path=sampler_path)
        sample_res = sampling_client.sample(
            prompt=types.ModelInput.from_ints(
                tokenizer.encode(TEST_PROMPTS[0], add_special_tokens=True)
            ),
            num_samples=1,
            sampling_params=types.SamplingParams(max_tokens=4, temperature=0.1, top_p=1.0),
        ).result(timeout=60)
        assert sample_res.sequences and sample_res.sequences[0].tokens
        _log(f"Checkpoint dir contents: {list(checkpoint_dir.rglob('*'))}")
        _log(
            f"FileRedis exists after save: {file_redis_path.exists()} size="
            f"{file_redis_path.stat().st_size if file_redis_path.exists() else 'n/a'}"
        )

        session_id = service_client.holder.get_session_id()
        rest_client = service_client.create_rest_client()
        session_before = rest_client.get_session(session_id).result(timeout=30)
        checkpoints_before = rest_client.list_checkpoints(training_client.model_id).result(
            timeout=30
        )
        checkpoint_ids_before = [c.checkpoint_id for c in checkpoints_before.checkpoints]
        assert checkpoint_name in checkpoint_ids_before

        _log("Restarting server...")
        _stop_server(server, thread, client)
        clear_ray_state()
        ray.init(
            ignore_reinit_error=True,
            runtime_env={"env_vars": {"TRANSFORMERS_NO_TORCHVISION": "1"}},
        )
        server, thread, base_url, client = _start_server(config, port)

        # Create a new service_client with the new server URL
        service_client = ServiceClient(
            api_key="tml-test-key",  # pragma: allowlist secret
            base_url=base_url,
            timeout=120,
        )
        rest_client = service_client.create_rest_client()
        sessions = rest_client.list_sessions().result(timeout=30)
        assert session_id in sessions.sessions

        session_after = rest_client.get_session(session_id).result(timeout=30)
        assert session_after.training_run_ids == session_before.training_run_ids
        assert session_after.sampler_ids == session_before.sampler_ids

        checkpoints_after = rest_client.list_checkpoints(training_client.model_id).result(
            timeout=30
        )
        checkpoint_ids_after = [c.checkpoint_id for c in checkpoints_after.checkpoints]
        assert checkpoint_name in checkpoint_ids_after

        # Use existing training_client to load state from checkpoint
        _log("Loading checkpoint into existing training client...")
        training_client.load_state_with_optimizer(checkpoint_path).result(timeout=60)
        _log(
            "Training client state loaded from checkpoint. "
            f"turn={training_client._turn_counter} "
            f"next_request={training_client._request_id_counter}"
        )
        training_client.forward_backward(train_data, "cross_entropy").result(timeout=60)
        training_client.optim_step(types.AdamParams(learning_rate=1e-4)).result(timeout=60)
        resumed_weights = training_client.save_weights_for_sampler("resume-sampler").result(
            timeout=60
        )
        assert resumed_weights.path.startswith("tinker://")
        resumed_sampling = service_client.create_sampling_client(model_path=resumed_weights.path)
        resumed_res = resumed_sampling.sample(
            prompt=types.ModelInput.from_ints(
                tokenizer.encode(TEST_PROMPTS[1], add_special_tokens=True)
            ),
            num_samples=1,
            sampling_params=types.SamplingParams(max_tokens=4, temperature=0.1, top_p=1.0),
        ).result(timeout=60)
        assert resumed_res.sequences and resumed_res.sequences[0].tokens
    finally:
        if service_client is not None:
            service_client.holder.close()
        if server and thread and client:
            _stop_server(server, thread, client)
        clear_ray_state()
        if file_redis_path.exists():
            file_redis_path.unlink()
