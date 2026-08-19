from __future__ import annotations

import os
import socket
import threading
import time
from pathlib import Path

import httpx
import pytest
import ray
import uvicorn
from tinker import types
from tinker.lib.public_interfaces.service_client import ServiceClient

from tuft.config import AppConfig, ModelConfig
from tuft.server import create_root_app

from .helpers import CPU_TEST_TIMEOUT, clear_ray_state


pytest.importorskip("h2")


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture(scope="module")
def server_endpoint(tmp_path_factory: pytest.TempPathFactory, request):
    # Clear TINKER_API_KEY to ensure the test uses the explicitly passed API key
    # instead of the environment variable
    saved_api_key = os.environ.pop("TINKER_API_KEY", None)
    ray.init(ignore_reinit_error=True)
    if request.config.getoption("--gpu"):
        assert "TUFT_TEST_MODEL" in os.environ, (
            "Environment variable TUFT_TEST_MODEL must be set for this test."
        )
        model_path = Path(os.environ.get("TUFT_TEST_MODEL", "Qwen/Qwen3-0.6B"))
    else:
        model_path = Path("/dummy/qwen-model")
    checkpoint_dir = tmp_path_factory.mktemp("checkpoints")
    config = AppConfig(checkpoint_dir=Path(checkpoint_dir))
    config.supported_models = [
        ModelConfig(
            model_name="Qwen/Qwen3-0.6B",
            model_path=model_path,
            max_model_len=4096,
            tensor_parallel_size=1,
            sampling_memory_fraction=0.5,
        )
    ]
    config.authorized_users = {
        "tml-test-key-1": "default",
        "tml-test-key-2": "default",
    }
    app = create_root_app(config)
    port = _find_free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    base_url = f"http://127.0.0.1:{port}"
    client = httpx.Client()
    for _ in range(120):
        try:
            response = client.get(f"{base_url}/api/v1/healthz", timeout=1)
            if response.status_code == 200:
                break
        except httpx.HTTPError:
            time.sleep(2)
    else:
        server.should_exit = True
        thread.join(timeout=5)
        client.close()
        raise RuntimeError("Server failed to start")

    yield base_url

    server.should_exit = True
    thread.join(timeout=5)
    client.close()
    clear_ray_state()
    # Restore TINKER_API_KEY if it was set
    if saved_api_key is not None:
        os.environ["TINKER_API_KEY"] = saved_api_key


@pytest.mark.integration
def test_training_and_sampling_round_trip(server_endpoint: str) -> None:
    service_client = ServiceClient(
        api_key="tml-test-key-1",  # pragma: allowlist secret
        base_url=server_endpoint,
        timeout=CPU_TEST_TIMEOUT,
    )
    try:
        capabilities = service_client.get_server_capabilities()
        assert capabilities.supported_models, "server did not report supported models"
        base_model = capabilities.supported_models[0].model_name or "Qwen/Qwen3-0.6B"

        training_client = service_client.create_lora_training_client(
            base_model=base_model, rank=8, train_unembed=False
        )
        datum = types.Datum(
            model_input=types.ModelInput.from_ints([11, 12, 13, 14]),
            loss_fn_inputs={
                "target_tokens": types.TensorData(data=[21, 22, 23, 24], dtype="int64", shape=[4]),
                "weights": types.TensorData(data=[1.0, 1.0, 1.0, 1.0], dtype="float32", shape=[4]),
            },
        )

        forward_result = training_client.forward([datum], "cross_entropy").result(
            timeout=CPU_TEST_TIMEOUT
        )
        assert forward_result.loss_fn_outputs

        fwdbwd_result = training_client.forward_backward([datum], "cross_entropy").result(
            timeout=CPU_TEST_TIMEOUT
        )
        assert fwdbwd_result.metrics["loss:sum"] >= 0

        optim_result = training_client.optim_step(types.AdamParams(learning_rate=1e-3)).result(
            timeout=CPU_TEST_TIMEOUT
        )
        # optim_result.metrics may be empty, so just check optim_result is not None
        assert optim_result is not None

        save_response = training_client.save_state("checkpoint-test").result(
            timeout=CPU_TEST_TIMEOUT
        )
        sampler_response = training_client.save_weights_for_sampler("sampler-test").result(
            timeout=CPU_TEST_TIMEOUT
        )
        assert save_response.path.startswith("tinker://")
        assert sampler_response.path.startswith("tinker://")

        rest_client = service_client.create_rest_client()
        model_id = training_client.model_id
        checkpoints = rest_client.list_checkpoints(model_id).result(timeout=CPU_TEST_TIMEOUT)
        assert len(checkpoints.checkpoints) >= 2

        rest_client.publish_checkpoint_from_tinker_path(save_response.path).result(
            timeout=CPU_TEST_TIMEOUT
        )
        refreshed = rest_client.list_checkpoints(model_id).result(timeout=CPU_TEST_TIMEOUT)
        published = [
            ckpt for ckpt in refreshed.checkpoints if ckpt.checkpoint_id == "checkpoint-test"
        ]
        assert published and published[0].public is True

        weights_info = rest_client.get_weights_info_by_tinker_path(save_response.path).result(
            timeout=CPU_TEST_TIMEOUT
        )
        assert weights_info.base_model == base_model

        archive = rest_client.get_checkpoint_archive_url(model_id, "checkpoint-test").result(
            timeout=CPU_TEST_TIMEOUT
        )
        assert archive.url.startswith("file:")

        # create sampling client from saved checkpoint
        sampling_client = service_client.create_sampling_client(model_path=sampler_response.path)

        sample_res = sampling_client.sample(
            prompt=types.ModelInput.from_ints([99, 5, 12]),
            num_samples=1,
            sampling_params=types.SamplingParams(max_tokens=5, temperature=0.5),
        ).result(timeout=60)

        assert sample_res.sequences and sample_res.sequences[0].tokens

        # create sampling client from base model
        sampling_client_base = service_client.create_sampling_client(base_model=base_model)
        sample_res = sampling_client_base.sample(
            prompt=types.ModelInput.from_ints([99, 5, 12]),
            num_samples=1,
            sampling_params=types.SamplingParams(max_tokens=5, temperature=0.5),
        ).result(timeout=CPU_TEST_TIMEOUT)
        assert sample_res.sequences and sample_res.sequences[0].tokens

        training_runs = rest_client.list_training_runs().result(timeout=CPU_TEST_TIMEOUT)
        assert training_runs.training_runs and training_runs.cursor.total_count >= 1

        session_id = service_client.holder.get_session_id()
        session_info = rest_client.get_session(session_id).result(timeout=CPU_TEST_TIMEOUT)
        assert model_id in session_info.training_run_ids

        sessions = rest_client.list_sessions().result(timeout=CPU_TEST_TIMEOUT)
        assert session_id in sessions.sessions

        all_checkpoints = rest_client.list_user_checkpoints().result(timeout=CPU_TEST_TIMEOUT)
        assert any(ckpt.public for ckpt in all_checkpoints.checkpoints)
    finally:
        service_client.holder.close()


@pytest.mark.integration
def test_forward_backward_custom_round_trip(server_endpoint: str) -> None:
    """Client-defined losses via the pinned SDK against an unmodified server.

    ``forward_backward_custom`` lowers onto two allowlisted ``cross_entropy``
    requests (a log-prob forward, then a backward whose weights carry the
    client-side gradients), so the server needs no custom-loss enum for it.
    This drives the real SDK entry point end to end over the protobuf wire
    protocol and checks the parts the client's autograd relies on: per-datum
    log-prob alignment, custom metrics surfacing on the combined future, and a
    normal optim_step afterwards.
    """
    import math

    import torch

    service_client = ServiceClient(
        api_key="tml-test-key-2",  # pragma: allowlist secret
        base_url=server_endpoint,
        timeout=CPU_TEST_TIMEOUT,
    )
    try:
        capabilities = service_client.get_server_capabilities()
        base_model = capabilities.supported_models[0].model_name or "Qwen/Qwen3-0.6B"
        training_client = service_client.create_lora_training_client(
            base_model=base_model, rank=8, train_unembed=False
        )

        # One datum without weights (the SDK zero-fills them for the forward
        # pass) and one with explicit weights (forwarded as-is); distinct
        # lengths make any row misalignment visible.
        data = [
            types.Datum(
                model_input=types.ModelInput.from_ints([11, 12, 13, 14]),
                loss_fn_inputs={
                    "target_tokens": types.TensorData(
                        data=[21, 22, 23, 24], dtype="int64", shape=[4]
                    ),
                },
            ),
            types.Datum(
                model_input=types.ModelInput.from_ints([15, 16, 17]),
                loss_fn_inputs={
                    "target_tokens": types.TensorData(data=[25, 26, 27], dtype="int64", shape=[3]),
                    "weights": types.TensorData(data=[1.0, 1.0, 0.0], dtype="float32", shape=[3]),
                },
            ),
        ]

        seen_shapes: list[tuple[int, ...]] = []

        def dpo_style_loss(loss_data, logprobs_list):
            seen_shapes.extend(tuple(lp.shape) for lp in logprobs_list)
            assert len(loss_data) == len(logprobs_list)
            chosen, rejected = logprobs_list[0].sum(), logprobs_list[1].sum()
            loss = -torch.nn.functional.logsigmoid(0.5 * (chosen - rejected))
            return loss, {"custom_dpo:sum": float(loss.item())}

        result = training_client.forward_backward_custom(data, dpo_style_loss).result(
            timeout=CPU_TEST_TIMEOUT
        )

        # The callback saw one log-prob row per datum, in order and trimmed to
        # each datum's own length.
        assert seen_shapes == [(4,), (3,)]
        # Client-side metrics are merged into the backward pass's result, next
        # to the server's own metrics for the surrogate loss.
        assert math.isfinite(result.metrics["custom_dpo:sum"])
        assert "loss:sum" in result.metrics
        assert len(result.loss_fn_outputs) == len(data)

        optim_result = training_client.optim_step(types.AdamParams(learning_rate=1e-3)).result(
            timeout=CPU_TEST_TIMEOUT
        )
        assert optim_result is not None

        # The SDK validates inputs before anything reaches the server: custom
        # losses accept only target_tokens and weights per datum.
        bad_datum = types.Datum(
            model_input=types.ModelInput.from_ints([11, 12]),
            loss_fn_inputs={
                "target_tokens": types.TensorData(data=[21, 22], dtype="int64", shape=[2]),
                "advantages": types.TensorData(data=[1.0, 1.0], dtype="float32", shape=[2]),
            },
        )
        with pytest.raises(ValueError, match="advantages"):
            training_client.forward_backward_custom([bad_datum], dpo_style_loss)

        # A callback failure surfaces client-side as well.
        def broken_loss(loss_data, logprobs_list):
            raise RuntimeError("client callback exploded")

        with pytest.raises(RuntimeError, match="client callback exploded"):
            training_client.forward_backward_custom(data, broken_loss)

        # The failed callback happened after pass 1 consumed a sequence ID. A
        # subsequent valid request must still run, proving that the abandoned
        # custom step did not strand the server-side sequence executor.
        recovered = training_client.forward_backward_custom(data, dpo_style_loss).result(
            timeout=CPU_TEST_TIMEOUT
        )
        assert math.isfinite(recovered.metrics["custom_dpo:sum"])
        training_client.optim_step(types.AdamParams(learning_rate=1e-3)).result(
            timeout=CPU_TEST_TIMEOUT
        )
    finally:
        service_client.holder.close()
