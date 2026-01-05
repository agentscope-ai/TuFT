from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from fastapi import HTTPException, status

from llm_rpc.config import AppConfig, ModelConfig
from llm_rpc.state import ServerState
from tinker import types


@pytest.fixture(scope="function", autouse=True)
def ray_cluster(request):
    if request.config.getoption("--gpu"):
        import ray

        ray.init(ignore_reinit_error=True)
        yield
        ray.shutdown()
        return
    yield


def _build_state(tmp_path, use_gpu: bool = False) -> ServerState:
    if use_gpu:
        assert (
            "LLM_RPC_TEST_MODEL" in os.environ
        ), "Environment variable LLM_RPC_TEST_MODEL must be set for this test."
        model_path = Path(os.environ.get("LLM_RPC_TEST_MODEL", "Qwen/Qwen3-0.6B"))
    else:
        model_path = Path("/path/to/model")

    config = AppConfig(checkpoint_dir=tmp_path)
    config.supported_base_models = [
        ModelConfig(
            model_name="Qwen/Qwen3-0.6B",
            model_path=model_path,
            max_model_len=2048,
            tensor_parallel_size=1,
        )
    ]
    return ServerState(config)


def _create_session(state: ServerState) -> str:
    session = state.create_session(
        types.CreateSessionRequest(tags=["test"], user_metadata=None, sdk_version="1.0"),
    )
    return session.session_id


@pytest.mark.asyncio
async def test_sampling_session_requires_seq_id(request, tmp_path) -> None:
    use_gpu = request.config.getoption("--gpu")
    state = _build_state(tmp_path, use_gpu)
    session_id = _create_session(state)
    sampling_session_id = state.create_sampling_session(
        session_id=session_id,
        base_model="Qwen/Qwen3-0.6B",
        model_path=None,
        session_seq_id=1,
    )
    request = types.SampleRequest(
        prompt=types.ModelInput.from_ints([1, 2, 3]),
        num_samples=1,
        sampling_params=types.SamplingParams(max_tokens=2, temperature=0.1),
        sampling_session_id=sampling_session_id,
    )
    with pytest.raises(HTTPException) as excinfo:
        await state.run_sample(request)
    assert excinfo.value.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT


@pytest.mark.asyncio
async def test_sampling_session_seq_id_must_increase(request, tmp_path) -> None:
    use_gpu = request.config.getoption("--gpu")
    state = _build_state(tmp_path, use_gpu)
    session_id = _create_session(state)
    sampling_session_id = state.create_sampling_session(
        session_id=session_id,
        base_model="Qwen/Qwen3-0.6B",
        model_path=None,
        session_seq_id=10,
    )
    first_request = types.SampleRequest(
        prompt=types.ModelInput.from_ints([5, 6, 7]),
        num_samples=1,
        sampling_params=types.SamplingParams(max_tokens=1, temperature=0.5),
        sampling_session_id=sampling_session_id,
        seq_id=1,
    )
    response = await state.run_sample(first_request)
    assert response.sequences
    record = state.sampling.sampling_sessions[sampling_session_id]
    assert record.last_seq_id == 1
    assert record.history and record.history[0].prompt_token_count == 3

    repeat_request = first_request.model_copy(update={"seq_id": 1})
    with pytest.raises(HTTPException) as excinfo:
        await state.run_sample(repeat_request)
    assert excinfo.value.status_code == status.HTTP_409_CONFLICT
    assert excinfo.value.detail == "sequence_conflict"


def test_training_seq_id_enforced(tmp_path) -> None:
    state = _build_state(tmp_path)
    session_id = _create_session(state)
    training = state.create_model(
        session_id,
        base_model="Qwen/Qwen3-0.6B",
        lora_rank=4,
        user_metadata=None,
    )
    datum = types.Datum(
        model_input=types.ModelInput.from_ints([11, 12, 13]),
        loss_fn_inputs={
            "target_tokens": types.TensorData(data=[21, 22, 23], dtype="int64", shape=[3])
        },
    )

    state.run_forward(
        training.training_run_id,
        [datum],
        "cross_entropy",
        None,
        seq_id=1,
        backward=False,
    )

    with pytest.raises(HTTPException) as excinfo:
        state.run_forward(
            training.training_run_id,
            [datum],
            "cross_entropy",
            None,
            seq_id=1,
            backward=False,
        )
    assert excinfo.value.status_code == status.HTTP_409_CONFLICT
    assert excinfo.value.detail == "sequence_conflict"

    state.run_forward(
        training.training_run_id,
        [datum],
        "cross_entropy",
        None,
        seq_id=2,
        backward=False,
    )


def test_checkpoint_metadata_persisted(tmp_path) -> None:
    state = _build_state(tmp_path)
    session_id = _create_session(state)
    training = state.create_model(
        session_id, base_model="Qwen/Qwen3-0.6B", lora_rank=4, user_metadata=None
    )

    checkpoint = state.save_checkpoint(training.training_run_id, "ckpt-metadata", "training")
    metadata = json.loads(checkpoint.path.read_text())
    assert metadata["name"] == "ckpt-metadata"
    assert metadata["session_id"] == session_id
    assert metadata["checkpoint_type"] == "training"
    assert metadata["tinker_path"].startswith("tinker://")
    assert metadata["public"] is False
    assert "model_state" in metadata and metadata["model_state"]

    state.set_checkpoint_visibility(training.training_run_id, "ckpt-metadata", public=True)
    updated = json.loads(checkpoint.path.read_text())
    assert updated["public"] is True
    assert updated["model_state"] == metadata["model_state"]

    listed = state.list_user_checkpoints()
    assert listed and listed[0].checkpoint_id == "ckpt-metadata"


def test_checkpoint_views_reflect_metadata(tmp_path) -> None:
    state = _build_state(tmp_path)
    session_id = _create_session(state)
    training = state.create_model(
        session_id, base_model="Qwen/Qwen3-0.6B", lora_rank=2, user_metadata=None
    )

    training_ckpt = state.save_checkpoint(training.training_run_id, None, "training")
    sampler_ckpt = state.save_checkpoint(training.training_run_id, None, "sampler")

    listed = state.list_checkpoints(training.training_run_id)
    assert {ckpt.checkpoint_type for ckpt in listed} == {"training", "sampler"}
    assert all(ckpt.size_bytes is not None and ckpt.size_bytes > 0 for ckpt in listed)

    metadata = json.loads(sampler_ckpt.path.read_text())
    assert metadata["checkpoint_type"] == "sampler"
    assert metadata["tinker_path"].endswith(sampler_ckpt.checkpoint_id)
    assert metadata["model_state"]

    info = state.get_weights_info(training_ckpt.to_api(training.training_run_id).tinker_path)
    assert info.base_model == "Qwen/Qwen3-0.6B"


def test_load_checkpoint_restores_state(tmp_path) -> None:
    state = _build_state(tmp_path)
    session_id = _create_session(state)
    training = state.create_model(
        session_id, base_model="Qwen/Qwen3-0.6B", lora_rank=4, user_metadata=None
    )

    datum = types.Datum(
        model_input=types.ModelInput.from_ints([3, 4, 5, 6]),
        loss_fn_inputs={
            "target_tokens": types.TensorData(data=[7, 8, 9, 10], dtype="int64", shape=[4])
        },
    )
    state.run_forward(
        training.training_run_id,
        [datum],
        "cross_entropy",
        None,
        seq_id=None,
        backward=True,
    )
    state.run_optim_step(training.training_run_id, types.AdamParams(), seq_id=None)

    checkpoint = state.save_checkpoint(training.training_run_id, "restore-test", "training")
    saved_state = training.backend.snapshot_state()

    mutated_state = dict(saved_state)
    mutated_state["weights"] = [value + 0.5 for value in saved_state["weights"]]
    training.backend.load_state(mutated_state)

    ckpt_path = checkpoint.to_api(training.training_run_id).tinker_path
    state.load_checkpoint(training.training_run_id, ckpt_path, optimizer=True)
    restored = training.backend.snapshot_state()
    assert restored["weights"] == saved_state["weights"]
