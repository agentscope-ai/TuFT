from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from tinker import types

from tuft.auth import User
from tuft.config import AppConfig, ModelConfig
from tuft.exceptions import (
    CheckpointAccessDeniedException,
    CheckpointIncompatibleException,
    InvalidRequestException,
    LossFunctionMissingInputException,
    MissingSequenceIDException,
    SequenceConflictException,
    UnknownModelException,
    UserMismatchException,
)
from tuft.state import ServerState
from tuft.training_controller import TrainingController, TrainingRunRecord

from .helpers import clear_ray_state


@pytest.fixture(scope="function", autouse=True)
def ray_cluster(request):
    if request.config.getoption("--gpu"):
        import ray

        ray.init(ignore_reinit_error=True)
        yield
        clear_ray_state()
        return
    yield


async def _build_state(
    tmp_path,
    use_gpu: bool = False,
    extra_base_models: list[str] | None = None,
    cpu_model_path: str = "/path/to/qwen-test-model",
) -> ServerState:
    """cpu_model_path picks the fake weights path for CPU runs.

    The default resolves as Qwen so every newly created run records concrete
    target geometry, as production runs are now required to do.
    """
    if use_gpu:
        assert "TUFT_TEST_MODEL" in os.environ, (
            "Environment variable TUFT_TEST_MODEL must be set for this test."
        )
        model_path = Path(os.environ.get("TUFT_TEST_MODEL", "Qwen/Qwen3-0.6B"))
    else:
        model_path = Path(cpu_model_path)

    config = AppConfig(checkpoint_dir=tmp_path)
    # extra_base_models reuse model_path: they exist to give a test a second
    # distinct base_model to route against, not a second set of weights.
    config.supported_models = [
        ModelConfig(
            model_name=model_name,
            model_path=model_path,
            max_model_len=2048,
            tensor_parallel_size=1,
            sampling_memory_fraction=0.6,
        )
        for model_name in ["Qwen/Qwen3-0.6B", *(extra_base_models or [])]
    ]
    state = ServerState(config)
    await state.async_init()
    return state


def _create_session(state: ServerState, user_id: str = "tester") -> str:
    session = state.create_session(
        types.CreateSessionRequest(tags=["test"], user_metadata=None, sdk_version="1.0"),
        user=User(user_id=user_id),
    )
    return session.session_id


def test_effective_target_modules_rejects_unknown_model_without_assert() -> None:
    """The request path keeps its typed error when Python assertions are disabled."""
    controller = object.__new__(TrainingController)
    controller.config = AppConfig()

    with pytest.raises(UnknownModelException, match="Unknown model: missing"):
        controller._effective_target_modules("missing", types.LoraConfig(rank=8))


def test_effective_target_modules_records_qv_only_mode() -> None:
    """Persisted runs record the geometry selected by the dedicated Q/V opt-in."""
    model_config = ModelConfig(
        model_name="qv",
        model_path=Path("/tmp/qwen-model"),
        max_model_len=1024,
        training_backend="fsdp",
        fsdp_qv_only=True,
    )
    controller = object.__new__(TrainingController)
    controller.config = AppConfig(supported_models=[model_config])

    assert controller._effective_target_modules("qv", types.LoraConfig(rank=8)) == [
        "q_proj",
        "v_proj",
    ]


def test_checkpoint_compatibility_rejects_destination_without_geometry() -> None:
    """A legacy destination produces a clean request error rather than a set(None) failure."""
    model_config = ModelConfig(
        model_name="base",
        model_path=Path("/tmp/qwen-model"),
        max_model_len=1024,
    )
    controller = object.__new__(TrainingController)
    controller.config = AppConfig(supported_models=[model_config])
    destination = TrainingRunRecord(
        training_run_id="destination",
        base_model="base",
        lora_rank=8,
        train_attn=True,
        train_mlp=True,
        train_unembed=True,
        target_modules=None,
        session_id="session",
        model_owner="tester",
    )
    checkpoint = MagicMock()
    checkpoint.saved_target_modules = ["q_proj", "v_proj"]
    metadata = MagicMock(base_model="base", lora_rank=8)

    with pytest.raises(InvalidRequestException, match="does not record effective LoRA"):
        controller._check_adapter_compatible(
            checkpoint_id="checkpoint",
            checkpoint=checkpoint,
            metadata=metadata,
            destination=destination,
        )


@pytest.mark.asyncio
async def test_sampling_session_requires_seq_id(request, tmp_path) -> None:
    use_gpu = request.config.getoption("--gpu")
    state = await _build_state(tmp_path, use_gpu)
    session_id = _create_session(state)
    sampling_session_id = await state.create_sampling_session(
        session_id=session_id,
        base_model="Qwen/Qwen3-0.6B",
        model_path=None,
        session_seq_id=1,
        user_id="tester",
    )
    request = types.SampleRequest(
        prompt=types.ModelInput.from_ints([1, 2, 3]),
        num_samples=1,
        sampling_params=types.SamplingParams(max_tokens=2, temperature=0.1),
        sampling_session_id=sampling_session_id,
    )
    with pytest.raises(MissingSequenceIDException) as excinfo:
        await state.run_sample(request, user_id="tester")
    assert excinfo.value.detail == "Missing sequence ID in the request."

    with pytest.raises(UserMismatchException) as excinfo2:
        await state.run_sample(
            request,
            user_id="different_user",
        )
    assert "You do not have permission" in str(excinfo2.value)


@pytest.mark.asyncio
async def test_sampling_session_wrong_user(request, tmp_path) -> None:
    """Test that sampling session access is restricted to the correct user."""
    use_gpu = request.config.getoption("--gpu")
    state = await _build_state(tmp_path, use_gpu)
    session_id = _create_session(state)
    sampling_session_id = await state.create_sampling_session(
        session_id=session_id,
        base_model="Qwen/Qwen3-0.6B",
        model_path=None,
        session_seq_id=1,
        user_id="tester",
    )
    request = types.SampleRequest(
        prompt=types.ModelInput.from_ints([1, 2, 3]),
        num_samples=1,
        sampling_params=types.SamplingParams(max_tokens=2, temperature=0.1),
        sampling_session_id=sampling_session_id,
        seq_id=1,
    )

    with pytest.raises(UserMismatchException) as excinfo:
        await state.run_sample(
            request,
            user_id="different_user",
        )
    assert "You do not have permission" in str(excinfo.value)


@pytest.mark.asyncio
async def test_sampling_session_cocurrent(request, tmp_path) -> None:
    use_gpu = request.config.getoption("--gpu")
    state = await _build_state(tmp_path, use_gpu)
    session_id = _create_session(state)
    sampling_session_id = await state.create_sampling_session(
        session_id=session_id,
        base_model="Qwen/Qwen3-0.6B",
        model_path=None,
        session_seq_id=10,
        user_id="tester",
    )
    requests = [
        types.SampleRequest(
            prompt=types.ModelInput.from_ints([5, 6, 7]),
            num_samples=1,
            sampling_params=types.SamplingParams(max_tokens=1, temperature=0.5),
            sampling_session_id=sampling_session_id,
            seq_id=i,
        )
        for i in range(10)
    ]
    response = await asyncio.gather(*[state.run_sample(req, user_id="tester") for req in requests])
    for resp in response:
        assert resp.sequences is not None
        assert len(resp.sequences) == 1
        assert resp.sequences[0].tokens is not None
        assert len(resp.sequences[0].tokens) > 0


@pytest.mark.asyncio
async def test_sampling_seq_id_history_is_monotonic(request, tmp_path) -> None:
    use_gpu = request.config.getoption("--gpu")
    state = await _build_state(tmp_path, use_gpu)
    session_id = _create_session(state)
    sampling_session_id = await state.create_sampling_session(
        session_id=session_id,
        base_model="Qwen/Qwen3-0.6B",
        model_path=None,
        session_seq_id=1,
        user_id="tester",
    )

    req1 = types.SampleRequest(
        prompt=types.ModelInput.from_ints([1, 2, 3]),
        num_samples=1,
        sampling_params=types.SamplingParams(max_tokens=1, temperature=0.1),
        sampling_session_id=sampling_session_id,
        seq_id=1,
    )
    req0 = types.SampleRequest(
        prompt=types.ModelInput.from_ints([4, 5, 6]),
        num_samples=1,
        sampling_params=types.SamplingParams(max_tokens=1, temperature=0.1),
        sampling_session_id=sampling_session_id,
        seq_id=0,
    )

    await state.run_sample(req1, user_id="tester")
    await state.run_sample(req0, user_id="tester")

    record = state.sampling.sampling_sessions[sampling_session_id]
    assert record.last_seq_id == 1
    assert [entry.seq_id for entry in record.history] == [0, 1]


@pytest.mark.asyncio
async def test_sampling_duplicate_seq_id_overwrites_history_entry(request, tmp_path) -> None:
    use_gpu = request.config.getoption("--gpu")
    state = await _build_state(tmp_path, use_gpu)
    session_id = _create_session(state)
    sampling_session_id = await state.create_sampling_session(
        session_id=session_id,
        base_model="Qwen/Qwen3-0.6B",
        model_path=None,
        session_seq_id=1,
        user_id="tester",
    )

    req = types.SampleRequest(
        prompt=types.ModelInput.from_ints([1, 2, 3]),
        num_samples=1,
        sampling_params=types.SamplingParams(max_tokens=1, temperature=0.1),
        sampling_session_id=sampling_session_id,
        seq_id=0,
    )

    await state.run_sample(req, user_id="tester")

    req_updated = types.SampleRequest(
        prompt=types.ModelInput.from_ints([9, 9, 9, 9]),
        num_samples=1,
        sampling_params=types.SamplingParams(max_tokens=1, temperature=0.1),
        sampling_session_id=sampling_session_id,
        seq_id=0,
    )

    await state.run_sample(req_updated, user_id="tester")

    record = state.sampling.sampling_sessions[sampling_session_id]
    assert record.last_seq_id == 0
    assert len(record.history) == 1
    assert record.history[0].seq_id == 0
    assert record.history[0].prompt_token_count == 4


@pytest.mark.asyncio
async def test_training_seq_id_enforced(request, tmp_path) -> None:
    use_gpu = request.config.getoption("--gpu")
    state = await _build_state(tmp_path, use_gpu)
    session_id = _create_session(state)
    training = await state.create_model(
        session_id,
        base_model="Qwen/Qwen3-0.6B",
        lora_config=types.LoraConfig(rank=4, train_unembed=False),
        model_owner="tester",
        user_metadata=None,
    )
    datum = types.Datum(
        model_input=types.ModelInput.from_ints([11, 12, 13]),
        loss_fn_inputs={
            "target_tokens": types.TensorData(data=[21, 22, 23], dtype="int64", shape=[3]),
            "weights": types.TensorData(data=[1.0, 1.0, 1.0], dtype="float32", shape=[3]),
        },
    )

    await state.run_forward(
        training.training_run_id,
        user_id="tester",
        data=[datum],
        loss_fn="cross_entropy",
        loss_fn_config=None,
        seq_id=1,
        backward=False,
    )

    with pytest.raises(SequenceConflictException) as excinfo:
        await state.run_forward(
            training.training_run_id,
            user_id="tester",
            data=[datum],
            loss_fn="cross_entropy",
            loss_fn_config=None,
            seq_id=1,
            backward=False,
        )
    assert excinfo.value.detail == "Sequence conflict: expected 2, got 1."

    # failed operatrion will not increase seq_id, so seq_id=2 is still expected
    with pytest.raises(LossFunctionMissingInputException) as excinfo:
        await state.run_forward(
            training.training_run_id,
            user_id="tester",
            data=[datum],
            loss_fn="importance_sampling",  # raise LossFunctionMissingInputException
            loss_fn_config={
                "raise_missing_input": 1.0,
            },
            seq_id=2,
            backward=False,
        )
    # should be executed successfully
    await state.run_forward(
        training.training_run_id,
        user_id="tester",
        data=[datum],
        loss_fn="cross_entropy",
        loss_fn_config=None,
        seq_id=2,
        backward=False,
    )


@pytest.mark.asyncio
async def test_training_user_mismatch(request, tmp_path) -> None:
    """Test that training operations are restricted to the correct user."""
    use_gpu = request.config.getoption("--gpu")
    state = await _build_state(tmp_path, use_gpu)
    session_id = _create_session(state)
    training = await state.create_model(
        session_id,
        base_model="Qwen/Qwen3-0.6B",
        lora_config=types.LoraConfig(rank=4, train_unembed=False),
        model_owner="tester",
        user_metadata=None,
    )
    datum = types.Datum(
        model_input=types.ModelInput.from_ints([31, 32, 33]),
        loss_fn_inputs={
            "target_tokens": types.TensorData(data=[41, 42, 43], dtype="int64", shape=[3]),
            "weights": types.TensorData(data=[1.0, 1.0, 1.0], dtype="float32", shape=[3]),
        },
    )

    with pytest.raises(UserMismatchException) as excinfo:
        await state.run_forward(
            training.training_run_id,
            user_id="wrong_user",
            data=[datum],
            loss_fn="cross_entropy",
            loss_fn_config=None,
            seq_id=1,
            backward=False,
        )

    with pytest.raises(UserMismatchException) as excinfo:
        await state.run_optim_step(
            training.training_run_id,
            user_id="wrong_user",
            params=types.AdamParams(),
            seq_id=1,
        )

    assert "You do not have permission" in str(excinfo.value)


@pytest.mark.asyncio
async def test_checkpoint_metadata_persisted(request, tmp_path) -> None:
    use_gpu = request.config.getoption("--gpu")
    state = await _build_state(tmp_path, use_gpu)
    session_id = _create_session(state)
    training = await state.create_model(
        session_id,
        base_model="Qwen/Qwen3-0.6B",
        lora_config=types.LoraConfig(rank=4, train_unembed=False),
        model_owner="tester",
        user_metadata=None,
    )

    checkpoint = await state.save_checkpoint(
        training.training_run_id,
        user_id="tester",
        name="ckpt-metadata",
        checkpoint_type="training",
    )
    metadata = checkpoint.metadata
    assert metadata.name == "ckpt-metadata"
    assert metadata.session_id == session_id
    assert metadata.checkpoint_type == "training"
    assert metadata.tinker_path.startswith("tinker://")
    assert metadata.public is False
    assert metadata.owner_name == "tester"

    state.set_checkpoint_visibility(
        training.training_run_id,
        user_id="tester",
        checkpoint_id="ckpt-metadata",
        public=True,
    )
    updated = checkpoint.metadata
    assert updated.public is True
    listed = state.list_user_checkpoints(user_id="tester")
    assert listed and listed[0].checkpoint_id == "ckpt-metadata"
    listed_different_user = state.list_user_checkpoints(user_id="other_user")
    assert not listed_different_user


@pytest.mark.asyncio
async def test_checkpoint_views_reflect_metadata(request, tmp_path) -> None:
    use_gpu = request.config.getoption("--gpu")
    state = await _build_state(tmp_path, use_gpu)
    session_id = _create_session(state)
    training = await state.create_model(
        session_id,
        model_owner="tester",
        base_model="Qwen/Qwen3-0.6B",
        lora_config=types.LoraConfig(rank=2, train_unembed=False),
        user_metadata=None,
    )

    training_ckpt = await state.save_checkpoint(
        training.training_run_id,
        user_id="tester",
        name=None,
        checkpoint_type="training",
    )
    sampler_ckpt = await state.save_checkpoint(
        training.training_run_id,
        user_id="tester",
        name=None,
        checkpoint_type="sampler",
    )

    listed = state.list_checkpoints(training.training_run_id, user_id="tester")
    assert {ckpt.checkpoint_type for ckpt in listed} == {"training", "sampler"}
    assert all(ckpt.size_bytes is not None and ckpt.size_bytes > 0 for ckpt in listed)

    metadata = sampler_ckpt.metadata
    assert metadata.checkpoint_type == "sampler"
    assert metadata.tinker_path.endswith(sampler_ckpt.checkpoint_id)

    info = state.get_weights_info(training_ckpt.tinker_checkpoint.tinker_path, user_id="tester")
    assert info.base_model == "Qwen/Qwen3-0.6B"


@pytest.mark.asyncio
async def test_load_checkpoint_restores_state(request, tmp_path) -> None:
    use_gpu = request.config.getoption("--gpu")
    state = await _build_state(tmp_path, use_gpu)
    session_id = _create_session(state)
    training = await state.create_model(
        session_id,
        model_owner="tester",
        base_model="Qwen/Qwen3-0.6B",
        lora_config=types.LoraConfig(rank=4, train_unembed=False),
        user_metadata=None,
    )

    datum = types.Datum(
        model_input=types.ModelInput.from_ints([3, 4, 5, 6]),
        loss_fn_inputs={
            "target_tokens": types.TensorData(data=[7, 8, 9, 10], dtype="int64", shape=[4]),
            "weights": types.TensorData(data=[1.0, 1.0, 1.0, 1.0], dtype="float32", shape=[4]),
        },
    )
    await state.run_forward(
        training.training_run_id,
        user_id="tester",
        data=[datum],
        loss_fn="cross_entropy",
        loss_fn_config=None,
        seq_id=None,
        backward=True,
    )
    await state.run_optim_step(
        training.training_run_id,
        user_id="tester",
        params=types.AdamParams(),
        seq_id=None,
    )

    checkpoint = await state.save_checkpoint(
        training.training_run_id,
        user_id="tester",
        name="restore-test",
        checkpoint_type="training",
    )

    ckpt_path = checkpoint.tinker_checkpoint.tinker_path
    await state.load_checkpoint(
        training.training_run_id, path=ckpt_path, user_id="tester", optimizer=True
    )

    with pytest.raises(CheckpointAccessDeniedException) as excinfo:
        await state.load_checkpoint(
            training.training_run_id, path=ckpt_path, user_id="wrong_user", optimizer=True
        )
    assert "Access to checkpoint restore-test is denied." in str(excinfo.value)


@pytest.mark.asyncio
async def test_load_checkpoint_into_new_run_uses_destination_sequence_and_adapter(
    request, tmp_path, monkeypatch
) -> None:
    use_gpu = request.config.getoption("--gpu")
    state = await _build_state(tmp_path, use_gpu)
    session_id = _create_session(state)
    source = await state.create_model(
        session_id,
        model_owner="tester",
        base_model="Qwen/Qwen3-0.6B",
        lora_config=types.LoraConfig(rank=4, train_unembed=False),
        user_metadata=None,
    )

    datum = types.Datum(
        model_input=types.ModelInput.from_ints([3, 4, 5, 6]),
        loss_fn_inputs={
            "target_tokens": types.TensorData(data=[7, 8, 9, 10], dtype="int64", shape=[4]),
            "weights": types.TensorData(data=[1.0, 1.0, 1.0, 1.0], dtype="float32", shape=[4]),
        },
    )
    await state.run_forward(
        source.training_run_id,
        user_id="tester",
        data=[datum],
        loss_fn="cross_entropy",
        loss_fn_config=None,
        seq_id=1,
        backward=False,
    )
    checkpoint = await state.save_checkpoint(
        source.training_run_id,
        user_id="tester",
        name="cross-run-restore-test",
        checkpoint_type="training",
        seq_id=2,
    )

    destination = await state.create_model(
        session_id,
        model_owner="tester",
        base_model="Qwen/Qwen3-0.6B",
        lora_config=types.LoraConfig(rank=4, train_unembed=False),
        user_metadata=None,
    )
    backend = state.training.training_backends["Qwen/Qwen3-0.6B"]
    original_load_state = backend.load_state
    loaded_lora_ids: list[str] = []

    async def recording_load_state(*, lora_id, checkpoint_record, optimizer):
        loaded_lora_ids.append(lora_id)
        await original_load_state(
            lora_id=lora_id,
            checkpoint_record=checkpoint_record,
            optimizer=optimizer,
        )

    monkeypatch.setattr(backend, "load_state", recording_load_state)

    await state.load_checkpoint(
        destination.training_run_id,
        path=checkpoint.tinker_checkpoint.tinker_path,
        user_id="tester",
        optimizer=True,
        seq_id=1,
    )

    source_record = state.get_training_run_record(source.training_run_id, "tester")
    destination_record = state.get_training_run_record(destination.training_run_id, "tester")
    assert source_record.next_seq_id == 3
    assert destination_record.next_seq_id == 2
    assert loaded_lora_ids == [destination.training_run_id]


@pytest.mark.asyncio
async def test_load_checkpoint_rejects_different_lora_rank(request, tmp_path) -> None:
    use_gpu = request.config.getoption("--gpu")
    state = await _build_state(tmp_path, use_gpu)
    session_id = _create_session(state)
    source = await state.create_model(
        session_id,
        model_owner="tester",
        base_model="Qwen/Qwen3-0.6B",
        lora_config=types.LoraConfig(rank=4, train_unembed=False),
        user_metadata=None,
    )
    checkpoint = await state.save_checkpoint(
        source.training_run_id,
        user_id="tester",
        name="rank-mismatch-test",
        checkpoint_type="training",
    )
    destination = await state.create_model(
        session_id,
        model_owner="tester",
        base_model="Qwen/Qwen3-0.6B",
        lora_config=types.LoraConfig(rank=8, train_unembed=False),
        user_metadata=None,
    )

    # The message names both ranks so the client knows which one to recreate at.
    with pytest.raises(
        InvalidRequestException, match="LoRA rank 4 into a training run with LoRA rank 8"
    ):
        await state.load_checkpoint(
            destination.training_run_id,
            path=checkpoint.tinker_checkpoint.tinker_path,
            user_id="tester",
            optimizer=True,
        )


@pytest.mark.asyncio
async def test_load_checkpoint_rejects_different_base_model(request, tmp_path) -> None:
    use_gpu = request.config.getoption("--gpu")
    state = await _build_state(tmp_path, use_gpu, extra_base_models=["Qwen/Qwen3-0.6B-other"])
    session_id = _create_session(state)
    source = await state.create_model(
        session_id,
        model_owner="tester",
        base_model="Qwen/Qwen3-0.6B",
        lora_config=types.LoraConfig(rank=4, train_unembed=False),
        user_metadata=None,
    )
    checkpoint = await state.save_checkpoint(
        source.training_run_id,
        user_id="tester",
        name="base-model-mismatch-test",
        checkpoint_type="training",
    )
    destination = await state.create_model(
        session_id,
        model_owner="tester",
        base_model="Qwen/Qwen3-0.6B-other",
        lora_config=types.LoraConfig(rank=4, train_unembed=False),
        user_metadata=None,
    )

    with pytest.raises(InvalidRequestException, match="Qwen/Qwen3-0.6B-other"):
        await state.load_checkpoint(
            destination.training_run_id,
            path=checkpoint.tinker_checkpoint.tinker_path,
            user_id="tester",
            optimizer=True,
        )


@pytest.mark.asyncio
async def test_load_checkpoint_rejects_different_lora_target_modules(request, tmp_path) -> None:
    """Same base model and rank, different target modules, is still incompatible.

    peft loads a checkpoint into an existing adapter without consulting the
    checkpoint's own adapter_config.json, so an unguarded load here would leave
    the unmatched modules at their random init without raising.

    Compatibility is checked from the concrete target modules persisted in the
    run and checkpoint, rather than reconstructing geometry from modifier flags.
    """
    use_gpu = request.config.getoption("--gpu")
    state = await _build_state(tmp_path, use_gpu)
    session_id = _create_session(state)
    source = await state.create_model(
        session_id,
        model_owner="tester",
        base_model="Qwen/Qwen3-0.6B",
        lora_config=types.LoraConfig(rank=4, train_mlp=True, train_unembed=False),
        user_metadata=None,
    )
    checkpoint = await state.save_checkpoint(
        source.training_run_id,
        user_id="tester",
        name="target-module-mismatch-test",
        checkpoint_type="training",
    )
    assert checkpoint.metadata.train_mlp is True

    destination = await state.create_model(
        session_id,
        model_owner="tester",
        base_model="Qwen/Qwen3-0.6B",
        lora_config=types.LoraConfig(rank=4, train_mlp=False, train_unembed=False),
        user_metadata=None,
    )

    with pytest.raises(InvalidRequestException, match="targeting LoRA modules"):
        await state.load_checkpoint(
            destination.training_run_id,
            path=checkpoint.tinker_checkpoint.tinker_path,
            user_id="tester",
            optimizer=True,
        )


@pytest.mark.asyncio
async def test_create_model_rejects_train_unembed_on_qwen(request, tmp_path) -> None:
    """Qwen has no unembed modules to train, so the flag fails instead of no-oping.

    Tinker's hosted service adapts embed_tokens on Qwen3.5-based models, so a
    silently accepted flag would train different weights than the same client
    code trains on Tinker. The SDK default is train_unembed=True, so a bare
    default LoraConfig is rejected too; the error says what to change.
    """
    use_gpu = request.config.getoption("--gpu")
    state = await _build_state(tmp_path, use_gpu, cpu_model_path="/path/to/qwen-test-model")
    session_id = _create_session(state)
    with pytest.raises(InvalidRequestException, match="train_unembed"):
        await state.create_model(
            session_id,
            model_owner="tester",
            base_model="Qwen/Qwen3-0.6B",
            lora_config=types.LoraConfig(rank=4, train_unembed=True),
            user_metadata=None,
        )
    with pytest.raises(InvalidRequestException, match="train_unembed=False to create"):
        await state.create_model(
            session_id,
            model_owner="tester",
            base_model="Qwen/Qwen3-0.6B",
            lora_config=types.LoraConfig(rank=4),
            user_metadata=None,
        )


@pytest.mark.asyncio
async def test_load_checkpoint_compares_resolved_module_sets(request, tmp_path) -> None:
    """The same resolved module set loads; a different one is rejected.

    Compatibility is decided from the concrete module sets persisted in the
    run and checkpoint, not by re-deriving geometry from raw flags.
    """
    use_gpu = request.config.getoption("--gpu")
    state = await _build_state(tmp_path, use_gpu, cpu_model_path="/path/to/qwen-test-model")
    session_id = _create_session(state)
    source = await state.create_model(
        session_id,
        model_owner="tester",
        base_model="Qwen/Qwen3-0.6B",
        lora_config=types.LoraConfig(rank=4, train_unembed=False),
        user_metadata=None,
    )
    checkpoint = await state.save_checkpoint(
        source.training_run_id,
        user_id="tester",
        name="module-set-test",
        checkpoint_type="training",
    )
    expected_modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    assert state.training.training_runs[source.training_run_id].target_modules == expected_modules
    assert checkpoint.metadata.target_modules == expected_modules

    same_geometry = await state.create_model(
        session_id,
        model_owner="tester",
        base_model="Qwen/Qwen3-0.6B",
        lora_config=types.LoraConfig(rank=4, train_unembed=False),
        user_metadata=None,
    )
    await state.load_checkpoint(
        same_geometry.training_run_id,
        path=checkpoint.tinker_checkpoint.tinker_path,
        user_id="tester",
        optimizer=True,
    )

    different_geometry = await state.create_model(
        session_id,
        model_owner="tester",
        base_model="Qwen/Qwen3-0.6B",
        lora_config=types.LoraConfig(rank=4, train_mlp=False, train_unembed=False),
        user_metadata=None,
    )
    with pytest.raises(InvalidRequestException, match="targeting LoRA modules"):
        await state.load_checkpoint(
            different_geometry.training_run_id,
            path=checkpoint.tinker_checkpoint.tinker_path,
            user_id="tester",
            optimizer=True,
        )


@pytest.mark.asyncio
async def test_legacy_run_is_invalid_but_checkpoint_can_seed_new_run(request, tmp_path) -> None:
    """Runs missing effective geometry are read-only checkpoint sources."""
    use_gpu = request.config.getoption("--gpu")
    state = await _build_state(tmp_path, use_gpu, cpu_model_path="/path/to/qwen-test-model")
    session_id = _create_session(state)
    source = await state.create_model(
        session_id,
        model_owner="tester",
        base_model="Qwen/Qwen3-0.6B",
        lora_config=types.LoraConfig(rank=4, train_mlp=False, train_unembed=False),
        user_metadata=None,
    )
    checkpoint = await state.save_checkpoint(
        source.training_run_id,
        user_id="tester",
        name="legacy-source-test",
        checkpoint_type="training",
    )
    checkpoint.adapter_path.mkdir(parents=True, exist_ok=True)
    (checkpoint.adapter_path / "adapter_config.json").write_text(
        json.dumps({"target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]}),
        encoding="utf-8",
    )
    checkpoint.save_metadata(
        base_model=source.base_model,
        session_id=source.session_id,
        lora_rank=source.lora_rank,
    )

    source_record = state.training.training_runs[source.training_run_id]
    source_record.train_attn = None
    source_record.train_mlp = None
    source_record.train_unembed = None
    restored = await state.training.restore_from_checkpoint(source.training_run_id)

    assert restored is None
    assert source_record.corrupted is True
    with pytest.raises(InvalidRequestException, match="does not record.*target geometry"):
        await state.save_checkpoint(
            source.training_run_id,
            user_id="tester",
            name="legacy-resume-test",
            checkpoint_type="training",
        )

    destination = await state.create_model(
        session_id,
        model_owner="tester",
        base_model="Qwen/Qwen3-0.6B",
        lora_config=types.LoraConfig(rank=4, train_mlp=False, train_unembed=False),
        user_metadata=None,
    )
    await state.load_checkpoint(
        destination.training_run_id,
        path=checkpoint.tinker_checkpoint.tinker_path,
        user_id="tester",
        optimizer=True,
    )

    incompatible_destination = await state.create_model(
        session_id,
        model_owner="tester",
        base_model="Qwen/Qwen3-0.6B",
        lora_config=types.LoraConfig(rank=4, train_mlp=True, train_unembed=False),
        user_metadata=None,
    )
    with pytest.raises(InvalidRequestException, match="targeting LoRA modules"):
        await state.load_checkpoint(
            incompatible_destination.training_run_id,
            path=checkpoint.tinker_checkpoint.tinker_path,
            user_id="tester",
            optimizer=True,
        )


@pytest.mark.asyncio
async def test_checkpoint_without_explicit_adapter_geometry_is_rejected(request, tmp_path) -> None:
    """Checkpoint metadata without concrete target modules is rejected clearly."""
    use_gpu = request.config.getoption("--gpu")
    state = await _build_state(tmp_path, use_gpu, cpu_model_path="/path/to/qwen-test-model")
    session_id = _create_session(state)
    source = await state.create_model(
        session_id,
        model_owner="tester",
        base_model="Qwen/Qwen3-0.6B",
        lora_config=types.LoraConfig(rank=4, train_unembed=False),
        user_metadata=None,
    )
    checkpoint = await state.save_checkpoint(
        source.training_run_id,
        user_id="tester",
        name="unverifiable-legacy-test",
        checkpoint_type="training",
    )
    checkpoint.save_metadata(
        base_model=source.base_model,
        session_id=source.session_id,
        lora_rank=source.lora_rank,
        train_attn=True,
        train_mlp=True,
        train_unembed=True,
    )
    # saved_target_modules treats adapter_config.json as ground truth and only
    # falls back to metadata, so clearing metadata alone leaves the real backend
    # (which always writes that file) fully verifiable. A regex-string target is
    # the shape this rejection exists for: PEFT accepts it, but it cannot be
    # checked against an allocated slot.
    checkpoint.adapter_path.mkdir(parents=True, exist_ok=True)
    (checkpoint.adapter_path / "adapter_config.json").write_text(
        json.dumps({"target_modules": r".*\.(q_proj|v_proj)"}),
        encoding="utf-8",
    )
    assert checkpoint.saved_target_modules is None, (
        "checkpoint still exposes a target-module list; the rejection under test is unreachable"
    )

    destination = await state.create_model(
        session_id,
        model_owner="tester",
        base_model="Qwen/Qwen3-0.6B",
        lora_config=types.LoraConfig(rank=4, train_unembed=False),
        user_metadata=None,
    )

    with pytest.raises(InvalidRequestException, match="without an explicit target-module list"):
        await state.load_checkpoint(
            destination.training_run_id,
            path=checkpoint.tinker_checkpoint.tinker_path,
            user_id="tester",
            optimizer=True,
        )


@pytest.mark.asyncio
async def test_restore_recreates_adapter_for_run_without_checkpoint(
    request, tmp_path, monkeypatch
) -> None:
    """A run with no checkpoint of its own must survive a restart.

    A run seeded only by load_weights owns no checkpoint, so restore has nothing
    to load; it still has to recreate the adapter or every later request fails
    with "Adapter not found".
    """
    use_gpu = request.config.getoption("--gpu")
    state = await _build_state(tmp_path, use_gpu)
    session_id = _create_session(state)
    training = await state.create_model(
        session_id,
        model_owner="tester",
        base_model="Qwen/Qwen3-0.6B",
        lora_config=types.LoraConfig(rank=4, train_mlp=False, train_unembed=False),
        user_metadata=None,
    )
    backend = state.training.training_backends["Qwen/Qwen3-0.6B"]
    # Simulate the restart: the record survives in Redis, the adapter does not.
    await backend.remove_adapter(training.training_run_id)

    original_create_adapter = backend.create_adapter
    created: list[tuple[str, types.LoraConfig]] = []

    async def recording_create_adapter(lora_id, lora_config):
        created.append((lora_id, lora_config))
        await original_create_adapter(lora_id, lora_config)

    monkeypatch.setattr(backend, "create_adapter", recording_create_adapter)

    restored = await state.training.restore_from_checkpoint(training.training_run_id)

    assert restored is None
    assert [lora_id for lora_id, _ in created] == [training.training_run_id]
    # Recreated with the run's own LoRA config, not a defaulted one.
    assert created[0][1].rank == 4
    assert created[0][1].train_mlp is False


@pytest.mark.asyncio
async def test_restore_marks_run_corrupted_when_recorded_geometry_is_stale(
    request, tmp_path, monkeypatch
) -> None:
    """A checkpoint-less run whose recorded modules no longer resolve is corrupted.

    Recreating its adapter would silently train the newly resolved modules
    while the record and all past metadata say otherwise.
    """
    use_gpu = request.config.getoption("--gpu")
    state = await _build_state(tmp_path, use_gpu)
    session_id = _create_session(state)
    training = await state.create_model(
        session_id,
        model_owner="tester",
        base_model="Qwen/Qwen3-0.6B",
        lora_config=types.LoraConfig(rank=4, train_unembed=False),
        user_metadata=None,
    )
    record = state.training.training_runs[training.training_run_id]
    backend = state.training.training_backends["Qwen/Qwen3-0.6B"]
    await backend.remove_adapter(training.training_run_id)
    # Simulate a record written by a release that resolved fewer modules.
    assert record.target_modules is not None
    record.target_modules = [m for m in record.target_modules if m != "down_proj"]

    created: list[str] = []
    original_create_adapter = backend.create_adapter

    async def recording_create_adapter(lora_id, lora_config):
        created.append(lora_id)
        await original_create_adapter(lora_id, lora_config)

    monkeypatch.setattr(backend, "create_adapter", recording_create_adapter)

    restored = await state.training.restore_from_checkpoint(training.training_run_id)

    assert restored is None
    assert record.corrupted is True
    assert created == []


@pytest.mark.asyncio
async def test_restore_releases_fallback_adapter_when_load_keeps_failing(
    request, tmp_path, monkeypatch
) -> None:
    """A corrupted run must not keep the slot its restore fallback allocated."""
    use_gpu = request.config.getoption("--gpu")
    state = await _build_state(tmp_path, use_gpu)
    session_id = _create_session(state)
    training = await state.create_model(
        session_id,
        model_owner="tester",
        base_model="Qwen/Qwen3-0.6B",
        lora_config=types.LoraConfig(rank=4, train_unembed=False),
        user_metadata=None,
    )
    await state.save_checkpoint(
        training.training_run_id, user_id="tester", name="ckpt", checkpoint_type="training"
    )
    record = state.training.training_runs[training.training_run_id]
    backend = state.training.training_backends["Qwen/Qwen3-0.6B"]

    removed: list[str] = []
    original_remove_adapter = backend.remove_adapter

    async def recording_remove_adapter(lora_id):
        removed.append(lora_id)
        await original_remove_adapter(lora_id)

    async def failing_load_state(lora_id, checkpoint_record, optimizer):
        raise RuntimeError("simulated load failure")

    monkeypatch.setattr(backend, "remove_adapter", recording_remove_adapter)
    monkeypatch.setattr(backend, "load_state", failing_load_state)

    restored = await state.training.restore_from_checkpoint(training.training_run_id)

    assert restored is not None
    assert record.corrupted is True
    assert removed == [training.training_run_id]


@pytest.mark.asyncio
async def test_rest_client(request, tmp_path) -> None:
    use_gpu = request.config.getoption("--gpu")
    state = await _build_state(tmp_path, use_gpu)
    session_id_1 = _create_session(state, "tester")
    training_1 = await state.create_model(
        session_id_1,
        model_owner="tester",
        base_model="Qwen/Qwen3-0.6B",
        lora_config=types.LoraConfig(rank=4, train_unembed=False),
        user_metadata=None,
    )
    session_id_2 = _create_session(state, "tester")
    training_2 = await state.create_model(
        session_id_2,
        model_owner="tester",
        base_model="Qwen/Qwen3-0.6B",
        lora_config=types.LoraConfig(rank=4, train_unembed=False),
        user_metadata=None,
    )
    session_id_3 = _create_session(state, "other_user")
    training_3 = await state.create_model(
        session_id_3,
        model_owner="other_user",
        base_model="Qwen/Qwen3-0.6B",
        lora_config=types.LoraConfig(rank=4, train_unembed=False),
        user_metadata=None,
    )

    with pytest.raises(UserMismatchException):
        await state.save_checkpoint(
            training_1.training_run_id,
            user_id="other_user",
            name="ckpt1",
            checkpoint_type="training",
        )

    await state.save_checkpoint(
        training_2.training_run_id,
        user_id="tester",
        name="ckpt2",
        checkpoint_type="training",
    )

    await state.save_checkpoint(
        training_3.training_run_id,
        user_id="other_user",
        name="ckpt3",
        checkpoint_type="training",
    )

    sampler_1 = await state.create_sampling_session(
        session_id=session_id_1,
        base_model="Qwen/Qwen3-0.6B",
        model_path=None,
        session_seq_id=2,
        user_id="tester",
    )

    with pytest.raises(UserMismatchException):
        await state.run_sample(
            types.SampleRequest(
                prompt=types.ModelInput.from_ints([1, 2, 3]),
                num_samples=1,
                sampling_params=types.SamplingParams(max_tokens=2, temperature=0.1),
                sampling_session_id=sampler_1,
                seq_id=0,
            ),
            user_id="other_user",
        )

    sampler_2 = await state.create_sampling_session(
        session_id=session_id_2,
        base_model="Qwen/Qwen3-0.6B",
        model_path=None,
        session_seq_id=2,
        user_id="tester",
    )

    await state.run_sample(
        types.SampleRequest(
            prompt=types.ModelInput.from_ints([1, 2, 3]),
            num_samples=1,
            sampling_params=types.SamplingParams(max_tokens=2, temperature=0.1),
            sampling_session_id=sampler_2,
            seq_id=0,
        ),
        user_id="tester",
    )

    assert len(state.list_sessions(user_id="tester").sessions) == 2
    assert len(state.list_sessions(user_id="other_user").sessions) == 1

    assert len(state.list_training_runs(user_id="tester").training_runs) == 2
    assert len(state.list_training_runs(user_id="other_user").training_runs) == 1

    assert len(state.list_user_checkpoints(user_id="tester")) == 1
    assert len(state.list_user_checkpoints(user_id="other_user")) == 1

    info = state.get_sampler_info(sampler_id=sampler_2, user_id="tester")
    assert info.sampler_id == sampler_2
    assert info.base_model == "Qwen/Qwen3-0.6B"
    assert info.model_path is None

    with pytest.raises(UserMismatchException):
        state.get_sampler_info(
            sampler_id=sampler_1,
            user_id="other_user",
        )


@pytest.mark.asyncio
async def test_load_checkpoint_rejects_mismatched_lora_alpha(request, tmp_path) -> None:
    """A checkpoint trained at a different lora_alpha must not load.

    LoRA update scale is lora_alpha / rank, so replaying weights trained at one
    alpha into an adapter built with another silently rescales every update.
    This is the pre-existing-checkpoint case: the 'hf' backend used
    lora_alpha = rank before lora_alpha_ratio existed, so a checkpoint from then
    records alpha 4 at rank 4 while the ratio-2 default builds alpha 8.
    """
    use_gpu = request.config.getoption("--gpu")
    state = await _build_state(tmp_path, use_gpu)
    session_id = _create_session(state)
    run = await state.create_model(
        session_id,
        model_owner="tester",
        base_model="Qwen/Qwen3-0.6B",
        lora_config=types.LoraConfig(rank=4, train_unembed=False),
        user_metadata=None,
    )
    checkpoint = await state.save_checkpoint(
        run.training_run_id,
        user_id="tester",
        name="alpha-mismatch-test",
        checkpoint_type="training",
    )
    # Saved under the ratio-2 default, then rewritten as a legacy ratio-1 checkpoint.
    assert checkpoint.metadata.lora_alpha == 8
    checkpoint.save_metadata(
        base_model=run.base_model,
        session_id=run.session_id,
        lora_rank=run.lora_rank,
        lora_alpha=4,
        train_attn=checkpoint.metadata.train_attn,
        train_mlp=checkpoint.metadata.train_mlp,
        train_unembed=checkpoint.metadata.train_unembed,
    )

    with pytest.raises(CheckpointIncompatibleException, match="lora_alpha=4"):
        await state.load_checkpoint(
            run.training_run_id,
            path=checkpoint.tinker_checkpoint.tinker_path,
            user_id="tester",
            optimizer=True,
        )


@pytest.mark.asyncio
async def test_load_checkpoint_accepts_matching_lora_alpha(request, tmp_path) -> None:
    """The alpha gate must not reject a checkpoint written by this same server."""
    use_gpu = request.config.getoption("--gpu")
    state = await _build_state(tmp_path, use_gpu)
    session_id = _create_session(state)
    run = await state.create_model(
        session_id,
        model_owner="tester",
        base_model="Qwen/Qwen3-0.6B",
        lora_config=types.LoraConfig(rank=8, train_unembed=False),
        user_metadata=None,
    )
    checkpoint = await state.save_checkpoint(
        run.training_run_id,
        user_id="tester",
        name="alpha-match-test",
        checkpoint_type="training",
    )
    assert checkpoint.metadata.lora_alpha == 16

    await state.load_checkpoint(
        run.training_run_id,
        path=checkpoint.tinker_checkpoint.tinker_path,
        user_id="tester",
        optimizer=True,
    )
