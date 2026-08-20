"""Tests for per-model training/sampling capability declarations (issue #159).

Covers the three deployment profiles (training-only, sampling-only, combined),
configuration validation, endpoint errors, persistence restore across
capability toggles, and backward-compatible defaults.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from tinker import types

from tuft.auth import User
from tuft.config import AppConfig, ModelCapability, ModelConfig
from tuft.exceptions import (
    CapabilityDisabledException,
    FutureCancelledException,
    InvalidRequestException,
    UnknownModelException,
)
from tuft.futures import FutureRecord
from tuft.persistence import (
    get_redis_store,
    save_config_signature,
    validate_config_signature,
)
from tuft.persistence.redis_store import ConfigSignature
from tuft.server import create_root_app
from tuft.state import ServerState


# The dummy weights path resolves as a Qwen model so newly created runs record
# concrete LoRA target geometry, matching what production runs must do.
CPU_MODEL_PATH = "/path/to/qwen-test-model"

TRAIN_ONLY = "qwen-train-only"
SAMPLE_ONLY = "qwen-sample-only"
BOTH = "qwen-both"


@pytest.fixture(autouse=True)
def _cpu_only(request):
    if request.config.getoption("--gpu"):
        pytest.skip("Capability profile tests cover the CPU dummy backends")


def _model_config(name: str, capabilities: list[ModelCapability] | None = None) -> ModelConfig:
    if capabilities is None:
        return ModelConfig(
            model_name=name,
            model_path=Path(CPU_MODEL_PATH),
            max_model_len=2048,
        )
    return ModelConfig(
        model_name=name,
        model_path=Path(CPU_MODEL_PATH),
        max_model_len=2048,
        capabilities=capabilities,
    )


def _profile_config(checkpoint_dir: Path) -> AppConfig:
    return AppConfig(
        checkpoint_dir=checkpoint_dir,
        supported_models=[
            _model_config(TRAIN_ONLY, ["training"]),
            _model_config(SAMPLE_ONLY, ["sampling"]),
            _model_config(BOTH, ["training", "sampling"]),
        ],
        authorized_users={"test-key": "tester"},
    )


async def _build_state(tmp_path: Path) -> ServerState:
    state = ServerState(_profile_config(tmp_path))
    await state.async_init()
    return state


def _create_session(state: ServerState, user_id: str = "tester") -> str:
    session = state.create_session(
        types.CreateSessionRequest(tags=["test"], user_metadata=None, sdk_version="1.0"),
        user=User(user_id=user_id),
    )
    return session.session_id


def _datum() -> types.Datum:
    return types.Datum(
        model_input=types.ModelInput.from_ints([11, 12, 13]),
        loss_fn_inputs={
            "target_tokens": types.TensorData(data=[21, 22, 23], dtype="int64", shape=[3]),
            "weights": types.TensorData(data=[1.0, 1.0, 1.0], dtype="float32", shape=[3]),
        },
    )


# =============================================================================
# Configuration validation
# =============================================================================


def test_default_capabilities_enable_both() -> None:
    model = _model_config("default")
    assert model.capabilities == ["training", "sampling"]
    assert model.training_enabled
    assert model.sampling_enabled


def test_capabilities_normalized_to_canonical_order() -> None:
    model = _model_config("dup", ["sampling", "training", "sampling"])
    assert model.capabilities == ["training", "sampling"]


def test_empty_capabilities_rejected() -> None:
    with pytest.raises(ValidationError, match="declares no capabilities"):
        _model_config("none", [])


def test_unknown_capability_rejected() -> None:
    with pytest.raises(ValidationError):
        _model_config("bad", ["training", "serving"])  # pyright: ignore[reportArgumentType]


def test_colocate_requires_both_capabilities() -> None:
    with pytest.raises(ValidationError, match="requires both capabilities"):
        ModelConfig(
            model_name="colo",
            model_path=Path(CPU_MODEL_PATH),
            max_model_len=2048,
            capabilities=["training"],
            colocate=True,
        )


def test_config_signature_ignores_capability_changes(tmp_path) -> None:
    """Toggling capabilities and restarting must not be flagged as config drift."""
    both = AppConfig(
        checkpoint_dir=tmp_path,
        supported_models=[_model_config("m", ["training", "sampling"])],
    )
    training_only = AppConfig(
        checkpoint_dir=tmp_path,
        supported_models=[_model_config("m", ["training"])],
    )
    sig_both = ConfigSignature.from_app_config(both)
    sig_training_only = ConfigSignature.from_app_config(training_only)
    assert sig_both.matches(sig_training_only)

    changed_model = AppConfig(
        checkpoint_dir=tmp_path,
        supported_models=[
            ModelConfig(model_name="m", model_path=Path(CPU_MODEL_PATH), max_model_len=4096)
        ],
    )
    assert not ConfigSignature.from_app_config(changed_model).matches(sig_both)


def test_validate_config_signature_allows_capability_toggle(tmp_path) -> None:
    if not get_redis_store().is_enabled:
        pytest.skip("Persistence disabled")
    both = AppConfig(
        checkpoint_dir=tmp_path,
        supported_models=[_model_config("m", ["training", "sampling"])],
    )
    save_config_signature(both)
    sampling_only = AppConfig(
        checkpoint_dir=tmp_path,
        supported_models=[_model_config("m", ["sampling"])],
    )
    # Must not raise ConfigMismatchError; returns False because data exists.
    assert validate_config_signature(sampling_only) is False


# =============================================================================
# Backend construction per profile
# =============================================================================


@pytest.mark.asyncio
async def test_backends_created_only_for_declared_capabilities(tmp_path) -> None:
    state = await _build_state(tmp_path)
    assert set(state.training.training_backends) == {TRAIN_ONLY, BOTH}
    assert set(state.sampling._base_backends) == {SAMPLE_ONLY, BOTH}


@pytest.mark.asyncio
async def test_build_supported_models_reflects_capabilities(tmp_path) -> None:
    state = await _build_state(tmp_path)
    by_name = {model.model_name: model for model in state.build_supported_models()}
    assert set(by_name) == {TRAIN_ONLY, SAMPLE_ONLY, BOTH}
    assert by_name[TRAIN_ONLY].capabilities == ["training"]
    assert by_name[SAMPLE_ONLY].capabilities == ["sampling"]
    assert by_name[BOTH].capabilities == ["training", "sampling"]
    assert by_name[BOTH].max_context_length == 2048


# =============================================================================
# Training-only profile
# =============================================================================


@pytest.mark.asyncio
async def test_training_only_model_trains_and_exports_sampler_weights(tmp_path) -> None:
    """Training and sampler-compatible export work without a live sampler."""
    state = await _build_state(tmp_path)
    session_id = _create_session(state)
    record = await state.create_model(
        session_id=session_id,
        base_model=TRAIN_ONLY,
        lora_config=types.LoraConfig(rank=8),
        model_owner="tester",
        user_metadata=None,
    )
    run_id = record.training_run_id

    await state.run_forward(
        model_id=run_id,
        user_id="tester",
        data=[_datum()],
        loss_fn="cross_entropy",
        loss_fn_config=None,
        seq_id=None,
        backward=True,
    )
    await state.run_optim_step(
        model_id=run_id,
        user_id="tester",
        params=types.AdamParams(learning_rate=1e-3),
        seq_id=None,
    )
    training_ckpt = await state.save_checkpoint(run_id, "tester", "train-ckpt", "training")
    sampler_ckpt = await state.save_checkpoint(run_id, "tester", "sampler-ckpt", "sampler")
    assert training_ckpt.checkpoint_type == "training"
    assert sampler_ckpt.checkpoint_type == "sampler"
    assert len(state.list_checkpoints(run_id, "tester")) == 2


@pytest.mark.asyncio
async def test_training_only_model_rejects_sampling(tmp_path) -> None:
    state = await _build_state(tmp_path)
    session_id = _create_session(state)

    with pytest.raises(CapabilityDisabledException, match="'sampling' capability"):
        await state.create_sampling_session(
            session_id=session_id,
            base_model=TRAIN_ONLY,
            model_path=None,
            user_id="tester",
            session_seq_id=1,
        )

    # Loading a sampler checkpoint exported from the training-only model is
    # equally rejected: the capability, not the checkpoint, is the gate.
    record = await state.create_model(
        session_id=session_id,
        base_model=TRAIN_ONLY,
        lora_config=types.LoraConfig(rank=8),
        model_owner="tester",
        user_metadata=None,
    )
    ckpt = await state.save_checkpoint(record.training_run_id, "tester", None, "sampler")
    with pytest.raises(CapabilityDisabledException, match="'sampling' capability"):
        await state.create_sampling_session(
            session_id=session_id,
            base_model=None,
            model_path=ckpt.tinker_checkpoint.tinker_path,
            user_id="tester",
            session_seq_id=1,
        )


# =============================================================================
# Sampling-only profile
# =============================================================================


@pytest.mark.asyncio
async def test_sampling_only_model_rejects_training(tmp_path) -> None:
    state = await _build_state(tmp_path)
    session_id = _create_session(state)
    with pytest.raises(CapabilityDisabledException, match="'training' capability"):
        await state.create_model(
            session_id=session_id,
            base_model=SAMPLE_ONLY,
            lora_config=types.LoraConfig(rank=8),
            model_owner="tester",
            user_metadata=None,
        )
    # Unknown models keep their distinct 404 error.
    with pytest.raises(UnknownModelException):
        await state.create_model(
            session_id=session_id,
            base_model="missing-model",
            lora_config=types.LoraConfig(rank=8),
            model_owner="tester",
            user_metadata=None,
        )


@pytest.mark.asyncio
async def test_sampling_only_model_serves_base_model(tmp_path) -> None:
    state = await _build_state(tmp_path)
    session_id = _create_session(state)
    sampling_session_id = await state.create_sampling_session(
        session_id=session_id,
        base_model=SAMPLE_ONLY,
        model_path=None,
        user_id="tester",
        session_seq_id=1,
    )
    response = await state.run_sample(
        types.SampleRequest(
            prompt=types.ModelInput.from_ints([1, 2, 3]),
            num_samples=1,
            sampling_params=types.SamplingParams(max_tokens=2, temperature=0.1),
            sampling_session_id=sampling_session_id,
            seq_id=0,
        ),
        user_id="tester",
    )
    assert response.sequences


@pytest.mark.asyncio
async def test_sampling_only_model_loads_existing_sampler_checkpoint(tmp_path) -> None:
    """Sampler checkpoints remain usable without a live training backend."""
    checkpoint_dir = tmp_path / "checkpoints"
    model_name = "qwen-checkpoint-toggle"

    combined = ServerState(
        AppConfig(
            checkpoint_dir=checkpoint_dir,
            supported_models=[_model_config(model_name, ["training", "sampling"])],
        )
    )
    await combined.async_init()
    source_session_id = _create_session(combined)
    run = await combined.create_model(
        session_id=source_session_id,
        base_model=model_name,
        lora_config=types.LoraConfig(rank=8),
        model_owner="tester",
        user_metadata=None,
    )
    checkpoint = await combined.save_checkpoint(
        run.training_run_id, "tester", "sampler-weights", "sampler"
    )
    await combined.future_store.shutdown()
    await combined.shutdown()

    sampling_only = ServerState(
        AppConfig(
            checkpoint_dir=checkpoint_dir,
            supported_models=[_model_config(model_name, ["sampling"])],
        )
    )
    await sampling_only.async_init()
    assert not sampling_only.training.training_backends
    session_id = _create_session(sampling_only)
    sampling_session_id = await sampling_only.create_sampling_session(
        session_id=session_id,
        base_model=None,
        model_path=checkpoint.tinker_checkpoint.tinker_path,
        user_id="tester",
        session_seq_id=1,
    )
    response = await sampling_only.run_sample(
        types.SampleRequest(
            prompt=types.ModelInput.from_ints([1, 2, 3]),
            num_samples=1,
            sampling_params=types.SamplingParams(max_tokens=2, temperature=0.1),
            sampling_session_id=sampling_session_id,
            seq_id=0,
        ),
        user_id="tester",
    )
    assert response.sequences
    await sampling_only.future_store.shutdown()
    await sampling_only.shutdown()


# =============================================================================
# Combined profile
# =============================================================================


@pytest.mark.asyncio
async def test_combined_model_keeps_current_behavior(tmp_path) -> None:
    state = await _build_state(tmp_path)
    session_id = _create_session(state)
    record = await state.create_model(
        session_id=session_id,
        base_model=BOTH,
        lora_config=types.LoraConfig(rank=8),
        model_owner="tester",
        user_metadata=None,
    )
    assert record.backend is not None
    sampling_session_id = await state.create_sampling_session(
        session_id=session_id,
        base_model=BOTH,
        model_path=None,
        user_id="tester",
        session_seq_id=1,
    )
    assert sampling_session_id in state.sampling.sampling_sessions


# =============================================================================
# HTTP endpoints
# =============================================================================


@pytest.fixture
def api_client(tmp_path):
    app = create_root_app(_profile_config(tmp_path / "checkpoints"))
    with TestClient(app) as client:
        client.headers["X-API-Key"] = "test-key"
        yield client


def test_get_server_capabilities_reports_capabilities(api_client) -> None:
    response = api_client.get("/api/v1/get_server_capabilities")
    assert response.status_code == 200
    by_name = {m["model_name"]: m for m in response.json()["supported_models"]}
    assert by_name[TRAIN_ONLY]["capabilities"] == ["training"]
    assert by_name[SAMPLE_ONLY]["capabilities"] == ["sampling"]
    assert by_name[BOTH]["capabilities"] == ["training", "sampling"]


def test_oai_models_lists_only_sampling_capable(api_client) -> None:
    response = api_client.get("/oai/api/v1/models")
    assert response.status_code == 200
    listed = {m["id"] for m in response.json()["data"]}
    assert SAMPLE_ONLY in listed
    assert BOTH in listed
    assert TRAIN_ONLY not in listed


def test_oai_inference_on_training_only_model_rejected(api_client) -> None:
    response = api_client.post(
        "/oai/api/v1/chat/completions",
        json={
            "model": TRAIN_ONLY,
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert response.status_code == 400
    assert "Capability disabled" in response.json()["detail"]


def test_create_model_endpoint_rejects_sampling_only_model(api_client) -> None:
    session = api_client.post(
        "/api/v1/create_session",
        json={"tags": [], "user_metadata": None, "sdk_version": "1.0"},
    )
    assert session.status_code == 201
    response = api_client.post(
        "/api/v1/create_model",
        json={
            "session_id": session.json()["session_id"],
            "model_seq_id": 0,
            "base_model": SAMPLE_ONLY,
            "lora_config": {"rank": 8},
            "user_metadata": None,
        },
    )
    assert response.status_code == 400
    assert "Capability disabled" in response.json()["detail"]


def test_save_and_get_sampler_rejects_before_writing_checkpoint(api_client) -> None:
    session = api_client.post(
        "/api/v1/create_session",
        json={"tags": [], "user_metadata": None, "sdk_version": "1.0"},
    )
    assert session.status_code == 201
    create = api_client.post(
        "/api/v1/create_model",
        json={
            "session_id": session.json()["session_id"],
            "model_seq_id": 0,
            "base_model": TRAIN_ONLY,
            "lora_config": {"rank": 8},
            "user_metadata": None,
        },
    )
    assert create.status_code == 202
    created = api_client.post(
        "/api/v1/retrieve_future",
        json={"request_id": create.json()["request_id"]},
    )
    assert created.status_code == 200
    model_id = created.json()["model_id"]

    save = api_client.post(
        "/api/v1/save_weights_for_sampler",
        json={
            "model_id": model_id,
            "sampling_session_seq_id": 1,
            "seq_id": 0,
        },
    )
    assert save.status_code == 202
    rejected = api_client.post(
        "/api/v1/retrieve_future",
        json={"request_id": save.json()["request_id"]},
    )
    assert rejected.status_code == 400
    assert "Capability disabled" in rejected.json()["detail"]

    state = api_client.app.state.server_state
    assert state.list_checkpoints(model_id, user_id="tester") == []


# =============================================================================
# Persistence across capability toggles
# =============================================================================


def _skip_without_persistence() -> None:
    if not get_redis_store().is_enabled:
        pytest.skip("Persistence disabled")


def _single_model_config(checkpoint_dir: Path, capabilities: list[ModelCapability]) -> AppConfig:
    return AppConfig(
        checkpoint_dir=checkpoint_dir,
        supported_models=[_model_config("qwen-toggled", capabilities)],
        authorized_users={"test-key": "tester"},
    )


@pytest.mark.asyncio
async def test_training_run_preserved_across_training_capability_toggle(tmp_path) -> None:
    _skip_without_persistence()
    checkpoint_dir = tmp_path / "checkpoints"

    # Phase A: training enabled - create a run and save a checkpoint.
    state = ServerState(_single_model_config(checkpoint_dir, ["training", "sampling"]))
    await state.async_init()
    session_id = _create_session(state)
    record = await state.create_model(
        session_id=session_id,
        base_model="qwen-toggled",
        lora_config=types.LoraConfig(rank=8),
        model_owner="tester",
        user_metadata=None,
    )
    run_id = record.training_run_id
    await state.save_checkpoint(run_id, "tester", "ckpt", "training")
    await state.future_store.shutdown()
    await state.shutdown()

    # Phase B: training disabled - the run is preserved but rejects training.
    state2 = ServerState(_single_model_config(checkpoint_dir, ["sampling"]))
    await state2.async_init()
    restored = state2.training.training_runs[run_id]
    assert restored.corrupted is False
    assert restored.backend is None
    assert "ckpt" in restored.checkpoints
    with pytest.raises(CapabilityDisabledException, match="'training' capability"):
        await state2.run_forward(
            model_id=run_id,
            user_id="tester",
            data=[_datum()],
            loss_fn="cross_entropy",
            loss_fn_config=None,
            seq_id=None,
            backward=True,
        )
    with pytest.raises(CapabilityDisabledException, match="'training' capability"):
        await state2.save_checkpoint(run_id, "tester", "ckpt-2", "training")
    # Read-only surfaces stay available.
    assert state2.list_checkpoints(run_id, "tester")
    await state2.future_store.shutdown()
    await state2.shutdown()

    # Phase C: training re-enabled - the run restores and trains again.
    state3 = ServerState(_single_model_config(checkpoint_dir, ["training", "sampling"]))
    await state3.async_init()
    revived = state3.training.training_runs[run_id]
    assert revived.corrupted is False
    assert revived.backend is not None
    result = await state3.run_forward(
        model_id=run_id,
        user_id="tester",
        data=[_datum()],
        loss_fn="cross_entropy",
        loss_fn_config=None,
        seq_id=None,
        backward=True,
    )
    assert result.loss_fn_outputs
    await state3.future_store.shutdown()
    await state3.shutdown()


@pytest.mark.asyncio
async def test_sampling_session_preserved_across_sampling_capability_toggle(tmp_path) -> None:
    _skip_without_persistence()
    checkpoint_dir = tmp_path / "checkpoints"

    # Phase A: sampling enabled - create a base-model sampling session.
    state = ServerState(_single_model_config(checkpoint_dir, ["training", "sampling"]))
    await state.async_init()
    session_id = _create_session(state)
    sampling_session_id = await state.create_sampling_session(
        session_id=session_id,
        base_model="qwen-toggled",
        model_path=None,
        user_id="tester",
        session_seq_id=1,
    )
    await state.future_store.shutdown()
    await state.shutdown()

    def _sample_request(seq_id: int) -> types.SampleRequest:
        return types.SampleRequest(
            prompt=types.ModelInput.from_ints([1, 2, 3]),
            num_samples=1,
            sampling_params=types.SamplingParams(max_tokens=2, temperature=0.1),
            sampling_session_id=sampling_session_id,
            seq_id=seq_id,
        )

    # Phase B: sampling disabled - the session is preserved but rejects requests.
    state2 = ServerState(_single_model_config(checkpoint_dir, ["training"]))
    await state2.async_init()
    assert sampling_session_id in state2.sampling.sampling_sessions
    with pytest.raises(CapabilityDisabledException, match="'sampling' capability"):
        await state2.run_sample(_sample_request(seq_id=0), user_id="tester")
    disabled_record = state2.sampling.sampling_sessions[sampling_session_id]
    assert disabled_record.history == []
    assert disabled_record.last_seq_id == -1
    await state2.future_store.shutdown()
    await state2.shutdown()

    # Phase C: sampling re-enabled - the exact rejected request can be retried.
    state3 = ServerState(_single_model_config(checkpoint_dir, ["training", "sampling"]))
    await state3.async_init()
    response = await state3.run_sample(_sample_request(seq_id=0), user_id="tester")
    assert response.sequences
    await state3.future_store.shutdown()
    await state3.shutdown()


@pytest.mark.asyncio
async def test_pending_futures_fail_deterministically_when_training_disabled(tmp_path) -> None:
    _skip_without_persistence()
    checkpoint_dir = tmp_path / "checkpoints"

    state = ServerState(_single_model_config(checkpoint_dir, ["training", "sampling"]))
    await state.async_init()
    session_id = _create_session(state)
    record = await state.create_model(
        session_id=session_id,
        base_model="qwen-toggled",
        lora_config=types.LoraConfig(rank=8),
        model_owner="tester",
        user_metadata=None,
    )
    run_id = record.training_run_id
    # Persist a pending training future as if the server crashed mid-operation.
    pending = FutureRecord(
        model_id=run_id,
        user_id="tester",
        operation_type="forward_backward",
    )
    state.future_store._store_record(pending)
    await state.future_store.shutdown()
    await state.shutdown()

    state2 = ServerState(_single_model_config(checkpoint_dir, ["sampling"]))
    await state2.async_init()
    with pytest.raises(FutureCancelledException, match="Training capability"):
        await state2.future_store.retrieve(request_id=pending.request_id, user_id="tester")
    await state2.future_store.shutdown()
    await state2.shutdown()


@pytest.mark.asyncio
async def test_pending_futures_fail_deterministically_when_run_is_corrupted(tmp_path) -> None:
    _skip_without_persistence()
    checkpoint_dir = tmp_path / "checkpoints"

    state = ServerState(_single_model_config(checkpoint_dir, ["training", "sampling"]))
    await state.async_init()
    session_id = _create_session(state)
    record = await state.create_model(
        session_id=session_id,
        base_model="qwen-toggled",
        lora_config=types.LoraConfig(rank=8),
        model_owner="tester",
        user_metadata=None,
    )
    pending = FutureRecord(
        model_id=record.training_run_id,
        user_id="tester",
        operation_type="forward_backward",
    )
    state.future_store._store_record(pending)
    await state.future_store.shutdown()
    await state.shutdown()

    # Removing the base model from the current configuration makes the
    # restored run corrupted: no backend can restore its adapter.
    replacement_config = AppConfig(
        checkpoint_dir=checkpoint_dir,
        supported_models=[_model_config("replacement-model")],
        authorized_users={"test-key": "tester"},
    )
    state2 = ServerState(replacement_config)
    await state2.async_init()
    restored = state2.training.training_runs[record.training_run_id]
    assert restored.corrupted is True
    with pytest.raises(FutureCancelledException, match="is corrupted"):
        await state2.future_store.retrieve(request_id=pending.request_id, user_id="tester")
    with pytest.raises(InvalidRequestException, match="is corrupted"):
        await state2.save_checkpoint(
            record.training_run_id,
            user_id="tester",
            name="corrupted-save",
            checkpoint_type="training",
        )
    assert state2.list_checkpoints(record.training_run_id, user_id="tester") == []
    assert not (checkpoint_dir / record.training_run_id / "corrupted-save").exists()
    await state2.future_store.shutdown()
    await state2.shutdown()
