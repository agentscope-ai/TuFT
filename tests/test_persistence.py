from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path

import pytest
import ray
from tinker import types
from tinker.types.try_again_response import TryAgainResponse

from tuft.auth import User
from tuft.config import AppConfig, ModelConfig, TelemetryConfig
from tuft.exceptions import ConfigMismatchError, UnknownModelException
from tuft.futures import FutureStore
from tuft.persistence import (
    ConfigCheckField,
    flush_all_data,
    get_redis_store,
    is_persistence_enabled,
    save_config_signature,
    validate_config_signature,
)
from tuft.persistence.redis_store import (
    DEFAULT_CHECK_FIELDS,
    ConfigSignature,
    has_existing_data,
    load_config_signature,
)
from tuft.sampling_controller import SamplingController, SamplingSessionRecord
from tuft.state import ServerState, SessionManager
from tuft.training_controller import TrainingController, TrainingRunRecord


def _is_gpu_mode() -> bool:
    """Check if running in GPU mode (not CPU test mode)."""
    return os.environ.get("TUFT_CPU_TEST") != "1"


def _create_test_config(checkpoint_dir: Path) -> AppConfig:
    """Create a minimal test config.

    In GPU mode, uses the real model path from TUFT_TEST_MODEL environment variable.
    In CPU mode, uses a dummy model path since the model won't actually be loaded.
    """
    if _is_gpu_mode():
        assert "TUFT_TEST_MODEL" in os.environ, (
            "Environment variable TUFT_TEST_MODEL must be set for GPU tests."
        )
        model_path = Path(os.environ.get("TUFT_TEST_MODEL", "Qwen/Qwen3-0.6B"))
    else:
        model_path = Path("/dummy/model")

    return AppConfig(
        checkpoint_dir=checkpoint_dir,
        supported_models=[
            ModelConfig(
                model_name="Qwen/Qwen3-0.6B",
                model_path=model_path,
                max_model_len=2048,
            )
        ],
    )


def _ensure_ray_shutdown():
    """Ensure Ray is completely shutdown."""
    if ray.is_initialized():
        ray.shutdown()
    # Give Ray time to fully shutdown
    time.sleep(0.5)


def _init_ray():
    """Initialize a fresh Ray cluster."""
    _ensure_ray_shutdown()
    ray.init()


def _create_test_datum() -> types.Datum:
    """Create a test datum for forward/backward operations."""
    return types.Datum(
        model_input=types.ModelInput.from_ints([11, 12, 13]),
        loss_fn_inputs={
            "target_tokens": types.TensorData(data=[21, 22, 23], dtype="int64", shape=[3]),
            "weights": types.TensorData(data=[1.0, 1.0, 1.0], dtype="float32", shape=[3]),
        },
    )


async def _wait_for_result(store: FutureStore, request_id: str, user_id: str, timeout: float = 5.0):
    """Wait for a future to complete and return the result."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = await store.retrieve(request_id, user_id=user_id, timeout=0.5)
        if not isinstance(result, TryAgainResponse):
            return result
        await asyncio.sleep(0.01)
    raise AssertionError("future did not complete in time")


def _skip_if_no_persistence():
    """Skip test if persistence is not enabled.

    Note: With the FileRedis fallback, persistence is always available
    unless explicitly disabled via --no-persistence.
    """
    if not is_persistence_enabled():
        pytest.skip("Persistence not enabled (use default or remove --no-persistence)")


# =============================================================================
# SessionManager Persistence Tests
# =============================================================================


@pytest.mark.persistence
class TestSessionManagerPersistence:
    """Test SessionManager save and restore."""

    @pytest.fixture
    def setup(self, tmp_path):
        _skip_if_no_persistence()
        store = get_redis_store()

        yield store

    def test_session_persisted_and_restored(self, setup):
        """Test that sessions are persisted and AUTOMATICALLY restored on manager creation.

        Simulates:
        1. Server starts, creates session
        2. Server crashes (SessionManager destroyed)
        3. Server restarts (new SessionManager created)
        4. Session should be automatically restored
        """
        store = setup

        # === Phase 1: Server running, create session ===
        manager = SessionManager()
        request = types.CreateSessionRequest(
            tags=["persistent"], user_metadata={"test": "yes"}, sdk_version="1.0"
        )
        record = manager.create_session(request, User(user_id="persistent_user"))
        session_id = record.session_id

        # Verify persisted to Redis
        key = store.build_key(SessionManager.REDIS_KEY_PREFIX, session_id)
        assert store.exists(key) is True

        # === Phase 2: Server crash (manager destroyed) ===
        del manager

        # === Phase 3: Server restart (new manager created) ===
        new_manager = SessionManager()

        assert session_id in new_manager._sessions, (
            "Session should be automatically restored from Redis on manager creation"
        )
        restored = new_manager._sessions[session_id]
        assert restored.tags == ["persistent"]
        assert restored.user_metadata == {"test": "yes"}


# =============================================================================
# FutureStore Persistence Tests
# =============================================================================


@pytest.mark.persistence
class TestFutureStorePersistence:
    """Test FutureStore save and restore."""

    @pytest.fixture
    def setup(self, tmp_path):
        _skip_if_no_persistence()
        redis_store = get_redis_store()
        yield redis_store

    @pytest.mark.asyncio
    async def test_future_persisted_and_restored_on_success(self, setup):
        """Test that successful futures are persisted and restored after restart."""
        redis_store = setup

        # === Phase 1: Server running, enqueue successful operation ===
        store = FutureStore()

        def _operation() -> types.SaveWeightsResponse:
            return types.SaveWeightsResponse(path="tinker://run/weights/ckpt")

        future = await store.enqueue(_operation, model_id="run", user_id="tester")
        request_id = future.request_id

        key = redis_store.build_key(FutureStore.REDIS_KEY_PREFIX, request_id)
        assert redis_store.exists(key) is True

        result = await _wait_for_result(store, request_id, user_id="tester")
        assert isinstance(result, types.SaveWeightsResponse)
        assert result.path.endswith("ckpt")

        # === Phase 2: Server crash ===
        await store.shutdown()
        del store

        # === Phase 3: Server restart ===
        new_store = FutureStore()
        record = new_store._records.get(request_id)
        assert record is not None
        assert record.status == "ready"
        assert record.payload is not None

        await new_store.shutdown()

    @pytest.mark.asyncio
    async def test_future_persisted_and_restored_on_failure(self, setup):
        """Test that failed futures are persisted and restored after restart."""
        _ = setup  # Ensure fixture is used

        # === Phase 1: Server running, enqueue failing operation ===
        store = FutureStore()

        def _operation() -> types.SaveWeightsResponse:
            raise UnknownModelException("unknown")

        future = await store.enqueue(_operation, user_id="tester")
        request_id = future.request_id

        result = await _wait_for_result(store, request_id, user_id="tester")
        assert isinstance(result, types.RequestFailedResponse)
        assert result.error == "Unknown model: unknown"
        assert result.category == types.RequestErrorCategory.User

        # === Phase 2: Server crash ===
        await store.shutdown()
        del store

        # === Phase 3: Server restart ===
        new_store = FutureStore()
        record = new_store._records.get(request_id)
        assert record is not None
        assert record.status == "failed"
        assert record.error is not None

        await new_store.shutdown()


# =============================================================================
# SamplingSession Persistence Tests
# =============================================================================


@pytest.mark.persistence
class TestSamplingSessionPersistence:
    """Test SamplingController sampling session save and restore."""

    @pytest.fixture
    def setup(self, tmp_path):
        _skip_if_no_persistence()
        store = get_redis_store()
        app_config = _create_test_config(tmp_path / "checkpoints")
        app_config.ensure_directories()
        yield store, app_config

    def test_sampling_session_with_history_persisted(self, setup):
        """Test that sampling session history is preserved after restart."""
        store, app_config = setup

        # === Phase 1: Create session with history ===
        controller = SamplingController(app_config)

        sampling_session_id = "test-sampling-with-history"
        from tuft.sampling_controller import SamplingHistoryEntry

        history = [
            SamplingHistoryEntry(seq_id=1, prompt_token_count=10, prompt_hash="abc123"),
            SamplingHistoryEntry(seq_id=2, prompt_token_count=20, prompt_hash="def456"),
        ]
        record = SamplingSessionRecord(
            sampling_session_id=sampling_session_id,
            session_id="session-002",
            user_id="history_user",
            model_id=sampling_session_id,
            base_model="Qwen/Qwen3-0.6B",
            model_path=None,
            session_seq_id=1,
            last_seq_id=2,
            history=history,
        )
        controller.sampling_sessions[sampling_session_id] = record
        controller._save_session(sampling_session_id)

        # === Phase 2: Crash and restart ===
        del controller
        new_controller = SamplingController(app_config)

        # === Phase 3: Verify history restored ===
        restored = new_controller.sampling_sessions[sampling_session_id]
        assert len(restored.history) == 2
        assert restored.history[0].seq_id == 1
        assert restored.history[0].prompt_hash == "abc123"
        assert restored.history[1].seq_id == 2
        assert restored.history[1].prompt_token_count == 20

    def test_max_submitted_seq_id_persisted_and_executor_restored(self, setup):
        """Test that max_submitted_seq_id is persisted and executor is restored correctly."""
        store, app_config = setup

        # === Phase 1: Create session with max_submitted_seq_id ===
        controller = SamplingController(app_config)

        sampling_session_id = "test-executor-restore"
        record = SamplingSessionRecord(
            sampling_session_id=sampling_session_id,
            session_id="session-003",
            user_id="executor_user",
            model_id=sampling_session_id,
            base_model="Qwen/Qwen3-0.6B",
            model_path=None,
            session_seq_id=1,
            last_seq_id=10,
            max_submitted_seq_id=15,  # Some requests were submitted but not yet processed
        )
        controller.sampling_sessions[sampling_session_id] = record
        controller._save_session(sampling_session_id)

        # Verify initial executor state
        assert record.executor.next_sequence_id == 0  # Default value

        # === Phase 2: Crash and restart ===
        del controller
        new_controller = SamplingController(app_config)

        # === Phase 3: Verify executor restored with correct next_sequence_id ===
        restored = new_controller.sampling_sessions[sampling_session_id]
        assert restored.max_submitted_seq_id == 15
        assert restored.last_seq_id == 10
        # executor.next_sequence_id should be max_submitted_seq_id + 1
        assert restored.executor.next_sequence_id == 16


# =============================================================================
# TrainingRun Persistence Tests (CPU)
# =============================================================================


@pytest.mark.persistence
class TestTrainingRunPersistence:
    """Test TrainingController training run save and restore."""

    @pytest.fixture
    def setup(self, tmp_path):
        _skip_if_no_persistence()
        store = get_redis_store()
        app_config = _create_test_config(tmp_path / "checkpoints")
        app_config.ensure_directories()
        yield store, app_config

    def test_training_run_persisted_and_restored(self, setup):
        """Test that training runs are persisted and restored on controller creation."""
        store, app_config = setup

        # === Phase 1: Create training controller and run ===
        controller = TrainingController(app_config)

        training_run_id = "test-training-run-001"
        record = TrainingRunRecord(
            training_run_id=training_run_id,
            base_model="Qwen/Qwen3-0.6B",
            lora_rank=8,
            session_id="session-001",
            model_owner="trainer_user",
            user_metadata={"experiment": "test"},
            next_seq_id=5,
        )
        controller.training_runs[training_run_id] = record
        controller._save_training_run(training_run_id)

        key = store.build_key(TrainingController.REDIS_KEY_PREFIX, training_run_id)
        assert store.exists(key) is True

        # === Phase 2: Server crash (controller destroyed) ===
        del controller

        # === Phase 3: Server restart (new controller created) ===
        new_controller = TrainingController(app_config)

        assert training_run_id in new_controller.training_runs, (
            "Training run should be automatically restored from Redis"
        )
        restored = new_controller.training_runs[training_run_id]
        assert restored.base_model == "Qwen/Qwen3-0.6B"
        assert restored.lora_rank == 8
        assert restored.model_owner == "trainer_user"
        assert restored.user_metadata == {"experiment": "test"}
        assert restored.next_seq_id == 5
        assert restored.session_id == "session-001"

    def test_training_run_with_checkpoint_persisted(self, setup):
        """Test that training run checkpoints are persisted and restored."""
        store, app_config = setup

        # === Phase 1: Create training run with checkpoint ===
        controller = TrainingController(app_config)

        training_run_id = "test-run-with-ckpt"
        record = TrainingRunRecord(
            training_run_id=training_run_id,
            base_model="Qwen/Qwen3-0.6B",
            lora_rank=4,
            session_id="session-002",
            model_owner="ckpt_user",
            next_seq_id=10,
        )
        controller.training_runs[training_run_id] = record

        # Create a checkpoint record
        from tuft.checkpoints import CheckpointRecord

        checkpoint = CheckpointRecord(
            checkpoint_id="my-checkpoint",
            owner_name="ckpt_user",
            checkpoint_type="training",
            training_run_id=training_run_id,
            path=app_config.checkpoint_dir / training_run_id / "my-checkpoint",
            size_bytes=1024,
        )
        record.checkpoints["my-checkpoint"] = checkpoint

        controller._save_training_run(training_run_id)
        controller._save_checkpoint(training_run_id, "my-checkpoint")

        run_key = store.build_key(TrainingController.REDIS_KEY_PREFIX, training_run_id)
        ckpt_key = controller._build_checkpoint_key(training_run_id, "my-checkpoint")
        assert store.exists(run_key) is True
        assert store.exists(ckpt_key) is True

        # === Phase 2: Crash and restart ===
        del controller
        new_controller = TrainingController(app_config)

        # === Phase 3: Verify restoration ===
        restored = new_controller.training_runs[training_run_id]
        assert "my-checkpoint" in restored.checkpoints
        restored_ckpt = restored.checkpoints["my-checkpoint"]
        assert restored_ckpt.checkpoint_id == "my-checkpoint"
        assert restored_ckpt.size_bytes == 1024
        assert restored_ckpt.checkpoint_type == "training"


# =============================================================================
# Full Server State Persistence Tests (Requires Ray and GPU)
# =============================================================================


@pytest.mark.persistence
@pytest.mark.gpu
class TestServerStatePersistence:
    """This test simulates actual server crash/restart."""

    @pytest.fixture
    def setup(self, tmp_path, request):
        if not request.config.getoption("--gpu"):
            pytest.skip("GPU mode required for this test")

        _skip_if_no_persistence()
        store = get_redis_store()

        app_config = _create_test_config(tmp_path / "checkpoints")
        app_config.ensure_directories()

        yield store, app_config

        # Cleanup Ray
        _ensure_ray_shutdown()

    @pytest.mark.asyncio
    async def test_full_training_workflow_survives_restart_with_two_crashes(self, setup):
        """Test complete training workflow with multiple restarts and checkpoint recovery.

        This comprehensive test simulates a full training session with crashes:
        1. Create session and training run
        2. First training cycle: fb + optim_step, save checkpoint-1
        3. Second training cycle: fb + optim_step, save checkpoint-2
        4. [CRASH 1] Restart and verify recovery to checkpoint-2
           - All 4 futures (fb1, o1, fb2, o2) should be ready
        5. Continue training - 1 more fb + optim_step (completed futures)
        6. [CRASH 2] Restart - futures after checkpoint-2 should be FAILED
           - fb1, o1, fb2, o2 should still be ready (before/at checkpoint-2)
           - fb3, o3 should be FAILED (after checkpoint-2)
        """
        store, app_config = setup
        datum = _create_test_datum()

        # =====================================================================
        # PHASE 1: Initial Setup - Create session and training run
        # =====================================================================
        _init_ray()
        state = ServerState(app_config)
        await state.async_init()

        session = state.create_session(
            types.CreateSessionRequest(
                tags=["full-workflow"], user_metadata={"experiment": "e2e"}, sdk_version="2.0"
            ),
            User(user_id="trainer"),
        )
        session_id = session.session_id

        training = await state.create_model(
            session_id=session.session_id,
            base_model="Qwen/Qwen3-0.6B",
            lora_config=types.LoraConfig(rank=8),
            model_owner="trainer",
            user_metadata={"model_version": "v1"},
        )
        training_run_id = training.training_run_id

        # =====================================================================
        # PHASE 2: First training cycle - fb + optim_step + checkpoint-1
        # =====================================================================

        # Helper to create async operation wrapper (lambda can't be async)
        def make_forward_op(st, run_id, data, sid):
            async def op():
                return await st.training.run_forward(
                    model_id=run_id,
                    user_id="trainer",
                    data=data,
                    loss_fn="cross_entropy",
                    loss_fn_config=None,
                    seq_id=sid,
                    backward=True,
                )

            return op

        def make_optim_op(st, run_id, sid):
            async def op():
                return await st.training.run_optim_step(
                    model_id=run_id,
                    user_id="trainer",
                    params=types.AdamParams(learning_rate=0.001),
                    seq_id=sid,
                )

            return op

        # Forward+backward pass 1 (seq_id=1)
        current_seq_id = state.training.training_runs[training_run_id].next_seq_id
        fb1_future = await state.future_store.enqueue(
            make_forward_op(state, training_run_id, [datum], current_seq_id),
            user_id="trainer",
            model_id=training_run_id,
            operation_type="forward_backward",
        )
        fb1_request_id = fb1_future.request_id
        await _wait_for_result(state.future_store, fb1_request_id, "trainer")

        # Optimizer step 1 (seq_id=2)
        current_seq_id = state.training.training_runs[training_run_id].next_seq_id
        o1_future = await state.future_store.enqueue(
            make_optim_op(state, training_run_id, current_seq_id),
            user_id="trainer",
            model_id=training_run_id,
            operation_type="optim_step",
        )
        o1_request_id = o1_future.request_id
        await _wait_for_result(state.future_store, o1_request_id, "trainer")

        # Save checkpoint-1
        ckpt1 = await state.save_checkpoint(
            model_id=training_run_id,
            user_id="trainer",
            name="checkpoint-1",
            checkpoint_type="training",
        )
        assert ckpt1.created_at is not None
        # ckpt1_time = ckpt1.created_at
        seq_id_at_ckpt1 = state.training.training_runs[training_run_id].next_seq_id
        assert seq_id_at_ckpt1 == 3  # After 2 operations

        # =====================================================================
        # PHASE 3: Second training cycle - fb + optim_step + checkpoint-2
        # =====================================================================

        # Forward+backward pass 2 (seq_id=3)
        current_seq_id = state.training.training_runs[training_run_id].next_seq_id
        fb2_future = await state.future_store.enqueue(
            make_forward_op(state, training_run_id, [datum], current_seq_id),
            user_id="trainer",
            model_id=training_run_id,
            operation_type="forward_backward",
        )
        fb2_request_id = fb2_future.request_id
        await _wait_for_result(state.future_store, fb2_request_id, "trainer")

        # Optimizer step 2 (seq_id=4)
        current_seq_id = state.training.training_runs[training_run_id].next_seq_id
        o2_future = await state.future_store.enqueue(
            make_optim_op(state, training_run_id, current_seq_id),
            user_id="trainer",
            model_id=training_run_id,
            operation_type="optim_step",
        )
        o2_request_id = o2_future.request_id
        await _wait_for_result(state.future_store, o2_request_id, "trainer")

        # Save checkpoint-2 (this is the final recovery point)
        ckpt2 = await state.save_checkpoint(
            model_id=training_run_id,
            user_id="trainer",
            name="checkpoint-2",
            checkpoint_type="training",
        )
        assert ckpt2.created_at is not None
        ckpt2_time = ckpt2.created_at
        seq_id_at_ckpt2 = state.training.training_runs[training_run_id].next_seq_id
        assert seq_id_at_ckpt2 == 5  # After 4 operations

        # Verify all 4 futures are successful before crash
        assert state.future_store._records[fb1_request_id].status == "ready"
        assert state.future_store._records[o1_request_id].status == "ready"
        assert state.future_store._records[fb2_request_id].status == "ready"
        assert state.future_store._records[o2_request_id].status == "ready"

        await state.future_store.shutdown()

        # =====================================================================
        # CRASH 1: First restart - verify recovery to checkpoint-2
        #          All 4 futures (fb1, o1, fb2, o2) should be ready
        # =====================================================================
        del state
        _ensure_ray_shutdown()

        _init_ray()
        state2 = ServerState(app_config)
        await state2.async_init()

        # Verify session restored
        assert session_id in state2.sessions._sessions
        restored_session = state2.sessions._sessions[session_id]
        assert restored_session.tags == ["full-workflow"]

        # Verify training run restored
        assert training_run_id in state2.training.training_runs
        restored_training = state2.training.training_runs[training_run_id]
        assert restored_training.lora_rank == 8
        assert restored_training.backend is not None

        # Verify both checkpoints restored
        assert "checkpoint-1" in restored_training.checkpoints
        assert "checkpoint-2" in restored_training.checkpoints

        # Verify seq_id preserved at checkpoint-2 state
        assert restored_training.next_seq_id == seq_id_at_ckpt2

        # *** Verify all 4 futures before/at checkpoint-2 are still ready ***
        assert state2.future_store._records[fb1_request_id].status == "ready", (
            "fb1 should remain 'ready' after first restart"
        )
        assert state2.future_store._records[o1_request_id].status == "ready", (
            "o1 should remain 'ready' after first restart"
        )
        assert state2.future_store._records[fb2_request_id].status == "ready", (
            "fb2 should remain 'ready' after first restart"
        )
        assert state2.future_store._records[o2_request_id].status == "ready", (
            "o2 should remain 'ready' after first restart"
        )

        # =====================================================================
        # PHASE 4: Continue training after checkpoint-2
        #          These operations complete but will be FAILED after next restart
        # =====================================================================

        # Forward+backward pass 3 (seq_id=5)
        current_seq_id = state2.training.training_runs[training_run_id].next_seq_id
        fb3_future = await state2.future_store.enqueue(
            make_forward_op(state2, training_run_id, [datum], current_seq_id),
            user_id="trainer",
            model_id=training_run_id,
            operation_type="forward_backward",
        )
        fb3_request_id = fb3_future.request_id
        await _wait_for_result(state2.future_store, fb3_request_id, "trainer")

        # Optimizer step 3 (seq_id=6)
        current_seq_id = state2.training.training_runs[training_run_id].next_seq_id
        o3_future = await state2.future_store.enqueue(
            make_optim_op(state2, training_run_id, current_seq_id),
            user_id="trainer",
            model_id=training_run_id,
            operation_type="optim_step",
        )
        o3_request_id = o3_future.request_id
        await _wait_for_result(state2.future_store, o3_request_id, "trainer")

        assert state2.future_store._records[fb3_request_id].status == "ready"
        assert state2.future_store._records[o3_request_id].status == "ready"

        assert state2.future_store._records[fb3_request_id].created_at > ckpt2_time
        assert state2.future_store._records[o3_request_id].created_at > ckpt2_time

        await state2.future_store.shutdown()

        # =====================================================================
        # CRASH 2: Second restart - recover to checkpoint-2
        #          Futures AFTER checkpoint-2 should be marked as FAILED
        #          (regardless of their previous status)
        # =====================================================================
        del state2
        _ensure_ray_shutdown()

        _init_ray()
        state3 = ServerState(app_config)
        await state3.async_init()

        assert training_run_id in state3.training.training_runs
        restored = state3.training.training_runs[training_run_id]

        assert "checkpoint-1" in restored.checkpoints
        assert "checkpoint-2" in restored.checkpoints

        assert restored.backend is not None

        # Note: next_seq_id is restored from Redis, not rolled back to checkpoint state.
        # It reflects the latest saved value (after fb3 and o3 operations = 7)
        # This is intentional: seq_id is always monotonically increasing to avoid conflicts.
        assert restored.next_seq_id == 7, (
            "next_seq_id should be the latest value from Redis (after fb3 and o3)"
        )

        # *** Verify futures BEFORE/AT checkpoint-2 are still ready ***
        assert state3.future_store._records[fb1_request_id].status == "ready", (
            "fb1 should remain 'ready' after second restart"
        )
        assert state3.future_store._records[o1_request_id].status == "ready", (
            "o1 should remain 'ready' after second restart"
        )
        assert state3.future_store._records[fb2_request_id].status == "ready", (
            "fb2 should remain 'ready' after second restart"
        )
        assert state3.future_store._records[o2_request_id].status == "ready", (
            "o2 should remain 'ready' after second restart"
        )

        # *** Verify futures AFTER checkpoint-2 are marked as FAILED ***
        # Even though they were "ready" before crash, they are now invalid
        assert state3.future_store._records[fb3_request_id].status == "failed", (
            "fb3 (after checkpoint-2) should be 'failed' after second restart"
        )
        assert state3.future_store._records[fb3_request_id].error is not None

        assert state3.future_store._records[o3_request_id].status == "failed", (
            "o3 (after checkpoint-2) should be 'failed' after second restart"
        )
        assert state3.future_store._records[o3_request_id].error is not None
        # =====================================================================
        # PHASE 5: Verify training can continue from checkpoint-2
        # =====================================================================
        current_seq_id = restored.next_seq_id

        new_forward = await state3.run_forward(
            training_run_id,
            user_id="trainer",
            data=[datum],
            loss_fn="cross_entropy",
            loss_fn_config=None,
            seq_id=current_seq_id,
            backward=True,
        )
        assert new_forward is not None
        assert "loss:sum" in new_forward.metrics

        await state3.future_store.shutdown()


# =============================================================================
# Config Signature Validation Tests
# =============================================================================


def _create_config_with_models(
    checkpoint_dir: Path,
    model_names: list[str],
    telemetry_enabled: bool = False,
    check_fields: list[str] | None = None,
) -> AppConfig:
    """Create a test config with specified model names and optional telemetry."""
    from tuft.persistence import PersistenceConfig, PersistenceMode
    from tuft.persistence.redis_store import DEFAULT_CHECK_FIELDS

    return AppConfig(
        checkpoint_dir=checkpoint_dir,
        supported_models=[
            ModelConfig(
                model_name=name,
                model_path=Path(f"/dummy/{name}"),
                max_model_len=2048,
            )
            for name in model_names
        ],
        telemetry=TelemetryConfig(enabled=telemetry_enabled),
        persistence=PersistenceConfig(
            mode=PersistenceMode.FILE,
            check_fields=check_fields if check_fields is not None else DEFAULT_CHECK_FIELDS.copy(),
        ),
    )


@pytest.mark.persistence
class TestConfigSignatureValidation:
    """Test configuration signature validation for persistence safety."""

    @pytest.fixture
    def setup(self, tmp_path):
        """Setup file-based persistence for testing."""
        _skip_if_no_persistence()
        store = get_redis_store()
        yield store, tmp_path

        # Cleanup after test
        flush_all_data()

    def test_config_signature_creation(self, setup):
        """Test that ConfigSignature is created correctly from AppConfig."""
        _, tmp_path = setup
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()

        config = _create_config_with_models(checkpoint_dir, ["ModelA", "ModelB", "ModelC"])

        signature = ConfigSignature.from_app_config(config)

        models = signature.config_data.get("supported_models", [])
        model_names = sorted([m["model_name"] for m in models])
        assert model_names == ["ModelA", "ModelB", "ModelC"]
        assert signature.created_at is not None

        # Verify other fields are also stored
        assert "checkpoint_dir" in signature.config_data
        assert "model_owner" in signature.config_data
        assert "telemetry" in signature.config_data

    def test_save_and_load_config_signature(self, setup):
        """Test that config signature can be saved and loaded from Redis."""
        _, tmp_path = setup
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()

        config = _create_config_with_models(checkpoint_dir, ["ModelA", "ModelB"])

        # Save signature
        result = save_config_signature(config)
        assert result is True

        # Load signature
        loaded = load_config_signature()
        assert loaded is not None
        models = loaded.config_data.get("supported_models", [])
        model_names = sorted([m["model_name"] for m in models])
        assert model_names == ["ModelA", "ModelB"]

    def test_validate_config_signature_matching(self, setup):
        """Test that validation succeeds when configs match."""
        _, tmp_path = setup
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()

        config = _create_config_with_models(checkpoint_dir, ["ModelA", "ModelB"])

        save_config_signature(config)

        is_fresh = validate_config_signature(config)
        assert is_fresh is False

    def test_validate_config_signature_mismatch_raises(self, setup):
        """Test that validation raises ConfigMismatchError on mismatch."""
        _, tmp_path = setup
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()

        config1 = _create_config_with_models(checkpoint_dir, ["ModelA", "ModelB"])
        config2 = _create_config_with_models(checkpoint_dir, ["ModelC"])

        save_config_signature(config1)

        with pytest.raises(ConfigMismatchError) as exc_info:
            validate_config_signature(config2)

        error = exc_info.value
        assert "supported_models" in str(error).lower() or "mismatch" in str(error).lower()

        model_diff = error.diff.get(ConfigCheckField.SUPPORTED_MODELS)
        assert model_diff is not None
        current_names = [m["model_name"] for m in model_diff["current"]]
        stored_names = [m["model_name"] for m in model_diff["stored"]]
        assert "ModelC" in current_names
        assert "ModelA" in stored_names
        assert "ModelB" in stored_names

    def test_flush_all_data_clears_signature(self, setup):
        """Test that flush_all_data clears the config signature."""
        _, tmp_path = setup
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()

        config = _create_config_with_models(checkpoint_dir, ["ModelA"])

        save_config_signature(config)
        assert load_config_signature() is not None

        flush_all_data()

        assert load_config_signature() is None

    def test_config_mismatch_prevents_silent_corruption(self, setup):
        """Test the full scenario: config mismatch is detected before corruption.

        This test verifies the fix for the original issue:
        1. Create training run with ModelA
        2. Try to restart with config that only has ModelC
        3. ConfigMismatchError should be raised BEFORE any corruption happens
        """
        store, tmp_path = setup
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()

        # === Phase 1: Create training run with ModelA ===
        config1 = _create_config_with_models(checkpoint_dir, ["ModelA", "ModelB"])
        config1.ensure_directories()

        # Validate (first run - returns True for fresh start)
        is_fresh = validate_config_signature(config1)
        assert is_fresh is True

        controller1 = TrainingController(config1)

        # Create a training run
        training_run_id = "test-run-validation"
        record = TrainingRunRecord(
            training_run_id=training_run_id,
            base_model="ModelA",
            lora_rank=8,
            session_id="session-001",
            model_owner="user1",
        )
        record.backend = controller1.training_backends.get("ModelA")
        controller1.training_runs[training_run_id] = record
        controller1._save_training_run(training_run_id)

        # Simulate successful init - save signature
        save_config_signature(config1)

        del controller1

        # === Phase 2: Try to restart with different config ===
        config2 = _create_config_with_models(checkpoint_dir, ["ModelC"])
        config2.ensure_directories()

        # This should raise ConfigMismatchError BEFORE TrainingController is created
        with pytest.raises(ConfigMismatchError):
            validate_config_signature(config2)

    def test_refresh_persistence_allows_restart_with_new_config(self, setup):
        """Test that after flush_all_data, a new config can be used."""
        _, tmp_path = setup
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()

        config1 = _create_config_with_models(checkpoint_dir, ["ModelA"])
        config2 = _create_config_with_models(checkpoint_dir, ["ModelC"])

        save_config_signature(config1)

        with pytest.raises(ConfigMismatchError):
            validate_config_signature(config2)

        flush_all_data()

        is_fresh = validate_config_signature(config2)
        assert is_fresh is True

        save_config_signature(config2)

        loaded = load_config_signature()
        assert loaded is not None
        models = loaded.config_data.get("supported_models", [])
        model_names = [m["model_name"] for m in models]
        assert model_names == ["ModelC"]

    def test_matches_with_custom_check_fields(self, setup):
        """Test matches() with custom check_fields parameter."""
        _, tmp_path = setup
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()

        # Same models, different telemetry
        config1 = _create_config_with_models(checkpoint_dir, ["ModelA"], telemetry_enabled=False)
        config2 = _create_config_with_models(checkpoint_dir, ["ModelA"], telemetry_enabled=True)

        sig1 = ConfigSignature.from_app_config(config1)
        sig2 = ConfigSignature.from_app_config(config2)

        # With default check_fields (only model names), they should match
        assert sig1.matches(sig2, check_fields=DEFAULT_CHECK_FIELDS)

        # With telemetry check included, they should NOT match
        check_with_telemetry = [
            ConfigCheckField.SUPPORTED_MODELS,
            ConfigCheckField.TELEMETRY,
        ]
        assert not sig1.matches(sig2, check_fields=check_with_telemetry)

    def test_get_diff_with_custom_check_fields(self, setup):
        """Test get_diff() with custom check_fields parameter."""
        _, tmp_path = setup
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()

        # Different models AND different telemetry
        config1 = _create_config_with_models(checkpoint_dir, ["ModelA"], telemetry_enabled=False)
        config2 = _create_config_with_models(checkpoint_dir, ["ModelB"], telemetry_enabled=True)

        sig1 = ConfigSignature.from_app_config(config1)
        sig2 = ConfigSignature.from_app_config(config2)

        # With default check_fields, only model diff is returned
        diff_default = sig1.get_diff(sig2, check_fields=DEFAULT_CHECK_FIELDS)
        assert ConfigCheckField.SUPPORTED_MODELS in diff_default
        assert ConfigCheckField.TELEMETRY not in diff_default

        # With telemetry check included, both diffs are returned
        check_with_telemetry = [
            ConfigCheckField.SUPPORTED_MODELS,
            ConfigCheckField.TELEMETRY,
        ]
        diff_full = sig1.get_diff(sig2, check_fields=check_with_telemetry)
        assert ConfigCheckField.SUPPORTED_MODELS in diff_full
        assert ConfigCheckField.TELEMETRY in diff_full

        # Check telemetry diff values (telemetry is a dict with 'enabled' field)
        telemetry_diff = diff_full[ConfigCheckField.TELEMETRY]
        assert telemetry_diff["current"]["enabled"] is False  # current value
        assert telemetry_diff["stored"]["enabled"] is True  # stored value

    def test_model_names_always_checked(self, setup):
        """Test that SUPPORTED_MODELS is always checked even if not in check_fields."""
        _, tmp_path = setup
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()

        config1 = _create_config_with_models(checkpoint_dir, ["ModelA"])
        config2 = _create_config_with_models(checkpoint_dir, ["ModelB"])

        sig1 = ConfigSignature.from_app_config(config1)
        sig2 = ConfigSignature.from_app_config(config2)

        # Even with only telemetry in check_fields, model names should be checked
        # because it's mandatory
        check_only_telemetry = [ConfigCheckField.TELEMETRY]
        assert not sig1.matches(sig2, check_fields=check_only_telemetry)

        diff = sig1.get_diff(sig2, check_fields=check_only_telemetry)
        # Model names should still be in the diff
        assert ConfigCheckField.SUPPORTED_MODELS in diff

    def test_validate_with_custom_check_fields(self, setup):
        """Test validate_config_signature with custom check_fields from persistence config."""
        _, tmp_path = setup
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()

        # Ensure clean state
        flush_all_data()

        # First run with telemetry disabled - save signature after successful init
        config1 = _create_config_with_models(checkpoint_dir, ["ModelA"], telemetry_enabled=False)
        save_config_signature(config1)

        # Second run with same model but telemetry enabled (default check_fields)
        config2_default = _create_config_with_models(
            checkpoint_dir, ["ModelA"], telemetry_enabled=True
        )

        # With default check_fields, should pass (only model names checked)
        validate_config_signature(config2_default)

        # With telemetry check in persistence config, should fail
        check_with_telemetry = [
            ConfigCheckField.SUPPORTED_MODELS,
            ConfigCheckField.TELEMETRY,
        ]
        config2_with_telemetry_check = _create_config_with_models(
            checkpoint_dir, ["ModelA"], telemetry_enabled=True, check_fields=check_with_telemetry
        )
        with pytest.raises(ConfigMismatchError):
            validate_config_signature(config2_with_telemetry_check)

    def test_has_existing_data_detects_data(self, setup):
        """Test that has_existing_data correctly detects existing data."""
        store, tmp_path = setup
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()

        # Ensure clean state
        flush_all_data()

        # Should be empty initially
        assert has_existing_data() is False

        # Save some data
        config = _create_config_with_models(checkpoint_dir, ["ModelA"])
        save_config_signature(config)

        # Now should have data
        assert has_existing_data() is True

        # Flush and check again
        flush_all_data()
        assert has_existing_data() is False

    def test_validate_detects_corrupted_state(self, setup):
        """Test that validation detects when data exists but no signature."""
        store, tmp_path = setup
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()

        # Ensure clean state
        flush_all_data()

        # Manually add some data without signature (simulating corrupted state)
        store.set(store.build_key("session", "test-session"), '{"test": "data"}')

        # Now validation should fail with corrupted state error
        config = _create_config_with_models(checkpoint_dir, ["ModelA"])
        with pytest.raises(ConfigMismatchError) as exc_info:
            validate_config_signature(config)

        error = exc_info.value
        assert "_state" in error.diff
        assert "missing signature" in str(error).lower() or "corrupted" in str(error).lower()
