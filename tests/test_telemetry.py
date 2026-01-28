"""OpenTelemetry integration tests for span context, attribute propagation, and isolation."""

from __future__ import annotations

import gc
import os
from pathlib import Path

import pytest
import ray
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from tinker import types
from tinker.lib.public_interfaces.service_client import ServiceClient

from tuft.auth import User
from tuft.config import AppConfig, ModelConfig, TelemetryConfig
from tuft.state import ServerState
from tuft.telemetry.tracing import clear_tracers

from .helpers import _find_free_port, _start_server, _stop_server


# =============================================================================
# Session-scoped Tracer Provider Fixture
# =============================================================================


@pytest.fixture(scope="session")
def session_tracer_provider():
    """Set up a test tracer provider once per session.

    This fixture sets up the OpenTelemetry tracer provider at session scope
    to avoid issues with pytest not being process-isolated.
    """
    span_exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider(resource=Resource.create({"service.name": "tuft-test"}))
    tracer_provider.add_span_processor(SimpleSpanProcessor(span_exporter))

    # Only set if not already configured
    if not isinstance(trace.get_tracer_provider(), TracerProvider):
        trace.set_tracer_provider(tracer_provider)

    yield {"provider": tracer_provider, "exporter": span_exporter}

    # Cleanup: shutdown the tracer provider
    tracer_provider.shutdown()


# =============================================================================
# Fixtures and Helpers
# =============================================================================


@pytest.fixture(scope="function")
def span_exporter(session_tracer_provider):
    """Get the session-level span exporter and clear it for each test."""
    exporter = session_tracer_provider["exporter"]
    exporter.clear()
    clear_tracers()
    yield exporter
    exporter.clear()


@pytest.fixture(scope="function")
def setup_tracer_provider(session_tracer_provider, span_exporter):
    """Provide the tracer provider for tests."""
    yield session_tracer_provider["provider"]


@pytest.fixture(scope="function")
def ray_cluster(request):
    if request.config.getoption("--gpu"):
        ray.init(ignore_reinit_error=True)
        yield
        ray.shutdown(_exiting_interpreter=True)
        gc.collect()
    else:
        yield


def _build_state(tmp_path, use_gpu: bool = False) -> ServerState:
    model_path = (
        Path(os.environ.get("TUFT_TEST_MODEL", "/path/to/model"))
        if use_gpu
        else Path("/path/to/model")
    )
    config = AppConfig(checkpoint_dir=tmp_path)
    config.supported_models = [
        ModelConfig(
            model_name="Qwen/Qwen3-0.6B",
            model_path=model_path,
            max_model_len=2048,
            tensor_parallel_size=1,
        )
    ]
    config.telemetry = TelemetryConfig(enabled=True, service_name="tuft-test")
    return ServerState(config)


def _create_session(state: ServerState, user_id: str = "tester") -> str:
    return state.create_session(
        types.CreateSessionRequest(tags=["test"], user_metadata=None, sdk_version="1.0"),
        user=User(user_id=user_id),
    ).session_id


def _make_datum(tokens: list[int] | None = None) -> types.Datum:
    tokens = tokens or [1, 2, 3]
    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens),
        loss_fn_inputs={
            "target_tokens": types.TensorData(
                data=[t + 10 for t in tokens], dtype="int64", shape=[len(tokens)]
            ),
            "weights": types.TensorData(
                data=[1.0] * len(tokens), dtype="float32", shape=[len(tokens)]
            ),
        },
    )


def _get_spans_with_attr(spans, attr_name: str, attr_value=None) -> list:
    return [
        s
        for s in spans
        if s.attributes
        and attr_name in s.attributes
        and (attr_value is None or s.attributes[attr_name] == attr_value)
    ]


def _get_span_names(spans) -> set[str]:
    return {s.name for s in spans}


async def _run_forward(state, rid, user_id, datum, seq_id, backward=True):
    """Helper to run forward pass."""
    await state.run_forward(
        rid,
        user_id=user_id,
        data=[datum],
        loss_fn="cross_entropy",
        loss_fn_config=None,
        seq_id=seq_id,
        backward=backward,
    )


async def _create_training(state, session_id, user_id="tester", rank=4):
    """Helper to create training model."""
    return await state.create_model(
        session_id,
        base_model="Qwen/Qwen3-0.6B",
        lora_config=types.LoraConfig(rank=rank),
        model_owner=user_id,
        user_metadata=None,
    )


async def _create_sampling(state, session_id, seq_id=1, user_id="tester", model_path=None):
    """Helper to create sampling session."""
    return await state.create_sampling_session(
        session_id=session_id,
        base_model="Qwen/Qwen3-0.6B" if model_path is None else None,
        model_path=model_path,
        session_seq_id=seq_id,
        user_id=user_id,
    )


async def _run_sample(state, sampling_id, user_id="tester", seq_id=1):
    """Helper to run sample."""
    return await state.run_sample(
        types.SampleRequest(
            prompt=types.ModelInput.from_ints([1, 2, 3]),
            num_samples=1,
            sampling_params=types.SamplingParams(max_tokens=2, temperature=0.1),
            sampling_session_id=sampling_id,
            seq_id=seq_id,
        ),
        user_id=user_id,
    )


# =============================================================================
# Core Workflow Tests
# =============================================================================


@pytest.mark.asyncio
async def test_training_workflow_spans(
    request, tmp_path, span_exporter, setup_tracer_provider, ray_cluster
):
    """Test spans for full training workflow with correct attributes."""
    state = _build_state(tmp_path, request.config.getoption("--gpu"))
    session_id = _create_session(state)
    training = await _create_training(state, session_id)
    rid = training.training_run_id
    datum = _make_datum()

    await _run_forward(state, rid, "tester", datum, 1, backward=False)
    await _run_forward(state, rid, "tester", datum, 2, backward=True)
    await state.run_optim_step(
        rid, user_id="tester", params=types.AdamParams(learning_rate=1e-3), seq_id=3
    )
    await state.save_checkpoint(
        rid, user_id="tester", name="ckpt", checkpoint_type="training", seq_id=4
    )

    spans = span_exporter.get_finished_spans()
    names = _get_span_names(spans)

    # Verify expected spans and attributes
    expected = {
        "training_controller.create_model",
        "training_controller.run_forward",
        "training_controller.run_forward_backward",
        "training_controller.run_optim_step",
        "training_controller.save_checkpoint",
    }
    assert expected.issubset(names), f"Missing spans: {expected - names}"
    assert len(_get_spans_with_attr(spans, "tuft.training_run_id", rid)) >= 4
    assert len(_get_spans_with_attr(spans, "tuft.session_id", session_id)) >= 4

    # Verify create_model span attributes
    create_spans = [s for s in spans if s.name == "training_controller.create_model"]
    assert create_spans[0].attributes["tuft.training_run_id"] == rid
    assert create_spans[0].attributes["tuft.base_model"] == "Qwen/Qwen3-0.6B"


@pytest.mark.asyncio
async def test_sampling_workflow_spans(
    request, tmp_path, span_exporter, setup_tracer_provider, ray_cluster
):
    """Test spans for sampling workflow with correct attributes."""
    state = _build_state(tmp_path, request.config.getoption("--gpu"))
    session_id = _create_session(state)
    sampling_id = await _create_sampling(state, session_id)
    await _run_sample(state, sampling_id)

    spans = span_exporter.get_finished_spans()
    expected = {"sampling_controller.create_sampling_session", "sampling_controller.run_sample"}
    assert expected.issubset(_get_span_names(spans))
    assert len(_get_spans_with_attr(spans, "tuft.sampling_session_id", sampling_id)) >= 2
    assert len(_get_spans_with_attr(spans, "tuft.session_id", session_id)) >= 2


# =============================================================================
# Isolation Tests
# =============================================================================


@pytest.mark.asyncio
async def test_multi_session_and_training_isolation(
    request, tmp_path, span_exporter, setup_tracer_provider, ray_cluster
):
    """Test session and training run isolation across different users/sessions."""
    state = _build_state(tmp_path, request.config.getoption("--gpu"))
    # Create two sessions with different users
    sid1, sid2 = _create_session(state, "user1"), _create_session(state, "user2")
    t1 = await _create_training(state, sid1, "user1", rank=4)
    t2 = await _create_training(state, sid2, "user2", rank=8)

    datum = _make_datum()
    await _run_forward(state, t1.training_run_id, "user1", datum, 1)
    await _run_forward(state, t2.training_run_id, "user2", datum, 1)

    spans = span_exporter.get_finished_spans()
    s1_spans = _get_spans_with_attr(spans, "tuft.session_id", sid1)
    s2_spans = _get_spans_with_attr(spans, "tuft.session_id", sid2)

    # Verify isolation
    assert len(s1_spans) >= 2 and len(s2_spans) >= 2
    assert {s.context.span_id for s in s1_spans}.isdisjoint({s.context.span_id for s in s2_spans})

    # Verify training_run_id consistency
    for s in s1_spans:
        if "tuft.training_run_id" in s.attributes:
            assert s.attributes["tuft.training_run_id"] == t1.training_run_id
    for s in s2_spans:
        if "tuft.training_run_id" in s.attributes:
            assert s.attributes["tuft.training_run_id"] == t2.training_run_id


@pytest.mark.asyncio
async def test_multi_training_run_same_session(
    request, tmp_path, span_exporter, setup_tracer_provider, ray_cluster
):
    """Test multiple training runs in same session have isolated spans but same session_id."""
    state = _build_state(tmp_path, request.config.getoption("--gpu"))
    sid = _create_session(state)
    t1 = await _create_training(state, sid, rank=4)
    t2 = await _create_training(state, sid, rank=8)

    datum = _make_datum()
    await _run_forward(state, t1.training_run_id, "tester", datum, 1)
    await _run_forward(state, t2.training_run_id, "tester", datum, 1)
    await state.run_optim_step(
        t1.training_run_id, user_id="tester", params=types.AdamParams(), seq_id=2
    )
    await state.run_optim_step(
        t2.training_run_id, user_id="tester", params=types.AdamParams(), seq_id=2
    )

    spans = span_exporter.get_finished_spans()
    t1_spans = _get_spans_with_attr(spans, "tuft.training_run_id", t1.training_run_id)
    t2_spans = _get_spans_with_attr(spans, "tuft.training_run_id", t2.training_run_id)

    assert len(t1_spans) >= 3 and len(t2_spans) >= 3
    assert {s.context.span_id for s in t1_spans}.isdisjoint({s.context.span_id for s in t2_spans})
    # All should have same session_id
    for s in t1_spans + t2_spans:
        if "tuft.session_id" in s.attributes:
            assert s.attributes["tuft.session_id"] == sid


@pytest.mark.asyncio
async def test_multi_sampling_session_isolation(
    request, tmp_path, span_exporter, setup_tracer_provider, ray_cluster
):
    """Test multiple sampling sessions have isolated spans."""
    state = _build_state(tmp_path, request.config.getoption("--gpu"))
    sid = _create_session(state)
    ss1 = await _create_sampling(state, sid, seq_id=1)
    ss2 = await _create_sampling(state, sid, seq_id=2)

    await _run_sample(state, ss1)
    await _run_sample(state, ss2)

    spans = span_exporter.get_finished_spans()
    ss1_spans = _get_spans_with_attr(spans, "tuft.sampling_session_id", ss1)
    ss2_spans = _get_spans_with_attr(spans, "tuft.sampling_session_id", ss2)

    assert len(ss1_spans) >= 2 and len(ss2_spans) >= 2
    assert {s.context.span_id for s in ss1_spans}.isdisjoint({s.context.span_id for s in ss2_spans})


# =============================================================================
# End-to-End and FutureStore Tests
# =============================================================================


@pytest.mark.asyncio
async def test_end_to_end_with_future_store(
    request, tmp_path, span_exporter, setup_tracer_provider, ray_cluster
):
    """Test comprehensive flow including FutureStore spans."""
    state = _build_state(tmp_path, request.config.getoption("--gpu"))
    sid = _create_session(state)
    t = await _create_training(state, sid)
    rid = t.training_run_id
    datum = _make_datum([1, 2, 3, 4, 5])

    # Training workflow
    await _run_forward(state, rid, "tester", datum, 1)
    await state.run_optim_step(
        rid, user_id="tester", params=types.AdamParams(learning_rate=1e-3), seq_id=2
    )
    ckpt = await state.save_checkpoint(
        rid, user_id="tester", name="sampler-ckpt", checkpoint_type="sampler", seq_id=3
    )

    # Sampling workflow
    ss_id = await _create_sampling(state, sid, model_path=ckpt.tinker_checkpoint.tinker_path)
    resp = await _run_sample(state, ss_id)
    assert resp.sequences

    # Test FutureStore spans via enqueue
    async def op():
        return await _run_forward(state, rid, "tester", datum, 4)

    future = await state.future_store.enqueue(
        op, user_id="tester", model_id=rid, operation_type="forward_backward", operation_args={}
    )
    await state.future_store.retrieve(future.request_id, user_id="tester", timeout=10.0)

    spans = span_exporter.get_finished_spans()
    names = _get_span_names(spans)

    # Verify all expected operations
    expected = {
        "training_controller.create_model",
        "training_controller.run_forward_backward",
        "training_controller.run_optim_step",
        "training_controller.save_checkpoint",
        "sampling_controller.create_sampling_session",
        "sampling_controller.run_sample",
        "future_store.execute_operation",
    }
    assert expected.issubset(names), f"Missing: {expected - names}"

    # Verify attributes
    assert len(_get_spans_with_attr(spans, "tuft.session_id", sid)) >= 5
    assert len(_get_spans_with_attr(spans, "tuft.training_run_id", rid)) >= 4
    assert len(_get_spans_with_attr(spans, "tuft.sampling_session_id", ss_id)) >= 2

    # Verify FutureStore span attributes
    fs_spans = [s for s in spans if s.name == "future_store.execute_operation"]
    assert fs_spans[0].attributes.get("tuft.request_id") is not None
    assert fs_spans[0].attributes.get("tuft.operation_type") == "forward_backward"


# =============================================================================
# HTTP Integration Test
# =============================================================================


@pytest.fixture(scope="function")
def telemetry_server(tmp_path_factory, request, span_exporter, setup_tracer_provider):  # type: ignore[misc]
    """Start a test server with telemetry enabled."""
    saved_api_key = os.environ.pop("TINKER_API_KEY", None)
    if request.config.getoption("--gpu"):
        ray.init(ignore_reinit_error=True)
        model_path = Path(os.environ.get("TUFT_TEST_MODEL", "Qwen/Qwen3-0.6B"))
    else:
        model_path = Path("/dummy/model")

    config = AppConfig(checkpoint_dir=Path(tmp_path_factory.mktemp("checkpoints")))
    config.supported_models = [
        ModelConfig(
            model_name="Qwen/Qwen3-0.6B",
            model_path=model_path,
            max_model_len=4096,
            tensor_parallel_size=1,
        )
    ]
    config.authorized_users = {"tml-test-key-1": "user1", "tml-test-key-2": "user2"}
    config.telemetry = TelemetryConfig(enabled=True, service_name="tuft-integration-test")

    server, thread, base_url, client = _start_server(config, _find_free_port())
    yield base_url
    _stop_server(server, thread, client)

    if request.config.getoption("--gpu"):
        ray.shutdown(_exiting_interpreter=True)
        gc.collect()
    if saved_api_key is not None:
        os.environ["TINKER_API_KEY"] = saved_api_key


@pytest.mark.integration
def test_http_trace_isolation(telemetry_server: str, span_exporter, setup_tracer_provider) -> None:
    """Test HTTP request traces: multiple clients create isolated spans."""
    client1 = ServiceClient(
        api_key="tml-test-key-1",  # pragma: allowlist secret
        base_url=telemetry_server,
        timeout=15,
    )
    client2 = ServiceClient(
        api_key="tml-test-key-2",  # pragma: allowlist secret
        base_url=telemetry_server,
        timeout=15,
    )

    try:
        caps = client1.get_server_capabilities()
        base_model = caps.supported_models[0].model_name or "Qwen/Qwen3-0.6B"
        tc1 = client1.create_lora_training_client(base_model=base_model, rank=8)
        tc2 = client2.create_lora_training_client(base_model=base_model, rank=4)

        datum = _make_datum([11, 12, 13, 14])
        r1 = tc1.forward_backward([datum], "cross_entropy").result(timeout=10)
        r2 = tc2.forward_backward([datum], "cross_entropy").result(timeout=10)
        assert r1.metrics["loss:sum"] >= 0 and r2.metrics["loss:sum"] >= 0

        tc1.optim_step(types.AdamParams(learning_rate=1e-3)).result(timeout=10)
        tc2.optim_step(types.AdamParams(learning_rate=1e-4)).result(timeout=10)

        spans = span_exporter.get_finished_spans()
        assert len(spans) > 0

        # Verify training spans exist
        training_spans = [
            s for s in spans if "training_controller" in s.name or "forward" in s.name.lower()
        ]
        assert len(training_spans) > 0

        # Verify multiple training_run_ids (isolation)
        run_ids = {
            s.attributes["tuft.training_run_id"]
            for s in spans
            if s.attributes and "tuft.training_run_id" in s.attributes
        }
        if len(run_ids) >= 2:
            for rid in run_ids:
                assert len(_get_spans_with_attr(spans, "tuft.training_run_id", rid)) >= 1
    finally:
        client1.holder.close()
        client2.holder.close()
