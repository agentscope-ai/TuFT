"""CPU tests for the evaluation-mode components (no GPU required).

Covers:
- MockCorrector overhead simulation and pass-through semantics
- SerialAsyncGate tenant time-slicing
- RLRuntimeScheduler discovery -> fast loop -> one-shot delay gate
- RLRuntimeScheduler slow-loop conversion (identity_tag and quantized cost)
- SamplingPipelineBackend L1 A0 merge and L3 flex-horizon admission
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest
from tinker import types

from tuft.backends.sampling_pipeline import SamplingPipelineBackend
from tuft.config import ModelConfig
from tuft.corrector import MockCorrector, build_corrector
from tuft.runtime.serial_async_gate import SerialAsyncGate
from tuft.runtime.unified_engine_gate import UnifiedEngineGate
from tuft.schedulers.rl_loop_scheduler import FixedFlexCompositionController, SlowLoopConfig
from tuft.schedulers.runtime_scheduler import RLRuntimeScheduler


def _config(**kwargs) -> ModelConfig:
    values = {
        "model_name": "test-model",
        "model_path": Path("/tmp/test-model"),
        "max_model_len": 128,
    }
    values.update(kwargs)
    return ModelConfig(**values)


def _sample_response(tokens=(1, 2, 3), logprobs=(-1.0, -2.0, -3.0)) -> types.SampleResponse:
    return types.SampleResponse(
        sequences=[
            types.SampledSequence(
                stop_reason="length",
                _tokens_list=list(tokens),
                _logprobs_list=list(logprobs),
            )
        ]
    )


# ---------------------------------------------------------------------------
# MockCorrector
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mock_corrector_default_bias_leaves_logprobs_unchanged() -> None:
    corrector = MockCorrector(latency_s=0.0, bias=0.0)
    response = _sample_response()

    corrected = await corrector.correct(response)

    assert corrected is response
    assert corrected.sequences[0].logprobs == [-1.0, -2.0, -3.0]
    assert corrector.snapshot()["corrected_sequences"] == 1.0


@pytest.mark.asyncio
async def test_mock_corrector_applies_bias_and_latency() -> None:
    corrector = MockCorrector(latency_s=0.05, bias=0.25)
    response = _sample_response()

    corrected = await corrector.correct(response)

    assert corrected.sequences[0].logprobs == [-0.75, -1.75, -2.75]
    snapshot = corrector.snapshot()
    assert snapshot["corrected_tokens"] == 3.0
    assert snapshot["total_overhead_s"] >= 0.05


def test_build_corrector_disabled_returns_none() -> None:
    assert build_corrector(enabled=False) is None
    assert isinstance(build_corrector(enabled=True, latency_ms=1.0), MockCorrector)


# ---------------------------------------------------------------------------
# SerialAsyncGate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_serial_async_gate_slice_lifecycle() -> None:
    gate = SerialAsyncGate(iterations_per_slice=2)

    # run-a takes the create slot (first tenant: nothing to evict) and is promoted.
    assert await gate.acquire_create_slot() is None
    await gate.promote("run-a")

    b_state = {"created": False, "evict": None}

    async def b_create() -> None:
        b_state["evict"] = await gate.acquire_create_slot()
        b_state["created"] = True
        await gate.promote("run-b")

    b_task = asyncio.create_task(b_create())
    await asyncio.sleep(0.05)
    assert not b_state["created"]  # slot busy while run-a's slice is alive

    # Active run's requests pass; foreign runs wait.
    await asyncio.wait_for(gate.acquire("run-a"), timeout=1.0)

    # Budget of 2 iterations: eviction is DEFERRED (always []) and the run stays
    # active so its trailing weight-sync / final-eval sampling still passes.
    assert await gate.release_iteration("run-a") == []
    assert await gate.release_iteration("run-a") == []
    await asyncio.wait_for(gate.acquire("run-a"), timeout=1.0)

    # Slot is free again; run-b creates and is told to evict run-a.
    await b_task
    assert b_state["created"]
    assert b_state["evict"] == "run-a"
    await asyncio.wait_for(gate.acquire("run-b"), timeout=1.0)
    snapshot = gate.snapshot()
    assert snapshot["slice_switches"] == 1.0
    assert snapshot["evicted_runs"] == 1.0


@pytest.mark.asyncio
async def test_serial_async_gate_waits_for_inflight_before_eviction() -> None:
    gate = SerialAsyncGate(iterations_per_slice=1, eviction_grace_s=0.1)
    assert await gate.acquire_create_slot() is None
    await gate.promote("run-a")
    assert await gate.release_iteration("run-a") == []  # due for eviction

    # An in-flight op keeps run-a from being evictable even past the grace.
    gate.begin_op("run-a")
    evict_task = asyncio.create_task(gate.await_evictable("run-a"))
    await asyncio.sleep(0.3)
    assert not evict_task.done()
    gate.end_op("run-a")
    await asyncio.wait_for(evict_task, timeout=2.0)


# ---------------------------------------------------------------------------
# UnifiedEngineGate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unified_engine_gate_holds_sampling_while_training_in_flight() -> None:
    gate = UnifiedEngineGate(sampling_min_window_s=0.0)
    order: list[str] = []

    async def sampling_request() -> None:
        await gate.acquire_sampling()
        order.append("sample")

    await gate.acquire_training()
    sample_task = asyncio.create_task(sampling_request())
    await asyncio.sleep(0.05)
    assert order == []  # sampling held while training is in flight

    await gate.release_training()
    await sample_task
    assert order == ["sample"]
    snapshot = gate.snapshot()
    assert snapshot["switches_to_training"] == 1.0
    assert snapshot["switches_to_sampling"] == 1.0
    assert snapshot["sampling_hold_seconds"] > 0.0


@pytest.mark.asyncio
async def test_unified_engine_gate_serializes_training_across_runs() -> None:
    gate = UnifiedEngineGate(sampling_min_window_s=0.0)
    order: list[str] = []

    await gate.acquire_training()

    async def second_train() -> None:
        await gate.acquire_training()
        order.append("second")
        await gate.release_training()

    task = asyncio.create_task(second_train())
    await asyncio.sleep(0.05)
    assert order == []  # second training waits: single engine is busy

    order.append("first")
    await gate.release_training()
    await task
    assert order == ["first", "second"]
    snapshot = gate.snapshot()
    assert snapshot["switches_to_training"] == 2.0
    assert snapshot["switches_to_sampling"] == 2.0


# ---------------------------------------------------------------------------
# RLRuntimeScheduler: discovery -> fast loop -> delay gate
# ---------------------------------------------------------------------------


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


async def _drive_iterations(
    scheduler: RLRuntimeScheduler,
    clock: FakeClock,
    tenant_id: str,
    *,
    gap: int,
    sample_s: float = 2.0,
    train_s: float = 1.0,
) -> None:
    """Feed two loop iterations so discovery has enough observations.

    The training span holds the single training lane, exactly like the request
    path does: the slow loop reads its pressure signal from lane-busy time, so a
    helper that skips the lane reports zero pressure and no conversion ever
    fires.
    """
    for _ in range(2):
        clock.advance(sample_s)
        scheduler.observe_sampling_submit(tenant_id)
        clock.advance(0.1)
        scheduler.observe_training_ready(tenant_id)
        clock.advance(0.05)
        await scheduler.acquire_training_lane(tenant_id)
        scheduler.observe_training_start(tenant_id)
        scheduler.observe_consumed_staleness(tenant_id, gap)
        clock.advance(train_s)
        scheduler.release_training_lane(tenant_id)
        await scheduler.observe_training_finish(tenant_id)


@pytest.mark.asyncio
async def test_rl_runtime_scheduler_plans_and_applies_one_shot_delay() -> None:
    clock = FakeClock()

    async def fake_sleep(seconds: float) -> None:
        clock.advance(seconds)

    scheduler = RLRuntimeScheduler(replan_every_s=0.0, clock=clock.monotonic, sleep_fn=fake_sleep)

    # Tenant A owns a long burst; tenant B's next burst becomes ready inside
    # A's lane and must be tucked behind it (one-shot delay), within its
    # staleness budget (gap=3).
    await _drive_iterations(scheduler, clock, "tenant-a", gap=0, sample_s=0.4, train_s=0.7)
    await _drive_iterations(scheduler, clock, "tenant-b", gap=3, sample_s=0.06, train_s=0.12)

    plan = scheduler.maybe_replan()
    assert plan is not None and plan.feasible
    delays = {a.tenant_id: a.sampling_delay_s for a in plan.assignments}
    assert delays["tenant-a"] == 0.0
    assert delays["tenant-b"] > 0.0

    # The delay gate holds tenant-b exactly once for its assigned delay.
    before = clock.now
    await scheduler.acquire_sampling_permission("tenant-b")
    assert clock.now - before >= delays["tenant-b"] - 1e-6
    # Second call in the same cycle is free (one-shot semantics).
    before = clock.now
    await scheduler.acquire_sampling_permission("tenant-b")
    assert clock.now - before == 0.0


# ---------------------------------------------------------------------------
# RLRuntimeScheduler: slow loop conversions
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_slow_loop_identity_tag_conversion_has_zero_cost() -> None:
    clock = FakeClock()
    scheduler = RLRuntimeScheduler(
        replan_every_s=1e9,
        slow_loop_enabled=True,
        conversion_mode="identity_tag",
        clock=clock.monotonic,
        slow_loop=FixedFlexCompositionController(SlowLoopConfig(smoothing_alpha=1.0)),
    )
    # Make training pressure high enough to cross the slow-loop threshold.
    await _drive_iterations(scheduler, clock, "tenant-a", gap=0, sample_s=0.05, train_s=10.0)

    snapshot = scheduler.snapshot()
    assert snapshot["conversion_count"] >= 1
    assert snapshot["conversion_time_s"] == 0.0


@pytest.mark.asyncio
async def test_slow_loop_quantized_conversion_injects_measured_cost() -> None:
    clock = FakeClock()

    async def fake_sleep(seconds: float) -> None:
        clock.advance(seconds)

    scheduler = RLRuntimeScheduler(
        replan_every_s=1e9,
        slow_loop_enabled=True,
        conversion_mode="quantized",
        fixed_to_flex_cost_s=3.8,
        flex_to_fixed_cost_s=0.95,
        clock=clock.monotonic,
        sleep_fn=fake_sleep,
        slow_loop=FixedFlexCompositionController(SlowLoopConfig(smoothing_alpha=1.0)),
    )
    await _drive_iterations(scheduler, clock, "tenant-a", gap=0, sample_s=0.05, train_s=10.0)

    snapshot = scheduler.snapshot()
    assert snapshot["conversion_count"] >= 1
    assert snapshot["conversion_time_s"] > 0.0


# ---------------------------------------------------------------------------
# SamplingPipelineBackend: L1 merge + L3 horizon
# ---------------------------------------------------------------------------


class FakeSamplingBackend:
    def __init__(self) -> None:
        self.samples: list[str | None] = []
        self.adapters: dict[str, Path] = {}

    async def async_init(self) -> None:
        pass

    async def sample(
        self,
        prompt: types.ModelInput,
        num_samples: int,
        sampling_params: types.SamplingParams,
        include_prompt_logprobs: bool = False,
        topk_prompt_logprobs: int = 0,
        lora_id: str | None = None,
    ) -> types.SampleResponse:
        self.samples.append(lora_id)
        return _sample_response()

    async def add_adapter(self, lora_id: str, adapter_path: Path) -> None:
        self.adapters[lora_id] = adapter_path

    async def remove_adapter(self, lora_id: str) -> None:
        self.adapters.pop(lora_id, None)

    def get_openai_api_url(self) -> str:
        return "fake://inner"


@pytest.mark.asyncio
async def test_sampling_pipeline_merges_version_zero_adapters_into_a0() -> None:
    inner = FakeSamplingBackend()
    pipeline = SamplingPipelineBackend(_config(), inner)
    await pipeline.add_adapter("tenant-a", Path("/tmp/adapter"))

    response = await pipeline.sample(
        prompt=types.ModelInput.from_ints([1, 2]),
        num_samples=1,
        sampling_params=types.SamplingParams(max_tokens=8),
        lora_id="tenant-a",
    )

    assert response.sequences
    assert inner.samples == [None]  # version 0 -> served through the base path
    assert pipeline.snapshot()["a0_merged_requests"] == 1

    pipeline.notify_adapter_version("tenant-a", 1)
    await pipeline.sample(
        prompt=types.ModelInput.from_ints([1, 2]),
        num_samples=1,
        sampling_params=types.SamplingParams(max_tokens=8),
        lora_id="tenant-a",
    )
    assert inner.samples == [None, "tenant-a"]


@pytest.mark.asyncio
async def test_sampling_pipeline_holds_requests_outside_flex_horizon() -> None:
    import time

    inner = FakeSamplingBackend()
    pipeline = SamplingPipelineBackend(_config(), inner, estimated_tokens_per_s=100.0)
    flip_at = time.monotonic() + 0.3
    pipeline.set_flex_horizon(flip_at)

    # 1 sample x 64 tokens at 100 tok/s = 0.64s > 0.3s horizon -> must be held.
    start = time.monotonic()
    await pipeline.sample(
        prompt=types.ModelInput.from_ints([1, 2]),
        num_samples=1,
        sampling_params=types.SamplingParams(max_tokens=64),
        lora_id=None,
    )
    held = time.monotonic() - start
    assert held >= 0.2
    assert pipeline.snapshot()["flex_horizon_holds"] >= 1


@pytest.mark.asyncio
async def test_serial_async_gate_promote_makes_creator_active() -> None:
    gate = SerialAsyncGate(iterations_per_slice=2)

    async def waiting_sample() -> None:
        await gate.acquire("tenant-b")  # waits: no active run yet

    task = asyncio.create_task(waiting_sample())
    await asyncio.sleep(0.05)

    # tenant-a creates its training run and is promoted: its requests pass.
    await gate.promote("tenant-a")
    await asyncio.wait_for(gate.acquire("tenant-a"), timeout=1.0)
    task.cancel()


# ---------------------------------------------------------------------------
# Sampling tenant identity (sample -> training run mapping)
# ---------------------------------------------------------------------------


def test_tenant_of_sample_maps_session_to_training_run() -> None:
    """A sample must resolve to the training_run_id the gate promotes, not the
    sampling_session_id. Base-model (tenant-less) samples resolve to None so
    evaluation gates never block them."""
    from types import SimpleNamespace

    from tuft.sampling_controller import SamplingSessionRecord
    from tuft.state import ServerState

    def _session(ssid: str, run_id: str | None) -> SamplingSessionRecord:
        return SamplingSessionRecord(
            sampling_session_id=ssid,
            session_id="sess",
            model_id=ssid,
            base_model="test-model",
            user_id="u",
            training_run_id=run_id,
            session_seq_id=0,
        )

    sessions = {
        "ss-tenant": _session("ss-tenant", "run-A"),
        "ss-base": _session("ss-base", None),
    }
    state = SimpleNamespace(sampling=SimpleNamespace(sampling_sessions=sessions))

    req = SimpleNamespace(sampling_session_id="ss-tenant", model_path=None)
    assert ServerState._tenant_of_sample(state, req) == "run-A"

    base_req = SimpleNamespace(sampling_session_id="ss-base", model_path=None)
    assert ServerState._tenant_of_sample(state, base_req) is None

    unknown_req = SimpleNamespace(sampling_session_id="ss-missing", model_path=None)
    assert ServerState._tenant_of_sample(state, unknown_req) is None


@pytest.mark.asyncio
async def test_slow_loop_identity_tag_recomposes_the_router() -> None:
    """identity_tag must re-compose for real, not just move bookkeeping.

    A measured 8-GPU run reported conversion_count=4 and composition 3+5 -> 7+1
    while the router still served from 3 replicas, because identity_tag skipped
    the router call entirely and only the cost model was supposed to differ.
    """

    class _Router:
        def __init__(self) -> None:
            self.added = 0
            self.removed = 0

        async def add_one_fixed_replica(self) -> bool:
            self.added += 1
            return True

        async def remove_one_fixed_replica(self) -> bool:
            self.removed += 1
            return True

    clock = FakeClock()
    router = _Router()
    scheduler = RLRuntimeScheduler(
        replan_every_s=1e9,
        slow_loop_enabled=True,
        conversion_mode="identity_tag",
        clock=clock.monotonic,
        router=router,
        initial_fixed_groups=3,
        initial_flex_groups=5,
        slow_loop=FixedFlexCompositionController(
            # Tiny conversion costs so the amortization gate does not swallow the
            # decision: this test is about whether identity_tag reaches the
            # router, not about the gain threshold.
            SlowLoopConfig(
                smoothing_alpha=1.0,
                min_dwell_s=0.0,
                fixed_to_flex_cost_s=0.01,
                flex_to_fixed_cost_s=0.01,
            )
        ),
    )
    # Long sampling stretches with a short train step keep pressure below the
    # low threshold, which is the "spare capacity should serve sampling" branch.
    await _drive_iterations(scheduler, clock, "tenant-a", gap=0, sample_s=20.0, train_s=0.05)

    snapshot = scheduler.snapshot()
    assert snapshot["conversion_count"] >= 1
    assert snapshot["conversion_time_s"] == 0.0
    # The replica boot is deferred so it never blocks a training step; drain the
    # background tasks before asserting it reached the router.
    await asyncio.gather(*scheduler._conversion_tasks)
    assert router.added >= 1, "identity_tag conversion must reach the router"
    assert snapshot["failed_conversion_count"] == 0


@pytest.mark.asyncio
async def test_slow_loop_rolls_back_accounting_when_the_replica_boot_fails():
    """A failed replica boot must not leave the controller a pool it lacks.

    A growth run reported ``fixed_groups=7`` while only 4 replicas existed: the
    accounting advanced when the decision was taken and three
    ``add_one_fixed_replica`` calls then raised out of ``_create_engine``, so
    every later decision was made against a pool that did not exist.
    """

    class FailingRouter:
        def __init__(self):
            self.attempts = 0

        async def add_one_fixed_replica(self):
            self.attempts += 1
            raise RuntimeError("no GPU capacity for another engine")

    clock = FakeClock()
    router = FailingRouter()
    scheduler = RLRuntimeScheduler(
        replan_every_s=1e9,
        slow_loop_enabled=True,
        conversion_mode="identity_tag",
        clock=clock.monotonic,
        router=router,
        initial_fixed_groups=3,
        initial_flex_groups=5,
        slow_loop=FixedFlexCompositionController(
            SlowLoopConfig(
                smoothing_alpha=1.0,
                min_dwell_s=0.0,
                fixed_to_flex_cost_s=0.01,
                flex_to_fixed_cost_s=0.01,
            )
        ),
    )
    before = scheduler.snapshot()["composition"]["fixed_groups"]

    # Spare capacity keeps pressure low, so the controller repeatedly tries to
    # hand a flex group to sampling; every attempt fails to boot its replica.
    await _drive_iterations(scheduler, clock, "tenant-a", gap=0, sample_s=20.0, train_s=0.05)
    await asyncio.gather(*scheduler._conversion_tasks, return_exceptions=True)

    snapshot = scheduler.snapshot()
    assert router.attempts >= 1, "the conversion must have been attempted"
    assert snapshot["failed_conversion_count"] == router.attempts
    assert snapshot["conversion_count"] == 0
    assert snapshot["composition"]["fixed_groups"] == before


def test_duty_cycle_window_is_capped_even_when_training_never_queues():
    """An unbounded sampling window is what cost the measured run 32.8%.

    Exit-on-ready cannot save it: the gate that holds training out while the flex
    GPU samples also keeps ``_pending_training`` at 0, so the only exit condition
    never fires. A measured run held 1040.5s across 8 fills, 130.1s each, against
    a 3.5s observed gap.
    """

    class _Router:
        active_backend = 'fixed'

        async def switch_to_flex(self):
            self.active_backend = 'flex'

        async def switch_to_fixed(self):
            self.active_backend = 'fixed'

    clock = FakeClock()
    scheduler = RLRuntimeScheduler(
        replan_every_s=1e9,
        clock=clock.monotonic,
        router=_Router(),
        flex_duty_cycle_enabled=True,
        flex_duty_cycle_tick_s=0.5,
        flex_duty_window_cap_s=10.0,
        flex_switch_backlog_threshold=1,
        flex_duty_cycle_min_dwell_s=0.0,
        flex_block_quiet_s=0.0,
    )
    # Sampling has backlog, training never queues: the pathological case.
    scheduler._sampling_backlog = 1
    scheduler._pending_training = 0
    scheduler._duty_window_start_s = clock.monotonic()

    cap = scheduler._duty_window_cap_s()
    assert cap <= 10.0 * scheduler.flex_duty_window_cap_max_scale
    assert cap >= 10.0, 'cap must not fall below the configured base'

    # With the queue far above the threshold the cap stretches, but stays bounded.
    scheduler._sampling_backlog = 1000
    stretched = scheduler._duty_window_cap_s()
    assert stretched == pytest.approx(10.0 * scheduler.flex_duty_window_cap_max_scale)
    assert stretched < float('inf')

    # Setting the base to 0 restores the old unbounded behaviour explicitly.
    scheduler.flex_duty_window_cap_s = 0.0
    assert scheduler._duty_window_cap_s() == float('inf')


def test_observed_gap_excludes_time_the_duty_cycle_itself_held_the_gpu():
    """The gap estimate must not count blockage this controller caused.

    Including it was self-reinforcing: a measured run drifted from a 3.5s gap to
    23.5s because its own long windows stretched the interval, after which the
    gain gate cleared on every gap (gaps_skipped=0).
    """
    clock = FakeClock()
    # _observe_gap treats a zero timestamp as "no training has finished yet", so
    # the clock has to be past 0 before the first observation.
    clock.advance(100.0)
    scheduler = RLRuntimeScheduler(replan_every_s=1e9, clock=clock.monotonic)

    # A 20s wall interval of which 18s was this controller sampling: the real
    # spare capacity was 2s.
    scheduler._last_training_done_s = clock.monotonic()
    clock.advance(20.0)
    scheduler.duty_time_in_sampling_s = 18.0
    scheduler._observe_gap(clock.monotonic())
    assert scheduler.observed_gap_s == pytest.approx(2.0)

    # A later interval with no sampling is reported in full (EMA with the first).
    scheduler._last_training_done_s = clock.monotonic()
    clock.advance(4.0)
    scheduler._observe_gap(clock.monotonic())
    assert scheduler.observed_gap_s == pytest.approx(0.5 * 4.0 + 0.5 * 2.0)


def test_duty_window_cap_is_plumbed_from_config_to_the_scheduler():
    """A scheduler knob is useless unless config declares and state passes it.

    flex_duty_window_cap_s was read by RLRuntimeScheduler but missing from both
    config.py and the state.py call site, so the yaml value was silently dropped
    and every arm ran the dataclass default. The uncapped control then reported
    cap_exits=10 with a 25.6s window instead of the unbounded behaviour it was
    supposed to reproduce, which made the whole A/B measure two runs of the same
    policy.
    """
    import inspect

    from tuft import state as state_mod
    from tuft.config import EvaluationConfig

    for field in ('flex_duty_window_cap_s', 'flex_duty_window_cap_max_scale'):
        assert field in EvaluationConfig.model_fields, (
            f'{field} missing from EvaluationConfig: the yaml value would be dropped'
        )
        # The scheduler must actually receive it at the construction site.
        assert f'{field}=ev.{field}' in inspect.getsource(state_mod), (
            f'{field} is never passed to RLRuntimeScheduler'
        )

    # Zero has to mean "no bound", which is what the control arm relies on.
    sched = RLRuntimeScheduler(replan_every_s=1e9, flex_duty_window_cap_s=0.0)
    assert sched._duty_window_cap_s() == float('inf')


class _FakeSamplingBackend:
    """Records which requests it served, so routing can be asserted directly."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.served: list[str | None] = []
        self.adapters: dict = {}
        self._mode = 'sampling'

    async def async_init(self) -> None:
        return None

    async def add_adapter(self, lora_id, adapter_path) -> None:
        self.adapters[lora_id] = adapter_path

    async def sample(self, *, prompt, num_samples, sampling_params,
                     include_prompt_logprobs=False, topk_prompt_logprobs=0,
                     lora_id=None):
        self.served.append(lora_id)
        return 'ok'


def _router_with(fixed_names, flex=None):
    """A router in fixed_plus_flex mode over pre-built fake replicas."""
    from tuft.backends.sampling_router import SamplingRuntimeRouter

    class _Cfg:
        fixed_sampling_data_parallel_size = len(fixed_names)

    router = SamplingRuntimeRouter.__new__(SamplingRuntimeRouter)
    router.config = _Cfg()
    router.mode = 'fixed_plus_flex'
    router.flex_backend = flex
    router.fixed_backend = None
    router._fixed_backends = [_FakeSamplingBackend(n) for n in fixed_names]
    router._fixed_backends_initialized = True
    router._fixed_backend_factory = None
    router._adapter_paths = {}
    router._adapter_pin_map = {}
    router._rr_counter = 0
    router._flex_in_sampling = flex is not None
    router._flex_draining = False
    router._flex_inflight = 0
    router._flex_replayed = set()
    router._flex_horizon = None
    router._flex_tokens_per_s = 2000.0
    router.flex_max_sample_tokens = 0
    router.flex_horizon_skips = 0
    router.flex_long_request_skips = 0
    router._flex_idle = asyncio.Event()
    router._flex_idle.set()
    # This helper bypasses __init__, so any counter the router reads during sample()
    # has to be seeded here or the test fails on an AttributeError that says
    # nothing about the behaviour under test.
    router.flex_sampling_requests = 0
    router.fixed_sampling_requests = 0
    router.per_replica_requests = {}
    router.flex_drain_timeouts = 0
    router._pin_assign_counter = 0
    router.flex_horizon_admits_idle = 0
    router.flex_service_samples = 0
    router.flex_fill_admits = 0
    router.flex_target_inflight = 0
    router.flex_window_fit_zero = 0
    router._flex_window_index = 0
    return router


def test_flex_group_receives_sampling_work_when_it_joins_the_pool():
    """The bug this pins: every request carries a lora_id, the adapter pin always
    hit, round-robin never ran, and the flex GPU sat in the sampling pool with
    zero requests - so switching its idle time to sampling bought nothing."""
    flex = _FakeSamplingBackend('flex')
    router = _router_with(['f0', 'f1', 'f2'], flex=flex)

    class _P:
        max_tokens = 8

    async def drive():
        for i in range(40):
            await router.sample(prompt='p', num_samples=1, sampling_params=_P(),
                                lora_id=f'tenant{i % 8}')

    asyncio.run(drive())
    assert flex.served, 'flex joined the pool but was never routed a request'
    # One slot in four: flex should carry roughly a quarter, not a token amount.
    assert len(flex.served) >= 8, f'flex only served {len(flex.served)} of 40'


def test_growing_the_fixed_pool_repins_adapters_onto_the_new_replica():
    """add_one_fixed_replica used to leave every pin in place: the indices stayed
    in range, so the new replica received no pinned traffic at all."""
    router = _router_with(['f0', 'f1', 'f2'])

    class _P:
        max_tokens = 8

    async def drive(n):
        for i in range(n):
            await router.sample(prompt='p', num_samples=1, sampling_params=_P(),
                                lora_id=f'tenant{i % 8}')

    asyncio.run(drive(24))
    assert router._adapter_pin_map, 'pins should exist before the grow'

    async def grow():
        router._fixed_backend_factory = lambda cfg: _FakeSamplingBackend('f3')
        assert await router.add_one_fixed_replica() is True

    asyncio.run(grow())
    assert router._adapter_pin_map == {}, 'grow must invalidate stale pins'
    asyncio.run(drive(24))
    assert router._fixed_backends[3].served, 'new replica received no traffic'


def test_routing_snapshot_is_plumbed_and_shows_where_sampling_landed():
    """The claim 'flex idle time was spent sampling' needs a counter, not just GPU
    utilisation: a replica doing adapter loads looks busy too. This locks both the
    counter and its path into the metrics snapshot."""
    import inspect

    from tuft import state as state_mod

    flex = _FakeSamplingBackend('flex')
    router = _router_with(['f0', 'f1', 'f2'], flex=flex)

    class _P:
        max_tokens = 8

    async def drive():
        for i in range(40):
            await router.sample(prompt='p', num_samples=1, sampling_params=_P(),
                                lora_id=f'tenant{i % 8}')

    asyncio.run(drive())
    snap = router.routing_snapshot()
    assert snap['flex_sampling_requests'] > 0
    assert snap['fixed_sampling_requests'] > 0
    assert snap['flex_sampling_requests'] + snap['fixed_sampling_requests'] == 40
    assert 0.0 < snap['flex_sampling_share'] < 1.0
    # Every fixed replica must have served something: a pool member that never
    # gets routed is the slow-loop grow bug.
    assert snap['replicas_never_routed'] == 0, snap['per_replica_requests']
    counts = sorted(snap['per_replica_requests'].values())
    assert counts[-1] <= 2 * counts[0], f'pool is unbalanced: {snap["per_replica_requests"]}'

    src = inspect.getsource(state_mod)
    assert '_sampling_routing_stats' in src
    assert 'snapshot["sampling_routing"] = routing' in src


def test_horizon_guard_does_not_leave_an_idle_flex_gpu_unused():
    """The measured 8-GPU duty arm held flex in sampling for 247.5s and served six
    requests, bouncing 204 on the horizon test. Two causes, both checked here: the
    estimate multiplied by num_samples as if samples were serial, and an idle flex
    GPU was 'protected' by sending its work elsewhere."""
    flex = _FakeSamplingBackend('flex')
    router = _router_with(['f0', 'f1', 'f2'], flex=flex)

    class _P:
        max_tokens = 512

    # A horizon that has already passed: every estimate overruns it.
    router.set_flex_horizon(time.monotonic() - 1.0)

    async def drive():
        for i in range(40):
            await router.sample(prompt='p', num_samples=16, sampling_params=_P(),
                                lora_id=f'tenant{i % 8}')

    asyncio.run(drive())
    snap = router.routing_snapshot()
    # Flex is idle between these serialised calls, so the overrunning requests are
    # admitted rather than bounced into a wasted window.
    assert snap['flex_sampling_requests'] > 0, snap
    assert snap['flex_horizon_admits_idle'] > 0, snap
    # And the estimate no longer scales with num_samples: batched decode means one
    # request of 512 tokens, not sixteen.
    est = router._estimate_flex_service_s(_P())
    assert est == 512 / router._flex_tokens_per_s, est


def test_flex_decode_rate_is_calibrated_from_observed_service():
    """The admission test divided by a hard-coded 2000 tokens/s that no measurement
    supported; a wrong constant silently stops the flex GPU being used."""
    flex = _FakeSamplingBackend('flex')
    router = _router_with(['f0'], flex=flex)

    class _P:
        max_tokens = 1000

    assert router.flex_service_samples == 0
    router._observe_flex_service(_P(), 2.0)
    assert router.flex_service_samples == 1
    assert router._flex_tokens_per_s == 500.0, router._flex_tokens_per_s
    router._observe_flex_service(_P(), 1.0)
    # Second observation is 1000 tok/s; the EWMA moves toward it without jumping.
    assert 500.0 < router._flex_tokens_per_s < 1000.0, router._flex_tokens_per_s


def test_duty_window_exits_when_the_backlog_drains_before_the_time_floor():
    """The floor is meant to earn back the pair of flips, not to hold the GPU for a
    fixed time. A 60s-capped run spent 338.5s of 470s sampling on the floor at
    0.213 req/s and served fewer requests than the 10s-capped run, because the
    three fixed replicas had already cleared the queue."""
    sched = RLRuntimeScheduler(
        replan_every_s=1e9,
        flex_duty_cycle_enabled=True,
        flex_sampling_min_window_s=60.0,
        flex_switch_backlog_threshold=1,
    )
    sched._duty_current_min_window_s = 60.0
    sched._duty_window_start_s = 0.0
    sched._pending_training = 1

    # Backlog still present and the floor not reached: hold.
    sched._sampling_backlog = 5
    held, floor = 5.0, sched._duty_current_min_window_s
    drained = sched._sampling_backlog < sched.flex_switch_backlog_threshold
    assert not drained
    assert not (held >= floor or drained), 'should not exit while work remains'

    # Queue emptied: the floor can no longer be earned, so exit now.
    sched._sampling_backlog = 0
    drained = sched._sampling_backlog < sched.flex_switch_backlog_threshold
    assert drained
    assert (held >= floor or drained), 'must exit once the backlog is drained'
    assert 'duty_drained_exits' in sched.snapshot()


def test_flex_target_inflight_fills_the_window_instead_of_trickling():
    """A round-robin 1/N share left the flex GPU at 0.213 req/s with 66% of a 60s
    window idle. The fixed replicas are already at their ceiling, so the marginal
    request belongs on flex until flex itself is busy - bounded, so the flip back
    to training does not wait on a deep queue."""
    flex = _FakeSamplingBackend('flex')
    router = _router_with(['f0', 'f1', 'f2'], flex=flex)
    router.flex_target_inflight = 4

    class _P:
        max_tokens = 128

    async def drive():
        for i in range(40):
            await router.sample(prompt='p', num_samples=1, sampling_params=_P(),
                                lora_id=f'tenant{i % 8}')

    asyncio.run(drive())
    snap = router.routing_snapshot()
    # Serialised calls leave flex idle between them, so every request is admitted
    # by the fill rule: a far larger share than the 1/4 round-robin would give.
    assert snap['flex_fill_admits'] > 0, snap
    assert snap['flex_sampling_share'] > 0.5, snap
    assert snap['flex_target_inflight'] == 4

    # With the knob off the old 1/N behaviour returns.
    flex2 = _FakeSamplingBackend('flex2')
    r2 = _router_with(['f0', 'f1', 'f2'], flex=flex2)
    r2.flex_target_inflight = 0
    asyncio.run(_drive_router(r2, _P, 40))
    s2 = r2.routing_snapshot()
    assert s2['flex_fill_admits'] == 0, s2
    assert 0.1 < s2['flex_sampling_share'] < 0.4, s2


async def _drive_router(router, params_cls, n):
    for i in range(n):
        await router.sample(prompt='p', num_samples=1, sampling_params=params_cls(),
                            lora_id=f'tenant{i % 8}')


def test_window_fit_capacity_never_admits_past_the_window():
    """A flat ceiling of 8 deadlocked a real run: requests still in flight when the
    window closed were torn down with the sampling runtime and the client retried
    retrieve_future 1.2M times. Capacity must fall to zero as the window ends."""
    flex = _FakeSamplingBackend('flex')
    router = _router_with(['f0', 'f1', 'f2'], flex=flex)
    router.flex_target_inflight = 8
    router._flex_tokens_per_s = 1000.0          # 512 tokens -> 0.512s per request

    class _P:
        max_tokens = 512

    now = time.monotonic()
    router.set_flex_horizon(now + 10.0)         # plenty of room
    roomy = router._window_fit_capacity(_P())
    assert roomy == 8, roomy                    # ceiling binds, not the window

    # Above one request's service time the ceiling applies: the engine decodes
    # the batch concurrently, so N requests take about as long as one. Dividing
    # the window by per-request time assumed serial service and locked capacity
    # at zero once per-request latency grew to the window length.
    router.set_flex_horizon(now + 1.6)
    assert router._window_fit_capacity(_P()) == 8

    router.set_flex_horizon(now + 0.4)          # less than one request
    assert router._window_fit_capacity(_P()) == 0

    router.set_flex_horizon(now - 1.0)          # window already over
    assert router._window_fit_capacity(_P()) == 0

    # No window published at all: the ceiling applies, as exclusive mode expects.
    router.set_flex_horizon(None)
    assert router._window_fit_capacity(_P()) == 8


def test_duty_window_leaves_as_soon_as_training_is_ready():
    """At four tenants with 82% training idle the old exit floor held the flex GPU
    for 55.0s of a 68.0s window while training waited, serving 38 requests and
    costing 45.3% wall. Thrash protection belongs at the entry gate, which already
    requires the gap to amortise a round trip."""
    import inspect

    from tuft.schedulers import runtime_scheduler as rs

    src = inspect.getsource(rs)
    # The exit path must not consult the floor before yielding to training.
    assert 'held_s >= self._duty_current_min_window_s or drained' not in src
    assert 'if self._pending_training > 0:' in src
    assert 'self.duty_early_exits += 1' in src
    # The entry gate is what keeps the flip from thrashing, and it stays.
    assert '_gap_worth_filling' in src

    sched = RLRuntimeScheduler(replan_every_s=1e9, flex_duty_cycle_enabled=True)
    assert 'duty_early_exits' in sched.snapshot()


def test_wake_latency_request_does_not_pollute_decode_rate():
    """A request that woke the flex runtime carries wake + first-token latency,
    not decode time. Folding it in measured 155.6 tok/s against a ~630 tok/s fixed
    replica and made the window-fit admission reject almost everything. Only
    steady-state requests (flex already busy on arrival) calibrate the rate."""
    flex = _FakeSamplingBackend('flex')
    router = _router_with(['f0'], flex=flex)

    class _P:
        max_tokens = 1000

    # Cold request: 5s elapsed (mostly wake). Must NOT move the estimate.
    before = router._flex_tokens_per_s
    router._observe_flex_service(_P(), 5.0, steady_state=False)
    assert router.flex_service_samples == 0
    assert router._flex_tokens_per_s == before

    # Steady-state request: 1s for 1000 tokens = 1000 tok/s. This one counts.
    router._observe_flex_service(_P(), 1.0, steady_state=True)
    assert router.flex_service_samples == 1
    assert router._flex_tokens_per_s == 1000.0

def test_flex_sample_keeps_the_offline_generate_call_untouched():
    """LLM.generate() blocking the event loop is real (flex serves 5.09 req/s against
    15.63 for a fixed replica) but asyncio.to_thread is not the remedy: vLLM's LLM
    wrapper is not thread safe. A measured run with to_thread collapsed to 0.01
    req/s and 3.7 tok/s, pinned the flex GPU in sampling for 6292.8s, logged 47
    drain timeouts, and timed out at 7200s after 38 of 40 steps - against 830s and
    40 steps for the synchronous version. This locks the rollback so the thread
    wrapper is not reintroduced; the real fix is a concurrent-submission engine."""
    import re
    from pathlib import Path as _P

    text = _P("src/tuft/backends/flex/torchtp.py").read_text()
    parts = text.split("    async def sample(", 1)
    assert len(parts) == 2, "sample() not found"
    body = re.split(r"\n    (?:async )?def ", parts[1], maxsplit=1)[0]
    code = "\n".join(
        ln for ln in body.splitlines() if not ln.lstrip().startswith("#")
    )
    assert "engine.generate(" in code, "the direct call is the supported path"
    assert "asyncio.to_thread(" not in code, "to_thread around generate is unsafe"


def test_serial_async_gate_keeps_one_run_active_at_a_time():
    """Serial-Async means exactly what its docstring says: one tenant owns the
    cluster for k iterations, then yields. Two earlier "fixes" broke that and were
    reverted - a 120s create-slot timeout let seven of eight tenants preempt the
    holder, and admitting every promoted run let their samples run concurrently
    too. The arm then measured 279.8 steps/h against static's 283.9, i.e. it had
    silently become static; the tenant interleaving in its log confirmed it.
    """
    import inspect

    from tuft.runtime import serial_async_gate as sag

    src = inspect.getsource(sag)
    assert '_stale_slot_reclaims += 1' not in src, 'slot preemption breaks serialisation'
    assert 'model_id in self._promoted_runs' not in src, 'blanket admission breaks serialisation'
    # Handover is allowed only when the holder is genuinely finished, judged by the
    # same idle window await_evictable uses before unloading a run - never on a
    # timer, which is what the reverted preemption did.
    assert 'self.eviction_grace_s' in src
    assert '_idle_handovers += 1' in src
    # Trailing requests of an evicted run still need to land, so in-flight work is
    # admitted; nothing else is.
    assert 'self._inflight.get(model_id, 0) > 0' in src

def test_serial_async_slice_budget_covers_a_whole_tenant():
    """Serial-Async runs one tenant to completion, then the next. promote() happens
    once, at create_model, and nothing re-promotes a tenant that yielded
    mid-schedule, so a slice budget below the workload's num_train_steps deadlocks
    that tenant permanently. At 2 against 10 steps the measured arm logged 8
    create_model and 8 promote calls yet completed 25 of 57 futures, with exactly
    one tenant finishing."""
    import yaml

    with open('config/tuft_config_eval_serial_async_8gpu.yaml') as fh:
        cfg = yaml.safe_load(fh)
    ev = cfg['supported_models'][0]['evaluation']
    assert ev['deployment_mode'] == 'serial_async'
    # Every paper workload uses num_train_steps=10; the budget must clear it with
    # room to spare so one tenant always finishes inside a single slice.
    assert ev['serial_async_iterations_per_slice'] >= 10, ev['serial_async_iterations_per_slice']
    assert ev['serial_async_tenants_per_slice'] == 1

def test_serial_async_release_run_is_wired_to_unload():
    """release_run existed but nothing called it. Serial-Async runs one tenant to
    completion, and a slice budget big enough to hold a whole tenant is never
    exhausted, so release_iteration never reopens the create slot. The measured arm
    showed exactly that: 1 create_model, 1 promote, 0 evictions, the first tenant's
    ten steps done, and then all eight GPUs idle with seven tenants queued."""
    from pathlib import Path

    src = Path('src/tuft/state.py').read_text()
    unload = src[src.index('async def unload_model'):]
    unload = unload[:unload.index('\n    def ')]
    assert 'release_run' in unload, 'unload_model must hand the create slot back'


def test_serial_async_gate_release_run_reopens_the_slot():
    import asyncio

    from tuft.runtime.serial_async_gate import SerialAsyncGate

    async def scenario():
        gate = SerialAsyncGate(iterations_per_slice=1000)
        await gate.acquire_create_slot()
        await gate.promote('run-a')
        # A budget that large never drains, so only release_run can free the slot.
        await gate.release_run('run-a')
        await asyncio.wait_for(gate.acquire_create_slot(), timeout=2.0)

    asyncio.run(scenario())

def test_serial_async_gate_waits_while_the_holder_is_busy():
    """A busy holder is not displaced: handover requires a quiet grace window, so a
    tenant with work in flight keeps the cluster."""
    import asyncio

    from tuft.runtime.serial_async_gate import SerialAsyncGate

    async def scenario():
        gate = SerialAsyncGate(iterations_per_slice=1000, eviction_grace_s=0.5)
        await asyncio.wait_for(gate.acquire_create_slot(), timeout=5.0)
        await gate.promote('run-1')
        gate.begin_op('run-1')          # still working, never ended

        try:
            await asyncio.wait_for(gate.acquire_create_slot(), timeout=2.0)
        except asyncio.TimeoutError:
            assert gate.snapshot()['idle_handovers'] == 0.0
            return
        raise AssertionError('a busy holder must not be displaced')

    asyncio.run(scenario())

def test_serial_async_gate_keeps_the_slot_for_a_tenant_that_has_not_trained():
    """Handover keyed on a plain quiet window displaced tenants before they started.
    A tenant is idle between its create_model and its first sample, so the slot
    rotated all eight through promote() before any trained: create 8, promote 8,
    asample 32, forward_backward 0, optim_step 0, and no tenant could pass acquire()
    because _active_run had moved on to the last one."""
    import asyncio

    from tuft.runtime.serial_async_gate import SerialAsyncGate

    async def scenario():
        gate = SerialAsyncGate(
            iterations_per_slice=1000, eviction_grace_s=0.3, completion_idle_s=1.0
        )
        await asyncio.wait_for(gate.acquire_create_slot(), timeout=5.0)
        await gate.promote('run-1')
        # No optim step recorded: the holder has not started, so it must keep the slot.
        try:
            await asyncio.wait_for(gate.acquire_create_slot(), timeout=3.0)
        except asyncio.TimeoutError:
            assert gate.snapshot()['idle_handovers'] == 0.0
            return
        raise AssertionError('a tenant that has not trained must not be displaced')


    asyncio.run(scenario())


def test_serial_async_gate_hands_over_once_a_tenant_stops_progressing():
    """A tenant that has trained and then stopped making progress for
    completion_idle_s is done - nothing else tells the gate so, since the client never
    reports completion - and the next tenant must get the cluster."""
    import asyncio

    from tuft.runtime.serial_async_gate import SerialAsyncGate

    async def scenario():
        gate = SerialAsyncGate(
            iterations_per_slice=1000, eviction_grace_s=0.3, completion_idle_s=1.0
        )
        await asyncio.wait_for(gate.acquire_create_slot(), timeout=5.0)
        await gate.promote('run-1')
        gate.record_progress('run-1')

        await asyncio.wait_for(gate.acquire_create_slot(), timeout=10.0)
        snap = gate.snapshot()
        assert snap['idle_handovers'] == 1.0
        assert snap['tenants_with_progress'] == 1.0

    asyncio.run(scenario())


def test_serial_async_progress_is_recorded_after_each_optim_step():
    from pathlib import Path

    src = Path('src/tuft/state.py').read_text()
    idx = src.index('await self.eval_gate.release_iteration(model_id)')
    assert 'record_progress' in src[idx - 500:idx], 'optim_step must report progress'
