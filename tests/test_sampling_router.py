from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
from tinker import types

from tuft.backends.base_backend import BaseSamplingBackend
from tuft.backends.sampling_backend import FixedSamplingBackend
from tuft.backends.sampling_router import SamplingRuntimeRouter
from tuft.config import ModelConfig


class FakeSamplingBackend:
    def __init__(self, name: str) -> None:
        self.name = name
        self.init_count = 0
        self.adapters: dict[str, Path] = {}
        self.removed: list[str] = []
        self.samples: list[str | None] = []

    async def async_init(self) -> None:
        self.init_count += 1

    async def sample(
        self,
        prompt: types.ModelInput,
        num_samples: int,
        sampling_params: types.SamplingParams,
        include_prompt_logprobs: bool = False,
        topk_prompt_logprobs: int = 0,
        lora_id: str | None = None,
    ):
        self.samples.append(lora_id)
        return SimpleNamespace(backend=self.name, lora_id=lora_id, num_samples=num_samples)

    async def add_adapter(self, lora_id: str, adapter_path: Path) -> None:
        self.adapters[lora_id] = adapter_path

    async def remove_adapter(self, lora_id: str) -> None:
        self.removed.append(lora_id)
        self.adapters.pop(lora_id, None)

    def get_openai_api_url(self) -> str:
        return f"fake://{self.name}"


def _config(**kwargs) -> ModelConfig:
    values = {
        "model_name": "test-model",
        "model_path": Path("/tmp/test-model"),
        "max_model_len": 128,
    }
    values.update(kwargs)
    return ModelConfig(**values)


def test_fixed_sampling_backend_uses_independent_quantized_config(monkeypatch) -> None:
    monkeypatch.setattr(FixedSamplingBackend, "_create_engine", lambda self, config: object())
    config = _config(
        colocate=True,
        sampling_backend="fixed",
        quantization=None,
        fixed_sampling_quantization="fp8",
    )

    backend = FixedSamplingBackend(config)

    assert backend.config.colocate is False
    assert backend.config.sampling_backend == "fixed"
    assert backend.config.quantization == "fp8"
    assert backend._actor_name_prefix == "fixed_sampling_model"


def test_fixed_sampling_backend_factory_registration(monkeypatch) -> None:
    monkeypatch.delenv("TUFT_CPU_TEST", raising=False)
    monkeypatch.setattr(FixedSamplingBackend, "_create_engine", lambda self, config: object())
    config = _config(sampling_backend="fixed", fixed_sampling_quantization="fp8")

    backend = BaseSamplingBackend.create_backend(config)

    assert isinstance(backend, FixedSamplingBackend)
    assert backend.config.quantization == "fp8"


@pytest.mark.asyncio
async def test_sampling_runtime_router_switches_between_flex_and_fixed() -> None:
    flex = FakeSamplingBackend("flex")
    fixed = FakeSamplingBackend("fixed")
    router = SamplingRuntimeRouter(_config(), flex_backend=flex, fixed_backend=fixed)
    adapter_path = Path("/tmp/adapter")

    await router.async_init()
    await router.add_adapter("adapter-a", adapter_path)
    flex_response = await router.sample(
        types.ModelInput.from_ints([1, 2]),
        1,
        types.SamplingParams(max_tokens=1),
        lora_id="adapter-a",
    )

    assert router.active_backend == "flex"
    assert flex.init_count == 1
    assert flex.adapters == {"adapter-a": adapter_path}
    assert flex_response.backend == "flex"

    await router.switch_to_fixed()
    fixed_response = await router.sample(
        types.ModelInput.from_ints([1, 2]),
        2,
        types.SamplingParams(max_tokens=1),
        lora_id="adapter-a",
    )

    assert router.active_backend == "fixed"
    assert fixed.adapters == {"adapter-a": adapter_path}
    assert fixed_response.backend == "fixed"
    assert fixed_response.num_samples == 2
    assert router.get_openai_api_url() == "fake://fixed"

    await router.switch_to_flex()
    assert router.active_backend == "flex"
    assert router.get_openai_api_url() == "fake://flex"


@pytest.mark.asyncio
async def test_sampling_runtime_router_lazy_initializes_fixed_backend() -> None:
    flex = FakeSamplingBackend("flex")
    fixed = FakeSamplingBackend("fixed")
    created: list[ModelConfig] = []

    def factory(config: ModelConfig) -> FakeSamplingBackend:
        created.append(config)
        return fixed

    router = SamplingRuntimeRouter(
        _config(fixed_sampling_quantization="fp8"),
        flex_backend=flex,
        fixed_backend_factory=factory,
    )
    await router.add_adapter("adapter-a", Path("/tmp/adapter"))

    assert fixed.init_count == 0
    assert fixed.adapters == {}

    await router.switch_to_fixed()

    assert len(created) == 1
    assert fixed.init_count == 1
    assert fixed.adapters == {"adapter-a": Path("/tmp/adapter")}

    await router.remove_adapter("adapter-a")
    assert flex.removed == ["adapter-a"]
    assert fixed.removed == ["adapter-a"]


class FakeFlexLikeBackend(FakeSamplingBackend):
    """Fake Flex backend exposing transform_to_training/transform_to_sampling."""

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.transform_to_training_calls = 0
        self.transform_to_sampling_calls = 0

    async def transform_to_training(self) -> None:
        self.transform_to_training_calls += 1

    async def transform_to_sampling(self) -> None:
        self.transform_to_sampling_calls += 1


class FakeSleepableFixedBackend(FakeSamplingBackend):
    """Fake Fixed backend exposing sleep(level=)/wake_up()."""

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.sleep_calls: list[int] = []
        self.wake_up_calls = 0

    async def sleep(self, level: int = 1) -> None:
        self.sleep_calls.append(level)

    async def wake_up(self) -> None:
        self.wake_up_calls += 1


@pytest.mark.asyncio
async def test_sampling_runtime_router_sleeps_inactive_backend_on_switch() -> None:
    flex = FakeFlexLikeBackend("flex")
    fixed = FakeSleepableFixedBackend("fixed")
    router = SamplingRuntimeRouter(_config(), flex_backend=flex, fixed_backend=fixed)

    # flex -> fixed: fixed was never asleep, so wake_up must NOT be called;
    # flex must be shrunk to its minimal (training-mode) footprint.
    await router.switch_to_fixed()
    assert flex.transform_to_training_calls == 1
    assert fixed.wake_up_calls == 0
    assert router.active_backend == "fixed"

    # fixed -> flex: fixed must be put to sleep(level=1) exactly once, and
    # flex must be woken back into sampling mode.
    await router.switch_to_flex()
    assert fixed.sleep_calls == [1]
    assert flex.transform_to_sampling_calls == 1
    assert router.active_backend == "flex"

    # flex -> fixed again: fixed was asleep, so wake_up must be called once.
    await router.switch_to_fixed()
    assert fixed.wake_up_calls == 1
    assert flex.transform_to_training_calls == 2


@pytest.mark.asyncio
async def test_sampling_runtime_router_never_double_sleeps_fixed_backend() -> None:
    """Guard against the known vLLM issue: calling sleep() twice without an
    intervening wake_up() can raise a CUDA invalid-argument error. The router
    must track sleep state and only call sleep() once per wake cycle.
    """
    flex = FakeFlexLikeBackend("flex")
    fixed = FakeSleepableFixedBackend("fixed")
    router = SamplingRuntimeRouter(
        _config(), flex_backend=flex, fixed_backend=fixed, initial_backend="flex"
    )

    await router.switch_to_flex()
    await router.switch_to_flex()
    await router.switch_to_flex()

    assert fixed.sleep_calls == [1]


@pytest.mark.asyncio
async def test_sampling_runtime_router_can_disable_sleep_based_switching() -> None:
    """sleep_inactive_backends=False only disables sleeping the fixed backend.

    Switching to flex must still call transform_to_sampling(), since that is what
    lets Flex serve requests at all (not merely a memory optimization). Switching
    back to fixed must still call transform_to_training() even with sleeping off:
    the optimal duty-cycle relies on it to leave flex able to run training
    forwards after a sampling window, so skipping it deadlocked the training lane.
    """
    flex = FakeFlexLikeBackend("flex")
    fixed = FakeSleepableFixedBackend("fixed")
    router = SamplingRuntimeRouter(
        _config(),
        flex_backend=flex,
        fixed_backend=fixed,
        sleep_inactive_backends=False,
    )

    await router.switch_to_fixed()
    await router.switch_to_flex()

    assert fixed.sleep_calls == []
    assert fixed.wake_up_calls == 0
    assert flex.transform_to_training_calls == 1
    assert flex.transform_to_sampling_calls == 1


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.mark.asyncio
async def test_switch_scheduler_drains_training_before_sampling() -> None:
    from tuft.schedulers.switch_scheduler import SwitchScheduler, SwitchSchedulerConfig

    clock = FakeClock()
    order: list[str] = []
    scheduler = SwitchScheduler(
        SwitchSchedulerConfig(sampling_min_window_s=0.0, idle_sleep_s=0.0),
        switch_to_sampling=lambda: order.append("switch_to_sampling"),
        switch_to_training=lambda: order.append("switch_to_training"),
        clock=clock,
    )
    scheduler.enqueue_training(lambda: order.append("train-1"))
    scheduler.enqueue_training(lambda: order.append("train-2"))
    scheduler.enqueue_sampling(lambda: order.append("sample-1"))

    stats = await scheduler.run_until_idle()

    assert order == ["train-1", "train-2", "switch_to_sampling", "sample-1"]
    assert stats.training_jobs_completed == 2
    assert stats.sampling_jobs_completed == 1
    assert stats.switches_to_sampling == 1


@pytest.mark.asyncio
async def test_switch_scheduler_min_sampling_window_blocks_early_train_switch() -> None:
    from tuft.schedulers.switch_scheduler import SwitchScheduler, SwitchSchedulerConfig

    clock = FakeClock()
    order: list[str] = []
    scheduler = SwitchScheduler(
        SwitchSchedulerConfig(sampling_min_window_s=5.0, idle_sleep_s=0.0),
        switch_to_sampling=lambda: order.append("switch_to_sampling"),
        switch_to_training=lambda: order.append("switch_to_training"),
        clock=clock,
    )

    def sample_then_enqueue_training() -> None:
        order.append("sample-1")
        scheduler.enqueue_training(lambda: order.append("train-late"))
        clock.advance(4.0)

    def sample_after_min_window() -> None:
        order.append("sample-2")
        clock.advance(1.1)

    scheduler.enqueue_training(lambda: order.append("train-initial"))
    scheduler.enqueue_sampling(sample_then_enqueue_training)
    scheduler.enqueue_sampling(sample_after_min_window)

    await scheduler.run_until_idle(max_cycles=2)

    assert order == [
        "train-initial",
        "switch_to_sampling",
        "sample-1",
        "sample-2",
        "switch_to_training",
        "train-late",
        "switch_to_sampling",
    ]


@pytest.mark.asyncio
async def test_switch_scheduler_keeps_sampling_when_no_training_pending() -> None:
    from tuft.schedulers.switch_scheduler import SwitchScheduler, SwitchSchedulerConfig

    clock = FakeClock()
    order: list[str] = []
    scheduler = SwitchScheduler(
        SwitchSchedulerConfig(sampling_min_window_s=0.0, idle_sleep_s=0.0),
        switch_to_sampling=lambda: order.append("switch_to_sampling"),
        switch_to_training=lambda: order.append("switch_to_training"),
        clock=clock,
    )
    scheduler.enqueue_sampling(lambda: order.append("sample-only"))

    await scheduler.run_until_idle()

    assert order == ["switch_to_sampling", "sample-only"]
    assert scheduler.mode.value == "sampling"


@pytest.mark.asyncio
async def test_adaptive_switch_scheduler_increases_window_when_switch_cost_exceeds_threshold() -> (
    None
):
    from tuft.schedulers.switch_scheduler import (
        SwitchPolicy,
        SwitchScheduler,
        SwitchSchedulerConfig,
    )

    clock = FakeClock()
    order: list[str] = []

    def costly_switch(name: str):
        def _inner() -> None:
            order.append(name)
            clock.advance(1.0)

        return _inner

    scheduler = SwitchScheduler(
        SwitchSchedulerConfig(
            sampling_min_window_s=2.0,
            policy=SwitchPolicy.ADAPTIVE,
            max_switch_overhead_fraction=0.10,
            adaptive_growth_factor=2.0,
            max_sampling_min_window_s=16.0,
            adaptive_window_s=100.0,
            idle_sleep_s=0.0,
        ),
        switch_to_sampling=costly_switch("switch_to_sampling"),
        switch_to_training=costly_switch("switch_to_training"),
        clock=clock,
    )
    scheduler.enqueue_training(lambda: order.append("train-1"))
    scheduler.enqueue_sampling(lambda: (order.append("sample-1"), clock.advance(2.1)))
    scheduler.enqueue_training(lambda: order.append("train-2"))
    scheduler.enqueue_sampling(lambda: (order.append("sample-2"), clock.advance(4.1)))

    stats = await scheduler.run_until_idle(max_cycles=2)

    assert stats.adaptive_adjustments >= 1
    assert scheduler.current_sampling_min_window_s > 2.0
    assert any(event.kind == "adaptive_window_increase" for event in scheduler.events)


@pytest.mark.asyncio
async def test_fixed_switch_scheduler_does_not_change_sampling_window() -> None:
    from tuft.schedulers.switch_scheduler import (
        SwitchPolicy,
        SwitchScheduler,
        SwitchSchedulerConfig,
    )

    clock = FakeClock()
    scheduler = SwitchScheduler(
        SwitchSchedulerConfig(
            sampling_min_window_s=2.0,
            policy=SwitchPolicy.FIXED,
            idle_sleep_s=0.0,
        ),
        switch_to_sampling=lambda: clock.advance(1.0),
        switch_to_training=lambda: clock.advance(1.0),
        clock=clock,
    )
    scheduler.enqueue_training(lambda: None)
    scheduler.enqueue_sampling(lambda: clock.advance(2.1))
    scheduler.enqueue_training(lambda: None)

    await scheduler.run_until_idle(max_cycles=2)

    assert scheduler.current_sampling_min_window_s == 2.0
    assert scheduler.stats.adaptive_adjustments == 0


@pytest.mark.asyncio
async def test_switch_to_fixed_survives_concurrent_adapter_registration() -> None:
    """Registering a tenant mid-switch must not abort the switch.

    Regression: the replay loop walked ``_adapter_paths`` across awaits, so a
    concurrent ``add_adapter`` raised "dictionary changed size during iteration".
    That aborted the duty-cycle flip back to training and wedged the flex GPU.
    """
    flex = FakeFlexLikeBackend("flex")
    fixed = FakeSamplingBackend("fixed")
    router = SamplingRuntimeRouter(
        _config(), flex_backend=flex, fixed_backend=fixed, sleep_inactive_backends=False
    )
    for i in range(20):
        router._adapter_paths[f"a{i}"] = Path(f"/tmp/a{i}")

    async def churn() -> None:
        for i in range(20, 40):
            await router.add_adapter(f"a{i}", Path(f"/tmp/a{i}"))

    await asyncio.gather(router.switch_to_fixed(), churn())
    assert router.active_backend == "fixed"


@pytest.mark.asyncio
async def test_switch_to_fixed_skips_adapter_replay_when_fixed_is_resident() -> None:
    """In fixed_plus_flex the replicas are already current, so a flip adds no calls.

    Replaying every adapter into every replica put len(adapters) x len(replicas)
    engine round-trips on the critical path of each duty-cycle switch.
    """
    flex = FakeFlexLikeBackend("flex")
    fixed = FakeSamplingBackend("fixed")
    router = SamplingRuntimeRouter(
        _config(),
        flex_backend=flex,
        fixed_backend=fixed,
        sleep_inactive_backends=False,
        mode="fixed_plus_flex",
    )
    router._fixed_backends = [fixed]
    router._fixed_backends_initialized = True
    for i in range(20):
        router._adapter_paths[f"a{i}"] = Path(f"/tmp/a{i}")

    await router.switch_to_fixed()
    assert fixed.adapters == {}


@pytest.mark.asyncio
async def test_flex_adapter_replay_is_incremental() -> None:
    """Replay runs on every sample() while flex serves, so it must only push new ids."""
    flex = FakeFlexLikeBackend("flex")
    fixed = FakeSamplingBackend("fixed")
    router = SamplingRuntimeRouter(
        _config(),
        flex_backend=flex,
        fixed_backend=fixed,
        sleep_inactive_backends=False,
        mode="fixed_plus_flex",
    )
    for i in range(5):
        router._adapter_paths[f"a{i}"] = Path(f"/tmp/a{i}")

    await router._replay_adapters_to_flex()
    assert sorted(flex.adapters) == [f"a{i}" for i in range(5)]

    flex.adapters.clear()
    await router._replay_adapters_to_flex()
    assert flex.adapters == {}

    router._adapter_paths["a5"] = Path("/tmp/a5")
    await router._replay_adapters_to_flex()
    assert sorted(flex.adapters) == ["a5"]
