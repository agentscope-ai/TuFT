"""Adversarial admission tests using explicit barriers instead of GPU timing."""

import asyncio
from pathlib import Path

import pytest
from pydantic import ValidationError

from tuft.config import AppConfig, ModelConfig, SamplingConfig
from tuft.persistence.redis_store import ConfigSignature
from tuft.scheduling import ResourceScheduler, SchedulingOverloadedError


async def spin_until(predicate):
    async with asyncio.timeout(2):
        while not predicate():
            await asyncio.sleep(0)


def config(**kwargs):
    return SamplingConfig.model_validate(
        {"borrowing": {"enabled": True, "max_inflight": 4}, **kwargs}
    )


async def test_round_robin_ignores_token_load_and_releases_on_failure():
    scheduler = ResourceScheduler(2, config())
    release = asyncio.Event()
    selected = []

    async def operation(index):
        selected.append(index)
        await release.wait()
        if len(selected) == 3:
            raise ValueError("backend failure")

    tasks = []
    for tenant, work in [("large", 1000), ("small", 10), ("small", 10)]:
        tasks.append(asyncio.create_task(scheduler.sample(tenant, tenant, work, True, operation)))
        await spin_until(lambda: len(selected) == len(tasks))
    assert selected == [0, 1, 0]
    release.set()
    results = await asyncio.gather(*tasks, return_exceptions=True)
    assert all(isinstance(r, ValueError) for r in results)
    assert scheduler.snapshot()["active"] == [0, 0]
    assert scheduler.snapshot()["reserved_tokens"] == [0, 0]
    await scheduler.close()


async def test_failed_phase_transition_never_dispatches_operation():
    scheduler = ResourceScheduler(2, config(), flex_index=0)

    async def phase(training):
        raise RuntimeError("transition failed")

    async def forbidden(*args):
        pytest.fail("GPU operation must not run after a failed transition")

    scheduler.set_phase_transition(phase)
    with pytest.raises(RuntimeError, match="transition failed"):
        await scheduler.training(forbidden, reclaim_sampler_memory=True)
    with pytest.raises(RuntimeError, match="restart the backend"):
        await scheduler.training(forbidden, reclaim_sampler_memory=True)
    assert scheduler.snapshot()["flex_failed"]

    async def fixed(index):
        assert index == 1

    await scheduler.sample("b", "b", 1, True, fixed)
    assert scheduler.snapshot()["active"] == [0, 0]
    assert scheduler.snapshot()["training_pending"] == 0
    await scheduler.close()


async def test_cancelled_training_still_owns_lane_until_backend_completes():
    scheduler = ResourceScheduler(2, config(), flex_index=0)
    release = asyncio.Event()
    started = asyncio.Event()

    async def train():
        started.set()
        await release.wait()

    task = asyncio.create_task(scheduler.training(train))
    await started.wait()
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    assert scheduler.snapshot()["training_active"]
    indices = []

    async def sample(index):
        indices.append(index)

    await scheduler.sample("a", "a", 10, True, sample)
    assert indices == [1]
    release.set()
    await task
    await scheduler.close()
    assert not scheduler.snapshot()["training_active"]


async def test_queue_capacity_timeout_and_cancel_cleanup():
    scheduler = ResourceScheduler(
        1, config(max_inflight_per_replica=1, max_queued_requests=1, queue_timeout_s=0.03)
    )
    release = asyncio.Event()
    started = asyncio.Event()

    async def sample(index):
        started.set()
        await release.wait()

    active = asyncio.create_task(scheduler.sample("a", "a", 1, True, sample))
    await started.wait()
    queued = asyncio.create_task(scheduler.sample("b", "b", 1, True, sample))
    await spin_until(lambda: scheduler.snapshot()["queued"] == 1)
    with pytest.raises(SchedulingOverloadedError, match="full") as rejected:
        await scheduler.sample("c", "c", 1, True, sample)
    assert rejected.value.status_code == 429
    with pytest.raises(SchedulingOverloadedError, match="expired"):
        await queued
    assert scheduler.snapshot()["queued"] == 0
    assert scheduler.snapshot()["active"] == [1]
    queued = asyncio.create_task(scheduler.sample("b", "b", 1, True, sample))
    await spin_until(lambda: scheduler.snapshot()["queued"] == 1)
    queued.cancel()
    with pytest.raises(asyncio.CancelledError):
        await queued
    assert scheduler.snapshot()["queued"] == 0
    release.set()
    await active
    await scheduler.close()


async def test_tenants_take_turns_and_preserve_fifo():
    scheduler = ResourceScheduler(1, config(max_inflight_per_replica=1))
    release = asyncio.Event()
    events = []

    async def operation(label, index):
        events.append(label)
        if label == "active":
            await release.wait()

    tasks = []
    for label in ["active", "a1", "a2", "a3", "b1", "b2"]:
        tasks.append(
            asyncio.create_task(
                scheduler.sample(
                    label[0], label[0], 10, True, lambda index, label=label: operation(label, index)
                )
            )
        )
    await spin_until(lambda: scheduler.snapshot()["queued"] == 5)
    release.set()
    await asyncio.gather(*tasks)
    assert events == ["active", "a1", "b1", "a2", "b2", "a3"]
    await scheduler.close()


async def test_long_work_never_uses_flex_and_close_rejects_queue():
    scheduler = ResourceScheduler(2, config(max_inflight_per_replica=1), flex_index=0)
    release = asyncio.Event()
    indices = []

    async def operation(index):
        indices.append(index)
        await release.wait()

    active = asyncio.create_task(scheduler.sample("a", "a", 1000, False, operation))
    await spin_until(lambda: indices == [1])
    queued = asyncio.create_task(scheduler.sample("b", "b", 1000, False, operation))
    await spin_until(lambda: scheduler.snapshot()["queued"] == 1)
    closing = asyncio.create_task(scheduler.close())
    with pytest.raises(RuntimeError, match="closed"):
        await queued
    assert not closing.done()
    release.set()
    await asyncio.gather(active, closing)
    with pytest.raises(RuntimeError, match="closed"):
        await scheduler.sample("c", None, 1, True, operation)
    assert indices == [1]


async def test_backend_timeout_is_not_reported_as_admission_timeout():
    scheduler = ResourceScheduler(1, config())

    async def fail(index):
        raise TimeoutError("engine timeout")

    with pytest.raises(TimeoutError, match="engine timeout"):
        await scheduler.sample("a", None, 1, True, fail)
    assert scheduler.snapshot().get("queue_timeouts", 0) == 0
    await scheduler.close()


async def test_close_cancellation_does_not_cancel_admitted_work():
    scheduler = ResourceScheduler(1, config())
    release = asyncio.Event()
    started = asyncio.Event()

    async def operation(index):
        started.set()
        await release.wait()
        return 42

    sampling = asyncio.create_task(scheduler.sample("a", None, 1, True, operation))
    await started.wait()
    closing = asyncio.create_task(scheduler.close())
    await asyncio.sleep(0)
    closing.cancel()
    await asyncio.sleep(0)
    assert scheduler.snapshot()["active"] == [1]
    release.set()
    assert await sampling == 42
    await closing


@pytest.mark.parametrize(
    "override",
    [
        {"colocate": False},
        {"training_backend": "fsdp", "fsdp_num_gpus": 2},
        {"data_parallel_size": 1},
        {"tensor_parallel_size": 2},
        {"sampling_memory_fraction": 1},
        {"capabilities": ["sampling"]},
    ],
)
def test_reject_invalid_hybrid_layout(override):
    values = dict(
        model_name="qwen",
        model_path=Path("qwen"),
        max_model_len=512,
        colocate=True,
        data_parallel_size=2,
        sampling_kv_cache_memory_bytes=16 * 1024**3,
        sampling={"policy": "shared_weight"},
    )
    values.update(override)
    with pytest.raises(ValidationError):
        ModelConfig.model_validate(values)


def test_shared_weights_require_explicit_cache_budget():
    values = dict(
        model_name="qwen",
        model_path=Path("qwen"),
        max_model_len=512,
        colocate=True,
        data_parallel_size=2,
        sampling={"policy": "shared_weight"},
    )
    with pytest.raises(ValidationError, match="sampling_kv_cache_memory_bytes"):
        ModelConfig.model_validate(values)
    values["sampling_kv_cache_memory_bytes"] = 16 * 1024**3
    assert ModelConfig.model_validate(values).sampling.share_base_weights


def test_shared_weight_sharing_is_required_and_legacy_config_migrates():
    for policy in ("round_robin", "borrow", "shared_weight"):
        expected = policy in {"borrow", "shared_weight"}
        current = SamplingConfig.model_validate({"policy": policy})
        legacy = SamplingConfig.model_validate({"policy": policy, "share_base_weights": expected})
        assert current.share_base_weights is expected
        assert current.model_dump() == legacy.model_dump()
        assert "share_base_weights" not in current.model_dump()
        with pytest.raises(ValidationError, match="always shares frozen base weights"):
            SamplingConfig.model_validate({"policy": policy, "share_base_weights": not expected})


def test_legacy_persistence_signature_allows_scheduler_change_but_not_geometry(tmp_path):
    model = ModelConfig(
        model_name="qwen", model_path=Path("qwen"), max_model_len=512, max_lora_rank=8
    )
    config = AppConfig(supported_models=[model], checkpoint_dir=tmp_path)
    # Reconstruct a pre-scheduler signature using the old serialization shape.
    legacy_data = config.model_dump(
        mode="json",
        exclude={
            "persistence": True,
            "authorized_users": True,
            "supported_models": {"__all__": {"capabilities"}},
        },
    )
    legacy_data["supported_models"][0].pop("sampling")
    legacy_data["supported_models"][0].pop("sampling_kv_cache_memory_bytes")
    for field in (
        "max_cpu_loras",
        "sampling_language_model_only",
        "sampling_moe_backend",
        "sampling_gdn_prefill_backend",
        "sampling_max_num_seqs",
        "sampling_enable_metrics",
    ):
        legacy_data["supported_models"][0].pop(field)
    legacy = ConfigSignature(config_data=legacy_data)
    assert legacy.matches(ConfigSignature.from_app_config(config))
    model.sampling_kv_cache_memory_bytes = 16 * 1024**3
    model.sampling_language_model_only = True
    model.sampling_moe_backend = "triton"
    model.sampling_gdn_prefill_backend = "triton"
    model.sampling_max_num_seqs = 64
    model.sampling_enable_metrics = True
    model.max_cpu_loras = 128
    assert legacy.matches(ConfigSignature.from_app_config(config))
    model.sampling = SamplingConfig.model_validate({"policy": "borrow"})
    assert legacy.matches(ConfigSignature.from_app_config(config))
    model.max_lora_rank = 16
    assert not legacy.matches(ConfigSignature.from_app_config(config))


async def test_admission_winning_timeout_race_returns_actual_result(monkeypatch):
    original_wait_for = asyncio.wait_for
    scheduler = ResourceScheduler(1, config())
    calls = []

    async def timeout_after_admission(awaitable, timeout):
        await original_wait_for(awaitable, timeout)
        raise TimeoutError("timer fired concurrently with admission")

    async def operation(index):
        calls.append(index)
        return "finished"

    monkeypatch.setattr(asyncio, "wait_for", timeout_after_admission)
    assert await scheduler.sample("tenant", None, 1, False, operation) == "finished"
    assert calls == [0]
    assert scheduler.snapshot().get("queue_timeouts", 0) == 0
    await scheduler.close()


async def test_waiting_phase_requests_do_not_retry_failed_transition():
    scheduler = ResourceScheduler(2, config(), flex_index=0)
    entered, release = asyncio.Event(), asyncio.Event()
    calls = []

    async def phase(training):
        calls.append(training)
        entered.set()
        await release.wait()
        raise RuntimeError("phase failed")

    scheduler.set_phase_transition(phase)
    first = asyncio.create_task(scheduler._enter_phase(False))
    await entered.wait()
    second = asyncio.create_task(scheduler._enter_phase(False))
    await asyncio.sleep(0)
    release.set()
    results = await asyncio.gather(first, second, return_exceptions=True)
    assert all(isinstance(result, RuntimeError) for result in results)
    assert calls == [False] and scheduler.snapshot()["flex_failed"]
    await scheduler.close()


@pytest.mark.parametrize("policy", ["borrow", "shared_weight"])
def test_borrow_policy_normalizes_and_round_trips(policy):
    raw = {"policy": policy}
    config = SamplingConfig.model_validate(raw)
    assert raw == {"policy": policy}
    assert config.borrowing.enabled
    assert config.share_base_weights
    assert "policy" not in config.model_dump()
    assert config.model_dump()["borrowing"]["enabled"] is True
    assert SamplingConfig.model_validate_json(config.model_dump_json()) == config


def test_removed_balanced_policy_is_rejected():
    with pytest.raises(ValidationError):
        SamplingConfig.model_validate({"policy": "balanced"})
