"""Adversarial admission tests using explicit barriers instead of GPU timing."""

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from tuft.config import AppConfig, ModelConfig, SchedulingConfig
from tuft.persistence.redis_store import ConfigSignature
from tuft.scheduling import LoopProfiler, ResourceScheduler, SchedulingOverloadedError


async def spin_until(predicate):
    async with asyncio.timeout(2):
        while not predicate():
            await asyncio.sleep(0)


def config(**kwargs):
    kwargs.setdefault("sampling_alignment_s", 0)
    kwargs.setdefault("adapter_batch_window_s", 0)
    return SchedulingConfig(policy="balanced", training_quiet_s=0, **kwargs)


def test_loop_profiler_requires_consistent_recent_observations_and_bounds_memory():
    profiler = LoopProfiler(max_tenants=2)
    for t in [0, 10, 20]:
        profiler.observe("regular", t)
    assert profiler.horizon(21) is None
    profiler.observe("regular", 30)
    assert profiler.horizon(31) == 40
    assert profiler.horizon(41) is None
    for t in [0, 10, 50, 60]:
        profiler.observe("irregular", t)
    assert profiler.horizon(61) is None
    profiler.observe("new", 70)
    assert len(profiler.arrivals) == 2
    assert "regular" not in profiler.arrivals


async def test_token_load_balances_and_releases_on_failure():
    scheduler = ResourceScheduler(2, config(locality_slack=0))
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
    assert selected == [0, 1, 1]
    release.set()
    results = await asyncio.gather(*tasks, return_exceptions=True)
    assert all(isinstance(r, ValueError) for r in results)
    assert scheduler.snapshot()["active"] == [0, 0]
    assert scheduler.snapshot()["reserved_tokens"] == [0, 0]
    await scheduler.close()


async def test_flex_drains_before_training_fixed_continues():
    scheduler = ResourceScheduler(2, config(max_inflight_per_replica=1), flex_index=0)
    release_flex = asyncio.Event()
    release_train = asyncio.Event()
    events = []

    async def sample(index):
        events.append(f"sample-{index}")
        if index == 0:
            await release_flex.wait()

    async def train():
        events.append("train")
        assert scheduler.snapshot()["active"][0] == 0
        await release_train.wait()

    sampling = asyncio.create_task(scheduler.sample("a", "a", 10, True, sample))
    await spin_until(lambda: events == ["sample-0"])
    training = asyncio.create_task(scheduler.training(train))
    await spin_until(lambda: scheduler.snapshot()["training_pending"] == 1)
    await scheduler.sample("b", "b", 10, True, sample)
    assert events == ["sample-0", "sample-1"]
    release_flex.set()
    await spin_until(lambda: "train" in events)
    await scheduler.sample("c", "c", 10, True, sample)
    assert events[-1] == "sample-1"
    release_train.set()
    await asyncio.gather(sampling, training)
    await scheduler.close()


async def test_memory_phase_transitions_wait_for_drain_and_wake_for_maintenance():
    scheduler = ResourceScheduler(2, config(), flex_index=0)
    release = asyncio.Event()
    events = []

    async def phase(training):
        events.append("sleep" if training else "wake")
        if training:
            assert scheduler.snapshot()["active"][0] == 0

    async def sample(index):
        events.append(f"sample-{index}")
        if index == 0:
            await release.wait()

    async def train():
        events.append("train")

    scheduler.set_phase_transition(phase)
    pending = asyncio.create_task(scheduler.sample("a", "a", 1, True, sample))
    await spin_until(lambda: "sample-0" in events)
    update = asyncio.create_task(scheduler.training(train, reclaim_sampler_memory=True))
    await spin_until(lambda: scheduler.snapshot()["training_pending"] == 1)
    await scheduler.sample("b", "b", 1, True, sample)
    assert events == ["wake", "sample-0", "sample-1"]
    release.set()
    await asyncio.gather(pending, update)
    assert events[-2:] == ["sleep", "train"]
    await scheduler.training(train)  # Adapter publication needs sampler allocations.
    assert events[-2:] == ["wake", "train"]
    await scheduler.close()


async def test_failed_phase_transition_never_dispatches_operation():
    scheduler = ResourceScheduler(2, config(), flex_index=0)

    async def phase(training):
        raise RuntimeError("transition failed")

    async def forbidden(*args):
        pytest.fail("GPU operation must not run after a failed transition")

    scheduler.set_phase_transition(phase)
    with pytest.raises(RuntimeError, match="transition failed"):
        await scheduler.sample("a", "a", 1, True, forbidden)
    with pytest.raises(RuntimeError, match="transition failed"):
        await scheduler.training(forbidden, reclaim_sampler_memory=True)
    assert scheduler.snapshot()["active"] == [0, 0]
    assert scheduler.snapshot()["training_pending"] == 0
    await scheduler.close()


async def test_sampling_wave_wakes_shared_engine_once():
    scheduler = ResourceScheduler(2, config(), flex_index=0)
    phases = []
    started = []
    release = asyncio.Event()

    async def phase(training):
        phases.append(training)
        await asyncio.sleep(0)

    async def sample(index):
        started.append(index)
        await release.wait()

    scheduler.set_phase_transition(phase)
    tasks = [asyncio.create_task(scheduler.sample(str(i), None, 1, True, sample)) for i in range(4)]
    await spin_until(lambda: len(started) == 4)
    assert started.count(0) == 2
    assert phases == [False]
    release.set()
    await asyncio.gather(*tasks)
    await scheduler.close()


async def test_cancelled_client_does_not_release_running_gpu_work():
    scheduler = ResourceScheduler(2, config(), flex_index=0)
    release = asyncio.Event()
    started = asyncio.Event()
    trained = asyncio.Event()

    async def sample(index):
        assert index == 0
        started.set()
        await release.wait()

    task = asyncio.create_task(scheduler.sample("a", "a", 10, True, sample))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert scheduler.snapshot()["active"][0] == 1

    async def train():
        trained.set()

    training = asyncio.create_task(scheduler.training(train))
    await spin_until(lambda: scheduler.snapshot()["training_pending"] == 1)
    assert not trained.is_set()
    release.set()
    await training
    assert trained.is_set()
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


async def test_predicted_training_horizon_keeps_long_estimate_on_fixed():
    import time

    scheduler = ResourceScheduler(2, config(), flex_index=0)
    now = time.monotonic()
    for offset in [-3.1, -2.1, -1.1, -0.1]:
        scheduler.profiler.observe("regular", now + offset)
    scheduler._seconds_per_token = 0.1
    indices = []

    async def operation(index):
        indices.append(index)

    await scheduler.sample("a", None, 10, True, operation)
    assert indices == [1]
    assert scheduler.snapshot()["window_rejections"] > 0
    await scheduler.close()


async def test_alignment_keeps_next_rollout_available_for_reclaimed_gpu():
    scheduler = ResourceScheduler(2, config(sampling_alignment_s=1), flex_index=0)
    training_started = asyncio.Event()
    release = asyncio.Event()
    indices = []

    async def train():
        training_started.set()
        await release.wait()

    async def sample(index):
        indices.append(index)

    training = asyncio.create_task(scheduler.training(train))
    await training_started.wait()
    sampling = asyncio.create_task(scheduler.sample("a", None, 1, True, sample))
    await spin_until(lambda: scheduler.snapshot()["queued"] == 1)
    assert not indices
    release.set()
    await asyncio.gather(training, sampling)
    assert indices == [0]
    await scheduler.close()


async def test_alignment_budget_expires_to_fixed_during_long_training():
    scheduler = ResourceScheduler(2, config(sampling_alignment_s=0.01), flex_index=0)
    training_started = asyncio.Event()
    release = asyncio.Event()

    async def train():
        training_started.set()
        await release.wait()

    async def sample(index):
        return index

    training = asyncio.create_task(scheduler.training(train))
    await training_started.wait()
    try:
        assert await asyncio.wait_for(scheduler.sample("a", None, 1, True, sample), 2) == 1
        assert scheduler.snapshot()["training_active"]
    finally:
        release.set()
        await training
        await scheduler.close()


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
        scheduling={"policy": "loopweave"},
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
        scheduling={"policy": "loopweave"},
    )
    with pytest.raises(ValidationError, match="sampling_kv_cache_memory_bytes"):
        ModelConfig.model_validate(values)
    values["sampling_kv_cache_memory_bytes"] = 16 * 1024**3
    assert ModelConfig.model_validate(values).scheduling.share_base_weights


def test_loopweave_sharing_is_required_and_legacy_config_migrates():
    for policy in ("round_robin", "balanced", "loopweave"):
        expected = policy == "loopweave"
        current = SchedulingConfig(policy=policy)
        legacy = SchedulingConfig.model_validate({"policy": policy, "share_base_weights": expected})
        assert current.share_base_weights is expected
        assert current.model_dump() == legacy.model_dump()
        assert "share_base_weights" not in current.model_dump()
        with pytest.raises(ValidationError, match="always shares frozen base weights"):
            SchedulingConfig.model_validate({"policy": policy, "share_base_weights": not expected})


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
    legacy_data["supported_models"][0].pop("scheduling")
    legacy_data["supported_models"][0].pop("sampling_kv_cache_memory_bytes")
    for field in (
        "sampling_language_model_only",
        "sampling_moe_backend",
        "sampling_gdn_prefill_backend",
        "sampling_max_num_seqs",
    ):
        legacy_data["supported_models"][0].pop(field)
    legacy = ConfigSignature(config_data=legacy_data)
    assert legacy.matches(ConfigSignature.from_app_config(config))
    model.sampling_kv_cache_memory_bytes = 16 * 1024**3
    model.sampling_language_model_only = True
    model.sampling_moe_backend = "triton"
    model.sampling_gdn_prefill_backend = "triton"
    model.sampling_max_num_seqs = 64
    assert legacy.matches(ConfigSignature.from_app_config(config))
    model.scheduling = SchedulingConfig(policy="balanced")
    assert legacy.matches(ConfigSignature.from_app_config(config))
    model.max_lora_rank = 16
    assert not legacy.matches(ConfigSignature.from_app_config(config))


async def test_window_service_estimate_counts_concurrent_gpu_interval_once(monkeypatch):
    import tuft.scheduling as module

    now = [0.0]
    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: now[0]))
    scheduler = ResourceScheduler(2, config(flex_max_inflight=2), flex_index=0)
    fixed_release, flex_release = asyncio.Event(), asyncio.Event()
    fixed_started = asyncio.Event()
    flex_indices = []

    async def fixed(index):
        assert index == 1
        fixed_started.set()
        await fixed_release.wait()

    async def flex(index):
        flex_indices.append(index)
        await flex_release.wait()

    fixed_task = asyncio.create_task(scheduler.sample("fixed", None, 10000, False, fixed))
    await fixed_started.wait()
    batch = [asyncio.create_task(scheduler.sample(str(i), None, 100, True, flex)) for i in range(2)]
    await spin_until(lambda: len(flex_indices) == 2)
    assert flex_indices == [0, 0]
    now[0] = 10.0
    flex_release.set()
    await asyncio.gather(*batch)
    # 200 reserved tokens served in one 10-second concurrent interval.
    assert scheduler._seconds_per_token == pytest.approx(10 / 200)

    async def fail(index):
        assert index == 0
        raise RuntimeError("backend rejected request")

    with pytest.raises(RuntimeError, match="rejected"):
        await scheduler.sample("bad", None, 100, True, fail)
    # Fast failures must not be learned as improved GPU throughput.
    assert scheduler._seconds_per_token == pytest.approx(10 / 200)
    fixed_release.set()
    await fixed_task
    await scheduler.close()


async def test_adapter_batch_groups_only_fairly_admitted_requests():
    scheduler = ResourceScheduler(
        1,
        config(
            adapter_batch_window_s=0.1, adapter_batch_max_requests=8, max_inflight_per_replica=8
        ),
    )
    release = asyncio.Event()
    seen = []

    async def operation(label, index):
        seen.append(label)
        await release.wait()
        return label

    tasks = []
    for i in range(8):
        adapter = f"snapshot-{i % 2}"
        label = (adapter, i)
        tasks.append(
            asyncio.create_task(
                scheduler.sample(
                    f"user-{i % 4}",
                    adapter,
                    1,
                    False,
                    lambda index, label=label: operation(label, index),
                )
            )
        )
    await spin_until(lambda: len(seen) == 8)
    assert seen == [("snapshot-0", i) for i in (0, 2, 4, 6)] + [
        ("snapshot-1", i) for i in (1, 3, 5, 7)
    ]
    stats = scheduler.snapshot()
    assert stats["adapter_dispatch_switches"] == 1  # FIFO would switch seven times.
    assert stats["adapter_grouped_requests"] == 6
    assert stats["active"] == [8]
    release.set()
    results = await asyncio.gather(*tasks)
    assert results == [(f"snapshot-{i % 2}", i) for i in range(8)]
    await scheduler.close()


async def test_batching_keeps_hot_tenant_from_taking_other_tenants_slots():
    scheduler = ResourceScheduler(
        1,
        config(
            adapter_batch_window_s=0.1, adapter_batch_max_requests=8, max_inflight_per_replica=4
        ),
    )
    release = asyncio.Event()
    seen = []

    async def operation(user, index):
        seen.append(user)
        await release.wait()

    users = ["hot"] * 5 + ["cold-a", "cold-b", "cold-c"]
    tasks = [
        asyncio.create_task(
            scheduler.sample(user, user, 1, False, lambda index, user=user: operation(user, index))
        )
        for user in users
    ]
    await spin_until(lambda: len(seen) == 4)
    assert seen == ["hot", "cold-a", "cold-b", "cold-c"]
    assert scheduler.snapshot()["queued"] == 4
    release.set()
    await asyncio.gather(*tasks)
    await scheduler.close()


async def test_batch_window_cancellation_and_shutdown_do_not_dispatch():
    scheduler = ResourceScheduler(1, config(adapter_batch_window_s=0.1))

    async def forbidden(index):
        pytest.fail("unadmitted request executed")

    cancelled = asyncio.create_task(scheduler.sample("a", "v1", 1, False, forbidden))
    pending = asyncio.create_task(scheduler.sample("b", "v2", 1, False, forbidden))
    await spin_until(lambda: scheduler.snapshot()["queued"] == 2)
    cancelled.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled
    await scheduler.close()
    with pytest.raises(RuntimeError, match="closed"):
        await pending
    assert scheduler.snapshot()["active"] == [0]
    assert scheduler.snapshot()["queued"] == 0


async def test_batch_deadline_is_not_extended_by_new_arrivals(monkeypatch):
    import tuft.scheduling as module

    now = [10.0]
    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: now[0]))
    scheduler = ResourceScheduler(1, config(adapter_batch_window_s=0.1))
    seen = []

    async def operation(index):
        seen.append(index)

    first = asyncio.create_task(scheduler.sample("a", "v1", 1, False, operation))
    await spin_until(lambda: scheduler.snapshot()["queued"] == 1)
    now[0] = 10.09
    second = asyncio.create_task(scheduler.sample("b", "v2", 1, False, operation))
    await spin_until(lambda: scheduler.snapshot()["queued"] == 2)
    assert not seen
    now[0] = 10.101
    scheduler._pump()
    await asyncio.gather(first, second)
    assert len(seen) == 2
    await scheduler.close()


async def test_batch_rechecks_training_before_borrowing():
    scheduler = ResourceScheduler(2, config(adapter_batch_window_s=0.1), flex_index=0)
    training_started, training_release = asyncio.Event(), asyncio.Event()
    indices = []

    async def sample(index):
        indices.append(index)

    async def train():
        training_started.set()
        await training_release.wait()

    request = asyncio.create_task(scheduler.sample("a", "v1", 1, True, sample))
    await spin_until(lambda: scheduler.snapshot()["queued"] == 1)
    update = asyncio.create_task(scheduler.training(train))
    await training_started.wait()
    await request
    assert indices == [1]
    training_release.set()
    await update
    await scheduler.close()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"adapter_batch_window_s": float("nan")},
        {"adapter_batch_window_s": -1},
        {"adapter_batch_max_requests": 0},
        {"adapter_batch_max_requests": 4097},
    ],
)
def test_invalid_adapter_batching_config(kwargs):
    with pytest.raises(ValidationError):
        config(**kwargs)
