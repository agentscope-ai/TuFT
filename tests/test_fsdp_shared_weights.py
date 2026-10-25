import asyncio
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from peft import LoraConfig, get_peft_model
from tinker import types
from transformers import Qwen3Config, Qwen3ForCausalLM

from tuft.backends.fsdp_shared_weights import shared_fsdp_placement
from tuft.backends.fsdp_training_backend import FSDPTrainingBackend
from tuft.backends.sampling_backend import VLLMSamplingBackend
from tuft.backends.vllm_engine import VLLMEngine
from tuft.config import ModelConfig, SchedulingConfig
from tuft.scheduling import ResourceScheduler


def test_fsdp_shared_layout_preserves_sharding_and_replicates_only_norms():
    config = Qwen3Config(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
    )
    model = get_peft_model(
        Qwen3ForCausalLM(config).to(torch.bfloat16),
        LoraConfig(r=2, target_modules=["q_proj", "down_proj"]),
    )
    placements, ignored = shared_fsdp_placement(model, 2)
    for name, p in model.named_parameters():
        if ".lora_" in name:
            assert p not in placements and p not in ignored
        elif p.ndim == 1:
            assert p in ignored
        else:
            expected = 1 if any(f".{part}." in name for part in ("o_proj", "down_proj")) else 0
            assert placements[p].dim == expected
    config.num_key_value_heads = 1
    with pytest.raises(ValueError, match="attention heads"):
        shared_fsdp_placement(model, 2)
    config.num_key_value_heads = 2
    config.vocab_size = 129
    with pytest.raises(ValueError, match="vocabulary"):
        shared_fsdp_placement(model, 2)


@pytest.mark.parametrize("world_size", [1, 2, 4])
def test_fsdp_shared_weight_requires_matching_groups(world_size):
    args = dict(
        model_name="Qwen/Qwen3-4B",
        model_path="Qwen/Qwen3-4B",
        max_model_len=1024,
        training_backend="fsdp",
        colocate=True,
        fsdp_num_gpus=world_size,
        tensor_parallel_size=world_size,
        data_parallel_size=2,
        scheduling={"policy": "shared_weight", "share_base_weights": True},
        sampling_kv_cache_memory_bytes=2**30,
    )
    assert ModelConfig.model_validate(args).fsdp_num_gpus == world_size
    args["tensor_parallel_size"] = world_size + 1
    with pytest.raises(ValueError, match="fsdp_num_gpus"):
        ModelConfig.model_validate(args)


async def test_fsdp_optimizer_waits_for_drain_and_survives_caller_cancellation():
    backend: Any = FSDPTrainingBackend.__new__(FSDPTrainingBackend)
    backend.scheduler = ResourceScheduler(
        2,
        SchedulingConfig(policy="shared_weight", training_quiet_s=0, cost_aware_borrowing=False),
        flex_index=0,
    )
    backend._lock = asyncio.Lock()
    backend._lora_id_to_adapter_name = {"tenant": "slot"}
    updates = []
    backend._worker = SimpleNamespace(optim_step=lambda *a: updates.append(a[0]) or {"steps": 1})
    events = []

    async def phase(training):
        events.append(training)

    backend.scheduler.set_phase_transition(phase)
    release = asyncio.Event()
    entered = asyncio.Event()

    async def sample(index):
        assert index == 0
        entered.set()
        await release.wait()

    sampling = asyncio.create_task(backend.scheduler.sample("user", None, 1, True, sample))
    await entered.wait()
    training = asyncio.create_task(
        backend.optim_step(types.AdamParams(learning_rate=1e-4), "tenant")
    )
    await asyncio.sleep(0)
    assert not updates
    training.cancel()
    release.set()
    await sampling
    result = await training
    assert result.metrics == {"steps": 1.0}
    assert updates == ["slot"] and events == [False, True]
    await backend.scheduler.close()


@pytest.mark.parametrize(
    "assignments",
    [
        [{"node": "a", "gpus": [0]}, {"node": "b", "gpus": [1]}],
        [{"node": "a", "gpus": [0]}, {"node": "a", "gpus": [0]}],
        [{"node": "a", "gpus": [0]}, {"node": "a", "gpus": []}],
    ],
)
def test_fsdp_sharing_rejects_invalid_physical_placement(assignments):
    sampler: Any = VLLMSamplingBackend.__new__(VLLMSamplingBackend)
    sampler.engine = None
    sampler.config = SimpleNamespace(tensor_parallel_size=2)
    with pytest.raises(ValueError, match="distinct GPUs on one node"):
        sampler.configure_fsdp_placement(assignments)


async def test_tp_verification_includes_every_rank():
    engine = VLLMEngine.__new__(VLLMEngine)
    engine._phase_stats = {}
    engine._sleeping = False

    async def collective_rpc(method):
        if method == "shared_weight_report":
            return [{"shared_storage_bytes": 10}, {"shared_storage_bytes": 20}]
        return [{"weight": "rank0"}, {"weight": "rank1"}]

    engine.async_llm = SimpleNamespace(collective_rpc=collective_rpc)
    report = await engine.shared_weight_report()
    assert report["shared_storage_bytes"] == 30 and len(report["ranks"]) == 2
    assert await engine.base_weight_fingerprints() == {
        "rank_0/weight": "rank0",
        "rank_1/weight": "rank1",
    }


async def test_failed_collective_blocks_queued_updates_and_flex_sampling(monkeypatch):
    import ray

    backend: Any = FSDPTrainingBackend.__new__(FSDPTrainingBackend)
    backend.scheduler = ResourceScheduler(
        2,
        SchedulingConfig(policy="shared_weight", training_quiet_s=0, cost_aware_borrowing=False),
        flex_index=0,
    )
    backend._worker = None
    backend._lora_id_to_adapter_name = {"tenant": "slot"}
    calls = []
    backend._actors = [
        SimpleNamespace(optim_step=SimpleNamespace(remote=lambda *a: calls.append(a) or "ref"))
    ]

    def failed_get(refs):
        raise RuntimeError("rank lost during collective")

    monkeypatch.setattr(ray, "get", failed_get)

    async def phase(training):
        backend.check_execution_health()

    backend.scheduler.set_phase_transition(phase)
    results = await asyncio.gather(
        backend.optim_step(types.AdamParams(learning_rate=1e-4), "tenant"),
        backend.optim_step(types.AdamParams(learning_rate=1e-4), "tenant"),
        return_exceptions=True,
    )
    assert all(isinstance(r, RuntimeError) for r in results)
    assert len(calls) == 1

    async def sample(index):
        return index

    assert await backend.scheduler.sample("user", None, 1, True, sample) == 1
    assert backend.scheduler.snapshot()["flex_failed"]
    assert await backend.scheduler.sample("user", None, 1, False, sample) == 1
    await backend.scheduler.close()


@pytest.mark.integration
async def test_fsdp_two_rank_update_telemetry_keeps_group_and_shards(monkeypatch):
    """Exercise real FSDP dispatch/aggregation with CPU stand-ins for Ray ranks."""
    import logging

    import ray

    backend: Any = FSDPTrainingBackend.__new__(FSDPTrainingBackend)
    backend.scheduler = ResourceScheduler(2, SchedulingConfig(policy="shared_weight"), flex_index=0)
    backend.config = SimpleNamespace(micro_batch_size=1)
    backend.logger = logging.getLogger("fsdp-test")
    backend._worker = None
    backend._world_size = 2
    backend._lora_id_to_adapter_name = {"a": "slot-a", "b": "slot-b"}
    calls = []
    phases = []

    async def phase(training):
        phases.append(training)
        assert backend.scheduler.snapshot()["active"][0] == 0

    def actor(rank):
        def forward(data, slot, *args):
            assert backend.scheduler.snapshot()["training_active"]
            calls.append((rank, "backward", slot, len(data)))
            return {"metrics": {}, "loss_fn_outputs": []}

        def optimizer(slot, *args):
            assert backend.scheduler.snapshot()["training_active"]
            calls.append((rank, "optimizer", slot, 0))
            return {"steps": 1}

        return SimpleNamespace(
            forward_backward=SimpleNamespace(remote=forward),
            optim_step=SimpleNamespace(remote=optimizer),
        )

    backend._actors = [actor(0), actor(1)]
    original_actors = tuple(backend._actors)
    monkeypatch.setattr(ray, "get", lambda refs: refs)
    backend.scheduler.set_phase_transition(phase)
    data = [
        types.Datum(model_input=types.ModelInput.from_ints([i, i + 1]), loss_fn_inputs={})
        for i in range(4)
    ]
    for _ in range(2):
        await backend.forward(data, "a", "cross_entropy", None, backward=True)
    assert not backend.scheduler.profiler.arrivals["a"]
    await backend.optim_step(types.AdamParams(), "a")
    assert backend.scheduler.profiler.states["a"].observations == 1
    assert "b" not in backend.scheduler.profiler.states
    assert calls == [
        (rank, op, "slot-a", size)
        for op, size in [("backward", 2), ("backward", 2), ("optimizer", 0)]
        for rank in range(2)
    ]
    assert tuple(backend._actors) == original_actors and backend._world_size == 2
    assert phases == [True]
    await backend.scheduler.close()


@pytest.mark.parametrize(
    "case,healthy",
    [
        ("input_all", True),
        ("write_partial", True),
        ("input_partial", False),
        ("input_pending", False),
        ("unclassified_all", False),
        ("actor_lost", False),
    ],
)
async def test_only_completed_safe_checkpoint_failures_preserve_fsdp_group(
    monkeypatch, case, healthy
):
    import ray
    from ray.exceptions import RayActorError

    from tuft.backends.fsdp_training_backend import (
        FSDPCheckpointInputError,
        FSDPCheckpointWriteError,
    )

    backend: Any = FSDPTrainingBackend.__new__(FSDPTrainingBackend)
    backend._group_failed = False
    backend.scheduler = ResourceScheduler(2, SchedulingConfig(policy="shared_weight"), flex_index=0)
    backend._worker = None
    backend._lora_id_to_adapter_name = {"tenant": "slot"}
    backend._actors = [SimpleNamespace(optim_step=SimpleNamespace(remote=lambda *a: "ok"))] * 2
    error = (
        FSDPCheckpointWriteError("disk full")
        if case == "write_partial"
        else RuntimeError("collective error")
        if case == "unclassified_all"
        else FSDPCheckpointInputError("bad checkpoint")
    )
    outcomes: list[BaseException | None] = [error, error]
    if case in {"input_partial", "write_partial"}:
        outcomes[1] = None
    if case == "actor_lost":
        outcomes[1] = RayActorError()

    def get(ref):
        if isinstance(ref, list):
            return [get(r) for r in ref]
        if ref == "ok":
            return {"step_count": 1}
        result = outcomes[ref]
        if isinstance(result, BaseException):
            raise result
        return result

    monkeypatch.setattr(ray, "get", get)
    monkeypatch.setattr(
        ray, "wait", lambda refs, **kw: ([0], [1]) if case == "input_pending" else (refs, [])
    )
    with pytest.raises(RuntimeError):
        await backend._await_workers([0, 1])
    assert backend._group_failed is not healthy
    assert backend.scheduler.snapshot()["flex_failed"] is not healthy
    if healthy:
        await backend.optim_step(types.AdamParams(), "tenant")
    else:
        with pytest.raises(RuntimeError, match="restart the backend"):
            await backend.optim_step(types.AdamParams(), "tenant")
    await backend.scheduler.close()


def test_missing_fsdp_checkpoint_is_rejected_before_gpu_mutation(tmp_path):
    from unittest.mock import Mock

    from tuft.backends.fsdp_training_backend import FSDPCheckpointInputError, MultiAdapterFSDPWorker

    worker: Any = MultiAdapterFSDPWorker.__new__(MultiAdapterFSDPWorker)
    worker._adapters = {"slot": SimpleNamespace(target_modules=["q_proj"], target_parameters=[])}
    worker._activate_adapter = Mock()
    with pytest.raises(FSDPCheckpointInputError):
        worker.load_checkpoint("slot", tmp_path, ["q_proj"], optimizer=False)
    worker._activate_adapter.assert_not_called()
