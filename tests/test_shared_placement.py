"""CPU contract tests for trainer-owned GPUs and explicit sampler placement."""

from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import pytest

from tuft.backends.sampling_backend import VLLMSamplingBackend
from tuft.backends.vllm_engine import VLLMEngine, VLLMEngineConfig
from tuft.config import AppConfig, ModelConfig, SchedulingConfig


def model(name: str = "qwen", backend: Literal["hf", "fsdp"] = "hf", ranks: int = 1):
    return ModelConfig(
        model_name=name,
        model_path=Path(name),
        max_model_len=512,
        colocate=True,
        training_backend=backend,
        fsdp_num_gpus=ranks,
        tensor_parallel_size=ranks,
        data_parallel_size=2,
        sampling_kv_cache_memory_bytes=1024**3,
        scheduling=SchedulingConfig(policy="shared_weight"),
    )


@pytest.mark.parametrize("backend,ranks", [("hf", 1), ("fsdp", 2)])
def test_borrowing_sampler_uses_exact_trainer_devices_without_new_gpu_reservation(
    monkeypatch, backend, ranks
):
    import ray

    options = {}
    created = []
    node = ray.NodeID.from_random().hex()

    class Actor:
        def options(self, **kwargs):
            options.update(kwargs)
            return self

        def remote(self, config):
            created.append(config)
            return "sampler-actor"

    monkeypatch.setattr(ray, "remote", lambda cls: Actor())
    sampler = VLLMSamplingBackend.__new__(VLLMSamplingBackend)
    sampler.config = model(backend=backend, ranks=ranks)
    sampler.engine = None
    sampler._worker_venv_path = None
    assignments = [{"node": node, "gpus": [i + 3]} for i in range(ranks)]
    sampler.configure_shared_placement(assignments)
    assert options["num_gpus"] == 0
    assert options["scheduling_strategy"].node_id == node
    assert options["scheduling_strategy"].soft is False
    assert created[0].shared_gpu_ids == [str(i + 3) for i in range(ranks)]
    assert created[0].tensor_parallel_size == ranks
    assert created[0].share_base_weights
    assert not created[0].enable_openai_api
    with pytest.raises(RuntimeError, match="initialization"):
        sampler.configure_shared_placement(assignments)


def test_hf_trainer_reserves_a_whole_gpu_for_each_shared_base(monkeypatch):
    import ray

    from tuft.backends.hf_training_model import HFTrainingModel

    options = []

    class Actor:
        def options(self, **kwargs):
            options.append(kwargs)
            return self

        def remote(self, config):
            return config.model_name

    monkeypatch.setattr(ray, "remote", lambda cls: Actor())
    for name in ["base-a", "base-b"]:
        assert HFTrainingModel.get_actor(model(name)) == name
    assert [o["num_gpus"] for o in options] == [1, 1]
    assert options[0]["name"] != options[1]["name"]


def test_shared_actor_reports_borrowed_devices_for_startup_validation(monkeypatch):
    import ray

    monkeypatch.setattr(
        ray, "get_runtime_context", lambda: SimpleNamespace(get_node_id=lambda: "trainer-node")
    )
    monkeypatch.setattr(ray, "get_gpu_ids", lambda: [])
    engine = VLLMEngine.__new__(VLLMEngine)
    engine.config = VLLMEngineConfig(model_path="test", shared_gpu_ids=["3"])
    assert engine.get_device_assignment() == {"node": "trainer-node", "gpus": [3]}


def test_multiple_models_require_explicit_shared_placement_for_colocation():
    AppConfig(supported_models=[model("a"), model("b", backend="fsdp", ranks=2)]).check_validity()
    legacy = ModelConfig(
        model_name="legacy", model_path=Path("legacy"), max_model_len=512, colocate=True
    )
    with pytest.raises(ValueError, match="Multiple models require shared_weight"):
        AppConfig(supported_models=[legacy, model("b")]).check_validity()


def test_nonshared_colocated_fsdp_creates_sampler_immediately(monkeypatch):
    import sys
    from types import ModuleType

    lora_module = ModuleType("vllm.lora.request")
    monkeypatch.setattr(lora_module, "LoRARequest", object, raising=False)
    monkeypatch.setitem(sys.modules, "vllm.lora.request", lora_module)
    created = []
    monkeypatch.setattr(
        VLLMSamplingBackend,
        "_create_engine",
        lambda self, config: created.append(config) or "engine",
    )
    config = ModelConfig(
        model_name="qwen",
        model_path=Path("qwen"),
        max_model_len=512,
        colocate=True,
        training_backend="fsdp",
        fsdp_num_gpus=1,
    )
    sampler = VLLMSamplingBackend(config)
    assert sampler.engine == "engine" and created == [config]
