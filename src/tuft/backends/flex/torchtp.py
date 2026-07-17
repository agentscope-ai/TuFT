from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Any, Optional

from tinker import types

from tuft.backends.flex.flex_backend import FlexBackend, FlexBackendMode
from tuft.backends.flex.torchtp_zero_copy import (
    TensorDescriptorFactory,
    call_collective_rpc,
    create_fused_vllm_state_dict,
    inject_cuda_ipc_alias,
    make_cuda_ipc_descriptor_dict,
    summarize_injection_results,
    tensor_to_cuda_ipc_descriptor,
)
from tuft.checkpoints import CheckpointRecord
from tuft.config import ModelConfig


StateDictBuilder = Callable[[Any], dict[str, Any]]
VLLMEngineFactory = Callable[[ModelConfig], Any]
DescriptorGatherer = Callable[[dict[str, dict[str, Any]]], list[dict[str, dict[str, Any]]]]
TrainingRuntimeFactory = Callable[
    [ModelConfig, list[Any], list[dict[str, dict[str, Any]]] | None],
    Any,
]
RuntimeReleaser = Callable[[Any], None]


class FusedTorchTPVLLMFlexBackend(FlexBackend):
    """FlexBackend for fused PyTorch-TP training weights and vLLM sampling aliasing.

    The training model is expected to already use the vLLM-compatible fused layout:
    qkv_proj and gate_up_proj are present in each decoder layer and each weight is a
    PyTorch-TP local shard. The transform path creates CUDA IPC descriptors from
    those local shards and injects them into vLLM workers via collective_rpc.
    """

    def __init__(
        self,
        config: ModelConfig,
        *,
        training_model: Any | None = None,
        vllm_engine: Any | None = None,
        state_dict_builder: StateDictBuilder = create_fused_vllm_state_dict,
        vllm_engine_factory: VLLMEngineFactory | None = None,
        descriptor_gatherer: DescriptorGatherer | None = None,
        training_runtime_factory: TrainingRuntimeFactory | None = None,
        training_runtime_releaser: RuntimeReleaser | None = None,
        sampling_runtime_releaser: RuntimeReleaser | None = None,
        rank: int | None = None,
        world_size: int | None = None,
        vocab_size: int | None = None,
        verify_inject: bool = False,
        require_cuda_ipc: bool = True,
        descriptor_factory: TensorDescriptorFactory = tensor_to_cuda_ipc_descriptor,
    ) -> None:
        super().__init__(config)
        self.training_model = training_model
        self.vllm_engine = vllm_engine
        self.state_dict_builder = state_dict_builder
        self.vllm_engine_factory = vllm_engine_factory
        self.descriptor_gatherer = descriptor_gatherer
        self.training_runtime_factory = training_runtime_factory
        self.training_runtime_releaser = training_runtime_releaser
        self.sampling_runtime_releaser = sampling_runtime_releaser
        self.rank = rank
        self.world_size = world_size
        self.vocab_size = vocab_size
        self.verify_inject = verify_inject
        self.require_cuda_ipc = require_cuda_ipc
        self.descriptor_factory = descriptor_factory
        self._base_storage_keepalive: list[Any] = []
        self._last_ipc_descriptors: dict[str, dict[str, Any]] | None = None
        self._last_all_rank_descriptors: list[dict[str, dict[str, Any]]] | None = None
        self._last_injection_results: list[dict[str, Any]] | None = None

    def _release_training_runtime(self) -> None:
        if self.training_model is not None and self.training_runtime_releaser is not None:
            self.training_runtime_releaser(self.training_model)
        self.training_model = None

    def _release_sampling_runtime(self) -> None:
        if self.vllm_engine is not None and self.sampling_runtime_releaser is not None:
            self.sampling_runtime_releaser(self.vllm_engine)
        self.vllm_engine = None

    async def async_init(self) -> None:
        if self.training_model is None:
            self.training_model = await asyncio.to_thread(self._load_default_training_model)

    def _load_default_training_model(self) -> Any:
        raise NotImplementedError(
            "Default fused PyTorch-TP model loading is environment-specific; pass "
            "training_model or subclass _load_default_training_model()."
        )

    def _create_default_vllm_engine(self) -> Any:
        from vllm import LLM

        return LLM(
            model=str(self.config.model_path),
            dtype="bfloat16",
            tensor_parallel_size=self._world_size(),
            gpu_memory_utilization=self.config.sampling_memory_fraction,
            trust_remote_code=True,
            enforce_eager=True,
            max_model_len=self.config.sampling_max_model_len or self.config.max_model_len,
            load_format="dummy",
        )

    def _rank(self) -> int:
        if self.rank is not None:
            return self.rank
        try:
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                return int(dist.get_rank())
        except Exception:
            pass
        return 0

    def _world_size(self) -> int:
        if self.world_size is not None:
            return self.world_size
        try:
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                return int(dist.get_world_size())
        except Exception:
            pass
        return int(self.config.tensor_parallel_size or 1)

    def _vocab_size(self) -> int | None:
        if self.vocab_size is not None:
            return self.vocab_size
        model = self.training_model
        config = getattr(model, "config", None)
        vocab_size = getattr(config, "vocab_size", None)
        return int(vocab_size) if vocab_size is not None else None

    def _gather_descriptors(
        self, descriptors: dict[str, dict[str, Any]]
    ) -> list[dict[str, dict[str, Any]]]:
        if self.descriptor_gatherer is not None:
            return self.descriptor_gatherer(descriptors)
        world_size = self._world_size()
        if world_size == 1:
            return [descriptors]
        import torch.distributed as dist

        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("Cannot gather descriptors: torch.distributed is not initialized")
        all_rank_descriptors: list[dict[str, dict[str, Any]] | None] = [None] * world_size
        dist.all_gather_object(all_rank_descriptors, descriptors)
        if any(item is None for item in all_rank_descriptors):
            raise RuntimeError("Descriptor gather returned an empty rank descriptor")
        return [item for item in all_rank_descriptors if item is not None]

    def _ensure_vllm_engine(self) -> Any:
        if self.vllm_engine is not None:
            return self.vllm_engine
        if self.vllm_engine_factory is not None:
            self.vllm_engine = self.vllm_engine_factory(self.config)
        else:
            self.vllm_engine = self._create_default_vllm_engine()
        return self.vllm_engine

    async def _transform_to_sampling_impl(self, *, force: bool = False) -> dict[str, float]:
        if self.training_model is None:
            await self.async_init()
        if self.training_model is None:
            raise RuntimeError("FusedTorchTPVLLMFlexBackend has no training_model")

        if hasattr(self.training_model, "eval"):
            self.training_model.eval()

        state_dict = self.state_dict_builder(self.training_model)
        descriptors, keepalive = make_cuda_ipc_descriptor_dict(
            state_dict,
            rank=self._rank(),
            world_size=self._world_size(),
            vocab_size=self._vocab_size(),
            descriptor_factory=self.descriptor_factory,
            require_cuda=self.require_cuda_ipc,
        )
        all_rank_descriptors = self._gather_descriptors(descriptors)
        engine = self._ensure_vllm_engine()
        injection_results = await call_collective_rpc(
            engine,
            inject_cuda_ipc_alias,
            args=(all_rank_descriptors, self.verify_inject),
        )
        if not isinstance(injection_results, list):
            injection_results = list(injection_results)

        self._base_storage_keepalive = keepalive
        self._release_training_runtime()
        self._last_ipc_descriptors = descriptors
        self._last_all_rank_descriptors = all_rank_descriptors
        self._last_injection_results = injection_results

        metrics = summarize_injection_results(injection_results)
        metrics["ipc_descriptors:sum"] = float(len(descriptors))
        metrics["ipc_ranks:sum"] = float(len(all_rank_descriptors))
        metrics["source_released"] = 1.0
        metrics["training_runtime_released:sum"] = 1.0
        metrics["force:sum"] = float(force)
        return metrics

    async def _transform_to_training_impl(self, *, force: bool = False) -> dict[str, float]:
        if self.training_runtime_factory is None:
            return {
                "base_transform_supported": 0.0,
                "zero_copy": float(self._vllm_base_alias_ready),
                "force:sum": float(force),
            }
        self.training_model = self.training_runtime_factory(
            self.config,
            self._base_storage_keepalive,
            self._last_all_rank_descriptors,
        )
        self._release_sampling_runtime()
        self._vllm_base_alias_ready = False
        return {
            "base_transform_supported": 1.0,
            "zero_copy": 1.0,
            "source_released": 1.0,
            "sampling_runtime_released:sum": 1.0,
            "force:sum": float(force),
        }

    async def sample(
        self,
        prompt: types.ModelInput,
        num_samples: int,
        sampling_params: types.SamplingParams,
        include_prompt_logprobs: bool = False,
        topk_prompt_logprobs: int = 0,
        lora_id: Optional[str] = None,
    ) -> types.SampleResponse:
        if self.mode != FlexBackendMode.SAMPLING:
            await self.transform_to_sampling()
        engine = self._ensure_vllm_engine()
        if not hasattr(engine, "generate"):
            raise NotImplementedError("Sampling requires a local vLLM LLM-compatible engine")
        from vllm import SamplingParams

        params = SamplingParams(
            max_tokens=sampling_params.max_tokens or self.config.default_max_tokens or 16,
            temperature=sampling_params.temperature,
            top_p=sampling_params.top_p,
            top_k=sampling_params.top_k,
            seed=sampling_params.seed,
            n=num_samples,
        )
        outputs = engine.generate([{"prompt_token_ids": prompt.to_ints()}], params)
        sequences: list[types.SampledSequence] = []
        for output in outputs[0].outputs:
            token_ids = list(output.token_ids)
            logprobs = [-0.0 for _ in token_ids]
            sequences.append(
                types.SampledSequence(
                    stop_reason=getattr(output, "finish_reason", "length") or "length",
                    _tokens_list=token_ids,
                    _logprobs_list=logprobs,
                )
            )
        return types.SampleResponse(sequences=sequences)

    async def forward(
        self,
        data: list[types.Datum],
        lora_id: str,
        loss_fn: types.LossFnType,
        loss_fn_config: dict[str, float] | None,
        backward: bool = False,
    ) -> types.ForwardBackwardOutput:
        raise NotImplementedError("FusedTorchTPVLLMFlexBackend training is not wired yet")

    async def create_adapter(self, lora_id: str, lora_config: types.LoraConfig) -> None:
        raise NotImplementedError("Adapter management is outside this transform implementation")

    async def remove_adapter(self, lora_id: str) -> None:
        raise NotImplementedError("Adapter management is outside this transform implementation")

    async def optim_step(
        self,
        adam_params: types.AdamParams,
        lora_id: str,
    ) -> types.OptimStepResponse:
        raise NotImplementedError("FusedTorchTPVLLMFlexBackend optimizer is not wired yet")

    async def save_state(
        self,
        lora_id: str,
        checkpoint_record: CheckpointRecord,
        optimizer: bool,
    ) -> None:
        raise NotImplementedError("Checkpointing is outside this transform implementation")

    async def load_state(
        self,
        lora_id: str,
        checkpoint_record: CheckpointRecord,
        optimizer: bool,
    ) -> None:
        raise NotImplementedError("Checkpointing is outside this transform implementation")

    async def add_adapter(self, lora_id: str, adapter_path: Path) -> None:
        raise NotImplementedError("Adapter management is outside this transform implementation")
