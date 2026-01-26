"""Sampling backend implementated using vLLM"""

import asyncio
from logging import getLogger
from pathlib import Path
from typing import Optional

from tinker import types

from ..config import ModelConfig
from .base_backend import BaseSamplingBackend

logger = getLogger(__name__)


class VLLMSamplingBackend(BaseSamplingBackend):
    """A sampling backend using vLLM.

    User side `sample`, `sample_async`, `compute_logprobs` and
    `compute_logprobs_async` are all supported by the sample method.
    """

    def __init__(self, config: ModelConfig) -> None:
        from vllm.lora.request import LoRARequest

        super().__init__(config)
        self.engine = self._create_engine(config)
        self.lora_adapters: dict[str, LoRARequest] = {}
        self._counter = 1
        self._lock = asyncio.Lock()

    def _create_engine(self, config: ModelConfig):
        import ray
        from ray.util.placement_group import placement_group
        from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
        from trinity.common.config import InferenceModelConfig
        from trinity.common.models.vllm_model import vLLMRolloutModel

        # create a placement group for this model
        pg = placement_group(
            [{"CPU": 1, "GPU": 1} for _ in range(config.tensor_parallel_size)],
            strategy="PACK",
        )
        ray.get(pg.ready(), timeout=10)
        return (
            ray.remote(vLLMRolloutModel)
            .options(
                name="sampling_model_" + self.base_model,
                num_gpus=0 if config.tensor_parallel_size > 1 else 1,
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_capture_child_tasks=True,
                ),
            )
            .remote(
                config=InferenceModelConfig(
                    model_path=str(config.model_path),
                    tensor_parallel_size=config.tensor_parallel_size,
                    max_model_len=config.max_model_len,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    top_k=config.top_k,
                    logprobs=config.logprobs,
                    min_response_tokens=config.min_response_tokens,
                    repetition_penalty=1.0,
                    enable_lora=True,
                    enable_runtime_lora_updating=True,
                    lora_kwargs={
                        "max_lora_rank": config.max_lora_rank,
                        "max_loras": config.max_loras,
                    },
                )
            )
        )

    async def async_init(self) -> None:
        """Initialize the backend for sampling."""
        await self.engine.prepare.remote()
        logger.info(f"SamplingBackend for model {self.base_model} initialized.")

    async def sample(
        self,
        prompt: types.ModelInput,
        num_samples: int,
        sampling_params: types.SamplingParams,
        include_prompt_logprobs: bool = False,
        topk_prompt_logprobs: int = 0,
        lora_id: Optional[str] = None,
    ) -> types.SampleResponse:
        """Sampling using vLLM engine."""
        async with self._lock:
            if lora_id is not None and lora_id not in self.lora_adapters:
                raise ValueError(f"LoRA adapter {lora_id} not found in backend.")
            lora_request = self.lora_adapters[lora_id] if lora_id is not None else None
        return await self.engine.sample.remote(
            prompt=prompt,
            num_samples=num_samples,
            sampling_params=sampling_params,
            include_prompt_logprobs=include_prompt_logprobs,
            topk_prompt_logprobs=topk_prompt_logprobs,
            lora_request=lora_request,
        )

    async def add_adapter(self, lora_id: str, adapter_path: Path) -> None:
        from vllm.lora.request import LoRARequest

        async with self._lock:
            self._counter += 1
            self.lora_adapters[lora_id] = LoRARequest(
                lora_int_id=self._counter + 1,
                lora_name=lora_id,
                lora_path=str(adapter_path),
            )

    async def remove_adapter(self, lora_id: str) -> None:
        async with self._lock:
            if lora_id in self.lora_adapters:
                del self.lora_adapters[lora_id]
        # TODO: unload LoRA from vLLM engine


class DummySamplingBackend(BaseSamplingBackend):
    """A dummy sampling backend that returns fixed responses for unittest."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self.lora_adapters: dict[str, Path] = {}
        self._counter = 1
        self._lock = asyncio.Lock()

    async def async_init(self) -> None:
        """No initialization needed for dummy backend."""
        pass

    async def sample(
        self,
        prompt: types.ModelInput,
        num_samples: int,
        sampling_params: types.SamplingParams,
        include_prompt_logprobs: bool = False,
        topk_prompt_logprobs: int = 0,
        lora_id: Optional[str] = None,
    ) -> types.SampleResponse:
        """Return a fixed dummy response."""
        prompt_tokens = prompt.to_ints()
        max_tokens = sampling_params.max_tokens or 16
        sequences: list[types.SampledSequence] = []
        for _ in range(num_samples):
            generated = self._generate_tokens(prompt_tokens, max_tokens)
            seq = types.SampledSequence(
                stop_reason="length",
                tokens=generated,
                logprobs=[-0.3 for _ in generated],
            )
            sequences.append(seq)
        prompt_logprobs = None
        topk_prompt = None
        if include_prompt_logprobs:
            prompt_logprobs = [-0.1 if tok is not None else None for tok in prompt_tokens]
        if topk_prompt_logprobs > 0:
            topk_prompt = [
                [
                    (token, round(-0.05 - idx * 0.01, 4))
                    for idx, token in enumerate(prompt_tokens[:topk_prompt_logprobs])
                ]
                if token is not None
                else None
                for token in prompt_tokens
            ]
        return types.SampleResponse(
            sequences=sequences,
            prompt_logprobs=prompt_logprobs,
            topk_prompt_logprobs=topk_prompt,
        )

    def _generate_tokens(self, prompt_tokens: list[int], max_tokens: int) -> list[int]:
        start = prompt_tokens[-1] if prompt_tokens else (abs(self.config.seed) % 32000) + 1
        return [(start + i) % 32000 for i in range(1, max_tokens + 1)]

    async def add_adapter(self, lora_id: str, adapter_path: Path) -> None:
        self.lora_adapters[lora_id] = adapter_path

    async def remove_adapter(self, lora_id: str) -> None:
        if lora_id in self.lora_adapters:
            del self.lora_adapters[lora_id]
