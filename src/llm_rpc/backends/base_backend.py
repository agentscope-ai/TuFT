import os
from abc import ABC, abstractmethod
from typing import Optional

from tinker import types

from ..config import ModelConfig


class BaseBackend(ABC):
    """Base class for all backends."""

    def __init__(self, config: ModelConfig) -> None:
        self.base_model = config.model_name
        self.config = config

    @abstractmethod
    async def async_init(self) -> None:
        """Asynchronous initialization if needed."""


class BaseSamplingBackend(BaseBackend):
    """Abstract sampling backend."""

    @abstractmethod
    async def sample(
        self,
        prompt: types.ModelInput,
        num_samples: int,
        sampling_params: types.SamplingParams,
        include_prompt_logprobs: bool = False,
        topk_prompt_logprobs: int = 0,
        lora_id: Optional[str] = None,
    ) -> types.SampleResponse:
        """Abstract method for sampling."""

    @classmethod
    def create_backend(cls, config: ModelConfig) -> "BaseSamplingBackend":
        """Factory method to create a sampling backend instance."""
        if os.getenv("LLM_RPC_CPU_TEST", "0") == "1":
            from ..backends.sampling_backend import DummySamplingBackend

            return DummySamplingBackend(config)
        else:
            from ..backends.sampling_backend import VLLMSamplingBackend

            return VLLMSamplingBackend(config)


class BaseTrainingBackend(BaseBackend):
    """Abstract training backend."""

    @abstractmethod
    async def forward(
        self,
        data: list[types.Datum],
        lora_id: str,
        loss_fn: types.LossFnType,
        loss_fn_config: dict[str, float] | None,
        backward: bool = False,
    ) -> types.ForwardBackwardOutput:
        """Abstract method for forward pass."""

    @abstractmethod
    async def optim_step(
        self,
        adam_params: types.AdamParams,
        lora_id: str,
    ) -> types.OptimStepResponse:
        """Abstract method for optimization step."""

    @abstractmethod
    async def save_state(self, lora_id: str, lora_path: str) -> None:
        """Abstract method for saving model state."""

    @abstractmethod
    async def load_state(self, lora_id: str, lora_path: str, optimizer: bool) -> None:
        """Abstract method for loading model state."""

    @classmethod
    def create_backend(cls, config: ModelConfig) -> "BaseTrainingBackend":
        """Factory method to create a training backend instance."""
        if os.getenv("LLM_RPC_CPU_TEST", "0") == "1":
            from ..backends.training_backend import DummyTrainingBackend

            return DummyTrainingBackend(config)
        else:
            from ..backends.training_backend import HFTrainingBackend

            return HFTrainingBackend(config)
