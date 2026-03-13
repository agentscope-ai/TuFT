import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from tinker import types

from ..checkpoints import CheckpointRecord
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

    @abstractmethod
    async def add_adapter(self, lora_id: str, adapter_path: Path) -> None:
        """Add LoRA adapter to the backend."""

    @abstractmethod
    async def remove_adapter(self, lora_id: str) -> None:
        """Remove LoRA adapter from the backend."""

    def get_openai_api_url(self) -> Optional[str]:
        """Return the vLLM OpenAI API base URL, or None if not available."""
        return None

    @classmethod
    def create_backend(
        cls, config: ModelConfig, worker_venv_path: Optional[str] = None
    ) -> "BaseSamplingBackend":
        """Factory method to create a sampling backend instance.

        TUFT_CPU_TEST=1: use DummySamplingBackend (no vLLM, for CPU-only unit tests).
        Otherwise: VLLMSamplingBackend (creates Ray/vLLM actor in __init__, may block startup).
        """
        if os.getenv("TUFT_CPU_TEST", "0") == "1":
            from ..backends.sampling_backend import DummySamplingBackend

            return DummySamplingBackend(config)
        from ..backends.sampling_backend import VLLMSamplingBackend

        return VLLMSamplingBackend(config, worker_venv_path=worker_venv_path)


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
    async def create_adapter(self, lora_id: str, lora_config: types.LoraConfig) -> None:
        """Abstract method for creating LoRA adapter."""

    @abstractmethod
    async def remove_adapter(self, lora_id: str) -> None:
        """Abstract method for removing LoRA adapter."""

    @abstractmethod
    async def optim_step(
        self,
        adam_params: types.AdamParams,
        lora_id: str,
    ) -> types.OptimStepResponse:
        """Abstract method for optimization step."""

    @abstractmethod
    async def save_state(
        self, lora_id: str, checkpoint_record: "CheckpointRecord", optimizer: bool
    ) -> None:
        """Abstract method for saving model state."""

    @abstractmethod
    async def load_state(
        self, lora_id: str, checkpoint_record: "CheckpointRecord", optimizer: bool
    ) -> None:
        """Abstract method for loading model state."""

    @classmethod
    def create_backend(
        cls,
        config: ModelConfig,
        fsdp_index: Optional[int] = None,
        worker_venv_path: Optional[str] = None,
    ) -> "BaseTrainingBackend":
        """Factory method to create a training backend instance.

        fsdp_index: For FSDP backends, master port is config.fsdp_master_port + fsdp_index.
        The base port is configurable via ModelConfig.fsdp_master_port (default 29500); multiple
        FSDP models use base, base+1, ... Pass the index of this model among FSDP models in
        supported_models order. Omit for non-FSDP or when only one FSDP model (uses config port).
        """
        if os.getenv("TUFT_CPU_TEST", "0") == "1":
            from ..backends.training_backend import DummyTrainingBackend

            return DummyTrainingBackend(config)
        training_backend = getattr(config, "training_backend", "hf")
        if training_backend == "fsdp":
            from ..backends.fsdp_training_backend import FSDPTrainingBackend

            return FSDPTrainingBackend(
                config, fsdp_index=fsdp_index, worker_venv_path=worker_venv_path
            )
        from ..backends.training_backend import HFTrainingBackend

        return HFTrainingBackend(config)
