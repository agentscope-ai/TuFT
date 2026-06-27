"""Abstract base class for training backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class AdapterConfig:
    """Configuration for a LoRA adapter."""

    lora_rank: int
    learning_rate: float


@dataclass
class SampleResult:
    """Result from a single sampling request."""

    tokens: List[int]
    logprobs: List[float]
    text: str


@dataclass
class TrainStepResult:
    """Result from a training step."""

    loss: Optional[float]
    weight_version: int


class TrainingBackend(ABC):
    """Abstract interface for training backends.

    All methods are async to support both service-based backends (Tinker)
    and in-process backends (verl) uniformly.
    """

    @abstractmethod
    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the backend connection/resources."""

    @abstractmethod
    async def create_adapter(self, adapter_id: str, adapter_config: AdapterConfig) -> None:
        """Create a LoRA adapter for training."""

    @abstractmethod
    async def sync_weights(self, adapter_id: str) -> int:
        """Sync trained weights to the serving path.

        Returns the weight version number after sync.
        """

    @abstractmethod
    async def sample(
        self,
        adapter_id: str,
        prompt_tokens: List[int],
        num_samples: int,
        max_tokens: int,
        temperature: float,
    ) -> List[SampleResult]:
        """Generate responses for given prompt tokens."""

    @abstractmethod
    async def train_step(
        self,
        adapter_id: str,
        training_datums: List[Dict[str, Any]],
    ) -> TrainStepResult:
        """Execute one training step.

        Each datum in training_datums is a dict with:
          - prompt_tokens: List[int]
          - response_tokens: List[int]
          - sampling_logprobs: List[float]
          - advantage: float
        """

    @abstractmethod
    def get_tokenizer(self) -> Any:
        """Return the tokenizer used by the model."""

    @abstractmethod
    async def compute_training_logprobs(
        self,
        adapter_id: str,
        items: List[Dict[str, Any]],
        temperature: float = 1.0,
    ) -> List[List[float]]:
        """Compute per-token training-side logprobs for a list of items.

        Each item is a dict with:
          - prompt_tokens: List[int]
          - response_tokens: List[int]
        Returns a list of per-response-token logprobs (one list per item),
        computed via the training code path (forward) on the CURRENT trainable
        weights (i.e. the same weights that will be used by the next
        forward_backward call). Must NOT modify any parameters.

        Args:
            temperature: Temperature to apply to logits before computing
                logprobs. Must match the temperature used during sampling
                so that IS ratios are computed consistently.
        """

    @abstractmethod
    async def cleanup(self, adapter_id: str) -> None:
        """Clean up resources for an adapter (delete checkpoints, etc.)."""
