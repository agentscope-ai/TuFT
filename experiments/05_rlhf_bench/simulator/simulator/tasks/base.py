"""Abstract base class for tasks (prompt generation + reward computation)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class TaskPrompt:
    """A prompt with associated metadata for reward computation."""

    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class Task(ABC):
    """Abstract task providing prompts and reward computation."""

    @abstractmethod
    def load_dataset(self) -> None:
        """Load the dataset (train + test splits)."""

    @abstractmethod
    def get_prompt(self) -> TaskPrompt:
        """Get the next training prompt (cycles through dataset)."""

    @abstractmethod
    def compute_reward(self, response_text: str, metadata: Dict[str, Any]) -> float:
        """Compute reward for a response. Returns value in [0, 1]."""

    @abstractmethod
    def get_eval_prompts(self, n: int) -> List[TaskPrompt]:
        """Get n evaluation prompts."""
