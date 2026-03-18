"""
base.py
abstract base class for finetuning tasks, defining the common interface and utilities.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional


class TaskStatus(Enum):
    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()


@dataclass
class TaskResult:
    """Result object returned by FinetuneTask.execute() to Simulator."""

    task_id: str
    task_type: str
    status: TaskStatus
    metrics_history: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
    elapsed_seconds: float = 0.0

    # Fill up by _build_summary() in FinetuneTask; can be customized by subclasses
    summary: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LoraParams:
    """General LoRA configuration parameters."""

    rank: int = 16
    train_mlp: bool = True
    train_attn: bool = True
    train_unembed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rank": self.rank,
            "train_mlp": self.train_mlp,
            "train_attn": self.train_attn,
            "train_unembed": self.train_unembed,
        }


class FinetuneTask(ABC):
    """
    Abstract base class for all finetune tasks.

    Subclasses must implement:
        - task_type:  Class attribute, task type name string
        - setup()  :  Initialize connections, load datasets, create training_client
        - run()    :  Execute full training, return metrics_history
        - teardown(): Cleanup resources
    """

    # example: sft, rlhf
    task_type: str = "base"

    def __init__(
        self,
        task_id: str,
        base_url: str,
        api_key: str,
        base_model: str,
        lora_params: LoraParams,
        num_steps: int,
        learning_rate: float,
        train_batch: int = 16,
        eval_batch: int = 16,
        max_length: int = 512,
        seed: int = 42,
        dataset: str = "",
        extra_cfg: Optional[Dict[str, Any]] = None,
    ):
        self.task_id = task_id
        self.base_url = base_url
        self.api_key = api_key
        self.base_model = base_model
        self.lora_params = lora_params
        self.num_steps = num_steps
        self.learning_rate = learning_rate
        self.train_batch = train_batch
        self.eval_batch = eval_batch
        self.max_length = max_length
        self.seed = seed
        self.dataset = dataset
        self.extra_cfg = extra_cfg or {}

        self.service_client = None
        self.training_client = None

    # ------------------------------------------------------------------
    # Core abstract methods to implement by subclasses
    # ------------------------------------------------------------------

    @abstractmethod
    def setup(self) -> None:
        """
        Initialization phase:
          - Connect to tinker service
          - Load datasets / tokenizer
          - Create training_client
        """

    @abstractmethod
    def run(self) -> List[Dict[str, Any]]:
        """
        Execute full training, return metrics_history.
        """

    def teardown(self) -> None:  # noqa: B027
        """
        Cleanup resources.
        """

    # ------------------------------------------------------------------
    # Called by Simulator
    # ------------------------------------------------------------------

    def execute(self) -> TaskResult:
        """
        Simulator calls this method to run the task.
        It handles the overall flow and error catching,
        while delegating specifics to setup/run/teardown.
        """
        result = TaskResult(
            task_id=self.task_id,
            task_type=self.task_type,
            status=TaskStatus.RUNNING,
        )
        t0 = time.time()

        try:
            self.setup()
            metrics_history = self.run()
            result.metrics_history = metrics_history
            result.summary = self._build_summary(metrics_history)
            result.status = TaskStatus.COMPLETED
        except Exception as exc:
            import traceback

            result.status = TaskStatus.FAILED
            result.error = traceback.format_exc()
            print(f"[Task {self.task_id}] FAILED: {exc}")
        finally:
            self.teardown()
            result.elapsed_seconds = time.time() - t0

        return result

    # ------------------------------------------------------------------
    # Optional override methods for customization
    # ------------------------------------------------------------------

    def _build_summary(self, metrics_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Default summary builder: extracts first/last step metrics
        and total steps. Subclasses can override."""
        if not metrics_history:
            return {}
        return {
            "first_step": metrics_history[0],
            "last_step": metrics_history[-1],
            "total_steps": len(metrics_history),
        }

    # ------------------------------------------------------------------
    # General utility methods for subclasses
    # ------------------------------------------------------------------

    def _connect(self):
        """Create and return a tinker service client."""
        import tinker

        print(f"[{self.task_id}] connect -> {self.base_url}")
        return tinker.ServiceClient(base_url=self.base_url, api_key=self.api_key)

    def _create_training_client(self, service_client, lora_params: Optional[LoraParams] = None):
        """Create a LoRA training client, using passed parameters
        or falling back to self.lora_params."""
        p = lora_params or self.lora_params
        print(f"[{self.task_id}] create_lora_training_client rank={p.rank}")
        return service_client.create_lora_training_client(
            base_model=self.base_model,
            rank=p.rank,
            train_mlp=p.train_mlp,
            train_attn=p.train_attn,
            train_unembed=p.train_unembed,
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"id={self.task_id!r}, "
            f"model={self.base_model!r}, "
            f"steps={self.num_steps}, "
            f"lora_rank={self.lora_params.rank})"
        )
