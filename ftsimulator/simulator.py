"""
simulator.py
A flexible simulator that can launch multiple concurrent finetuning tasks of different types.
"""

from __future__ import annotations

import concurrent.futures
import os
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type, TypeVar

from ft_tasks.ft_base import FinetuneTask, LoraParams, TaskResult, TaskStatus


# Registry for task types
TaskClass = TypeVar("TaskClass", bound=FinetuneTask)


class TaskRegistry:
    """
    Registry for task types. Allows dynamic registration of new task types
    without modifying the simulator code.
    """

    _registry: Dict[str, Type[FinetuneTask]] = {}

    @classmethod
    def register(cls, task_type: str, task_class: Type[FinetuneTask]) -> None:
        """Register a new task type."""
        if task_type in cls._registry:
            raise ValueError(f"Task type '{task_type}' is already registered")
        cls._registry[task_type] = task_class
        print(f"[TaskRegistry] Registered task type: {task_type} -> {task_class.__name__}")

    @classmethod
    def get(cls, task_type: str) -> Type[FinetuneTask]:
        """Get task class by type name."""
        if task_type not in cls._registry:
            available = ", ".join(cls._registry.keys())
            raise KeyError(f"Unknown task type '{task_type}'. Available: {available}")
        return cls._registry[task_type]

    @classmethod
    def list_types(cls) -> List[str]:
        """List all registered task types."""
        return list(cls._registry.keys())

    @classmethod
    def is_registered(cls, task_type: str) -> bool:
        """Check if a task type is registered."""
        return task_type in cls._registry


# Auto-discover and register built-in tasks
def _register_builtin_tasks():
    """Automatically register built-in task types."""
    try:
        from ft_tasks.sft.sft_chat import SFTChatTask

        TaskRegistry.register("sft_chat", SFTChatTask)
    except ImportError as e:
        print(f"[TaskRegistry] Warning: Could not import SFTChatTask: {e}")

    try:
        from ft_tasks.rlhf.rlhf_countdown import RLHFCountdownTask

        TaskRegistry.register("rlhf_countdown", RLHFCountdownTask)
    except ImportError as e:
        print(f"[TaskRegistry] Warning: Could not import RLHFCountdownTask: {e}")


# Register built-in tasks on module load
_register_builtin_tasks()


@dataclass
class TaskConfig:
    """
    Configuration for a single task instance.
    Flexible dataclass that accepts any task-specific parameters.
    """

    task_type: str
    count: int = 1  # Number of instances to create

    # Common parameters (used by all tasks)
    base_url: str = "http://localhost:8000"
    api_key: str = ""
    base_model: str = "meta-llama/Llama-2-7b-hf"
    lora_params: LoraParams = field(default_factory=LoraParams)
    num_steps: int = 100
    learning_rate: float = 1e-4
    train_batch: int = 16
    eval_batch: int = 16
    max_length: int = 512
    seed: int = 42
    dataset: str = ""

    # Task-specific extra configuration
    extra_cfg: Dict[str, Any] = field(default_factory=dict)

    # Optional: custom task ID prefix (auto-generated if not provided)
    task_id_prefix: Optional[str] = None

    def to_task_kwargs(self, task_idx: int) -> Dict[str, Any]:
        """Convert to kwargs for FinetuneTask constructor."""
        # Generate unique task ID
        prefix = self.task_id_prefix or f"{self.task_type}"
        task_id = f"{prefix}_{uuid.uuid4().hex[:8]}_{task_idx}"

        return {
            "task_id": task_id,
            "base_url": self.base_url,
            "api_key": self.api_key,
            "base_model": self.base_model,
            "lora_params": self.lora_params,
            "num_steps": self.num_steps,
            "learning_rate": self.learning_rate,
            "train_batch": self.train_batch,
            "eval_batch": self.eval_batch,
            "max_length": self.max_length,
            "seed": self.seed + task_idx,  # Auto-increment seed for diversity
            "dataset": self.dataset,
            "extra_cfg": self.extra_cfg,
        }


@dataclass
class SimulatorConfig:
    """
    Configuration for the simulator.
    Contains multiple task configurations to run concurrently.
    """

    # List of task configurations
    tasks: List[TaskConfig] = field(default_factory=list)

    # Execution parameters
    max_workers: Optional[int] = None  # None = use ThreadPoolExecutor default

    # Optional: global defaults that can be overridden per-task
    base_url: str = "http://localhost:10610"
    api_key: str = "tml-tuft-dev-key"

    def add_task(self, task_type: str, count: int = 1, **kwargs) -> TaskConfig:
        """Add a task configuration. Returns the config for further modification."""
        # Apply global defaults for unspecified common params
        if "base_url" not in kwargs:
            kwargs["base_url"] = self.base_url
        if "api_key" not in kwargs:
            kwargs["api_key"] = self.api_key

        config = TaskConfig(task_type=task_type, count=count, **kwargs)
        self.tasks.append(config)
        return config


@dataclass
class SimulatorResult:
    """Result from running the simulator."""

    all_results: List[TaskResult] = field(default_factory=list)
    completed: List[TaskResult] = field(default_factory=list)
    failed: List[TaskResult] = field(default_factory=list)

    def __post_init__(self):
        # Categorize results
        self.completed = [r for r in self.all_results if r.status == TaskStatus.COMPLETED]
        self.failed = [r for r in self.all_results if r.status == TaskStatus.FAILED]

    @property
    def success_rate(self) -> float:
        if not self.all_results:
            return 0.0
        return len(self.completed) / len(self.all_results)

    def summary(self) -> Dict[str, Any]:
        """Generate a summary of all results."""
        by_type: Dict[str, Dict[str, Any]] = {}

        for result in self.all_results:
            ttype = result.task_type
            if ttype not in by_type:
                by_type[ttype] = {"total": 0, "completed": 0, "failed": 0, "avg_time": 0.0}

            by_type[ttype]["total"] += 1
            if result.status == TaskStatus.COMPLETED:
                by_type[ttype]["completed"] += 1
            else:
                by_type[ttype]["failed"] += 1
            by_type[ttype]["avg_time"] += result.elapsed_seconds

        # Compute averages
        for stats in by_type.values():
            if stats["total"] > 0:
                stats["avg_time"] /= stats["total"]

        return {
            "total_tasks": len(self.all_results),
            "completed": len(self.completed),
            "failed": len(self.failed),
            "success_rate": self.success_rate,
            "by_type": by_type,
        }

    def print_summary(self) -> None:
        """Print a formatted summary."""
        summary = self.summary()
        print("\n" + "=" * 60)
        print("SIMULATOR RESULTS SUMMARY")
        print("=" * 60)
        print(f"Total tasks:  {summary['total_tasks']}")
        print(f"Completed:    {summary['completed']}")
        print(f"Failed:       {summary['failed']}")
        print(f"Success rate: {summary['success_rate']:.1%}")
        print("-" * 60)
        print("By task type:")
        for ttype, stats in summary["by_type"].items():
            print(f"  {ttype}:")
            print(
                f"    Total: {stats['total']}, "
                f"OK: {stats['completed']}, "
                f"Fail: {stats['failed']}, "
                f"Avg time: {stats['avg_time']:.2f}s"
            )
        print("=" * 60)

        if self.failed:
            print("\nFailed tasks:")
            for r in self.failed:
                print(f"  - {r.task_id}: {r.error[:200] if r.error else 'Unknown error'}...")


class Simulator:
    """
    Main simulator class that manages concurrent execution of multiple
    finetuning tasks of potentially different types.
    """

    def __init__(self, config: SimulatorConfig):
        self.config = config
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate that all task types are registered."""
        for task_cfg in self.config.tasks:
            if not TaskRegistry.is_registered(task_cfg.task_type):
                available = ", ".join(TaskRegistry.list_types())
                raise ValueError(
                    f"Task type '{task_cfg.task_type}' not registered. Available: {available}"
                )

    def _create_task(self, task_cfg: TaskConfig, task_idx: int) -> FinetuneTask:
        """Create a single task instance."""
        task_class = TaskRegistry.get(task_cfg.task_type)
        kwargs = task_cfg.to_task_kwargs(task_idx)
        return task_class(**kwargs)

    def _run_single_task(self, task: FinetuneTask) -> TaskResult:
        """Execute a single task (runs in worker thread/process)."""
        print(f"[Simulator] Starting {task.task_id} ({task.task_type})")
        result = task.execute()
        status_str = "COMPLETED" if result.status == TaskStatus.COMPLETED else "FAILED"
        print(f"[Simulator] Finished {task.task_id}: {status_str} in {result.elapsed_seconds:.2f}s")
        return result

    def run(self) -> SimulatorResult:
        """
        Execute all configured tasks concurrently.

        Returns:
            SimulatorResult containing all task results.
        """
        # Build list of all task instances to run
        all_tasks: List[FinetuneTask] = []
        for task_cfg in self.config.tasks:
            for i in range(task_cfg.count):
                task = self._create_task(task_cfg, i)
                all_tasks.append(task)

        print(
            f"[Simulator] Launching {len(all_tasks)} total tasks "
            f"with max_workers={self.config.max_workers}"
        )

        # Execute tasks concurrently
        all_results: List[TaskResult] = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(self._run_single_task, task): task for task in all_tasks
            }

            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result()
                    all_results.append(result)
                except Exception as exc:
                    # Handle unexpected errors in task execution wrapper
                    import traceback

                    error_result = TaskResult(
                        task_id=task.task_id,
                        task_type=task.task_type,
                        status=TaskStatus.FAILED,
                        error=f"Simulator execution error: {traceback.format_exc()}",
                    )
                    all_results.append(error_result)
                    print(f"[Simulator] Critical error in {task.task_id}: {exc}")

        return SimulatorResult(all_results=all_results)

    def run_sequential(self) -> SimulatorResult:
        """
        Run all tasks sequentially (useful for debugging).
        """
        all_results: List[TaskResult] = []

        # Build and run tasks one by one
        for task_cfg in self.config.tasks:
            for i in range(task_cfg.count):
                task = self._create_task(task_cfg, i)
                result = self._run_single_task(task)
                all_results.append(result)

        return SimulatorResult(all_results=all_results)


# ============================================================================
# Convenience functions for common use cases
# ============================================================================


def create_simulator(
    max_workers: Optional[int] = None,
    base_url: str = "http://localhost:8000",
    api_key: str = "",
) -> SimulatorConfig:
    """Create a new simulator configuration with defaults."""
    return SimulatorConfig(
        max_workers=max_workers,
        base_url=base_url,
        api_key=api_key,
    )


def quick_run(
    task_configs: List[Dict[str, Any]], max_workers: Optional[int] = None, **global_defaults
) -> SimulatorResult:
    """
    Quick run function for simple use cases.

    Args:
        task_configs: List of dicts, each with 'task_type', 'count', and other params
        max_workers: Max concurrent workers
        **global_defaults: Default values for all tasks (base_url, api_key, etc.)

    Example:
        results = quick_run([
            {"task_type": "sft_chat", "count": 2, "dataset": "no_robots"},
            {"task_type": "rlhf_countdown", "count": 1, "dataset": "Jiayi-Pan/Countdown"},
        ], max_workers=4, base_url="http://localhost:8000")
    """
    sim_config = SimulatorConfig(max_workers=max_workers, **global_defaults)

    for cfg in task_configs:
        task_type = cfg.pop("task_type")
        count = cfg.pop("count", 1)
        sim_config.add_task(task_type, count, **cfg)

    simulator = Simulator(sim_config)
    return simulator.run()


# ============================================================================
# Example usage and demonstration
# ============================================================================

if __name__ == "__main__":
    # Example 1: Using the config-based API
    print("=" * 60)
    print("Example 1: Config-based API")
    print("=" * 60)

    config = SimulatorConfig(
        max_workers=4,
        api_key=os.environ["TINKER_API_KEY"],
        base_url=os.environ["TINKER_BASE_URL"],
    )

    # Add SFT tasks
    config.add_task(
        task_type="sft_chat",
        count=2,
        base_model="Qwen/Qwen3-4B",
        dataset="no_robots",
        num_steps=50,  # Reduced for demo
        lora_params=LoraParams(rank=8),
    )

    # Add RLHF tasks
    config.add_task(
        task_type="rlhf_countdown",
        count=1,
        base_model="Qwen/Qwen3-4B",
        dataset="Jiayi-Pan/Countdown-Tasks-3to4",
        num_steps=30,  # Reduced for demo
        learning_rate=5e-5,
        lora_params=LoraParams(rank=8),
        extra_cfg={
            "group_size": 8,
            "eval_every": 5,
        },
    )

    # Create and run simulator
    simulator = Simulator(config)
    results = simulator.run()  # Uncomment to actually run
    results.print_summary()

    print("\nConfiguration created successfully!")
    print(f"Task types registered: {TaskRegistry.list_types()}")

    # Example 2: Quick run API
    print("\n" + "=" * 60)
    print("Example 2: Quick run API")
    print("=" * 60)

    # results = quick_run(
    #     [
    #         {
    #             "task_type": "sft_chat", "count": 2,
    #             "dataset": "no_robots", "num_steps": 10,
    #             "base_model": "Qwen/Qwen3-4B",
    #             "lora_params": LoraParams(rank=8),
    #         },
    #         {
    #             "task_type": "rlhf_countdown", "count": 1,
    #             "dataset": "Jiayi-Pan/Countdown-Tasks-3to4",
    #             "num_steps": 10,
    #             "base_model": "Qwen/Qwen3-4B",
    #             "lora_params": LoraParams(rank=8),
    #         },
    #     ],
    #     max_workers=4,
    #     base_url="http://localhost:10610",
    # )
    # results.print_summary()

    print("\nQuick run API example shown (commented out)")

    # Example 3: Custom task registration
    print("\n" + "=" * 60)
    print("Example 3: Extending with custom tasks")
    print("=" * 60)

    # Define a custom task (normally in separate file)
    class CustomPretrainTask(FinetuneTask):
        task_type = "custom_pretrain"  # Will be auto-registered

        def setup(self) -> None:
            print(f"[{self.task_id}] Custom pretrain setup")
            self.service_client = self._connect()

        def run(self) -> List[Dict[str, Any]]:
            print(f"[{self.task_id}] Running custom pretraining")
            # Dummy implementation
            return [{"step": i, "loss": 1.0 / (i + 1)} for i in range(self.num_steps)]

    # Register the custom task
    TaskRegistry.register("custom_pretrain", CustomPretrainTask)

    print(f"After custom registration: {TaskRegistry.list_types()}")
