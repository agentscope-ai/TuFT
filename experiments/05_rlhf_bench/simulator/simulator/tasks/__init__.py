from .agent_task import AgentTask, EpisodeTrace, StepResult
from .base import Task, TaskPrompt


TASK_REGISTRY = {
    "countdown": "simulator.tasks.countdown.CountdownTask",
    "gsm8k": "simulator.tasks.gsm8k.GSM8KTask",
    "math": "simulator.tasks.math_task.MATHTask",
    "mbpp": "simulator.tasks.mbpp.MBPPTask",
    "humaneval": "simulator.tasks.humaneval.HumanEvalTask",
    "ifeval": "simulator.tasks.ifeval.IFEvalTask",
    "hotpotqa": "simulator.tasks.hotpotqa.HotpotQATask",
    "math_agent": "simulator.tasks.math_agent.MathAgentTask",
    "triviaqa": "simulator.tasks.triviaqa.TriviaQATask",
    "apibank": "simulator.tasks.apibank.APIBankTask",
    "toolbench": "simulator.tasks.toolbench.ToolBenchTask",
}


def create_task(task_name: str, **kwargs) -> Task:
    """Create a task instance by name. Imports are deferred to avoid heavy deps at module level."""
    if task_name not in TASK_REGISTRY:
        raise ValueError(f"Unknown task: {task_name}. Available: {list(TASK_REGISTRY.keys())}")

    if task_name == "countdown":
        from .countdown import CountdownTask

        return CountdownTask(**kwargs)
    elif task_name == "gsm8k":
        from .gsm8k import GSM8KTask

        return GSM8KTask(**kwargs)
    elif task_name == "math":
        from .math_task import MATHTask

        return MATHTask(**kwargs)
    elif task_name == "mbpp":
        from .mbpp import MBPPTask

        return MBPPTask(**kwargs)
    elif task_name == "humaneval":
        from .humaneval import HumanEvalTask

        return HumanEvalTask(**kwargs)
    elif task_name == "ifeval":
        from .ifeval import IFEvalTask

        return IFEvalTask(**kwargs)
    elif task_name == "hotpotqa":
        from .hotpotqa import HotpotQATask

        return HotpotQATask(**kwargs)
    elif task_name == "math_agent":
        from .math_agent import MathAgentTask

        return MathAgentTask(**kwargs)
    elif task_name == "triviaqa":
        from .triviaqa import TriviaQATask

        return TriviaQATask(**kwargs)
    elif task_name == "apibank":
        from .apibank import APIBankTask

        return APIBankTask(**kwargs)
    elif task_name == "toolbench":
        from .toolbench import ToolBenchTask

        return ToolBenchTask(**kwargs)


__all__ = [
    "Task",
    "TaskPrompt",
    "AgentTask",
    "StepResult",
    "EpisodeTrace",
    "TASK_REGISTRY",
    "create_task",
]
