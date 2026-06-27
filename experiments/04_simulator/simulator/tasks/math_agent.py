"""Math Agent RL task: arithmetic problem solving with Calculator tool.

The agent solves GSM8K-style math problems using a step-by-step ReAct loop
with a Calculator tool for arithmetic operations. This tests the agent's
ability to decompose problems and leverage tools for computation.

Tools available to the agent:
- Calculate[expression]: Evaluate an arithmetic expression (e.g., Calculate[3 * 15 + 7])
- Finish[number]: Submit the final numerical answer

Reward:
- 1.0 if final answer matches ground truth exactly
- 0.5 partial credit if within 1% relative error
- 0.1 format reward for calling Finish[] with any number
"""

from __future__ import annotations

import random
import re
from typing import Any, Dict, List, Optional

import datasets

from .agent_task import AgentTask, StepResult
from .base import TaskPrompt


# ─── System prompt ─────────────────────────────────────────────────

MATH_AGENT_SYSTEM_PROMPT = """\
You are a math problem solver. Solve the given problem step by step.

You have access to the following tools:
- Calculate[expression]: Evaluate a math expression. Supports +, -, *, /, **, and parentheses.
- Finish[number]: Submit your final numerical answer.

Use the following format for each step:
Thought: <your reasoning about the next calculation needed>
Action: <one of Calculate[...] or Finish[...]>

Rules:
- Break the problem into small calculation steps.
- Use Calculate[] for ALL arithmetic — do NOT compute in your head.
- Call Finish[number] with ONLY the final number (no units, no text).
"""

MATH_AGENT_QUESTION_TEMPLATE = "Problem: {question}\n\nThought:"

# ─── Regex ──────────────────────────────────────────────────────────

_ACTION_RE = re.compile(r"Action:\s*(Calculate|Finish)\[(.+?)\]", flags=re.IGNORECASE | re.DOTALL)
_ALLOWED_CALC_RE = re.compile(r"^[\d+\-*/().,%\s\^eE]+$")
_NUMBER_RE = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def _safe_calculate(expression: str) -> Optional[float]:
    """Safely evaluate a math expression."""
    # Normalize common patterns
    expr = expression.strip()
    expr = expr.replace("^", "**")
    expr = expr.replace(",", "")
    expr = expr.replace("%", "/100")

    if not _ALLOWED_CALC_RE.match(expr):
        return None

    try:
        result = eval(expr, {"__builtins__": None}, {})
        if isinstance(result, (int, float)) and not (
            result != result  # NaN check
        ):
            return float(result)
        return None
    except Exception:
        return None


def _extract_number(text: str) -> Optional[float]:
    """Extract a number from text."""
    text = text.strip().replace(",", "").replace("$", "")
    match = _NUMBER_RE.search(text)
    if match:
        try:
            return float(match.group())
        except ValueError:
            return None
    return None


def _extract_gsm8k_answer(solution: str) -> Optional[float]:
    """Extract the final answer from GSM8K solution format (#### NUMBER)."""
    match = re.search(r"####\s*([-+]?[\d,]+\.?\d*)", solution)
    if match:
        return float(match.group(1).replace(",", ""))
    return None


class MathAgentTask(AgentTask):
    """Math problem solving with Calculator tool.

    Dataset: openai/gsm8k (reused as agent task)
    Interaction: Multi-turn with Calculate/Finish tools
    Reward: Exact match on final numerical answer
    """

    def __init__(self, dataset_name: str = "openai/gsm8k", seed: int = 42):
        self._dataset_name = dataset_name
        self._seed = seed
        self._train_data: List[Dict[str, Any]] = []
        self._test_data: List[Dict[str, Any]] = []
        self._train_idx = 0
        self._test_idx = 0

    def load_dataset(self) -> None:
        """Load GSM8K dataset."""
        ds = datasets.load_dataset(self._dataset_name, "main", split="train")

        all_data = []
        for row in ds:
            answer = _extract_gsm8k_answer(row["answer"])
            if answer is not None:
                all_data.append(
                    {
                        "question": row["question"],
                        "answer": answer,
                        "solution": row["answer"],
                    }
                )

        # Shuffle and split
        rng = random.Random(self._seed)
        rng.shuffle(all_data)

        split_point = int(len(all_data) * 0.9)
        self._train_data = all_data[:split_point]
        self._test_data = all_data[split_point:]

    def reset_episode(self) -> TaskPrompt:
        """Start a new math problem episode."""
        if not self._train_data:
            self.load_dataset()

        item = self._train_data[self._train_idx % len(self._train_data)]
        self._train_idx += 1

        prompt_text = (
            MATH_AGENT_SYSTEM_PROMPT
            + "\n"
            + MATH_AGENT_QUESTION_TEMPLATE.format(question=item["question"])
        )

        return TaskPrompt(
            text=prompt_text,
            metadata={
                "question": item["question"],
                "answer": item["answer"],
                "solution": item["solution"],
            },
        )

    def step(self, action_text: str, metadata: Dict[str, Any]) -> StepResult:
        """Execute Calculate or Finish action."""
        match = _ACTION_RE.search(action_text)
        if not match:
            return StepResult(
                observation="Invalid action format. Use: Action: Calculate[expression] or Action: Finish[number]",  # noqa: E501
                reward=0.0,
                done=False,
                info={"action_type": "invalid"},
            )

        action_type = match.group(1).lower()
        action_arg = match.group(2).strip()

        if action_type == "calculate":
            return self._do_calculate(action_arg)
        elif action_type == "finish":
            return self._do_finish(action_arg, metadata)
        else:
            return StepResult(
                observation=f"Unknown action: {action_type}",
                reward=0.0,
                done=False,
                info={"action_type": "unknown"},
            )

    def _do_calculate(self, expression: str) -> StepResult:
        """Execute a calculation."""
        result = _safe_calculate(expression)
        if result is None:
            return StepResult(
                observation=f"Error: Cannot evaluate '{expression}'. Use valid arithmetic (+, -, *, /, **, parentheses).",  # noqa: E501
                reward=0.0,
                done=False,
                info={"action_type": "calculate", "success": False},
            )

        # Format result nicely
        if result == int(result):
            result_str = str(int(result))
        else:
            result_str = f"{result:.6g}"

        return StepResult(
            observation=f"Result: {result_str}",
            reward=0.0,
            done=False,
            info={"action_type": "calculate", "success": True, "result": result},
        )

    def _do_finish(self, answer_text: str, metadata: Dict[str, Any]) -> StepResult:
        """Submit the final answer."""
        ground_truth = metadata["answer"]

        predicted = _extract_number(answer_text)
        if predicted is None:
            return StepResult(
                observation=f"Episode complete. Your answer: {answer_text}",
                reward=0.05,
                done=True,
                info={"action_type": "finish", "answer": answer_text, "valid_number": False},
            )

        # Check exact match (with tolerance for floating point)
        if abs(predicted - ground_truth) < 1e-3:
            reward = 1.0
        elif ground_truth != 0 and abs(predicted - ground_truth) / abs(ground_truth) < 0.01:
            # Within 1% relative error
            reward = 0.5
        else:
            reward = 0.1  # At least finished with a number

        return StepResult(
            observation=f"Episode complete. Your answer: {predicted}",
            reward=reward,
            done=True,
            info={
                "action_type": "finish",
                "answer": predicted,
                "ground_truth": ground_truth,
                "exact_match": abs(predicted - ground_truth) < 1e-3,
            },
        )

    def compute_episode_reward(self, steps: List[StepResult], metadata: Dict[str, Any]) -> float:
        """Episode reward from Finish action."""
        for step in reversed(steps):
            if step.info.get("action_type") == "finish":
                return step.reward
        return 0.0  # Didn't finish

    def get_eval_prompts(self, n: int) -> List[TaskPrompt]:
        """Get evaluation prompts."""
        if not self._test_data:
            self.load_dataset()

        prompts = []
        for i in range(min(n, len(self._test_data))):
            idx = (self._test_idx + i) % len(self._test_data)
            item = self._test_data[idx]
            prompt_text = (
                MATH_AGENT_SYSTEM_PROMPT
                + "\n"
                + MATH_AGENT_QUESTION_TEMPLATE.format(question=item["question"])
            )
            prompts.append(
                TaskPrompt(
                    text=prompt_text,
                    metadata={
                        "question": item["question"],
                        "answer": item["answer"],
                        "solution": item["solution"],
                    },
                )
            )
        self._test_idx = (self._test_idx + n) % len(self._test_data)
        return prompts
