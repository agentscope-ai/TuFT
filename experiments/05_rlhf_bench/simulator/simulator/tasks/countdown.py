"""Countdown task: reach a target number using arithmetic expressions."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import datasets

from .base import Task, TaskPrompt


COUNTDOWN_INSTRUCTION = (
    "Using the numbers {nums_str}, reach the target number {target}. "
    "You may use +, -, *, / and parentheses, and each number can only be used once. "
    "Put ONLY the final expression inside <answer>...</answer>. "
    "Example: <answer>(1+2)/3</answer>."
)

COUNTDOWN_FEWSHOT = (
    "Q: Using the numbers 2, 3, 7, reach the target number 13. "
    "You may use +, -, *, / and parentheses, and each number can only be used once. "
    "Put ONLY the final expression inside <answer>...</answer>. "
    "Example: <answer>(1+2)/3</answer>.\n"
    "A: <answer>(2*3)+7</answer>\n\n"
)

_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", flags=re.DOTALL)
_ALLOWED_EVAL_RE = re.compile(r"^[\d+\-*/().\s]+$")


def _extract_solution(text: str) -> Optional[str]:
    """Extract the last <answer>...</answer> content."""
    if "Assistant:" in text:
        text = text.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in text:
        text = text.split("<|im_start|>assistant", 1)[1]
    matches = list(_ANSWER_RE.finditer(text))
    if not matches:
        return None
    return matches[-1].group(1).strip()


def _validate_equation(equation_str: str, available_numbers: List[int]) -> bool:
    """Check if equation uses exactly the provided numbers."""
    try:
        numbers_in_eq = [int(n) for n in re.findall(r"\d+", equation_str)]
        return sorted(numbers_in_eq) == sorted(available_numbers)
    except Exception:
        return False


def _evaluate_equation(equation_str: str) -> Optional[float]:
    """Safely evaluate arithmetic expression."""
    try:
        if not _ALLOWED_EVAL_RE.match(equation_str):
            return None
        return eval(equation_str, {"__builtins__": None}, {})
    except Exception:
        return None


class CountdownTask(Task):
    """Countdown arithmetic task."""

    def __init__(self, dataset_name: str = "Jiayi-Pan/Countdown-Tasks-3to4", seed: int = 42):
        self._dataset_name = dataset_name
        self._seed = seed
        self._train_data: List[Dict[str, Any]] = []
        self._test_data: List[Dict[str, Any]] = []
        self._train_idx = 0
        self._test_idx = 0

    def load_dataset(self) -> None:
        ds = datasets.load_dataset(self._dataset_name, split="train")
        test_size = min(512, len(ds) // 10)

        test_ds = ds.select(range(test_size))
        train_ds = ds.select(range(test_size, len(ds))).shuffle(seed=self._seed)

        self._train_data = [
            {
                "target": int(row["target"]),
                "nums": list(row["nums"]),
            }
            for row in train_ds
        ]
        self._test_data = [
            {
                "target": int(row["target"]),
                "nums": list(row["nums"]),
            }
            for row in test_ds
        ]

    def get_prompt(self) -> TaskPrompt:
        if not self._train_data:
            self.load_dataset()

        item = self._train_data[self._train_idx % len(self._train_data)]
        self._train_idx += 1

        nums_str = ", ".join(map(str, item["nums"]))
        question = COUNTDOWN_INSTRUCTION.format(nums_str=nums_str, target=item["target"])
        prompt_text = COUNTDOWN_FEWSHOT + f"Q: {question}\nA:"

        return TaskPrompt(
            text=prompt_text,
            metadata={"target": item["target"], "nums": item["nums"]},
        )

    def compute_reward(self, response_text: str, metadata: Dict[str, Any]) -> float:
        target = metadata["target"]
        nums = metadata["nums"]

        equation = _extract_solution(response_text)
        if equation is None:
            return 0.0

        if not _validate_equation(equation, nums):
            return 0.1  # format score

        result = _evaluate_equation(equation)
        if result is None:
            return 0.1

        err = abs(result - target)
        if err < 1e-5:
            return 1.0

        # Continuous shaping
        return 0.1 + 0.9 * (1.0 / (1.0 + err))

    def get_eval_prompts(self, n: int) -> List[TaskPrompt]:
        if not self._test_data:
            self.load_dataset()

        prompts = []
        for i in range(min(n, len(self._test_data))):
            idx = (self._test_idx + i) % len(self._test_data)
            item = self._test_data[idx]
            nums_str = ", ".join(map(str, item["nums"]))
            question = COUNTDOWN_INSTRUCTION.format(nums_str=nums_str, target=item["target"])
            prompt_text = COUNTDOWN_FEWSHOT + f"Q: {question}\nA:"
            prompts.append(
                TaskPrompt(
                    text=prompt_text,
                    metadata={"target": item["target"], "nums": item["nums"]},
                )
            )
        self._test_idx = (self._test_idx + n) % len(self._test_data)
        return prompts
