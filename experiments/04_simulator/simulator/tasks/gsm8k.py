"""GSM8K task: grade school math word problems."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import datasets

from .base import Task, TaskPrompt


GSM8K_INSTRUCTION = (
    "Solve the following math problem step by step. "
    "Put your final numerical answer inside \\boxed{}.\n\n"
)

_BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")


def _extract_boxed_answer(text: str) -> Optional[str]:
    """Extract the last \\boxed{...} content from response."""
    matches = list(_BOXED_RE.finditer(text))
    if not matches:
        return None
    return matches[-1].group(1).strip()


def _normalize_number(s: str) -> Optional[float]:
    """Normalize a number string to float for comparison."""
    try:
        s = s.replace(",", "").replace(" ", "").strip()
        # Handle negative
        s = s.replace("−", "-")
        return float(s)
    except (ValueError, TypeError):
        return None


def _extract_gsm8k_answer(answer_text: str) -> Optional[float]:
    """Extract the numeric answer from GSM8K answer field.

    GSM8K answers end with '#### <number>'.
    """
    match = re.search(r"####\s*(.+)$", answer_text)
    if match:
        return _normalize_number(match.group(1))
    return None


class GSM8KTask(Task):
    """GSM8K grade school math task."""

    def __init__(self, dataset_name: str = "openai/gsm8k", seed: int = 42):
        self._dataset_name = dataset_name
        self._seed = seed
        self._train_data: List[Dict[str, Any]] = []
        self._test_data: List[Dict[str, Any]] = []
        self._train_idx = 0
        self._test_idx = 0

    def load_dataset(self) -> None:
        train_ds = datasets.load_dataset(self._dataset_name, "main", split="train")
        test_ds = datasets.load_dataset(self._dataset_name, "main", split="test")

        train_ds = train_ds.shuffle(seed=self._seed)

        self._train_data = [
            {"question": row["question"], "answer": row["answer"]} for row in train_ds
        ]
        self._test_data = [
            {"question": row["question"], "answer": row["answer"]} for row in test_ds
        ]

    def get_prompt(self) -> TaskPrompt:
        if not self._train_data:
            self.load_dataset()

        item = self._train_data[self._train_idx % len(self._train_data)]
        self._train_idx += 1

        prompt_text = GSM8K_INSTRUCTION + f"Q: {item['question']}\nA:"

        target_value = _extract_gsm8k_answer(item["answer"])
        return TaskPrompt(
            text=prompt_text,
            metadata={"answer_text": item["answer"], "target_value": target_value},
        )

    def compute_reward(self, response_text: str, metadata: Dict[str, Any]) -> float:
        target_value = metadata.get("target_value")
        if target_value is None:
            return 0.0

        predicted = _extract_boxed_answer(response_text)
        if predicted is None:
            return 0.0

        predicted_value = _normalize_number(predicted)
        if predicted_value is None:
            return 0.0

        if abs(predicted_value - target_value) < 1e-5:
            return 1.0

        return 0.0

    def get_eval_prompts(self, n: int) -> List[TaskPrompt]:
        if not self._test_data:
            self.load_dataset()

        prompts = []
        for i in range(min(n, len(self._test_data))):
            idx = (self._test_idx + i) % len(self._test_data)
            item = self._test_data[idx]
            prompt_text = GSM8K_INSTRUCTION + f"Q: {item['question']}\nA:"
            target_value = _extract_gsm8k_answer(item["answer"])
            prompts.append(
                TaskPrompt(
                    text=prompt_text,
                    metadata={"answer_text": item["answer"], "target_value": target_value},
                )
            )
        self._test_idx = (self._test_idx + n) % len(self._test_data)
        return prompts
