"""MBPP task: Python programming problems with test execution."""

from __future__ import annotations

import re
import subprocess
import sys
from typing import Any, Dict, List, Optional

import datasets

from .base import Task, TaskPrompt


MBPP_INSTRUCTION = (
    "Write a Python function to solve the following problem. "
    "Put your complete solution inside a ```python ... ``` code block.\n\n"
)

_CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", flags=re.DOTALL)


def _extract_code(response_text: str) -> Optional[str]:
    """Extract Python code from markdown code blocks or bare function definitions."""
    # Match complete ```python ... ``` or ``` ... ``` blocks
    matches = list(_CODE_BLOCK_RE.finditer(response_text))
    if matches:
        return matches[-1].group(1).strip()

    # Fallback: handle truncated code block where closing ``` is missing
    # (model sometimes outputs code without the closing fence)
    trunc_match = re.search(r"```(?:python)?\s*\n(.*)", response_text, flags=re.DOTALL)
    if trunc_match:
        candidate = trunc_match.group(1).strip()
        if "def " in candidate:
            return candidate

    # Fallback: extract from first `def` to end of contiguous indented block
    lines = response_text.split("\n")
    code_lines: List[str] = []
    in_code = False
    for line in lines:
        if line.strip().startswith("def "):
            in_code = True
        if in_code:
            code_lines.append(line)
    if code_lines:
        return "\n".join(code_lines)
    return None


def _execute_tests(code: str, test_cases: List[str], timeout: float = 5.0) -> float:
    """Execute test cases against the code in a subprocess.

    Returns the fraction of test cases that pass [0, 1].
    """
    if not test_cases:
        return 0.0

    passed = 0
    for test in test_cases:
        full_code = code + "\n\n" + test
        try:
            result = subprocess.run(
                [sys.executable, "-c", full_code],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            if result.returncode == 0:
                passed += 1
        except subprocess.TimeoutExpired:
            continue
        except Exception:
            continue

    return passed / len(test_cases)


class MBPPTask(Task):
    """MBPP Python programming task."""

    def __init__(self, dataset_name: str = "google-research-datasets/mbpp", seed: int = 42):
        self._dataset_name = dataset_name
        self._seed = seed
        self._train_data: List[Dict[str, Any]] = []
        self._test_data: List[Dict[str, Any]] = []
        self._train_idx = 0
        self._test_idx = 0

    def load_dataset(self) -> None:
        # MBPP has 'train', 'validation', 'test' splits
        train_ds = datasets.load_dataset(self._dataset_name, "sanitized", split="train")
        test_ds = datasets.load_dataset(self._dataset_name, "sanitized", split="test")

        train_ds = train_ds.shuffle(seed=self._seed)

        self._train_data = [
            {
                "text": row["prompt"],
                "code": row["code"],
                "test_list": row["test_list"],
            }
            for row in train_ds
        ]
        self._test_data = [
            {
                "text": row["prompt"],
                "code": row["code"],
                "test_list": row["test_list"],
            }
            for row in test_ds
        ]

    def get_prompt(self) -> TaskPrompt:
        if not self._train_data:
            self.load_dataset()

        item = self._train_data[self._train_idx % len(self._train_data)]
        self._train_idx += 1

        prompt_text = MBPP_INSTRUCTION + f"Problem: {item['text']}\n\nSolution:"

        return TaskPrompt(
            text=prompt_text,
            metadata={
                "test_list": item["test_list"],
                "reference_code": item["code"],
            },
        )

    def compute_reward(self, response_text: str, metadata: Dict[str, Any]) -> float:
        test_cases = metadata.get("test_list", [])
        if not test_cases:
            return 0.0

        code = _extract_code(response_text)
        if code is None:
            return 0.0

        return _execute_tests(code, test_cases)

    def get_eval_prompts(self, n: int) -> List[TaskPrompt]:
        if not self._test_data:
            self.load_dataset()

        prompts = []
        for i in range(min(n, len(self._test_data))):
            idx = (self._test_idx + i) % len(self._test_data)
            item = self._test_data[idx]
            prompt_text = MBPP_INSTRUCTION + f"Problem: {item['text']}\n\nSolution:"
            prompts.append(
                TaskPrompt(
                    text=prompt_text,
                    metadata={
                        "test_list": item["test_list"],
                        "reference_code": item["code"],
                    },
                )
            )
        self._test_idx = (self._test_idx + n) % len(self._test_data)
        return prompts
