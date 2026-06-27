"""HumanEval task: function-level code generation with test execution."""

from __future__ import annotations

import re
import subprocess
import sys
from typing import Any, Dict, List, Optional

import datasets

from .base import Task, TaskPrompt


HUMANEVAL_INSTRUCTION = (
    "Complete the following Python function. "
    "Put your complete solution inside a ```python ... ``` code block.\n\n"
)

_CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", flags=re.DOTALL)


def _extract_code(response_text: str, prompt: str, entry_point: str) -> Optional[str]:
    """Extract Python code from the model response.

    Tries multiple strategies:
    1. Extract from ```python ... ``` code blocks
    2. Extract from truncated code block (missing closing ```)
    3. Fall back to raw response prefixed with the original prompt
    """
    # Strategy 1: complete code blocks
    matches = list(_CODE_BLOCK_RE.finditer(response_text))
    if matches:
        code = matches[-1].group(1).strip()
        # If the code block contains the full function, use it directly
        if f"def {entry_point}" in code:
            return code
        # Otherwise, it might be just the body — prepend the prompt
        return prompt + code

    # Strategy 2: truncated code block
    trunc_match = re.search(r"```(?:python)?\s*\n(.*)", response_text, flags=re.DOTALL)
    if trunc_match:
        candidate = trunc_match.group(1).strip()
        if f"def {entry_point}" in candidate:
            return candidate
        return prompt + candidate

    # Strategy 3: raw response (model continued the prompt directly)
    stripped = response_text.strip()
    if stripped:
        if f"def {entry_point}" in stripped:
            return stripped
        return prompt + stripped

    return None


def _execute_humaneval_tests(
    code: str, test_code: str, entry_point: str, timeout: float = 10.0
) -> bool:
    """Execute HumanEval test suite against the generated code.

    HumanEval tests are structured as:
        def check(candidate):
            assert candidate(...) == ...
        check(<entry_point>)

    Returns True if all tests pass, False otherwise.
    """
    full_code = code + "\n\n" + test_code + f"\ncheck({entry_point})\n"
    try:
        result = subprocess.run(
            [sys.executable, "-c", full_code],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False


class HumanEvalTask(Task):
    """HumanEval function-level code generation task.

    Dataset: openai/openai_humaneval (164 problems)
    Reward: binary (1.0 if all tests pass, 0.0 otherwise)
    """

    def __init__(self, dataset_name: str = "openai/openai_humaneval", seed: int = 42):
        self._dataset_name = dataset_name
        self._seed = seed
        self._train_data: List[Dict[str, Any]] = []
        self._test_data: List[Dict[str, Any]] = []
        self._train_idx = 0
        self._test_idx = 0

    def load_dataset(self) -> None:
        ds = datasets.load_dataset(self._dataset_name, split="test")

        all_data = [
            {
                "task_id": row["task_id"],
                "prompt": row["prompt"],
                "canonical_solution": row["canonical_solution"],
                "test": row["test"],
                "entry_point": row["entry_point"],
            }
            for row in ds
        ]

        # HumanEval only has a "test" split (164 problems).
        # Use 80% for training prompts and 20% for evaluation.
        import random

        rng = random.Random(self._seed)
        indices = list(range(len(all_data)))
        rng.shuffle(indices)

        split_point = int(len(all_data) * 0.8)
        self._train_data = [all_data[i] for i in indices[:split_point]]
        self._test_data = [all_data[i] for i in indices[split_point:]]

    def get_prompt(self) -> TaskPrompt:
        if not self._train_data:
            self.load_dataset()

        item = self._train_data[self._train_idx % len(self._train_data)]
        self._train_idx += 1

        prompt_text = HUMANEVAL_INSTRUCTION + item["prompt"]

        return TaskPrompt(
            text=prompt_text,
            metadata={
                "function_prompt": item["prompt"],
                "test": item["test"],
                "entry_point": item["entry_point"],
                "canonical_solution": item["canonical_solution"],
            },
        )

    def compute_reward(self, response_text: str, metadata: Dict[str, Any]) -> float:
        function_prompt = metadata["function_prompt"]
        test_code = metadata["test"]
        entry_point = metadata["entry_point"]

        code = _extract_code(response_text, function_prompt, entry_point)
        if code is None:
            return 0.0

        if _execute_humaneval_tests(code, test_code, entry_point):
            return 1.0
        return 0.0

    def get_eval_prompts(self, n: int) -> List[TaskPrompt]:
        if not self._test_data:
            self.load_dataset()

        prompts = []
        for i in range(min(n, len(self._test_data))):
            idx = (self._test_idx + i) % len(self._test_data)
            item = self._test_data[idx]
            prompt_text = HUMANEVAL_INSTRUCTION + item["prompt"]
            prompts.append(
                TaskPrompt(
                    text=prompt_text,
                    metadata={
                        "function_prompt": item["prompt"],
                        "test": item["test"],
                        "entry_point": item["entry_point"],
                        "canonical_solution": item["canonical_solution"],
                    },
                )
            )
        self._test_idx = (self._test_idx + n) % len(self._test_data)
        return prompts
