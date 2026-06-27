"""MATH task: competition-level mathematics problems."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import datasets

from .base import Task, TaskPrompt


MATH_INSTRUCTION = (
    "Solve the following math problem. Show your work step by step. "
    "Put your final answer inside \\boxed{}.\n\n"
)

_BOXED_RE = re.compile(r"\\boxed\{([^}]*(?:\{[^}]*\}[^}]*)*)\}")


def _extract_boxed_answer(text: str) -> Optional[str]:
    """Extract the last \\boxed{...} content, handling nested braces."""
    matches = list(_BOXED_RE.finditer(text))
    if not matches:
        return None
    return matches[-1].group(1).strip()


def _normalize_math_answer(answer: str) -> str:
    """Normalize a math answer for comparison."""
    # Remove surrounding whitespace
    answer = answer.strip()
    # Remove common LaTeX formatting
    answer = answer.replace("\\$", "")
    answer = answer.replace("\\%", "")
    answer = answer.replace("\\,", "")
    answer = answer.replace("\\;", "")
    answer = answer.replace("\\!", "")
    answer = answer.replace("\\ ", "")
    # Remove \\text{...}
    answer = re.sub(r"\\text\{([^}]*)\}", r"\1", answer)
    # Remove \\mathrm{...}
    answer = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", answer)
    # Normalize fractions: \\frac{a}{b} -> a/b
    answer = re.sub(r"\\frac\{([^}]*)\}\{([^}]*)\}", r"(\1)/(\2)", answer)
    # Remove \\left and \\right
    answer = answer.replace("\\left", "").replace("\\right", "")
    # Normalize whitespace
    answer = re.sub(r"\s+", " ", answer).strip()
    return answer


def _answers_match(predicted: str, target: str) -> bool:
    """Check if predicted answer matches target (with normalization)."""
    pred_norm = _normalize_math_answer(predicted)
    target_norm = _normalize_math_answer(target)

    # Direct string match
    if pred_norm == target_norm:
        return True

    # Try numeric comparison
    try:
        pred_val = float(pred_norm.replace(",", "").replace(" ", ""))
        target_val = float(target_norm.replace(",", "").replace(" ", ""))
        if abs(pred_val - target_val) < 1e-5:
            return True
    except (ValueError, TypeError):
        pass

    return False


class MATHTask(Task):
    """MATH competition-level math task."""

    # EleutherAI/hendrycks_math has separate configs per subject; load all subjects
    _SUBJECTS = [
        "algebra",
        "counting_and_probability",
        "geometry",
        "intermediate_algebra",
        "number_theory",
        "prealgebra",
        "precalculus",
    ]
    # Primary dataset name; falls back to hendrycks/competition_math if available
    _DEFAULT_DATASET = "EleutherAI/hendrycks_math"

    def __init__(self, dataset_name: str = _DEFAULT_DATASET, seed: int = 42):
        self._dataset_name = dataset_name
        self._seed = seed
        self._train_data: List[Dict[str, Any]] = []
        self._test_data: List[Dict[str, Any]] = []
        self._train_idx = 0
        self._test_idx = 0

    def load_dataset(self) -> None:
        import logging

        logger = logging.getLogger(__name__)

        train_rows: List[Dict[str, Any]] = []
        test_rows: List[Dict[str, Any]] = []

        # EleutherAI/hendrycks_math requires a subject config argument
        load_kwargs: dict = {}
        if self._dataset_name == self._DEFAULT_DATASET:
            for subject in self._SUBJECTS:
                try:
                    tr = datasets.load_dataset(self._dataset_name, subject, split="train")
                    te = datasets.load_dataset(self._dataset_name, subject, split="test")
                    train_rows.extend(
                        {"problem": r["problem"], "solution": r["solution"]} for r in tr
                    )
                    test_rows.extend(
                        {"problem": r["problem"], "solution": r["solution"]} for r in te
                    )
                except Exception as e:
                    logger.warning(f"MATH: failed to load subject '{subject}': {e}")
        else:
            # Generic fallback: try loading directly (e.g. hendrycks/competition_math)
            try:
                tr = datasets.load_dataset(self._dataset_name, split="train")
                te = datasets.load_dataset(self._dataset_name, split="test")
                train_rows = [{"problem": r["problem"], "solution": r["solution"]} for r in tr]
                test_rows = [{"problem": r["problem"], "solution": r["solution"]} for r in te]
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load MATH dataset '{self._dataset_name}': {e}"
                ) from e

        if not train_rows:
            raise RuntimeError(
                f"No training data loaded for MATH from '{self._dataset_name}'. "
                "Check network access or dataset name."
            )

        import random

        rng = random.Random(self._seed)
        rng.shuffle(train_rows)

        self._train_data = train_rows
        self._test_data = test_rows
        logger.info(f"MATH dataset loaded: train={len(train_rows)}, test={len(test_rows)}")

    def _extract_target_answer(self, solution: str) -> Optional[str]:
        """Extract the final boxed answer from the dataset solution."""
        return _extract_boxed_answer(solution)

    def get_prompt(self) -> TaskPrompt:
        if not self._train_data:
            self.load_dataset()

        item = self._train_data[self._train_idx % len(self._train_data)]
        self._train_idx += 1

        prompt_text = MATH_INSTRUCTION + f"Problem: {item['problem']}\n\nSolution:"

        target_answer = self._extract_target_answer(item["solution"])
        return TaskPrompt(
            text=prompt_text,
            metadata={"solution": item["solution"], "target_answer": target_answer},
        )

    def compute_reward(self, response_text: str, metadata: Dict[str, Any]) -> float:
        target_answer = metadata.get("target_answer")
        if target_answer is None:
            return 0.0

        predicted = _extract_boxed_answer(response_text)
        if predicted is None:
            return 0.0

        if _answers_match(predicted, target_answer):
            return 1.0

        return 0.0

    def get_eval_prompts(self, n: int) -> List[TaskPrompt]:
        if not self._test_data:
            self.load_dataset()

        prompts = []
        for i in range(min(n, len(self._test_data))):
            idx = (self._test_idx + i) % len(self._test_data)
            item = self._test_data[idx]
            prompt_text = MATH_INSTRUCTION + f"Problem: {item['problem']}\n\nSolution:"
            target_answer = self._extract_target_answer(item["solution"])
            prompts.append(
                TaskPrompt(
                    text=prompt_text,
                    metadata={"solution": item["solution"], "target_answer": target_answer},
                )
            )
        self._test_idx = (self._test_idx + n) % len(self._test_data)
        return prompts
