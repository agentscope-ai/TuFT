from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import datasets
from datasets import Dataset
from tinker import types


COUNTDOWN_FEWSHOT = (
    "Q: Using the numbers 2, 3, 7, reach the target number 13. "
    "You may use +, -, *, / and parentheses, and each number can only be used once. "
    "Put ONLY the final expression inside <answer>...</answer>. "
    "Example: <answer>(1+2)/3</answer>.\n"
    "A: <answer>(2*3)+7</answer>\n\n"
)


def load_countdown_splits(
    dataset_name: str,
    split: str,
    test_size: int,
    seed: int,
) -> Tuple[Dataset, Dataset]:
    """Load Countdown dataset and build prompt-style question strings.
    Split policy (deterministic):
      - Test = first test_size rows
      - Train = remaining rows (shuffled)
    """
    ds = datasets.load_dataset(dataset_name, split=split)
    if len(ds) <= test_size:
        raise ValueError(f"Dataset too small: len={len(ds)} <= test_size={test_size}")

    test_ds = ds.select(range(test_size))
    train_ds = ds.select(range(test_size, len(ds)))

    def preprocess_fn(example, _idx):
        target = int(example["target"])
        nums = list(example["nums"])
        nums_str = ", ".join(map(str, nums))

        question = (
            f"Using the numbers {nums_str}, reach the target number {target}. "
            f"You may use +, -, *, / and parentheses, and each number can only be used once. "
            f"Put ONLY the final expression inside <answer>...</answer>. "
            f"Example: <answer>(1+2)/3</answer>."
        )
        return {"question": question, "target": target, "nums": nums}

    train_ds = train_ds.map(preprocess_fn, with_indices=True).shuffle(seed=seed)
    test_ds = test_ds.map(preprocess_fn, with_indices=True)
    return train_ds, test_ds


@dataclass
class Problem:
    question: str
    target: int
    nums: List[int]


class CountdownDatasetLoader:
    """Simple dataset wrapper with sequential batching for train/test."""

    def __init__(self, dataset_name: str, test_size: int, seed: int):
        train_ds, test_ds = load_countdown_splits(
            dataset_name=dataset_name,
            split="train",
            test_size=test_size,
            seed=seed,
        )
        self.train = train_ds
        self.test = test_ds
        self.train_idx = 0
        self.test_idx = 0

    def get_batch(self, batch_size: int, split: str = "train") -> List[Problem]:
        ds = self.train if split == "train" else self.test
        idx = self.train_idx if split == "train" else self.test_idx

        problems: List[Problem] = []
        for _ in range(batch_size):
            if idx >= len(ds):
                idx = 0
            row = ds[idx]
            idx += 1
            problems.append(
                Problem(
                    question=f"Q: {row['question']}\nA:",
                    target=int(row["target"]),
                    nums=list(row["nums"]),
                )
            )

        if split == "train":
            self.train_idx = idx
        else:
            self.test_idx = idx
        return problems


_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", flags=re.DOTALL)
_ALLOWED_EVAL_RE = re.compile(r"^[\d+\-*/().\s]+$")


def extract_solution(text: str) -> Optional[str]:
    """Extract the last <answer>...</answer> content from a model response."""
    if "Assistant:" in text:
        text = text.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in text:
        text = text.split("<|im_start|>assistant", 1)[1]

    matches = list(_ANSWER_RE.finditer(text))
    if not matches:
        return None
    return matches[-1].group(1).strip()


def validate_equation(equation_str: str, available_numbers: List[int]) -> bool:
    """Check if equation uses exactly the provided numbers (multiset match)."""
    try:
        numbers_in_eq = [int(n) for n in re.findall(r"\d+", equation_str)]
        return sorted(numbers_in_eq) == sorted(available_numbers)
    except Exception:
        return False


def evaluate_equation(equation_str: str) -> Optional[float]:
    """Safely evaluate arithmetic expression if it matches a restricted character set."""
    try:
        if not _ALLOWED_EVAL_RE.match(equation_str):
            return None
        return eval(equation_str, {"__builtins__": None}, {})
    except Exception:
        return None


def compute_reward(
    response_text: str,
    target: int,
    nums: List[int],
    format_score: float,
    use_continuous_shaping: bool,
) -> float:
    """Compute reward for a Countdown response.

    - 0.0 if no <answer>
    - format_score if invalid numbers or invalid eval
    - 1.0 if exact
    - otherwise optionally use continuous shaping
    """
    equation = extract_solution(response_text)
    if equation is None:
        return 0.0

    if not validate_equation(equation, nums):
        return float(format_score)

    result = evaluate_equation(equation)
    if result is None:
        return float(format_score)

    err = abs(result - target)
    if err < 1e-5:
        return 1.0

    if not use_continuous_shaping:
        return float(format_score)

    shaped = format_score + (1.0 - format_score) * (1.0 / (1.0 + err))
    return float(shaped)


def make_prompt_model_input(tokenizer, text: str) -> types.ModelInput:
    toks = tokenizer.encode(text, add_special_tokens=False)
    return types.ModelInput(chunks=[types.EncodedTextChunk(tokens=toks)])
