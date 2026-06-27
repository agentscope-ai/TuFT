"""IFEval task: instruction-following evaluation with rule-based rewards.

Implements deterministic, verifiable constraint checkers for the IFEval benchmark.
Each prompt contains a set of format/content constraints (e.g., word count, keyword
presence, case requirements) that can be verified without an LLM judge.

Reference: Zhou et al., "Instruction-Following Evaluation for Large Language Models", 2023.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, List

import datasets

from .base import Task, TaskPrompt


IFEVAL_INSTRUCTION = (
    "Follow the instruction below precisely. "
    "Pay close attention to ALL formatting and content requirements.\n\n"
)

# ---------------------------------------------------------------------------
# Constraint checker functions
# ---------------------------------------------------------------------------
# Each checker takes (response_text: str, **kwargs) -> bool


def _check_keywords_existence(response: str, keywords: List[str], **_) -> bool:
    """Check that ALL specified keywords appear in the response."""
    lower = response.lower()
    return all(kw.lower() in lower for kw in keywords)


def _check_keywords_frequency(
    response: str, keyword: str, frequency: int, relation: str = "at least", **_
) -> bool:
    """Check keyword frequency constraint."""
    count = response.lower().count(keyword.lower())
    if relation == "at least":
        return count >= frequency
    elif relation == "at most":
        return count <= frequency
    elif relation == "exactly":
        return count == frequency
    return count >= frequency


def _check_keywords_forbidden(response: str, forbidden_words: List[str], **_) -> bool:
    """Check that NONE of the forbidden words appear in the response."""
    lower = response.lower()
    return not any(fw.lower() in lower for fw in forbidden_words)


def _check_keywords_letter_frequency(
    response: str, letter: str, let_frequency: int, let_relation: str = "at least", **_
) -> bool:
    """Check letter frequency constraint."""
    count = response.lower().count(letter.lower())
    if let_relation == "at least":
        return count >= let_frequency
    elif let_relation == "at most":
        return count <= let_frequency
    elif let_relation == "exactly":
        return count == let_frequency
    return count >= let_frequency


def _check_number_words(response: str, num_words: int, relation: str = "at least", **_) -> bool:
    """Check word count constraint."""
    word_count = len(response.split())
    if relation == "at least":
        return word_count >= num_words
    elif relation == "at most":
        return word_count <= num_words
    elif relation == "exactly":
        return word_count == num_words
    return word_count >= num_words


def _check_number_sentences(
    response: str, num_sentences: int, relation: str = "at least", **_
) -> bool:
    """Check sentence count constraint."""
    # Split by sentence-ending punctuation
    sentences = [s.strip() for s in re.split(r"[.!?]+", response) if s.strip()]
    count = len(sentences)
    if relation == "at least":
        return count >= num_sentences
    elif relation == "at most":
        return count <= num_sentences
    elif relation == "exactly":
        return count == num_sentences
    return count >= num_sentences


def _check_number_paragraphs(response: str, num_paragraphs: int, **_) -> bool:
    """Check paragraph count constraint."""
    paragraphs = [p.strip() for p in response.split("\n\n") if p.strip()]
    return len(paragraphs) >= num_paragraphs


def _check_nth_paragraph_first_word(
    response: str, num_paragraphs: int, nth_paragraph: int, first_word: str, **_
) -> bool:
    """Check the first word of the nth paragraph."""
    paragraphs = [p.strip() for p in response.split("\n\n") if p.strip()]
    if nth_paragraph > len(paragraphs) or nth_paragraph < 1:
        return False
    words = paragraphs[nth_paragraph - 1].split()
    if not words:
        return False
    return words[0].lower().rstrip(".,;:!?") == first_word.lower()


def _check_postscript(response: str, postscript_marker: str = "P.S.", **_) -> bool:
    """Check if response contains a postscript section."""
    return postscript_marker in response


def _check_number_bullet_lists(response: str, num_bullets: int, **_) -> bool:
    """Check that response contains at least num_bullets bullet points."""
    bullets = re.findall(r"^[\s]*[-*•]\s+", response, flags=re.MULTILINE)
    return len(bullets) >= num_bullets


def _check_constrained_response(response: str, starter: str = None, **_) -> bool:
    """Check that response starts with a specific string."""
    if starter is None:
        return True
    return response.strip().startswith(starter)


def _check_number_highlighted_sections(response: str, num_highlights: int, **_) -> bool:
    """Check that response contains highlighted sections (text wrapped in *..*)."""
    highlights = re.findall(r"\*[^*]+\*", response)
    return len(highlights) >= num_highlights


def _check_multiple_sections(
    response: str, section_spliter: str = "Section", num_sections: int = 2, **_
) -> bool:
    """Check that response contains multiple sections."""
    count = len(re.findall(re.escape(section_spliter), response, flags=re.IGNORECASE))
    return count >= num_sections


def _check_json_format(response: str, **_) -> bool:
    """Check that the response is valid JSON or contains a valid JSON block."""
    # Try the whole response first
    try:
        json.loads(response.strip())
        return True
    except (json.JSONDecodeError, ValueError):
        pass
    # Try extracting from code block
    match = re.search(r"```(?:json)?\s*\n(.*?)```", response, flags=re.DOTALL)
    if match:
        try:
            json.loads(match.group(1).strip())
            return True
        except (json.JSONDecodeError, ValueError):
            pass
    return False


def _check_title(response: str, **_) -> bool:
    """Check that response contains a title (wrapped in << >> or starting with #)."""
    if re.search(r"<<[^>]+>>", response):
        return True
    if re.search(r"^#\s+\S+", response, flags=re.MULTILINE):
        return True
    return False


def _check_two_responses(response: str, **_) -> bool:
    """Check that response contains two parts separated by ****."""
    parts = response.split("****")
    return len(parts) >= 2 and all(p.strip() for p in parts[:2])


def _check_repeat_prompt(response: str, prompt_to_repeat: str = None, **_) -> bool:
    """Check that the response repeats the original prompt."""
    if prompt_to_repeat is None:
        return True
    return prompt_to_repeat.strip() in response


def _check_end_checker(response: str, end_phrase: str = None, **_) -> bool:
    """Check that the response ends with a specific phrase."""
    if end_phrase is None:
        return True
    return response.strip().endswith(end_phrase.strip())


def _check_english_capital(response: str, **_) -> bool:
    """Check that all words are capitalized (title case or all-caps)."""
    words = re.findall(r"[a-zA-Z]+", response)
    if not words:
        return False
    # At least 80% of words should be capitalized (allowing for articles etc.)
    capitalized = sum(1 for w in words if w[0].isupper())
    return capitalized / len(words) >= 0.8


def _check_english_lowercase(response: str, **_) -> bool:
    """Check that response is predominantly lowercase."""
    alpha = re.findall(r"[a-zA-Z]", response)
    if not alpha:
        return True
    lower_count = sum(1 for c in alpha if c.islower())
    return lower_count / len(alpha) >= 0.9


def _check_no_comma(response: str, **_) -> bool:
    """Check that the response contains no commas."""
    return "," not in response


# ---------------------------------------------------------------------------
# Constraint registry: instruction_id -> checker function
# ---------------------------------------------------------------------------

_CONSTRAINT_CHECKERS: Dict[str, Callable[..., bool]] = {
    "keywords:existence": _check_keywords_existence,
    "keywords:frequency": _check_keywords_frequency,
    "keywords:forbidden_words": _check_keywords_forbidden,
    "keywords:letter_frequency": _check_keywords_letter_frequency,
    "length_constraints:number_words": _check_number_words,
    "length_constraints:number_sentences": _check_number_sentences,
    "length_constraints:number_paragraphs": _check_number_paragraphs,
    "length_constraints:nth_paragraph_first_word": _check_nth_paragraph_first_word,
    "detectable_content:postscript": _check_postscript,
    "detectable_format:number_bullet_lists": _check_number_bullet_lists,
    "detectable_format:constrained_response": _check_constrained_response,
    "detectable_format:number_highlighted_sections": _check_number_highlighted_sections,
    "detectable_format:multiple_sections": _check_multiple_sections,
    "detectable_format:json_format": _check_json_format,
    "detectable_format:title": _check_title,
    "combination:two_responses": _check_two_responses,
    "combination:repeat_prompt": _check_repeat_prompt,
    "startend:end_checker": _check_end_checker,
    "change_case:english_capital": _check_english_capital,
    "change_case:english_lowercase": _check_english_lowercase,
    "punctuation:no_comma": _check_no_comma,
}


def evaluate_constraints(
    response: str,
    instruction_ids: List[str],
    kwargs_list: List[Dict[str, Any]],
) -> float:
    """Evaluate all constraints and return the fraction satisfied [0, 1]."""
    if not instruction_ids:
        return 0.0

    passed = 0
    total = len(instruction_ids)

    for inst_id, kw in zip(instruction_ids, kwargs_list, strict=False):
        checker = _CONSTRAINT_CHECKERS.get(inst_id)
        if checker is None:
            # Unknown constraint — skip (don't penalize for unsupported ones)
            total -= 1
            continue
        try:
            if checker(response, **(kw or {})):
                passed += 1
        except Exception:
            # Checker error — treat as failed
            pass

    if total <= 0:
        return 0.0
    return passed / total


# ---------------------------------------------------------------------------
# IFEval Task
# ---------------------------------------------------------------------------


class IFEvalTask(Task):
    """IFEval instruction-following task with deterministic, rule-based rewards.

    Dataset: google/IFEval (541 prompts)
    Reward: fraction of satisfied constraints per prompt [0, 1]
    """

    def __init__(self, dataset_name: str = "google/IFEval", seed: int = 42):
        self._dataset_name = dataset_name
        self._seed = seed
        self._train_data: List[Dict[str, Any]] = []
        self._test_data: List[Dict[str, Any]] = []
        self._train_idx = 0
        self._test_idx = 0

    def load_dataset(self) -> None:
        ds = datasets.load_dataset(self._dataset_name, split="train")

        all_data = [
            {
                "prompt": row["prompt"],
                "instruction_id_list": row["instruction_id_list"],
                "kwargs": row["kwargs"],
            }
            for row in ds
        ]

        # IFEval has only one split (541 prompts).
        # Use 80% for training and 20% for evaluation.
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

        prompt_text = IFEVAL_INSTRUCTION + item["prompt"]

        return TaskPrompt(
            text=prompt_text,
            metadata={
                "instruction_id_list": item["instruction_id_list"],
                "kwargs": item["kwargs"],
                "original_prompt": item["prompt"],
            },
        )

    def compute_reward(self, response_text: str, metadata: Dict[str, Any]) -> float:
        instruction_ids = metadata.get("instruction_id_list", [])
        kwargs_list = metadata.get("kwargs", [])

        # Parse kwargs if stored as JSON strings
        parsed_kwargs = []
        for kw in kwargs_list:
            if isinstance(kw, str):
                try:
                    parsed_kwargs.append(json.loads(kw))
                except (json.JSONDecodeError, ValueError):
                    parsed_kwargs.append({})
            elif isinstance(kw, dict):
                parsed_kwargs.append(kw)
            else:
                parsed_kwargs.append({})

        return evaluate_constraints(response_text, instruction_ids, parsed_kwargs)

    def get_eval_prompts(self, n: int) -> List[TaskPrompt]:
        if not self._test_data:
            self.load_dataset()

        prompts = []
        for i in range(min(n, len(self._test_data))):
            idx = (self._test_idx + i) % len(self._test_data)
            item = self._test_data[idx]
            prompt_text = IFEVAL_INSTRUCTION + item["prompt"]
            prompts.append(
                TaskPrompt(
                    text=prompt_text,
                    metadata={
                        "instruction_id_list": item["instruction_id_list"],
                        "kwargs": item["kwargs"],
                        "original_prompt": item["prompt"],
                    },
                )
            )
        self._test_idx = (self._test_idx + n) % len(self._test_data)
        return prompts
