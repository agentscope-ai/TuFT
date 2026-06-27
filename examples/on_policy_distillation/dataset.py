"""
Data + prompt helpers for the on-policy distillation (OPD) example.

Task: grade-school math word problems (GSM8K). The TEACHER is the base model guided by a
strong system prompt + a few worked few-shot examples (so it reasons step by step and ends
with ``#### <number>``). The STUDENT sees only a bare prompt (no examples). On-policy
distillation trains the student's LoRA to match the teacher's per-token distribution on the
student's OWN samples — internalising the few-shot behaviour into the weights, so the student
reasons well WITHOUT the examples.

Teacher and student are the SAME base model (the teacher's only edge is the few-shot prompt),
so the whole thing fits on one GPU. This "context distillation" gives a strong, clean OPD
signal; a *different, larger* teacher helps only in proportion to the per-token capability gap
(see the user guide), not raw model size.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch
from tinker import types
from tinker.types.tensor_data import TensorData


# --------------------------------------------------------------------------------------
# Prompts: the ONLY difference between teacher and student is the few-shot "context".
# --------------------------------------------------------------------------------------

STUDENT_SYSTEM = "You are a helpful assistant. Solve the math problem."

TEACHER_SYSTEM = (
    "You are an expert math tutor. Solve each problem step by step with short, clear "
    "reasoning, then give the final numeric answer on its own line in the exact format "
    "'#### <number>'."
)

# A handful of worked GSM8K-style exemplars shown ONLY to the teacher. These create the
# capability gap that on-policy distillation transfers into the student's weights.
FEWSHOT: List[tuple[str, str]] = [
    (
        "Natalia sold clips to 48 of her friends in April, and then she sold half as "
        "many clips in May. How many clips did she sell altogether in April and May?",
        "In April she sold 48 clips.\n"
        "In May she sold half as many: 48 / 2 = 24 clips.\n"
        "Altogether: 48 + 24 = 72 clips.\n"
        "#### 72",
    ),
    (
        "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of "
        "babysitting. How much did she earn?",
        "Per minute she earns 12 / 60 = $0.2.\nFor 50 minutes: 50 * 0.2 = $10.\n#### 10",
    ),
    (
        "Betty is saving money for a new wallet which costs $100. Betty has only half of "
        "the money she needs. Her parents decided to give her $15, and her grandparents "
        "twice as much as her parents. How much more money does Betty need?",
        "Betty has half of $100: 100 / 2 = $50.\n"
        "Her grandparents give twice the parents' $15: 2 * 15 = $30.\n"
        "Now she has 50 + 15 + 30 = $95.\n"
        "She still needs 100 - 95 = $5.\n"
        "#### 5",
    ),
    (
        "James writes a 3-page letter to 2 different friends twice a week. How many pages "
        "does he write a year?",
        "Each time he writes 3 * 2 = 6 pages.\n"
        "Twice a week: 6 * 2 = 12 pages per week.\n"
        "In a year: 12 * 52 = 624 pages.\n"
        "#### 624",
    ),
]


@dataclass(frozen=True)
class Problem:
    question: str
    gold: str  # the gold final answer, e.g. "72"


# --------------------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------------------

_GOLD_RE = re.compile(r"####\s*(.+)")
_NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _gold_from_answer(answer: str) -> str:
    m = _GOLD_RE.search(answer)
    raw = m.group(1) if m else answer
    return raw.strip().replace(",", "").replace("$", "")


def load_gsm8k(n_train: int, n_eval: int, seed: int = 0) -> tuple[List[Problem], List[Problem]]:
    """Load small train/eval slices of GSM8K (downloaded via `datasets`)."""
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main")
    train_rows = list(ds["train"])
    eval_rows = list(ds["test"])
    rng = random.Random(seed)
    rng.shuffle(train_rows)
    rng.shuffle(eval_rows)

    train = [
        Problem(r["question"].strip(), _gold_from_answer(r["answer"])) for r in train_rows[:n_train]
    ]
    evals = [
        Problem(r["question"].strip(), _gold_from_answer(r["answer"])) for r in eval_rows[:n_eval]
    ]
    return train, evals


# --------------------------------------------------------------------------------------
# Prompt rendering
# --------------------------------------------------------------------------------------


def _render_ids(tokenizer, messages: list[dict]) -> List[int]:
    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:  # older templates without enable_thinking
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return tokenizer.encode(text, add_special_tokens=False)


def student_prompt_ids(tokenizer, question: str) -> List[int]:
    """Bare prompt the STUDENT is trained on — no few-shot examples."""
    messages = [
        {"role": "system", "content": STUDENT_SYSTEM},
        {"role": "user", "content": question},
    ]
    return _render_ids(tokenizer, messages)


def teacher_prompt_ids(tokenizer, question: str) -> List[int]:
    """Rich prompt the TEACHER is conditioned on — system instruction + few-shot examples."""
    messages: list[dict] = [{"role": "system", "content": TEACHER_SYSTEM}]
    for q, a in FEWSHOT:
        messages.append({"role": "user", "content": q})
        messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": question})
    return _render_ids(tokenizer, messages)


# --------------------------------------------------------------------------------------
# Answer extraction + scoring (for the visible before/after accuracy)
# --------------------------------------------------------------------------------------


def extract_pred(text: str) -> Optional[str]:
    """Pull the predicted number: prefer the '#### N' marker, else the last number."""
    m = _GOLD_RE.search(text)
    if m:
        nums = _NUM_RE.findall(m.group(1))
        if nums:
            return nums[-1].replace(",", "")
    nums = _NUM_RE.findall(text)
    return nums[-1].replace(",", "") if nums else None


def _as_float(s: str) -> Optional[float]:
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def is_correct(text: str, gold: str) -> bool:
    pred = extract_pred(text)
    if pred is None:
        return False
    pf, gf = _as_float(pred), _as_float(gold)
    if pf is not None and gf is not None:
        return abs(pf - gf) < 1e-4
    return pred.strip() == gold.strip()


# --------------------------------------------------------------------------------------
# Datum builder — identical alignment to examples/countdown_rl, but with a PER-TOKEN
# advantage (the on-policy distillation signal) instead of one scalar advantage.
# --------------------------------------------------------------------------------------


def build_distillation_datum(
    *,
    prompt: types.ModelInput,
    answer_tokens: Sequence[int],
    sampling_logprobs: Sequence[float],
    advantages: Sequence[float],
) -> types.Datum:
    """Build an importance_sampling Datum for one student rollout.

    The server computes the (differentiable) current-policy logprob of ``target_tokens``;
    the importance_sampling loss is ``-(exp(target_logprobs - logprobs) * advantages).sum()``.
    With ``advantages[t] = teacher_logprob[t] - student_logprob[t]`` this is the single-sample
    REINFORCE estimator of the per-token reverse-KL gradient — i.e. on-policy distillation.
    """
    toks = list(answer_tokens)
    # model_input excludes the final answer token (next-token prediction).
    model_input = prompt.append(types.EncodedTextChunk(tokens=toks[:-1]))
    ob_len = prompt.length - 1

    target_tokens = [0] * ob_len + toks
    padded_logprobs = [0.0] * ob_len + list(sampling_logprobs)
    padded_advantages = [0.0] * ob_len + list(advantages)

    if not (
        model_input.length == len(target_tokens) == len(padded_logprobs) == len(padded_advantages)
    ):
        raise ValueError(
            f"length mismatch: model_input={model_input.length} target={len(target_tokens)} "
            f"logprobs={len(padded_logprobs)} adv={len(padded_advantages)}"
        )

    return types.Datum(
        model_input=model_input,
        loss_fn_inputs={
            "target_tokens": TensorData.from_torch(torch.tensor(target_tokens, dtype=torch.long)),
            "logprobs": TensorData.from_torch(torch.tensor(padded_logprobs, dtype=torch.float32)),
            "advantages": TensorData.from_torch(
                torch.tensor(padded_advantages, dtype=torch.float32)
            ),
        },
    )
