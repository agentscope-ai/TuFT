from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import datasets
import torch
from datasets import Dataset
from ft_tasks.ft_base import FinetuneTask, LoraParams
from tinker import types
from tinker.types.tensor_data import TensorData


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


@dataclass
class RLHFParams:
    """Extra RLHF-specific parameters."""

    batch_size: int = 4
    group_size: int = 16
    max_tokens: int = 128
    temperature: float = 0.9

    test_size: int = 512

    format_score: float = 0.1
    continuous_shaping: bool = True

    eval_every: int = 30
    eval_batch_size: int = 64
    eval_group_size: int = 1
    eval_temperature: float = 0.1
    reward_ema_alpha: float = 0.1

    sampler_name: Optional[str] = None
    checkpoint_name: Optional[str] = None
    plot_path: Optional[str] = None

    def with_defaults(self, task_id: str) -> "RLHFParams":
        return RLHFParams(
            batch_size=self.batch_size,
            group_size=self.group_size,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            test_size=self.test_size,
            format_score=self.format_score,
            continuous_shaping=self.continuous_shaping,
            eval_every=self.eval_every,
            eval_batch_size=self.eval_batch_size,
            eval_group_size=self.eval_group_size,
            eval_temperature=self.eval_temperature,
            reward_ema_alpha=self.reward_ema_alpha,
            sampler_name=self.sampler_name or f"{task_id}-sampler",
            checkpoint_name=self.checkpoint_name or f"{task_id}-state",
            plot_path=self.plot_path,
        )


def _build_rlhf_params(extra_cfg: Optional[Dict[str, Any]], task_id: str) -> RLHFParams:
    cfg = extra_cfg or {}
    params = RLHFParams(
        batch_size=cfg.get("batch_size", 4),
        group_size=cfg.get("group_size", 16),
        max_tokens=cfg.get("max_tokens", 128),
        temperature=cfg.get("temperature", 0.9),
        test_size=cfg.get("test_size", 512),
        format_score=cfg.get("format_score", 0.1),
        continuous_shaping=cfg.get("continuous_shaping", True),
        eval_every=cfg.get("eval_every", 30),
        eval_batch_size=cfg.get("eval_batch_size", 64),
        eval_group_size=cfg.get("eval_group_size", 1),
        eval_temperature=cfg.get("eval_temperature", 0.1),
        reward_ema_alpha=cfg.get("reward_ema_alpha", 0.1),
        sampler_name=cfg.get("sampler_name"),
        checkpoint_name=cfg.get("checkpoint_name"),
        plot_path=cfg.get("plot_path"),
    )
    return params.with_defaults(task_id)


def build_importance_sampling_datum(
    *,
    prompt: types.ModelInput,
    ob_len: int,
    toks: List[int],
    lps: List[float],
    adv: float,
) -> types.Datum:
    """Build one importance-sampling datum for RL training."""
    model_input = prompt.append(types.EncodedTextChunk(tokens=toks[:-1]))

    target_tokens = [0] * ob_len + toks
    padded_sampling_logprobs = [0.0] * ob_len + lps
    padded_advantages = [0.0] * ob_len + [adv] * (model_input.length - ob_len)

    if not (
        model_input.length
        == len(target_tokens)
        == len(padded_sampling_logprobs)
        == len(padded_advantages)
    ):
        raise RuntimeError(
            f"Length mismatch: model_input={model_input.length} "
            f"target={len(target_tokens)} logprobs={len(padded_sampling_logprobs)} "
            f"adv={len(padded_advantages)}"
        )

    target_tokens_t = torch.tensor(target_tokens, dtype=torch.long)
    logprobs_t = torch.tensor(padded_sampling_logprobs, dtype=torch.float32)
    advantages_t = torch.tensor(padded_advantages, dtype=torch.float32)

    return types.Datum(
        model_input=model_input,
        loss_fn_inputs={
            "target_tokens": TensorData.from_torch(target_tokens_t),
            "logprobs": TensorData.from_torch(logprobs_t),
            "advantages": TensorData.from_torch(advantages_t),
        },
    )


class RLHFCountdownTask(FinetuneTask):
    task_type = "rlhf_countdown"

    def setup(self) -> None:
        self.rlhf_params = _build_rlhf_params(self.extra_cfg, self.task_id)

        random.seed(self.seed)
        torch.manual_seed(self.seed)

        self.service_client = self._connect()

        self.dataset_loader = CountdownDatasetLoader(
            self.dataset,
            self.rlhf_params.test_size,
            self.seed,
        )

        self.training_client = self._create_training_client(
            self.service_client,
            LoraParams(
                rank=self.lora_params.rank,
                train_mlp=self.lora_params.train_mlp,
                train_attn=self.lora_params.train_attn,
                train_unembed=True,
            ),
        )

        self.tokenizer = self.training_client.get_tokenizer()

        self.sampling_params_train = types.SamplingParams(
            max_tokens=self.rlhf_params.max_tokens,
            temperature=self.rlhf_params.temperature,
        )
        self.sampling_params_eval = types.SamplingParams(
            max_tokens=self.rlhf_params.max_tokens,
            temperature=self.rlhf_params.eval_temperature,
        )
        self.adam_params = types.AdamParams(
            learning_rate=self.learning_rate,
            beta1=0.9,
            beta2=0.95,
            eps=1e-8,
        )

    def _compute_reward(self, response_text: str, target: int, nums: List[int]) -> float:
        return float(
            compute_reward(
                response_text=response_text,
                target=target,
                nums=nums,
                format_score=self.rlhf_params.format_score,
                use_continuous_shaping=self.rlhf_params.continuous_shaping,
            )
        )

    def _do_eval(self, step: int) -> float:
        eval_path = (
            self.training_client.save_weights_for_sampler(name=f"{self.task_id}_eval_{step:06d}")
            .result()
            .path
        )
        eval_client = self.service_client.create_sampling_client(model_path=eval_path)

        probs = self.dataset_loader.get_batch(
            self.rlhf_params.eval_batch_size,
            split="test",
        )
        rewards: List[float] = []

        for prob in probs:
            prompt_text = COUNTDOWN_FEWSHOT + prob.question
            prompt = make_prompt_model_input(self.tokenizer, prompt_text)

            res = eval_client.sample(
                prompt=prompt,
                num_samples=self.rlhf_params.eval_group_size,
                sampling_params=self.sampling_params_eval,
            ).result()

            for seq in res.sequences:
                resp_text = self.tokenizer.decode(seq.tokens, skip_special_tokens=True)
                rewards.append(
                    self._compute_reward(
                        response_text=resp_text,
                        target=prob.target,
                        nums=prob.nums,
                    )
                )

        return sum(rewards) / max(1, len(rewards))

    def _final_eval_and_save(self, metrics_history: List[Dict[str, Any]]) -> None:
        p = self.rlhf_params

        sampler_path = (
            self.training_client.save_weights_for_sampler(
                name=p.sampler_name or f"{self.task_id}-sampler"
            )
            .result()
            .path
        )
        final_client = self.service_client.create_sampling_client(model_path=sampler_path)

        test_problems = self.dataset_loader.get_batch(1, split="test")

        for _i, problem in enumerate(test_problems):
            prompt_text = COUNTDOWN_FEWSHOT + problem.question
            prompt_input = make_prompt_model_input(self.tokenizer, prompt_text)

            try:
                result = final_client.sample(
                    prompt=prompt_input,
                    num_samples=1,
                    sampling_params=types.SamplingParams(
                        max_tokens=p.max_tokens,
                        temperature=0.0,
                    ),
                ).result()
            except Exception:
                result = final_client.sample(
                    prompt=prompt_input,
                    num_samples=1,
                    sampling_params=types.SamplingParams(
                        max_tokens=p.max_tokens,
                        temperature=0.1,
                    ),
                ).result()

            response = self.tokenizer.decode(
                result.sequences[0].tokens,
                skip_special_tokens=True,
            )
            self._compute_reward(
                response_text=response,
                target=problem.target,
                nums=problem.nums,
            )

        checkpoint_name = p.checkpoint_name or f"{self.task_id}-state"
        self.training_client.save_state(name=checkpoint_name).result()

    def run(self) -> List[Dict[str, Any]]:
        metrics_history: List[Dict[str, Any]] = []
        ema_eval_reward: Optional[float] = None
        p = self.rlhf_params

        for step in range(self.num_steps):
            problems = self.dataset_loader.get_batch(
                p.batch_size,
                split="train",
            )

            save_result = self.training_client.save_weights_for_sampler(
                name=f"{self.task_id}_rl_step_{step:06d}"
            ).result()
            sampling_client = self.service_client.create_sampling_client(
                model_path=save_result.path
            )

            datums: List[types.Datum] = []
            mean_rewards_per_problem: List[float] = []
            kept_rollouts = 0
            skipped_problems = 0

            for prob in problems:
                prompt_text = COUNTDOWN_FEWSHOT + prob.question
                prompt = make_prompt_model_input(self.tokenizer, prompt_text)

                sample_res = sampling_client.sample(
                    prompt=prompt,
                    num_samples=p.group_size,
                    sampling_params=self.sampling_params_train,
                ).result()

                rewards_g: List[float] = []
                tokens_g: List[List[int]] = []
                logprobs_g: List[List[float]] = []

                for seq in sample_res.sequences:
                    toks = list(seq.tokens)
                    lps = seq.logprobs
                    if lps is None:
                        raise RuntimeError("Sampling did not return logprobs.")

                    response_text = self.tokenizer.decode(toks, skip_special_tokens=True)
                    reward = self._compute_reward(
                        response_text=response_text,
                        target=prob.target,
                        nums=prob.nums,
                    )

                    rewards_g.append(reward)
                    tokens_g.append(toks)
                    logprobs_g.append(list(lps))

                mean_r = sum(rewards_g) / len(rewards_g)
                mean_rewards_per_problem.append(mean_r)

                var_r = sum((r - mean_r) ** 2 for r in rewards_g) / max(1, len(rewards_g))
                std_r = var_r**0.5

                if std_r < 1e-8:
                    skipped_problems += 1
                    continue

                advantages_g = [(r - mean_r) / (std_r + 1e-6) for r in rewards_g]
                ob_len = prompt.length - 1

                for toks, lps, adv in zip(tokens_g, logprobs_g, advantages_g, strict=True):
                    datums.append(
                        build_importance_sampling_datum(
                            prompt=prompt,
                            ob_len=ob_len,
                            toks=toks,
                            lps=lps,
                            adv=adv,
                        )
                    )
                    kept_rollouts += 1

            train_mean_reward = sum(mean_rewards_per_problem) / max(
                1, len(mean_rewards_per_problem)
            )

            if datums:
                self.training_client.forward_backward(
                    datums,
                    loss_fn="importance_sampling",
                ).result()
                self.training_client.optim_step(self.adam_params).result()

            eval_mean_reward = None
            ema_now = None
            if p.eval_every > 0 and (step % p.eval_every == 0):
                eval_mean_reward = self._do_eval(step)

                if ema_eval_reward is None:
                    ema_eval_reward = eval_mean_reward
                else:
                    a = p.reward_ema_alpha
                    ema_eval_reward = (1 - a) * ema_eval_reward + a * eval_mean_reward
                ema_now = ema_eval_reward

            metrics = {
                "step": step,
                "train_mean_reward": float(train_mean_reward),
                "eval_mean_reward": None if eval_mean_reward is None else float(eval_mean_reward),
                "ema_eval_reward": None if ema_now is None else float(ema_now),
                "kept_rollouts": int(kept_rollouts),
                "skipped_problems": int(skipped_problems),
            }
            metrics_history.append(metrics)
        first_reward = metrics_history[0]["train_mean_reward"] if metrics_history else "N/A"
        final_reward = metrics_history[-1]["train_mean_reward"] if metrics_history else "N/A"
        print(f"First reward: {first_reward} Final reward: {final_reward}")
        self._final_eval_and_save(metrics_history)
        return metrics_history

    def teardown(self) -> None:
        pass

    def _build_summary(self, metrics_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not metrics_history:
            return {}

        first = metrics_history[0]
        last = metrics_history[-1]

        eval_points = [m for m in metrics_history if m.get("eval_mean_reward") is not None]
        best_eval = max((m["eval_mean_reward"] for m in eval_points), default=None)

        return {
            "first_step": first,
            "last_step": last,
            "total_steps": len(metrics_history),
            "initial_train_mean_reward": first.get("train_mean_reward"),
            "final_train_mean_reward": last.get("train_mean_reward"),
            "best_eval_mean_reward": best_eval,
        }
