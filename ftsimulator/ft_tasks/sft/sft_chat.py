from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, cast

import numpy as np
from datasets import load_dataset
from ft_tasks.ft_base import FinetuneTask
from ft_tasks.utils import load_tokenizer
from tinker import types


Messages = List[Dict[str, Any]]


@dataclass
class ChatDataset:
    """Simple chat dataset with batching."""

    data: List[Messages]
    index: int = 0

    def get_batch(self, batch_size: int) -> List[Messages]:
        batch: List[Messages] = []
        for _ in range(batch_size):
            if self.index >= len(self.data):
                self.index = 0
                random.shuffle(self.data)
            batch.append(self.data[self.index])
            self.index += 1
        return batch

    def __len__(self) -> int:
        return len(self.data)


def load_chat_dataset(dataset_name: str, seed: int = 42) -> Tuple[ChatDataset, ChatDataset]:
    """Load train/test chat dataset."""
    random.seed(seed)

    if dataset_name == "no_robots":
        ds = load_dataset("HuggingFaceH4/no_robots")

        train_split = ds["train"]
        test_split = ds["test"]

        # Pyright sometimes infers HF rows as list-like, so we cast to dict
        train_data = [cast(Messages, cast(dict[str, Any], row)["messages"]) for row in train_split]
        test_data = [cast(Messages, cast(dict[str, Any], row)["messages"]) for row in test_split]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    random.shuffle(train_data)
    return ChatDataset(train_data), ChatDataset(test_data)


def tokenize_conversation(
    messages: Messages,
    tokenizer,
    max_length: int,
) -> Tuple[List[int], np.ndarray]:
    """Tokenize a conversation and compute per-token loss weights (assistant=1, user/system=0)."""
    all_tokens: List[int] = []
    all_weights: List[float] = []

    for i, msg in enumerate(messages):
        partial = messages[: i + 1]
        text = tokenizer.apply_chat_template(
            partial,
            tokenize=False,
            add_generation_prompt=False,
        )
        tokens = tokenizer.encode(text, add_special_tokens=False)

        prev_len = len(all_tokens)
        new_tokens = tokens[prev_len:]

        is_assistant = msg.get("role") == "assistant"
        weight = 1.0 if is_assistant else 0.0

        all_tokens.extend(new_tokens)
        all_weights.extend([weight] * len(new_tokens))

    if len(all_tokens) > max_length:
        all_tokens = all_tokens[:max_length]
        all_weights = all_weights[:max_length]

    return all_tokens, np.array(all_weights, dtype=np.float32)


def conversation_to_datum(
    messages: Messages,
    tokenizer,
    max_length: int,
) -> types.Datum:
    """Convert a conversation into next-token-prediction Datum with shifted targets/weights."""
    tokens, weights = tokenize_conversation(messages, tokenizer, max_length)
    if len(tokens) < 2:
        raise ValueError("Conversation too short")

    input_tokens = tokens[:-1]
    target_tokens = tokens[1:]
    target_weights = weights[1:]

    return types.Datum(
        model_input=types.ModelInput.from_ints(input_tokens),
        loss_fn_inputs={
            "target_tokens": types.TensorData(
                data=target_tokens,
                dtype="int64",
                shape=[len(target_tokens)],
            ),
            "weights": types.TensorData(
                data=target_weights.tolist(),
                dtype="float32",
                shape=[len(target_weights)],
            ),
        },
    )


def _to_list(value):
    if isinstance(value, types.TensorData):
        return value.data
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def compute_weighted_nll_from_outputs(loss_fn_outputs, datums) -> float:
    total_loss = 0.0
    total_weight = 0.0

    for i, out in enumerate(loss_fn_outputs):
        logprobs = _to_list(out["logprobs"])
        weights = _to_list(datums[i].loss_fn_inputs["weights"])

        for lp, wt in zip(logprobs, weights, strict=False):
            total_loss += -lp * wt
            total_weight += wt

    return total_loss / max(total_weight, 1.0)


def build_datums(
    batch: List[List[Dict[str, Any]]],
    tokenizer,
    max_length: int,
) -> List[types.Datum]:
    datums: List[types.Datum] = []
    for messages in batch:
        try:
            datums.append(conversation_to_datum(messages, tokenizer, max_length))
        except ValueError:
            continue
    return datums


class SFTChatTask(FinetuneTask):
    task_type = "sft_chat"

    def setup(self):
        self.service_client = self._connect()
        self.tokenizer = load_tokenizer(self.base_model)
        self.train_dataset, self.test_dataset = load_chat_dataset(
            dataset_name=self.dataset, seed=self.seed
        )
        self.training_client = self._create_training_client(self.service_client, self.lora_params)

    def run(self) -> List[Dict[str, Any]]:
        metrics_history: List[Dict[str, Any]] = []

        for step in range(self.num_steps):
            batch = self.train_dataset.get_batch(self.train_batch)
            datums = build_datums(batch, self.tokenizer, self.max_length)

            if not datums:
                continue

            fwdbwd = self.training_client.forward_backward(datums, loss_fn="cross_entropy").result()
            loss = compute_weighted_nll_from_outputs(fwdbwd.loss_fn_outputs, datums)

            self.training_client.optim_step(
                types.AdamParams(learning_rate=self.learning_rate)
            ).result()
            metrics_history.append({"step": step, "loss": loss})
        first_loss = metrics_history[0]["loss"] if metrics_history else "N/A"
        final_loss = metrics_history[-1]["loss"] if metrics_history else "N/A"
        print(f"First loss: {first_loss} Final loss: {final_loss}")
        sampling_path = self.training_client.save_weights_for_sampler(self.task_id).result().path
        sampling_client = self.service_client.create_sampling_client(model_path=sampling_path)
        test_messages = [{"role": "user", "content": "Write a haiku about programming."}]
        prompt_text = self.tokenizer.apply_chat_template(
            test_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_tokens = self.tokenizer.encode(prompt_text, add_special_tokens=False)

        sampling_client.sample(
            prompt=types.ModelInput.from_ints(prompt_tokens),
            num_samples=1,
            sampling_params=types.SamplingParams(max_tokens=128, temperature=0.7),
        ).result()

        self.training_client.save_state(name=self.task_id).result()
        return metrics_history

    def teardown(self):
        pass  # No special teardown needed
