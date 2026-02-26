from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, cast

import numpy as np
from datasets import load_dataset
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
