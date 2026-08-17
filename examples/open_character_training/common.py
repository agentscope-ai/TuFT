"""Shared TuFT, tokenization, sampling, and resumable-cache helpers."""

from __future__ import annotations

import os
import re
import time
import unicodedata
import urllib.error
import urllib.request
from pathlib import Path
from typing import Sequence

import numpy as np
import settings as C
import tinker
from tinker import types
from transformers import AutoTokenizer


_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)
_STOPS = ("<|im_end|>", "<|endoftext|>", "<|im_start|>", "</s>")


class Cache:
    """A resumable JSON request cache with metadata and hash-keyed items."""

    def __init__(self, path: Path, metadata: dict | None = None):
        self.path = path
        if path.exists():
            payload = C.read_json(path)
        else:
            payload = {"metadata": {}, "items": {}}
        self.metadata = payload.get("metadata", {})
        self.items: dict[str, dict] = payload.get("items", {})
        if metadata:
            self.metadata.update(metadata)

    def get(self, key: str) -> dict | None:
        return self.items.get(key)

    def put(self, key: str, value: dict) -> None:
        self.items[key] = value

    def flush(self) -> None:
        C.write_json(self.path, {"metadata": self.metadata, "items": self.items})

    def __len__(self) -> int:
        return len(self.items)


def clean(text: str) -> str:
    for stop in _STOPS:
        if stop in text:
            text = text.split(stop)[0]
    return _THINK.sub("", text).strip()


def complete_response(text: str) -> bool:
    """Accept responses ending in punctuation or a symbol such as an emoji."""
    text = text.rstrip()
    if not text:
        return False
    category = unicodedata.category(text[-1])
    return category.startswith("P") or category.startswith("S")


def wait_healthy(base_url: str, timeout: float = 900.0) -> None:
    url = base_url.rstrip("/") + "/api/v1/healthz"
    deadline = time.time() + timeout
    print(f"[wait] {url}", flush=True)
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=10) as response:
                if response.status == 200:
                    print("[wait] TuFT is healthy", flush=True)
                    return
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(3)
    raise SystemExit("TuFT did not become healthy before the timeout")


def connect(base_url: str, api_key: str) -> tinker.ServiceClient:
    if not api_key or not api_key.startswith("tml-"):
        raise SystemExit("--api-key or TINKER_API_KEY must start with 'tml-'")
    return tinker.ServiceClient(base_url=base_url, api_key=api_key)


def tokenizer(model: str = C.MODEL):
    return AutoTokenizer.from_pretrained(model, trust_remote_code=True)


def apply_chat_template(tok, messages: list[dict], *, generation: bool) -> str:
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": generation,
    }
    try:
        return tok.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tok.apply_chat_template(messages, **kwargs)


def _tokenize_with_last_assistant_mask(messages: list[dict], tok) -> tuple[list[int], list[float]]:
    """Tokenize a chat and supervise only its final assistant message."""
    final_assistant = max(
        (index for index, message in enumerate(messages) if message["role"] == "assistant"),
        default=-1,
    )
    if final_assistant < 0:
        raise ValueError("conversation has no assistant response")

    tokens: list[int] = []
    weights: list[float] = []
    for index, _message in enumerate(messages):
        rendered = apply_chat_template(tok, messages[: index + 1], generation=False)
        cumulative = tok.encode(rendered, add_special_tokens=False)
        if cumulative[: len(tokens)] != tokens:
            raise ValueError("chat template is not prefix-stable for incremental masking")
        new_tokens = cumulative[len(tokens) :]
        weight = 1.0 if index == final_assistant else 0.0
        tokens.extend(new_tokens)
        weights.extend([weight] * len(new_tokens))
    return tokens, weights


def conversation_to_datum(
    messages: list[dict], tok, max_length: int, *, trim_old_turns: bool = False
):
    """Build a next-token Datum with a raw 0/1 final-response mask.

    For long introspection conversations, whole early turns are dropped until the
    final assistant target fits. DPO examples are never truncated; callers use
    ``trim_old_turns=False`` and filter over-length pairs instead.
    """
    current = list(messages)
    while True:
        tokens, weights = _tokenize_with_last_assistant_mask(current, tok)
        if len(tokens) <= max_length and sum(weights[1:]) > 0:
            break
        if not trim_old_turns:
            raise ValueError(f"conversation is {len(tokens)} tokens; limit is {max_length}")
        system_count = 1 if current and current[0]["role"] == "system" else 0
        if len(current) - system_count <= 2:
            raise ValueError("final assistant turn does not fit the token limit")
        current = current[:system_count] + current[system_count + 1 :]

    input_tokens = tokens[:-1]
    target_tokens = tokens[1:]
    target_weights = np.asarray(weights[1:], dtype=np.float32)
    return types.Datum(
        model_input=types.ModelInput.from_ints(input_tokens),
        loss_fn_inputs={
            "target_tokens": types.TensorData(
                data=target_tokens, dtype="int64", shape=[len(target_tokens)]
            ),
            "weights": types.TensorData(
                data=target_weights.tolist(),
                dtype="float32",
                shape=[len(target_weights)],
            ),
        },
    )


def rendered_length(messages: list[dict], tok) -> int:
    text = apply_chat_template(tok, messages, generation=False)
    return len(tok.encode(text, add_special_tokens=False))


def truncate_generation_context(messages: list[dict], tok, budget: int) -> list[dict]:
    """Drop old dialogue turns while retaining the system prompt and recent context."""
    current = list(messages)
    while rendered_length(current, tok) > budget:
        system_count = 1 if current and current[0]["role"] == "system" else 0
        if len(current) - system_count <= 1:
            break
        current = current[:system_count] + current[system_count + 1 :]
    return current


def sample_chat(
    sampler,
    tok,
    conversations: Sequence[Sequence[dict]],
    *,
    max_tokens: int,
    seeds: Sequence[int],
    progress: str,
) -> list[dict]:
    """Sample conversations in bounded chunks, preserving deterministic seeds."""
    outputs: list[dict] = []
    for start in range(0, len(conversations), C.SAMPLE_CHUNK):
        chunk = conversations[start : start + C.SAMPLE_CHUNK]
        chunk_seeds = seeds[start : start + C.SAMPLE_CHUNK]
        pending = []
        for messages, seed in zip(chunk, chunk_seeds, strict=True):
            rendered = apply_chat_template(tok, list(messages), generation=True)
            prompt = types.ModelInput.from_ints(tok.encode(rendered, add_special_tokens=False))
            pending.append(
                sampler.sample(
                    prompt=prompt,
                    num_samples=1,
                    sampling_params=types.SamplingParams(
                        max_tokens=max_tokens,
                        temperature=C.GEN_TEMPERATURE,
                        top_p=C.GEN_TOP_P,
                        seed=int(seed),
                    ),
                )
            )
        for future in pending:
            sequence = future.result(timeout=900).sequences[0]
            text = clean(tok.decode(sequence.tokens, skip_special_tokens=False))
            outputs.append(
                {
                    "text": text,
                    "tokens": len(sequence.tokens),
                    "hit_cap": len(sequence.tokens) >= max_tokens,
                }
            )
        print(f"[sample] {progress}: {len(outputs)}/{len(conversations)}", flush=True)
    return outputs


def seed(*parts) -> int:
    return int(C.stable_hash(list(parts))[:12], 16) % (2**31 - 1)


def add_connection_arguments(parser) -> None:
    parser.add_argument(
        "--base-url", default=os.getenv("TINKER_BASE_URL", "http://localhost:10610")
    )
    parser.add_argument("--api-key", default=os.getenv("TINKER_API_KEY"))
    parser.add_argument("--model", default=C.MODEL)
    parser.add_argument("--work-dir", type=Path, default=C.HERE / "work")


def write_checkpoint_record(work_dir: Path, stage: str, value: dict) -> None:
    C.write_json(work_dir / "checkpoints" / f"{stage}.json", value)


def read_checkpoint_record(work_dir: Path, stage: str) -> dict:
    path = work_dir / "checkpoints" / f"{stage}.json"
    if not path.exists():
        raise SystemExit(f"missing {path}; complete the {stage.upper()} stage first")
    return C.read_json(path)
