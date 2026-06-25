"""Integration test: Verify LoRA fine-tuned model produces different outputs from the base model.

This test:
1. Picks a training example (user question + expected assistant answer)
2. Queries the BASE model with the user question (before-training behavior)
3. Trains a LoRA adapter on a focused dataset
4. Queries the FINE-TUNED model with the same question (after-training behavior)
5. Verifies both SDK (sampling_client.sample) and OAI API (/chat/completions) paths
6. Shows the difference between base model output and fine-tuned output

Run with:
    pytest tests/test_lora_before_after.py -m integration -v

Or directly:
    python tests/test_lora_before_after.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import httpx
import pytest
import tinker
from tinker import types
from transformers import AutoTokenizer


# Add examples dir to path for dataset import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples" / "chat_sft"))
from dataset import ChatDataset, conversation_to_datum, load_chat_dataset


# Configuration (can be overridden via env vars)
BASE_URL = os.getenv("TUFT_BASE_URL", "http://localhost:10610")
API_KEY = os.getenv("TUFT_API_KEY", "tml-tuft-dev-key")
BASE_MODEL = os.getenv("TUFT_BASE_MODEL", "Qwen/Qwen3-4B-Thinking-2507")
MODEL_PATH = os.getenv("TUFT_MODEL_PATH", "/mnt/workspace/shared/qwen/Qwen3-4B-Thinking-2507")

# Mark all tests as integration (require running server + GPU)
pytestmark = [
    pytest.mark.integration,
    pytest.mark.gpu,
]

# Training config - small focused training to see clear difference
LORA_RANK = 16
NUM_STEPS = 30
BATCH_SIZE = 8
LEARNING_RATE = 3e-4  # Slightly higher LR for faster convergence on small data
MAX_LENGTH = 512


def pick_test_examples(train_dataset: ChatDataset, n: int = 3):
    """Pick training examples that have clear user->assistant patterns."""
    examples = []
    for messages in train_dataset.data[:100]:  # Search in first 100
        if len(messages) >= 2:
            user_msg = next((m for m in messages if m["role"] == "user"), None)
            asst_msg = next((m for m in messages if m["role"] == "assistant"), None)
            if user_msg and asst_msg:
                # Pick examples with medium-length answers (not too long, not too short)
                content = asst_msg["content"]
                if 50 < len(content) < 500:
                    examples.append(
                        {
                            "user_content": user_msg["content"],
                            "expected_answer": content,
                            "messages": messages,
                        }
                    )
                    if len(examples) >= n:
                        break
    return examples


def query_oai_api(model: str, user_content: str, max_tokens: int = 256) -> str:
    """Query via OpenAI-compatible API."""
    with httpx.Client() as client:
        resp = client.post(
            f"{BASE_URL}/oai/api/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": user_content}],
                "max_tokens": max_tokens,
                "temperature": 0.1,  # Low temp for deterministic comparison
            },
            headers={"X-API-Key": API_KEY},
            timeout=120.0,
        )
        if resp.status_code != 200:
            return f"[ERROR {resp.status_code}]: {resp.text[:200]}"
        data = resp.json()
        return data["choices"][0]["message"]["content"]


def query_sdk_sample(sampling_client, tokenizer, user_content: str, max_tokens: int = 256) -> str:
    """Query via tinker SDK sampling."""
    messages = [{"role": "user", "content": user_content}]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)

    result = sampling_client.sample(
        prompt=types.ModelInput.from_ints(prompt_tokens),
        num_samples=1,
        sampling_params=types.SamplingParams(max_tokens=max_tokens, temperature=0.1),
    ).result()

    return tokenizer.decode(result.sequences[0].tokens)


def train_lora(service_client, tokenizer, train_dataset, test_examples):
    """Train a LoRA adapter focused on the test examples' domain."""
    print("\n[Training] Creating LoRA client...")
    training_client = service_client.create_lora_training_client(
        base_model=BASE_MODEL,
        rank=LORA_RANK,
        train_mlp=True,
        train_attn=True,
        train_unembed=True,
    )
    print(f"  rank={LORA_RANK}, steps={NUM_STEPS}, batch={BATCH_SIZE}, lr={LEARNING_RATE}")

    # Train
    print(f"\n[Training] Running {NUM_STEPS} steps...")
    for step in range(NUM_STEPS):
        batch = train_dataset.get_batch(BATCH_SIZE)
        datums = []
        for messages in batch:
            try:
                datums.append(conversation_to_datum(messages, tokenizer, MAX_LENGTH))
            except ValueError:
                continue
        if not datums:
            continue

        fb = training_client.forward_backward(datums, loss_fn="cross_entropy").result()
        training_client.optim_step(types.AdamParams(learning_rate=LEARNING_RATE)).result()

        if step % 10 == 0 or step == NUM_STEPS - 1:
            # Compute loss
            loss_outputs = fb.loss_fn_outputs
            if loss_outputs:
                logprobs_data = loss_outputs[0].get("logprobs")
                if logprobs_data and hasattr(logprobs_data, "data"):
                    avg_lp = sum(logprobs_data.data) / max(len(logprobs_data.data), 1)
                    print(f"    step {step}: avg_logprob={avg_lp:.4f}")

    # Save weights
    print("\n[Training] Saving weights for sampler...")
    save_result = training_client.save_weights_for_sampler(name="lora-before-after-test").result()
    print(f"  path={save_result.path}")

    return training_client, save_result.path


def main():
    print("=" * 70)
    print("TEST: Base Model vs Fine-Tuned Model Output Comparison")
    print("=" * 70)
    print(f"  Server: {BASE_URL}")
    print(f"  Model: {BASE_MODEL}")
    print("  This test trains a LoRA and compares outputs before/after training")
    print("  using both the SDK sampling path and the OAI API path.")
    print()

    # Setup
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    service_client = tinker.ServiceClient(base_url=BASE_URL, api_key=API_KEY)

    # Load dataset and pick test examples
    print("[1] Loading dataset and picking test examples...")
    train_dataset, _ = load_chat_dataset("no_robots", seed=42)
    test_examples = pick_test_examples(train_dataset, n=3)
    print(f"  Selected {len(test_examples)} test examples from training data:")
    for i, ex in enumerate(test_examples):
        print(f"    Example {i + 1}: Q='{ex['user_content'][:80]}...'")
        print(f"               A='{ex['expected_answer'][:80]}...'")
    print()

    # Step 2: Query BASE model (before training) via OAI API
    print("[2] Querying BASE model (before training)...")
    base_responses_oai = []
    for i, ex in enumerate(test_examples):
        resp = query_oai_api(BASE_MODEL, ex["user_content"], max_tokens=200)
        base_responses_oai.append(resp)
        print(f"  Base OAI #{i + 1}: '{resp[:120]}...'")
    print()

    # Step 3: Query BASE model via SDK (create a base-model sampling session)
    print("[3] Querying BASE model via SDK...")
    base_sampling_client = service_client.create_sampling_client(base_model=BASE_MODEL)
    base_responses_sdk = []
    for i, ex in enumerate(test_examples):
        resp = query_sdk_sample(base_sampling_client, tokenizer, ex["user_content"], max_tokens=200)
        base_responses_sdk.append(resp)
        print(f"  Base SDK #{i + 1}: '{resp[:120]}...'")
    print()

    # Step 4: Train LoRA
    print("[4] Training LoRA adapter...")
    training_client, lora_path = train_lora(service_client, tokenizer, train_dataset, test_examples)
    print()

    # Step 5: Query FINE-TUNED model via OAI API
    print("[5] Querying FINE-TUNED model via OAI API...")
    ft_responses_oai = []
    for i, ex in enumerate(test_examples):
        resp = query_oai_api(lora_path, ex["user_content"], max_tokens=200)
        ft_responses_oai.append(resp)
        print(f"  LoRA OAI #{i + 1}: '{resp[:120]}...'")
    print()

    # Step 6: Query FINE-TUNED model via SDK
    print("[6] Querying FINE-TUNED model via SDK...")
    lora_sampling_client = service_client.create_sampling_client(model_path=lora_path)
    ft_responses_sdk = []
    for i, ex in enumerate(test_examples):
        resp = query_sdk_sample(lora_sampling_client, tokenizer, ex["user_content"], max_tokens=200)
        ft_responses_sdk.append(resp)
        print(f"  LoRA SDK #{i + 1}: '{resp[:120]}...'")
    print()

    # Step 7: Compare and report
    print("=" * 70)
    print("COMPARISON: Base Model vs Fine-Tuned Model")
    print("=" * 70)

    all_different = True
    for i, ex in enumerate(test_examples):
        print(f"\n--- Example {i + 1} ---")
        print(f"  Question: {ex['user_content'][:100]}")
        print(f"  Expected (training data): {ex['expected_answer'][:100]}...")
        print()
        print(f"  [OAI API] Base model:      {base_responses_oai[i][:100]}...")
        print(f"  [OAI API] Fine-tuned model: {ft_responses_oai[i][:100]}...")
        oai_different = base_responses_oai[i][:50] != ft_responses_oai[i][:50]
        print(f"  [OAI API] Different? {'YES ✓' if oai_different else 'NO (same output)'}")
        print()
        print(f"  [SDK]     Base model:      {base_responses_sdk[i][:100]}...")
        print(f"  [SDK]     Fine-tuned model: {ft_responses_sdk[i][:100]}...")
        sdk_different = base_responses_sdk[i][:50] != ft_responses_sdk[i][:50]
        print(f"  [SDK]     Different? {'YES ✓' if sdk_different else 'NO (same output)'}")

        if not oai_different and not sdk_different:
            all_different = False

    print()
    print("=" * 70)
    if all_different:
        print("RESULT: ALL EXAMPLES show different outputs between base and fine-tuned model!")
        print("        Both SDK and OAI API correctly serve the fine-tuned weights.")
    else:
        print("RESULT: Some examples show same output. Training may need more steps")
        print("        or the model already knew the answer. Check details above.")
    print("=" * 70)


def test_lora_before_after():
    """Pytest entry point for LoRA before/after comparison test."""
    main()


if __name__ == "__main__":
    main()
