"""Integration tests for Data Parallel inference load balancing and training.

These tests require:
- A running TuFT server with data_parallel_size > 1
- Multiple GPUs available

Run with:
    pytest tests/test_data_parallel.py -m integration -v

Or directly:
    python tests/test_data_parallel.py
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx
import pytest
import tinker
from tinker import types
from transformers import AutoTokenizer

# Configuration (can be overridden via env vars)
BASE_URL = os.getenv("TUFT_BASE_URL", "http://localhost:10610")
API_KEY = os.getenv("TUFT_API_KEY", "tml-tuft-dev-key")
BASE_MODEL = os.getenv("TUFT_BASE_MODEL", "Qwen/Qwen3-4B-Thinking-2507")
MODEL_PATH = os.getenv("TUFT_MODEL_PATH", "/mnt/workspace/shared/qwen/Qwen3-4B-Thinking-2507")
NUM_CONCURRENT_REQUESTS = 16  # Send enough requests to hit all DP instances

# Mark all tests as integration (require running server + GPU)
pytestmark = [
    pytest.mark.integration,
    pytest.mark.gpu,
]


def get_gpu_utilization():
    """Get GPU memory and compute utilization."""
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
    )
    gpus = []
    for line in result.stdout.strip().split("\n"):
        parts = [p.strip() for p in line.split(",")]
        gpus.append(
            {"index": int(parts[0]), "memory_used_mib": int(parts[1]), "gpu_util_pct": int(parts[2])}
        )
    return gpus


def send_chat_request(client: httpx.Client, request_id: int) -> dict:
    """Send a single chat completion request."""
    payload = {
        "model": BASE_MODEL,
        "messages": [
            {"role": "user", "content": f"Write a short poem about the number {request_id}. Be creative."}
        ],
        "max_tokens": 200,
        "temperature": 0.8,
    }
    start = time.perf_counter()
    resp = client.post(
        f"{BASE_URL}/oai/api/v1/chat/completions",
        json=payload,
        headers={"X-API-Key": API_KEY},
        timeout=120.0,
    )
    elapsed = time.perf_counter() - start
    data = resp.json()
    return {
        "request_id": request_id,
        "status": resp.status_code,
        "elapsed_s": round(elapsed, 2),
        "response_id": data.get("id", ""),
        "content_preview": (
            data.get("choices", [{}])[0].get("message", {}).get("content", "")[:80]
            if resp.status_code == 200
            else data.get("detail", str(data))[:80]
        ),
    }


def test_concurrent_inference():
    """Test 1: Send concurrent requests to verify DP load balancing."""
    print("=" * 70)
    print("TEST 1: Concurrent Inference Load Balancing (Data Parallel)")
    print("=" * 70)
    print(f"  Sending {NUM_CONCURRENT_REQUESTS} concurrent requests to {BASE_URL}")
    print(f"  Model: {BASE_MODEL}")
    print(f"  Expected: requests distributed across 4 DP vLLM instances")
    print()

    # Check GPU state before
    print("[Before] GPU utilization:")
    for gpu in get_gpu_utilization():
        status = "LOADED" if gpu["memory_used_mib"] > 5000 else "idle"
        print(f"  GPU {gpu['index']}: mem={gpu['memory_used_mib']}MiB  util={gpu['gpu_util_pct']}%  [{status}]")
    print()

    # Send concurrent requests
    results = []
    start_all = time.perf_counter()

    with httpx.Client() as client:
        with ThreadPoolExecutor(max_workers=NUM_CONCURRENT_REQUESTS) as executor:
            futures = {
                executor.submit(send_chat_request, client, i): i
                for i in range(NUM_CONCURRENT_REQUESTS)
            }
            for future in as_completed(futures):
                results.append(future.result())

    total_time = time.perf_counter() - start_all
    results.sort(key=lambda x: x["request_id"])

    # Print results
    print(f"[Results] {NUM_CONCURRENT_REQUESTS} requests completed in {total_time:.2f}s")
    print()
    successes = sum(1 for r in results if r["status"] == 200)
    failures = NUM_CONCURRENT_REQUESTS - successes
    print(f"  Successes: {successes}/{NUM_CONCURRENT_REQUESTS}")
    if failures:
        print(f"  Failures: {failures}")
    print()

    # Show first few results
    print("  Sample responses:")
    for r in results[:4]:
        print(f"    req#{r['request_id']}: {r['elapsed_s']}s | id={r['response_id'][:50]}...")
        print(f"      -> {r['content_preview'][:60]}...")
    print()

    # Check GPU state during/after (need a small delay for nvidia-smi to capture)
    time.sleep(1)
    print("[After] GPU utilization:")
    for gpu in get_gpu_utilization():
        status = "LOADED" if gpu["memory_used_mib"] > 5000 else "idle"
        print(f"  GPU {gpu['index']}: mem={gpu['memory_used_mib']}MiB  util={gpu['gpu_util_pct']}%  [{status}]")
    print()

    # Verify: all requests should succeed
    assert successes == NUM_CONCURRENT_REQUESTS, f"Some requests failed: {failures} failures"
    print("  [PASS] All concurrent inference requests succeeded")
    print()
    return True


def test_training():
    """Test 2: Verify training still works on its dedicated GPU."""
    print("=" * 70)
    print("TEST 2: Training (LoRA forward/backward + optim step)")
    print("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    service_client = tinker.ServiceClient(base_url=BASE_URL, api_key=API_KEY)
    print(f"  Connected to {BASE_URL}")

    # Create training client
    training_client = service_client.create_lora_training_client(
        base_model=BASE_MODEL,
        rank=8,
        train_mlp=True,
        train_attn=True,
        train_unembed=True,
    )
    print(f"  Created LoRA training client (rank=8)")

    # Prepare a training datum
    messages = [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "2+2 equals 4."},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    tokens = tokenizer.encode(text, add_special_tokens=False)
    # Shift: input = tokens[:-1], target = tokens[1:]
    input_tokens = tokens[:-1]
    target_tokens = tokens[1:]
    # Simple weights: 0 for prompt prefix, 1 for last portion
    n = len(target_tokens)
    weights = [0.0] * max(0, n - 10) + [1.0] * min(10, n)

    datum = types.Datum(
        model_input=types.ModelInput.from_ints(input_tokens),
        loss_fn_inputs={
            "target_tokens": types.TensorData(
                data=target_tokens, dtype="int64", shape=[len(target_tokens)]
            ),
            "weights": types.TensorData(
                data=weights, dtype="float32", shape=[len(weights)]
            ),
        },
    )
    print(f"  Prepared training datum: {len(input_tokens)} input tokens")

    # Forward + backward
    start = time.perf_counter()
    fb_output = training_client.forward_backward([datum], loss_fn="cross_entropy").result()
    fb_time = time.perf_counter() - start
    print(f"  Forward+Backward completed in {fb_time:.2f}s")

    # Get loss value
    loss_outputs = fb_output.loss_fn_outputs
    if loss_outputs:
        logprobs = loss_outputs[0].get("logprobs")
        if logprobs and hasattr(logprobs, "data"):
            avg_logprob = sum(logprobs.data) / max(len(logprobs.data), 1)
            print(f"  Average logprob: {avg_logprob:.4f}")

    # Optim step
    start = time.perf_counter()
    optim_resp = training_client.optim_step(
        adam_params=types.AdamParams(learning_rate=1e-4)
    ).result()
    optim_time = time.perf_counter() - start
    grad_norm = getattr(optim_resp, 'grad_norm', None) or getattr(optim_resp, 'gradient_norm', None)
    print(f"  Optim step completed in {optim_time:.2f}s")
    if grad_norm is not None:
        print(f"  Grad norm: {grad_norm:.4f}")

    print()
    print("  [PASS] Training forward/backward + optim step succeeded")
    print()
    return True


def test_inference_after_training():
    """Test 3: Verify inference still works after training (save weights + sample via tinker SDK)."""
    print("=" * 70)
    print("TEST 3: Save Weights & Sample via tinker SDK (inference after training)")
    print("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    service_client = tinker.ServiceClient(base_url=BASE_URL, api_key=API_KEY)

    training_client = service_client.create_lora_training_client(
        base_model=BASE_MODEL,
        rank=8,
        train_mlp=True,
        train_attn=True,
        train_unembed=True,
    )

    # Save weights for sampler
    start = time.perf_counter()
    save_result = training_client.save_weights_for_sampler(name="dp-test-sampler").result()
    save_time = time.perf_counter() - start
    print(f"  Saved weights in {save_time:.2f}s, path={save_result.path}")

    # Create sampling client and sample
    sampling_client = service_client.create_sampling_client(model_path=save_result.path)

    messages = [{"role": "user", "content": "Explain data parallelism in one sentence."}]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)

    start = time.perf_counter()
    sample_result = sampling_client.sample(
        prompt=types.ModelInput.from_ints(prompt_tokens),
        num_samples=1,
        sampling_params=types.SamplingParams(max_tokens=128, temperature=0.7),
    ).result()
    sample_time = time.perf_counter() - start

    response = tokenizer.decode(sample_result.sequences[0].tokens)
    print(f"  Sampled in {sample_time:.2f}s")
    print(f"  Response: {response[:200]}")
    print()
    print("  [PASS] Inference after training (tinker SDK path) succeeded")
    print()
    return True


def test_oai_lora_dp_routing():
    """Test 3b: Verify OAI API LoRA inference works across ALL DP instances.

    This tests the fix for the bug where _ensure_lora_loaded() only loaded
    the LoRA on one DP instance. We send multiple OAI requests with the
    tinker:// model path, which should hit different DP instances via round-robin.
    All must succeed for the fix to be validated.
    """
    print("=" * 70)
    print("TEST 3b: OAI API LoRA Inference Across ALL DP Instances")
    print("=" * 70)

    service_client = tinker.ServiceClient(base_url=BASE_URL, api_key=API_KEY)

    # Create training client and save weights to get a tinker:// path
    training_client = service_client.create_lora_training_client(
        base_model=BASE_MODEL,
        rank=8,
        train_mlp=True,
        train_attn=True,
        train_unembed=True,
    )
    save_result = training_client.save_weights_for_sampler(name="dp-oai-lora-test").result()
    tinker_path = save_result.path
    print(f"  LoRA saved at: {tinker_path}")

    # Send 8 OAI requests using tinker:// path as model name.
    # With 4 DP instances and round-robin, this ensures each instance gets at least 2 requests.
    # If LoRA is not loaded on all instances, some requests will fail.
    num_requests = 8
    print(f"  Sending {num_requests} OAI requests with LoRA model (round-robin across 4 DP instances)")
    print(f"  Each instance should handle ~2 requests with the LoRA adapter loaded.")
    print()

    results = []
    with httpx.Client() as client:
        for i in range(num_requests):
            payload = {
                "model": tinker_path,
                "messages": [{"role": "user", "content": f"Say hello #{i}"}],
                "max_tokens": 32,
                "temperature": 0.5,
            }
            start = time.perf_counter()
            resp = client.post(
                f"{BASE_URL}/oai/api/v1/chat/completions",
                json=payload,
                headers={"X-API-Key": API_KEY},
                timeout=120.0,
            )
            elapsed = time.perf_counter() - start
            results.append({
                "id": i,
                "status": resp.status_code,
                "elapsed": round(elapsed, 2),
                "ok": resp.status_code == 200,
                "detail": resp.json().get("detail", "")[:80] if resp.status_code != 200 else "",
            })

    successes = sum(1 for r in results if r["ok"])
    failures = num_requests - successes
    print(f"  Results: {successes}/{num_requests} succeeded")
    for r in results:
        status_str = "OK" if r["ok"] else f"FAIL({r['status']}: {r['detail']})"
        print(f"    req#{r['id']}: {r['elapsed']}s [{status_str}]")
    print()

    if failures > 0:
        print(f"  [FAIL] {failures} requests failed - LoRA not loaded on all DP instances!")
        return False

    print("  [PASS] All OAI LoRA requests succeeded across all DP instances")
    print()
    return True


def test_sustained_concurrent_load():
    """Test 4: Sustained concurrent load for 30 seconds to show all GPUs active."""
    print("=" * 70)
    print("TEST 4: Sustained Concurrent Load (30 seconds)")
    print("=" * 70)

    duration_seconds = 30
    concurrency = 14  # 2 requests per DP instance in flight at any time
    print(f"  Duration: {duration_seconds}s")
    print(f"  Concurrency: {concurrency} in-flight requests")
    print(f"  DP instances: 7 (each handling ~2 concurrent requests)")
    print()

    results = []
    errors = []
    start_time = time.perf_counter()
    request_counter = [0]  # mutable counter for threads

    def send_request(client: httpx.Client) -> dict:
        req_id = request_counter[0]
        request_counter[0] += 1
        payload = {
            "model": BASE_MODEL,
            "messages": [
                {"role": "user", "content": f"Write a creative haiku about number {req_id % 100}. Be brief."}
            ],
            "max_tokens": 128,
            "temperature": 0.9,
        }
        req_start = time.perf_counter()
        try:
            resp = client.post(
                f"{BASE_URL}/oai/api/v1/chat/completions",
                json=payload,
                headers={"X-API-Key": API_KEY},
                timeout=120.0,
            )
            elapsed = time.perf_counter() - req_start
            return {"id": req_id, "status": resp.status_code, "elapsed": elapsed, "ok": resp.status_code == 200}
        except Exception as e:
            elapsed = time.perf_counter() - req_start
            return {"id": req_id, "status": 0, "elapsed": elapsed, "ok": False, "error": str(e)}

    # Sustained load loop
    with httpx.Client() as client:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = []
            # Keep submitting requests for the duration
            while time.perf_counter() - start_time < duration_seconds:
                # Maintain concurrency level
                # Remove completed futures
                done = [f for f in futures if f.done()]
                for f in done:
                    results.append(f.result())
                    futures.remove(f)

                # Submit new requests to maintain concurrency
                while len(futures) < concurrency and (time.perf_counter() - start_time) < duration_seconds:
                    futures.append(executor.submit(send_request, client))

                time.sleep(0.05)  # Small sleep to avoid busy-waiting

            # Wait for remaining futures
            for f in futures:
                results.append(f.result())

    total_time = time.perf_counter() - start_time
    successes = sum(1 for r in results if r["ok"])
    failures = len(results) - successes
    avg_latency = sum(r["elapsed"] for r in results) / max(len(results), 1)
    p50 = sorted(r["elapsed"] for r in results)[len(results) // 2] if results else 0
    p99 = sorted(r["elapsed"] for r in results)[int(len(results) * 0.99)] if results else 0
    throughput = len(results) / total_time

    print(f"  === Results ===")
    print(f"  Total requests:  {len(results)}")
    print(f"  Successes:       {successes}/{len(results)}")
    print(f"  Failures:        {failures}")
    print(f"  Duration:        {total_time:.1f}s")
    print(f"  Throughput:      {throughput:.1f} req/s")
    print(f"  Avg latency:     {avg_latency:.2f}s")
    print(f"  P50 latency:     {p50:.2f}s")
    print(f"  P99 latency:     {p99:.2f}s")
    print()

    # Check GPU utilization
    print("  [Final] GPU utilization:")
    for gpu in get_gpu_utilization():
        status = "ACTIVE" if gpu["gpu_util_pct"] > 0 else ("LOADED" if gpu["memory_used_mib"] > 5000 else "idle")
        print(f"    GPU {gpu['index']}: mem={gpu['memory_used_mib']}MiB  util={gpu['gpu_util_pct']}%  [{status}]")
    print()

    assert successes == len(results), f"{failures} requests failed"
    print(f"  [PASS] Sustained load test: {throughput:.1f} req/s with {concurrency} concurrency")
    print()
    return True


def main():
    print()
    print("TuFT Data Parallel Inference + Training Integration Test")
    print("=" * 70)
    print(f"  Server: {BASE_URL}")
    print(f"  Model: {BASE_MODEL}")
    print(f"  Config: data_parallel_size=7, tensor_parallel_size=1, fsdp_num_gpus=1")
    print(f"  GPU layout: GPU 0 = training, GPU 1-7 = inference DP instances")
    print()

    all_pass = True

    # Test 1: Concurrent inference
    try:
        test_concurrent_inference()
    except Exception as e:
        print(f"  [FAIL] {e}")
        all_pass = False

    # Test 2: Training
    try:
        test_training()
    except Exception as e:
        print(f"  [FAIL] {e}")
        all_pass = False

    # Test 3: Inference after training
    try:
        test_inference_after_training()
    except Exception as e:
        print(f"  [FAIL] {e}")
        all_pass = False

    # Test 3b: OAI API LoRA across all DP instances
    try:
        if not test_oai_lora_dp_routing():
            all_pass = False
    except Exception as e:
        print(f"  [FAIL] {e}")
        all_pass = False

    # Test 4: Sustained load
    try:
        test_sustained_concurrent_load()
    except Exception as e:
        print(f"  [FAIL] {e}")
        all_pass = False

    # Summary
    print("=" * 70)
    if all_pass:
        print("ALL TESTS PASSED - DP inference + training working correctly!")
    else:
        print("SOME TESTS FAILED - check output above")
    print("=" * 70)


if __name__ == "__main__":
    main()
