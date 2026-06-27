# 06. Multi-LoRA Scheduling Benchmark

## Motivation

In a multi-tenant RLHF system, multiple LoRA adapters are served concurrently. If requests are scheduled **sequentially per adapter** (each adapter gets a dedicated batch), throughput is suboptimal. If requests are scheduled as a **mixed batch across adapters** (multi-LoRA), the engine can utilize GPU compute more efficiently — but this is the condition that introduces the batch-position mismatch characterized in `02_request_batch_mismatch`.

This experiment benchmarks the **throughput tradeoff**: sequential-per-adapter vs. multi-LoRA batched inference on vLLM, quantifying the speedup that motivates the multi-tenant scheduling design.

## Story Connection

This provides the "why we batch together" motivation: mixed-adapter batching gives significant throughput gains (shown here), but introduces logprob non-determinism (shown in `02_request_batch_mismatch`). The two together make the case for needing bias correction.

## Files

| File | Description |
|------|-------------|
| `benchmark.py` | Core benchmark: multi-LoRA batched vs. sequential-per-adapter, using vLLM offline engine |
| `benchmark_api.py` | Same benchmark via OpenAI-compatible API (online serving mode) |
| `analyze_results.py` | Parse and plot benchmark output JSON |
| `run_benchmark.sh` | One-click: create adapters → start server → run → kill → analyze |
| `start_vllm_server.sh` | Start vLLM server with multi-LoRA config |
| `create_mock_loras.py` | Generate random LoRA adapter weights for testing |
| `prepare_data.py` | Prepare request dataset from HuggingFaceH4/no_robots |
| `data/` | Cached request data (small) |

## Running

```bash
# Full pipeline (requires GPU)
bash run_benchmark.sh --tp-size 1 --port 8000 --max-tokens 256

# Or step by step:
python create_mock_loras.py --num-adapters 5 --rank 16 --base-model Qwen/Qwen3-8B
bash start_vllm_server.sh
python benchmark_api.py
python analyze_results.py
```

## Notes

- `mock_loras/` (~147 MB, random adapter weights) is **not included** in the repo.
  Regenerate with: `python create_mock_loras.py`
- Base model tested: `Qwen/Qwen3-8B` (local)
- Metrics: throughput (tokens/sec), average latency (sec/request)
