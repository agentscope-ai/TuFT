"""Experiment configuration for mode-switch micro-benchmark.

Compares 4 GPU sharing approaches for RLHF-style sampling + training workloads:
  1. Coexistence (no switch)
  2. Process-level switch (kill & reload)
  3. Sleep/Wake switch (vLLM sleep + FSDP offload simulation)
  4. FlexGPU zero-copy switch (CUDA IPC alias)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────

EXP_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = EXP_ROOT / "results"
BENCH_ROOT = Path("/mnt/nas/hanzhang.yhz/lora_rl_bench")
TUFT_ROOT = Path("/mnt/nas/hanzhang.yhz/flex_backend/TuFT")
QWEN3_ROOT = Path("/mnt/cpfs/shared/checkpoints/qwen/qwen3")

# ─── Model Registry ───────────────────────────────────────────────────────────

MODELS = {
    "qwen3-4b": {
        "name": "Qwen3-4B-Base",
        "path": str(QWEN3_ROOT / "Qwen3-4B-Base"),
        "hidden_size": 2560,
        "num_layers": 36,
        "num_heads": 32,
        "num_kv_heads": 8,
        "intermediate_size": 9728,
        "vocab_size": 151936,
        "tie_word_embeddings": True,
        "dtype": "bfloat16",
        # Approximate model weight in GB (bf16)
        "weight_gb": 8.0,
    },
    "qwen3-32b": {
        "name": "qwen3-32B",
        "path": str(QWEN3_ROOT / "qwen3-32B"),
        "hidden_size": 5120,
        "num_layers": 64,
        "num_heads": 64,
        "num_kv_heads": 8,
        "intermediate_size": 25600,
        "vocab_size": 151936,
        "tie_word_embeddings": False,
        "dtype": "bfloat16",
        "weight_gb": 64.0,
    },
}

# ─── Workload Parameters ──────────────────────────────────────────────────────
# Sized to saturate GPU compute for meaningful throughput comparison.


@dataclass
class WorkloadConfig:
    """Training and sampling workload parameters."""

    train_batch_size: int = 8
    train_seq_len: int = 2048
    num_prompts: int = 64
    max_tokens: int = 512
    max_model_len: int = 4096
    # LoRA config (same across all approaches for fairness)
    lora_rank: int = 8
    lora_alpha: int = 16
    lr: float = 1e-4
    # Measurement
    warmup_rounds: int = 1
    measure_rounds: int = 3
    seed: int = 42


WORKLOADS = {
    "qwen3-4b": WorkloadConfig(
        train_batch_size=4,
        train_seq_len=1024,
        num_prompts=64,
        max_tokens=512,
        max_model_len=4096,
    ),
    "qwen3-32b": WorkloadConfig(
        train_batch_size=2,
        train_seq_len=512,
        num_prompts=32,
        max_tokens=256,
        max_model_len=2048,
    ),
}

# ─── TP/DP Configurations ─────────────────────────────────────────────────────


@dataclass
class ParallelConfig:
    """TP/DP configuration for one experiment run."""

    tp_size: int
    dp_size: int
    num_gpus: int = 4  # total GPUs used = tp_size * dp_size (but we run 1 group)


EXPERIMENTS = {
    "qwen3-4b": [
        ParallelConfig(tp_size=1, dp_size=1, num_gpus=1),  # single-GPU 4B experiment only
    ],
    "qwen3-32b": [
        ParallelConfig(tp_size=2, dp_size=2),  # run 1 TP=2 group, multiply by dp=2
        ParallelConfig(tp_size=4, dp_size=1),  # all 4 GPUs
    ],
}

# ─── Approach Definitions ─────────────────────────────────────────────────────

APPROACHES = {
    "coexistence": {
        "id": 1,
        "name": "Coexistence (No Switch)",
        "description": "vLLM sampling + FSDP training coexist in GPU memory. No switching overhead.",
        "training_backend": "fsdp",
        "sampling_backend": "vllm",
    },
    "process_switch": {
        "id": 2,
        "name": "Process-level Switch",
        "description": "Kill and reload processes for mode switch. Full model reload from disk.",
        "training_backend": "fsdp",
        "sampling_backend": "vllm",
    },
    "sleep_wake": {
        "id": 3,
        "name": "Sleep/Wake Switch",
        "description": "vLLM sleep/wake + simulated FSDP CPU offload/reload.",
        "training_backend": "fsdp",
        "sampling_backend": "vllm",
    },
    "flexgpu": {
        "id": 4,
        "name": "FlexGPU Zero-Copy",
        "description": "PyTorch-TP training + vLLM sampling with CUDA IPC zero-copy switch.",
        "training_backend": "pytorch_tp",
        "sampling_backend": "vllm_zerocopy",
    },
}

# ─── GPU Memory Utilization Strategy ─────────────────────────────────────────
# For Approach 1 (coexistence), vLLM gets a fraction of GPU memory.
# The rest is left for FSDP training. If training OOMs, we record it.

GPU_TOTAL_GB = 80.0  # A100-80GB


def get_coexistence_gpu_mem_util(model_key: str, tp_size: int) -> float:
    """Compute gpu_memory_utilization for coexistence approach.

    In coexistence mode, BOTH vLLM and FSDP load model weights onto the same GPU.
    vLLM's gpu_memory_utilization controls its TOTAL budget (model weights + KV cache).

    Per-GPU memory budget:
      vLLM:  gpu_mem_util * 80GB (includes model_weight/tp + KV cache)
      FSDP:  model_weight/tp + activations + overhead
      Total: must fit in 80GB

    If both model copies don't fit, returns a minimal value and training will OOM.
    """
    model = MODELS[model_key]
    weight_per_gpu_gb = model["weight_gb"] / tp_size

    # FSDP training memory estimate
    workload = WORKLOADS[model_key]
    # Activation memory (rough: batch * seq * hidden * layers * bytes * factor / tp)
    act_gb = (
        workload.train_batch_size
        * workload.train_seq_len
        * model["hidden_size"]
        * 2  # bytes (bf16)
        * 6  # factor for forward+backward+grad+residual
        / (1024**3)
    )
    # FSDP shards model across GPUs; optimizer states are small (LoRA only)
    fsdp_need_gb = weight_per_gpu_gb + act_gb + 3.0  # +3GB overhead

    # vLLM needs at minimum: model weights per GPU + some KV cache
    vllm_min_gb = weight_per_gpu_gb + 2.0  # absolute minimum for vLLM

    # Check if both can fit at all
    total_min_gb = fsdp_need_gb + vllm_min_gb
    if total_min_gb > GPU_TOTAL_GB - 2.0:
        # Both model copies barely fit or don't fit.
        # Give vLLM minimal allocation; training will likely OOM.
        # This is the expected result for large models (e.g., 32B TP=2).
        return max(0.10, vllm_min_gb / GPU_TOTAL_GB)

    # vLLM gets remaining memory after FSDP
    vllm_budget_gb = GPU_TOTAL_GB - fsdp_need_gb - 2.0  # 2GB safety margin
    util = vllm_budget_gb / GPU_TOTAL_GB

    # Ensure vLLM at least gets enough for model weights + minimal KV cache
    min_util = (weight_per_gpu_gb + 1.0) / GPU_TOTAL_GB
    return max(min_util, min(util, 0.85))


def get_exclusive_gpu_mem_util(model_key: str, tp_size: int) -> float:
    """gpu_memory_utilization when vLLM has exclusive GPU access (approaches 2, 3)."""
    model = MODELS[model_key]
    weight_per_gpu_gb = model["weight_gb"] / tp_size
    # Leave room for model weights + KV cache
    if weight_per_gpu_gb > 30:
        return 0.85
    return 0.90


def get_flexgpu_gpu_mem_util(model_key: str, tp_size: int) -> float:
    """gpu_memory_utilization for FlexGPU zero-copy vLLM dummy runtime.

    FlexGPU keeps PyTorch base storage alive while vLLM allocates KV cache.
    vLLM's gpu_memory_utilization is interpreted against total GPU memory,
    so it must leave room for the shared base storage that is not owned by vLLM.
    """
    model = MODELS[model_key]
    weight_per_gpu_gb = model["weight_gb"] / tp_size
    safety_gb = 4.0
    util = (GPU_TOTAL_GB - weight_per_gpu_gb - safety_gb) / GPU_TOTAL_GB
    # Historical stable configs: 32B TP=4 works well at ~0.70 for long outputs.
    if model_key == "qwen3-32b" and tp_size == 4:
        return min(util, 0.70)
    return max(0.10, min(util, 0.85))


# ─── Sampling Prompts ─────────────────────────────────────────────────────────

SAMPLING_PROMPTS = [
    "What is machine learning and how does it differ from traditional programming?",
    "Explain quantum computing in simple terms that a beginner can understand.",
    "Summarize the key benefits and challenges of distributed training systems.",
    "Describe how the attention mechanism works in transformer architectures.",
    "Explain zero-copy memory sharing and why it matters for GPU computing.",
    "Write a comprehensive overview of reinforcement learning from human feedback.",
    "Compare tensor parallelism and data parallelism for large language models.",
    "Explain why batching and continuous batching improve inference throughput.",
    "Describe the architecture of modern large language models in detail.",
    "What are the main challenges in deploying LLMs in production environments?",
    "Explain the concept of mixture of experts in neural network architectures.",
    "How does gradient checkpointing reduce memory usage during training?",
    "Describe the differences between autoregressive and masked language models.",
    "Explain how KV cache works and why it is important for LLM inference.",
    "What are the trade-offs between model parallelism and pipeline parallelism?",
    "Describe the process of fine-tuning a large language model with LoRA.",
]


def get_prompts(n: int) -> list[str]:
    """Get n sampling prompts (cycling through the pool)."""
    return [SAMPLING_PROMPTS[i % len(SAMPLING_PROMPTS)] for i in range(n)]
