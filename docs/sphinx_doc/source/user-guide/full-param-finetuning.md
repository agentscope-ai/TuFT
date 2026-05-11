# Full-Parameter Fine-Tuning

This guide explains how to perform **full-parameter fine-tuning** (i.e. training **all** model weights) using TuFT, as an alternative to LoRA-based fine-tuning. The dataset/loss setup is the same as the LoRA SFT example in `examples/chat_sft/chat_sft.ipynb` — only the model-creation step differs (using `rank=None` instead of a LoRA rank).

---

## Table of Contents

1. [When to Use Full-Param vs. LoRA](#when-to-use-full-param-vs-lora)
2. [Prerequisites](#prerequisites)
3. [Server Configuration](#server-configuration)
4. [Minimal Training Example](#minimal-training-example)
5. [Key Differences from LoRA](#key-differences-from-lora)
6. [Important Notes](#important-notes)

---

## When to Use Full-Param vs. LoRA

| Aspect | Full-Parameter Fine-Tuning | LoRA Fine-Tuning |
|---|---|---|
| **What is trained** | All model weights | Low-rank adapter matrices only |
| **GPU memory** | High — entire model gradients + optimizer states | Low — only adapter parameters |
| **Training speed** | Slower per step (more parameters) | Faster per step |
| **Best for** | Maximum quality, domain transfer, small models | Quick experiments, multi-tenant, large models |
| **Multi-tenant** | Single shared instance — no per-tenant isolation (see below) | Multiple adapters share one base model |
| **Checkpoint size** | Full model copy | Small adapter files (MBs) |

**Rule of thumb**
- Use **LoRA** when you need fast iteration, limited GPU budget, or multi-tenant adapter serving.
- Use **full-param** when you need maximum model capacity, are performing deep domain adaptation, or the model is small enough to fit in memory with full gradients.

---

## Prerequisites

1. **FSDP backend only** — Full-parameter training is implemented exclusively in the FSDP backend. The HF backend does **not** support it. Set `training_backend: fsdp` in your model config.
2. **Multi-GPU recommended** — Full-param training loads the entire model with optimizer states. FSDP shards these across GPUs. While single-GPU works for small models, multi-GPU (via `fsdp_num_gpus`) is strongly recommended.
3. **`allow_full_param: true`** — You must explicitly enable full-param training in the model config (see next section). This is a safety flag to prevent accidental full-param requests.

---

## Server Configuration

Add `allow_full_param: true` to the model entry in your `tuft_config.yaml`:

```yaml
supported_models:
  - model_name: Qwen/Qwen3-4B
    model_path: Qwen/Qwen3-4B
    max_model_len: 32768
    training_backend: fsdp       # required — only FSDP supports full-param
    fsdp_num_gpus: 2             # number of GPUs for FSDP sharding
    allow_full_param: true       # enable full-parameter training for this model
    max_lora_rank: 16            # LoRA is still available alongside full-param
    max_loras: 1
```

If `allow_full_param` is not set (defaults to `false`), a full-param training request returns HTTP 400:

```
400 Bad Request: Model does not support full-parameter training.
```

---

## Minimal Training Example

### Embedded mode (recommended)

Using TuFT's embedded Python API, set `rank=None` to request full-parameter training:

```python
import tuft

tuft.init(model="/path/to/Qwen3-4B")

training_client = tuft.create_training_client(
    base_model="Qwen3-4B",
    rank=None,  # None = full-parameter training (no LoRA)
)

# The rest of the training loop is identical to LoRA SFT
fwdbwd = training_client.forward_backward(datums, loss_fn="cross_entropy").result()

from tinker import types
training_client.optim_step(types.AdamParams(learning_rate=1e-5)).result()
```

### Service mode (Tinker SDK)

When connecting to a running TuFT server, send a `CreateModelRequest` with `lora_config=None` via the Tinker SDK:

```python
import tinker
from tinker import types

service_client = tinker.ServiceClient(
    base_url="http://localhost:10610",
    api_key=TINKER_API_KEY,
)

# Create a full-param model (lora_config=None signals full-parameter training)
model = service_client.models.create(
    types.CreateModelRequest(
        base_model="Qwen/Qwen3-4B",
        lora_config=None,  # no LoRA → full-param
    )
)

training_client = tinker.TrainingClient(
    service_client=service_client,
    model_id=model.id,
)

# Training loop — same as LoRA
fwdbwd = training_client.forward_backward(datums, loss_fn="cross_entropy").result()
training_client.optim_step(types.AdamParams(learning_rate=1e-5)).result()
```

---

## Key Differences from LoRA

| Feature | Full-Param | LoRA |
|---|---|---|
| **Model creation** | `rank=None` / `lora_config=None` | `rank=32` / `lora_config=LoraConfig(...)` |
| **`forward_backward()`** | Same API | Same API |
| **`optim_step()`** | Same API | Same API |
| **Checkpoint** | Saves full model weights | Saves adapter weights only |
| **`get_model_info()`** | `is_lora=False`, `lora_rank=None` | `is_lora=True`, `lora_rank=<rank>` |

The training loop (forward/backward, optimizer step, evaluation) uses exactly the same API. Only the model-creation step differs.

---

## Important Notes

### Multi-tenancy limitation

Unlike LoRA — where each tenant gets an isolated adapter (independent parameter subset) sharing one frozen base model — full-parameter training modifies **all** model weights. The current implementation maintains a **single shared `FullParamVerlWorker`** for all full-param tenants:

- Multiple `model_id`s registered via `init_full_param()` share the **same** weights and optimizer.
- If tenant A trains a step, the weights that tenant B sees on the next forward are already mutated.
- There is **no parameter-level isolation** between full-param tenants.

**Why**: Isolating full-param tenants would require maintaining separate full copies of the model (one per tenant), which is prohibitively expensive in GPU memory.

**Future roadmap for true multi-tenant full-param isolation:**

1. Maintain multiple `FullParamVerlWorker` instances (one per tenant).
2. Only one worker's parameters reside on GPU at any time.
3. On tenant switch: current worker `engine.to("cpu")` to offload; incoming worker `engine.to("cuda")` to reload.
4. A scheduler at the backend level manages the swap queue and prevents OOM.
5. Expected cost: full model CPU↔GPU transfer on every switch (seconds for multi-GB models), unsuitable for rapid interleaved workloads.

For now, if you need independent full-param training for multiple users, deploy **separate TuFT instances** (each with its own GPU allocation).

### Performance not optimized

TuFT's full-parameter fine-tuning support is **functional but not performance-optimized**. The current implementation focuses on correctness and API compatibility. Compared to dedicated full-param training frameworks (e.g. DeepSpeed ZeRO, Megatron-LM), you may observe:

- Lower GPU utilization / throughput
- Higher memory overhead per GPU
- Lack of advanced optimizations such as gradient checkpointing, mixed-precision sharding strategies, or communication overlap

If training throughput is critical for your workload, consider using a specialized distributed training framework for full-param training, and reserve TuFT for LoRA-based fine-tuning where it excels.

### Memory requirements

Full-parameter training requires significantly more GPU memory than LoRA because:
- All model parameters require gradients (vs. only adapter parameters in LoRA).
- The optimizer (e.g. Adam) maintains momentum and variance for every parameter.
- FSDP shards these across GPUs, but total memory is still much larger.

**Recommendation**: Use at least 2× the number of GPUs you would use for LoRA on the same model. For a 4B-parameter model, 2× A100-80GB is a reasonable starting point.

### Learning rate

Full-param fine-tuning typically uses a **lower learning rate** than LoRA:
- LoRA default: `1e-4`
- Full-param recommended: `1e-5` to `5e-5`

Higher learning rates with full-param can cause catastrophic forgetting or training instability.

### HF backend not supported

If you request full-param training on a model configured with `training_backend: hf`, the server will reject the request:
- If `allow_full_param: false` (default): HTTP 400
- If `allow_full_param: true` but the backend is HF: HTTP 500 (the HF backend raises `NotImplementedError`)

Always use `training_backend: fsdp` for full-parameter training.

### Checkpoint format

Full-param checkpoints contain the **entire model weights**, unlike LoRA checkpoints which only contain adapter weights. This means:
- Checkpoint files are much larger (GBs vs. MBs).
- Loading a checkpoint restores all model weights, not just adapter deltas.
