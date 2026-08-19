# Custom Losses (Client-Side)

This guide documents **client-defined training objectives** on TuFT via the Tinker SDK's
`forward_backward_custom` — for example a composite objective that mixes **DPO** with a weighted
**NLL** anchor. The workflow is **officially supported on both training backends** (`hf` and
`fsdp`) with an **unmodified TuFT server**: your loss stays a plain PyTorch function in *your*
process, and the server only ever executes its built-in, allowlisted loss functions.

```python
def my_loss(data: list[types.Datum], logprobs_list: list[torch.Tensor]):
    loss = ...  # any differentiable torch scalar over logprobs_list
    return loss, {"my_metric:sum": loss.item()}


training_client.forward_backward_custom(data, my_loss, loss_type_input="logprobs")
```

---

## What You'll Learn

1. When to reach for `forward_backward_custom` instead of a built-in `loss_fn`
2. The **two-pass mechanism** the SDK lowers it onto, and why the gradients are exact
3. The **supported inputs** and the datum-level alignment contract
4. Two minimal runnable callbacks: **weighted-NLL SFT** and a **composite DPO** objective
5. The **performance cost**, **error behavior**, and **model-state invariants** to rely on

---

## Table of Contents

1. [When to Use a Custom Loss](#when-to-use-a-custom-loss)
2. [How It Works: Two-Pass Semantics](#how-it-works-two-pass-semantics)
3. [Minimal Example: SFT as a Custom Loss](#minimal-example-sft-as-a-custom-loss)
4. [Composite Example: DPO + NLL](#composite-example-dpo--nll)
5. [Supported Inputs](#supported-inputs)
6. [Performance Costs](#performance-costs)
7. [Error Handling](#error-handling)
8. [Model-State Invariants](#model-state-invariants)
9. [Q&A](#qa)

---

## When to Use a Custom Loss

TuFT ships five server-side loss functions — `cross_entropy`, `importance_sampling`, `ppo`,
`cispo`, and `dro` — selected by name in `forward_backward(data, loss_fn=...)`. The server
**allowlists these names at the wire protocol** and rejects anything else, so tenants can never
run arbitrary code on shared infrastructure.

Reach for `forward_backward_custom` when your objective is a differentiable function of the
**per-token target log-probabilities** but is not one of those five. Typical cases:

- **Preference objectives** (DPO and variants) that couple *pairs* of datums through one scalar.
- **Composite objectives**, e.g. DPO plus a weighted NLL anchor, or policy/reference
  log-ratio regularizers.
- Quick research iterations on bespoke losses without touching (or forking) the server.

If your objective *is* one of the built-ins, prefer `forward_backward`: it costs one fewer
forward pass per step ([details below](#performance-costs)).

```{admonition} Requires
:class: note

`tinker` ≥ 0.25 (the SDK version TuFT pins) and `torch` installed **on the client**. The custom
callback runs in your process only — the TuFT server never imports or executes client Python,
and no new server-side loss names are introduced.
```

---

## How It Works: Two-Pass Semantics

`forward_backward_custom(data, loss_fn, loss_type_input="logprobs")` is client-side sugar. The
SDK lowers it onto **two logical passes** that any TuFT server already serves (each pass may be
split into multiple requests when the SDK chunks a large batch):

```text
 client                                             TuFT server
 ──────                                             ───────────
 1. forward(data, loss_fn="cross_entropy")   ──►    forward pass only (no_grad)
    ◄──  loss_fn_outputs[i]["logprobs"]             per-datum target log-probs

 2. loss, metrics = loss_fn(data, logprobs)         (your PyTorch code, local autograd)
    grads[i] = ∂loss/∂logprobs[i]

 3. forward_backward(data', "cross_entropy") ──►    forward + backward, accumulates grads
       with weights[i] = -grads[i]
    ◄──  ForwardBackwardOutput (+ your metrics merged in by the SDK)
```

The trick in step 3: the server's cross-entropy is the **linear** form
`L = -(target_logprobs * weights).sum()`, so `∂L/∂logprobs = -weights = ∂loss/∂logprobs`. By
the chain rule the backward pass accumulates **exactly** `∂loss/∂θ` — the same gradients you
would get differentiating your composite loss straight through the model. TuFT's test suite
pins this equivalence on both backends (`tests/test_custom_loss.py`), including that a custom
weighted-NLL callback reproduces the built-in `cross_entropy` gradients to numerical precision.

Because both passes evaluate the *current* weights, the log-probs your callback sees are exact,
not stale as long as the model state does not change between them. The SDK awaits the forward
result before submitting the backward pass, and TuFT executes each training run's requests in
sequence-ID order.

---

## Minimal Example: SFT as a Custom Loss

Runs against any TuFT server (HF or FSDP backend) — start one as in the
[Quickstart](../getting-started/quickstart.md), then:

```python
import tinker
import torch
from tinker import types

service_client = tinker.ServiceClient(base_url="http://localhost:10610", api_key="local-dev-key")
base_model = service_client.get_server_capabilities().supported_models[0].model_name
training_client = service_client.create_lora_training_client(base_model=base_model, rank=8)
tokenizer = training_client.get_tokenizer()


def make_datum(prompt: str, completion: str) -> types.Datum:
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
    completion_tokens = tokenizer.encode(completion, add_special_tokens=False)
    tokens = prompt_tokens + completion_tokens
    # Standard next-token setup: model_input = tokens[:-1], target_tokens = tokens[1:],
    # and weights masking the prompt so only completion tokens are supervised.
    weights = [0.0] * (len(prompt_tokens) - 1) + [1.0] * len(completion_tokens)
    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens[:-1]),
        loss_fn_inputs={
            "target_tokens": types.TensorData(data=tokens[1:], dtype="int64"),
            "weights": types.TensorData(data=weights, dtype="float32"),
        },
    )


data = [
    make_datum("English: hello world\nPig Latin:", " ello-hay orld-way\n"),
    make_datum("English: banana split\nPig Latin:", " anana-bay plit-say\n"),
]


def sft_loss(data: list[types.Datum], logprobs_list: list[torch.Tensor]):
    """Batch mean of each example's response-token mean NLL."""
    per_example_nlls = []
    for datum, logprobs in zip(data, logprobs_list, strict=True):
        weights = datum.loss_fn_inputs["weights"].to_torch()
        response_token_count = weights.sum().clamp_min(1)
        per_example_nlls.append(-(logprobs * weights).sum() / response_token_count)
    loss = torch.stack(per_example_nlls).mean()
    return loss, {"sft_nll:mean": loss.item()}


for _ in range(10):
    result = training_client.forward_backward_custom(data, sft_loss).result()
    training_client.optim_step(types.AdamParams(learning_rate=1e-4)).result()
    print(result.metrics["sft_nll:mean"])
```

The callback receives one `logprobs_list[i]` per `data[i]`, **in order**, each a 1-D float32
tensor of length `len(data[i].model_input)` — `logprobs_list[i][t]` is the log-probability the
current model assigns to `target_tokens[t]` at position `t`. Your metrics dictionary is merged
into the returned `result.metrics` alongside the server's metrics for the surrogate loss.

## Composite Example: DPO + NLL

A cross-example objective no per-datum server loss could express: datums arrive as
(chosen, rejected) pairs sharing a prompt, and the loss couples each pair through one sigmoid,
then adds a chosen-response NLL anchor and a squared policy/reference log-ratio regularizer.
Capture the reference log-probs once before updating the training client; they remain constants
inside the callback.

```python
import torch.nn.functional as F

# data = [chosen_0, rejected_0, chosen_1, rejected_1, ...], built with make_datum,
# where weights mask the shared prompt (0.0) and cover the completion (1.0).
reference_result = training_client.forward(data, "cross_entropy").result()
reference_logprobs = [
    output["logprobs"].to_torch().float()
    for output in reference_result.loss_fn_outputs
]

BETA = 0.1
NLL_COEF = 0.1
LOGRATIO_MSE_COEF = 0.001


def dpo_composite_loss(data: list[types.Datum], logprobs_list: list[torch.Tensor]):
    policy_reference_sums = []
    response_nlls = []
    squared_logratios = []
    for datum, logprobs, reference in zip(
        data, logprobs_list, reference_logprobs, strict=True
    ):
        weights = datum.loss_fn_inputs["weights"].to_torch()
        logratio = logprobs - reference
        policy_reference_sums.append((logratio * weights).sum())
        response_nlls.append(-(logprobs * weights).sum() / weights.sum())
        squared_logratios.append(logratio[weights.bool()].square().mean())

    dpo_terms = []
    for i in range(0, len(policy_reference_sums), 2):
        chosen = policy_reference_sums[i]
        rejected = policy_reference_sums[i + 1]
        dpo_terms.append(-F.logsigmoid(BETA * (chosen - rejected)))

    dpo = torch.stack(dpo_terms).mean()
    chosen_nll = torch.stack(response_nlls[::2]).mean()
    logratio_mse = torch.stack(squared_logratios).mean()
    loss = dpo + NLL_COEF * chosen_nll + LOGRATIO_MSE_COEF * logratio_mse
    return loss, {
        "dpo:mean": dpo.item(),
        "chosen_nll:mean": chosen_nll.item(),
        "logratio_mse:mean": logratio_mse.item(),
        "composite:mean": loss.item(),
    }


result = training_client.forward_backward_custom(data, dpo_composite_loss).result()
training_client.optim_step(types.AdamParams(learning_rate=1e-4)).result()
```

Here the training client's initial state is the frozen reference. You can instead obtain the
reference log-probs from a separate model, as long as every reference row stays aligned with
the corresponding datum and target-token position.

---

## Supported Inputs

`loss_type_input="logprobs"` is the only supported input space (and the default). Per datum,
`loss_fn_inputs` may contain **exactly**:

| Key | Required | Dtype | Constraint |
|---|---|---|---|
| `target_tokens` | yes | `int64` | same length as `model_input` |
| `weights` | no | `float32` | same length as `target_tokens` |

The SDK **rejects any other key client-side** (e.g. `advantages`) before a request is sent.
`weights` serve two roles: your callback can read them back (e.g. as a prompt mask, as above),
and the server uses them for the pass-1 surrogate loss metric. When omitted, the SDK sends
zeros for pass 1; pass 2 always overwrites them with the gradient-derived values.

Your callback must return a tuple of a **scalar torch tensor** (differentiable w.r.t. the
log-prob tensors) and a **`dict[str, float]`** of metrics. Use names such as `"name:sum"` or
`"name:mean"` to follow the Tinker metric convention. Custom metrics are computed once over
the whole logical batch and merged client-side; TuFT does not reduce them across micro-batches.

## Performance Costs

- **One extra forward pass.** Each `forward_backward_custom` costs *two* forward passes plus
  one backward (versus one forward + one backward for a built-in loss), and one extra
  round-trip shipping log-probs to the client and weights back. The wall-clock multiplier
  depends on the model, hardware, batch size, and network latency.
- **No hidden activation memory.** Pass 1 runs under `torch.no_grad()` on both backends, so it
  allocates no autograd graph; peak memory is set by the backward pass, same as built-in
  losses.
- **Payload size.** Weights are dense float32 per token; for very large batches the SDK
  automatically chunks requests, and TuFT preserves datum order within and across chunks.
- **FSDP multi-GPU.** Both passes shard the batch across ranks, so each of the two requests
  must satisfy `len(data) >= fsdp_num_gpus` — the same constraint as `forward_backward`.

## Error Handling

- **Unknown loss names never reach a model.** `/forward_backward` validates `loss_fn` against
  the built-in allowlist at protobuf decode time and answers **422**; `forward_backward_custom`
  needs no new names, so unmodified servers accept it.
- **Malformed datum inputs** are rejected with a `loss_fn_inputs`-specific error. Unsupported
  keys are rejected by the SDK before pass 1; the server validates mismatched keys, shapes,
  and dtypes. `target_tokens` and `weights` must match the model-input length.
- **Callback exceptions stay client-side.** If your loss function raises (or an input datum
  carries an unsupported key), the error surfaces in your process. When this happens between
  the two passes, no harm is done server-side: pass 1 accumulated nothing, so the training
  run's gradient state is unchanged and you can simply retry.

## Model-State Invariants

- **Pass 1 is a pure read.** It changes no weights and accumulates no gradients (pinned by
  tests on both backends). A crashed or abandoned custom step leaves the run exactly as it was.
- **Pass 2 behaves like any `forward_backward`.** Gradients accumulate until the next
  `optim_step`, so you can mix custom and built-in objectives in one accumulation window (e.g.
  sum DPO and RL gradients before a single optimizer step).
- **Don't slip an `optim_step` between the passes.** The SDK orders the forward before the
  backward, and TuFT executes each run's requests in sequence-ID order — but another thread
  can enqueue an `optim_step` while the client callback is running. That would make pass 2
  apply gradients computed against pre-step log-probs. Sequential usage (await the returned
  future, then call `optim_step`) is safe.
- **Custom code never runs server-side.** Multi-tenant deployments keep their security
  boundary: the server executes only its five built-in loss functions.

---

## Q&A

**Q: Which SDK versions work?**
`forward_backward_custom` with `loss_type_input="logprobs"` is validated against the Tinker SDK
release TuFT pins (0.25.x, see `pyproject.toml`). The mechanism relies only on the stable
`forward` / `forward_backward` wire contract.

**Q: Can I use inputs other than log-probs (e.g. full logits)?**
Not currently — `"logprobs"` is the only `loss_type_input` the SDK offers, and TuFT returns
per-token *target* log-probs only. Losses needing full-vocabulary distributions (e.g. exact KL
to a teacher) can often be re-expressed against sampled targets; see the
[On-Policy Distillation guide](on-policy-distillation.md) for that pattern.

**Q: Do custom metrics aggregate across micro-batches?**
Your callback sees the *whole* request at once (the server's micro-batching is invisible to
it), computes metrics once, and the SDK merges them into the final result — no server-side
reduction is applied to them.

**Q: How do I debug a suspicious gradient?**
Compare against the direct computation: run your callback's math straight through a local copy
of the model (or against `training_client.forward(...)` log-probs) and check the loss values
match. `tests/test_custom_loss.py` shows the pattern used to validate TuFT itself.
