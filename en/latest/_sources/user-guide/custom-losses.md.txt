# Custom Losses

Use the Tinker SDK's `forward_backward_custom` method to write a TuFT training loss as an ordinary
PyTorch function. For example, you can combine Direct Preference Optimization (DPO) with negative
log-likelihood (NLL). Custom losses work with both training backends (`hf` and `fsdp`). Your
function runs in the client process; the server runs only its built-in losses.

```python
def my_loss(data: list[types.Datum], logprobs_list: list[torch.Tensor]):
    # Combine the per-token log-probabilities into one differentiable scalar.
    loss = ...
    # Metrics must be plain floats; they are returned with the training result.
    return loss, {"my_metric:sum": loss.item()}


training_client.forward_backward_custom(data, my_loss, loss_type_input="logprobs")
```

---

## What You'll Learn

1. When to reach for `forward_backward_custom` instead of a built-in `loss_fn`
2. What the client and server do during a custom-loss call
3. Which inputs are supported and how returned scores line up with each datum
4. Two runnable examples: supervised fine-tuning and DPO + NLL
5. The runtime cost, common errors, and what changes during a custom step

---

## Table of Contents

1. [When to Use a Custom Loss](#when-to-use-a-custom-loss)
2. [How Custom Losses Work](#how-custom-losses-work)
3. [Supervised Fine-Tuning Example](#supervised-fine-tuning-example)
4. [DPO + NLL Example](#dpo--nll-example)
5. [Supported Inputs](#supported-inputs)
6. [Runtime Cost](#runtime-cost)
7. [Error Handling](#error-handling)
8. [What Changes During a Custom Step](#what-changes-during-a-custom-step)
9. [Q&A](#qa)

---

## When to Use a Custom Loss

TuFT ships five server-side loss functions — `cross_entropy`, `importance_sampling`, `ppo`,
`cispo`, and `dro` — selected by name in `forward_backward(data, loss_fn=...)`. The server accepts
only these names, so a client cannot run arbitrary Python on shared infrastructure.

Reach for `forward_backward_custom` when your objective is a differentiable function of the
**per-token target log-probabilities** but is not one of those five. Typical cases:

- **Preference losses** such as DPO that compare chosen and rejected responses.
- **Combined losses**, such as DPO plus weighted NLL or a policy/reference stability term.
- Quick research iterations on new losses without changing the server.

If your objective *is* one of the built-ins, prefer `forward_backward`: it costs one fewer
forward pass per step ([details below](#runtime-cost)).

```{admonition} Requires
:class: note

`tinker` ≥ 0.25 (the SDK version TuFT pins) and `torch` installed **on the client**. The custom
callback runs in your process only — the TuFT server never imports or executes client Python,
and no new server-side loss names are introduced.
```

---

## How Custom Losses Work

A **log-probability** is the model's score for a target token, expressed on a logarithmic scale.
A **gradient** says how the loss changes when a model value changes. TuFT uses these two pieces to
train with your loss without sending your Python function to the server.

```{figure} ../../_static/images/custom-loss-two-pass.svg
:alt: A custom-loss call first reads token log-probabilities from the server, computes the loss and its derivatives on the client, then sends token weights back for a second server pass that accumulates the model gradient.
:width: 820px
:align: center

The client computes the custom loss between two server passes.
```

1. The server scores every target token and returns those log-probabilities. It does not update the
   model or save gradients in this pass.
2. Your PyTorch callback combines the scores into one loss. PyTorch calculates how much the loss
   changes with each returned score.
3. The client sends those values back as token weights. The server scores the same data again and
   runs backpropagation, which accumulates the custom loss's model gradient.

Here is why the last step is equivalent to training on your loss. Let $\theta$ represent all model
parameters, and let $\ell_i(\theta)$ be the log-probability for target token $i$. Your callback
combines all of those values into a custom loss $C$. On the client, PyTorch computes one derivative
per token:

```{math}
g_i = \frac{\partial C}{\partial \ell_i}.
```

$g_i$ says how a small change in token $i$'s log-probability would change the custom loss. The
client sends $w_i = -g_i$ to the server. The server treats each weight as a fixed number and
computes the helper loss:

```{math}
H(\theta) = -\sum_i w_i\,\ell_i(\theta).
```

When the server differentiates $H$ with respect to the model parameters, the two minus signs
cancel:

```{math}
\nabla_\theta H
= -\sum_i w_i\,\nabla_\theta \ell_i
= \sum_i g_i\,\nabla_\theta \ell_i
= \nabla_\theta C.
```

The final equality is the chain rule: each token affects the custom loss by $g_i$, and each model
parameter affects that token by $\nabla_\theta \ell_i$. Adding those paths gives the custom loss's
parameter gradient. The numeric values of $H$ and $C$ do not need to match; the optimizer uses their
parameter gradients, which do match.

This equality requires both server passes to use the same model parameters. Normal sequential use
does this: wait for `forward_backward_custom` to finish, and only then call `optim_step`.

---

## Supervised Fine-Tuning Example

Runs against any TuFT server (HF or FSDP backend) — start one as in the
[Quickstart](../getting-started/quickstart.md), then:

```python
import tinker
import torch
from tinker import types

service_client = tinker.ServiceClient(base_url="http://localhost:10610", api_key="local-dev-key")
base_model = service_client.get_server_capabilities().supported_models[0].model_name
training_client = service_client.create_lora_training_client(
    base_model=base_model, rank=8, train_unembed=False
)
tokenizer = training_client.get_tokenizer()


def make_datum(prompt: str, completion: str) -> types.Datum:
    # Include special tokens only once, at the start of the full sequence.
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
    completion_tokens = tokenizer.encode(completion, add_special_tokens=False)
    tokens = prompt_tokens + completion_tokens
    # Shift by one token: each input position predicts the following target token.
    # A zero weight ignores prompt tokens; a one trains on completion tokens.
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
        # Reuse the datum weights as a binary response-token mask.
        weights = datum.loss_fn_inputs["weights"].to_torch()
        # Normalize each response separately so long responses do not dominate the batch.
        response_token_count = weights.sum().clamp_min(1)
        per_example_nlls.append(-(logprobs * weights).sum() / response_token_count)
    # Give every example equal weight, regardless of response length.
    loss = torch.stack(per_example_nlls).mean()
    return loss, {"sft_nll:mean": loss.item()}


for _ in range(10):
    # Accumulate gradients for this batch, then update the LoRA parameters once.
    result = training_client.forward_backward_custom(data, sft_loss).result()
    training_client.optim_step(types.AdamParams(learning_rate=1e-4)).result()
    print(result.metrics["sft_nll:mean"])
```

The callback receives one `logprobs_list[i]` per `data[i]`, **in order**, each a 1-D float32
tensor of length `len(data[i].model_input)` — `logprobs_list[i][t]` is the log-probability the
current model assigns to `target_tokens[t]` at position `t`. Your metrics dictionary is merged
into the returned `result.metrics` alongside the server's helper-loss metrics.

---

## DPO + NLL Example

Direct Preference Optimization (DPO) trains the model to score a chosen response above a rejected
response. The original [DPO paper](https://arxiv.org/abs/2305.18290) defines the preference term.
[Open Character Training](https://arxiv.org/abs/2511.01689) combines it with mean negative
log-likelihood (NLL) on chosen responses and a small per-token stability penalty. Adding chosen
NLL to DPO has also been studied as
[Regularized Preference Optimization](https://arxiv.org/abs/2405.16436).

For each response, the example first sums the difference between the policy and reference
log-probabilities over response tokens. DPO compares those sums within each chosen/rejected pair.
The full batch loss is:

```{math}
L = L_{\mathrm{DPO}} + \lambda_{\mathrm{NLL}} L_{\mathrm{NLL}}
    + \lambda_{\mathrm{proxy}} L_{\mathrm{proxy}}.
```

`L_proxy` is the mean squared policy/reference log-ratio on the sampled response tokens. The Open
Character Training code uses this as a lightweight KL-divergence proxy. It is not the exact KL
over the full vocabulary, which cannot be computed from target-token log-probabilities alone.

Place datums in `(chosen, rejected)` order and capture the reference log-probabilities once before
the first optimizer update:

```python
import torch.nn.functional as F

# Each adjacent pair shares a prompt. The chosen response must come first.
# make_datum gives prompt tokens weight 0.0 and response tokens weight 1.0.
# data = [chosen_0, rejected_0, chosen_1, rejected_1, ...]

# Snapshot the initial policy. These tensors stay fixed while the policy trains.
reference_result = training_client.forward(data, "cross_entropy").result()
reference_logprobs = [
    output["logprobs"].to_torch().float()
    for output in reference_result.loss_fn_outputs
]

# These values reproduce the Open Character Training recipe; tune them for your data.
BETA = 0.1  # Scales how strongly DPO separates chosen and rejected responses.
NLL_COEF = 0.1  # Keeps the chosen response likely under the policy.
KL_PROXY_COEF = 0.001  # Limits drift from the reference on sampled tokens.


def dpo_composite_loss(data: list[types.Datum], logprobs_list: list[torch.Tensor]):
    if not data or len(data) % 2 != 0:
        raise ValueError("DPO data must contain one or more complete chosen/rejected pairs")

    sequence_logratios = []
    response_nlls = []
    kl_proxy_terms = []
    for datum, logprobs, reference in zip(
        data, logprobs_list, reference_logprobs, strict=True
    ):
        # Select response tokens only; prompt tokens must not affect the loss.
        response_mask = datum.loss_fn_inputs["weights"].to_torch().bool()
        if not response_mask.any().item():
            raise ValueError("each DPO response must contain at least one token")

        # DPO uses the sequence log-ratio: log π_policy(response) - log π_ref(response).
        logratio = logprobs - reference
        sequence_logratios.append(logratio[response_mask].sum())

        # Mean response NLL gives short and long responses equal weight in the batch.
        response_nlls.append(-logprobs[response_mask].mean())

        # This sampled-token squared log-ratio is the recipe's lightweight KL proxy.
        kl_proxy_terms.append(logratio[response_mask].square().mean())

    dpo_terms = []
    for pair_start in range(0, len(sequence_logratios), 2):
        # Even rows are chosen; the following odd rows are rejected.
        chosen_logratio = sequence_logratios[pair_start]
        rejected_logratio = sequence_logratios[pair_start + 1]
        preference_margin = chosen_logratio - rejected_logratio
        dpo_terms.append(-F.logsigmoid(BETA * preference_margin))

    # Average pair losses, then add NLL on chosen rows only (0, 2, 4, ...).
    dpo = torch.stack(dpo_terms).mean()
    chosen_nll = torch.stack(response_nlls[::2]).mean()
    kl_proxy = torch.stack(kl_proxy_terms).mean()
    loss = dpo + NLL_COEF * chosen_nll + KL_PROXY_COEF * kl_proxy
    return loss, {
        "dpo:mean": dpo.item(),
        "chosen_nll:mean": chosen_nll.item(),
        "kl_proxy:mean": kl_proxy.item(),
        "composite:mean": loss.item(),
    }


result = training_client.forward_backward_custom(data, dpo_composite_loss).result()
training_client.optim_step(types.AdamParams(learning_rate=1e-4)).result()
```

Here the training client's initial state is the reference. You can instead score the data with a
separate reference model. In either case, keep every reference row aligned with the same datum and
target-token position.

---

## Supported Inputs

`loss_type_input="logprobs"` is the only supported input type (and the default). Per datum,
`loss_fn_inputs` may contain **exactly**:

| Key | Required | Dtype | Constraint |
|---|---|---|---|
| `target_tokens` | yes | `int64` | same length as `model_input` |
| `weights` | no | `float32` | same length as `target_tokens` |

The SDK rejects any other key on the client (for example, `advantages`) before sending a request.
`weights` serve two roles: your callback can read them back (for example, as a prompt mask), and
the server uses them for the first pass's helper-loss metric. When omitted, the SDK sends zeros
for the first pass; the second pass replaces them with the gradient-derived values.

Your callback must return a **scalar torch tensor** (differentiable with respect to the
log-prob tensors) and a **`dict[str, float]`** of metrics. Use names such as `"name:sum"` or
`"name:mean"` to follow the Tinker metric convention. Custom metrics are computed once over
the full input batch and merged on the client. TuFT does not combine them again when it splits
the request into smaller server-side batches.

## Runtime Cost

- **One extra forward pass.** Each `forward_backward_custom` costs *two* forward passes plus
  one backward (versus one forward + one backward for a built-in loss), and one extra
  round trip to send log-probabilities to the client and weights back. The actual time depends
  on the model, hardware, batch size, and network latency.
- **Memory.** The first pass runs under `torch.no_grad()` on both backends, so PyTorch does not
  save a gradient graph. Peak memory is set by the backward pass, as it is for built-in losses.
- **Payload size.** The client sends one float32 weight per token. For a large batch, the SDK may
  split the data into smaller requests while preserving datum order.
- **FSDP multi-GPU.** Both passes shard the batch across ranks, so each of the two requests
  must satisfy `len(data) >= fsdp_num_gpus` — the same constraint as `forward_backward`.

## Error Handling

- **Unknown loss names never reach a model.** `/forward_backward` accepts only the five built-in
  names and returns **422** for anything else. `forward_backward_custom` uses an existing built-in
  loss for its server work, so it needs no new server-side name.
- **Malformed datum inputs** are rejected with a `loss_fn_inputs`-specific error. Unsupported
  keys are rejected by the SDK before pass 1; the server validates mismatched keys, shapes,
  and dtypes. `target_tokens` and `weights` must match the model-input length.
- **Callback exceptions stay client-side.** If your loss function raises (or an input datum
  carries an unsupported key), the error surfaces in your process. When this happens between
  the two passes, no harm is done server-side: pass 1 accumulated nothing, so the training
  run's gradient state is unchanged and you can simply retry.

## What Changes During a Custom Step

- **The first server pass only reads the model.** It changes no weights and accumulates no
  gradients. If the callback fails, the training run is unchanged and you can retry.
- **Pass 2 behaves like any `forward_backward`.** Gradients accumulate until the next
  `optim_step`, so you can add custom and built-in gradients before one optimizer update.
- **Do not update the model between the two passes.** Another thread must not call `optim_step`
  while the client callback is running. Otherwise, the returned log-probabilities describe the
  old model while the backward pass uses the new model. Sequential use is safe: wait for the
  custom call, then update the optimizer.
- **Custom code never runs server-side.** Shared servers still execute only the five built-in
  loss functions.

---

## Q&A

**Q: Which SDK versions work?**
`forward_backward_custom` with `loss_type_input="logprobs"` is validated against the Tinker SDK
release TuFT pins (0.25.x, see `pyproject.toml`). The mechanism relies only on the stable
`forward` and `forward_backward` requests.

**Q: Can I use inputs other than log-probs (e.g. full logits)?**
Not currently — `"logprobs"` is the only `loss_type_input` the SDK offers, and TuFT returns
per-token *target* log-probs only. Losses needing full-vocabulary distributions (e.g. exact KL
to a teacher) can often be re-expressed against sampled targets; see the
[On-Policy Distillation guide](on-policy-distillation.md) for that pattern.

**Q: Do custom metrics aggregate across micro-batches?**
Your callback sees the full input batch at once, even if the server processes it in smaller
batches. The callback computes metrics once, and the SDK merges them into the final result.

**Q: How do I debug a suspicious gradient?**
Compare against the direct computation: run your callback's math straight through a local copy
of the model and compare its parameter gradients with the custom call. For a quick value check,
run the same math on log-probabilities returned by `training_client.forward(...)`.
