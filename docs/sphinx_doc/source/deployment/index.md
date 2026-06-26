# Deployment

```{admonition} 🚀 No GPU? No problem!
:class: tip

You don't need to own a GPU to run TuFT. These guides show how to stand up a TuFT
server on **pay-as-you-go cloud providers** — rent a GPU on demand and release it when
you're done. Train from your laptop (no local GPU) by pointing the
[Tinker](https://github.com/thinking-machines-lab/tinker) SDK at the cloud server.
```

TuFT is a single, standard server (`tuft launch`). The deploy helpers in the
[`deploy/`](https://github.com/agentscope-ai/TuFT/tree/main/deploy) directory just run
that exact server on rented compute and wire up storage, ports, and secrets for you —
they don't change anything about the product. Pick the backend that fits your workflow:

```{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Modal
:link: modal
:link-type: doc
:shadow: none

**Serverless, scale-to-zero.** Deploys a web endpoint that scales to zero when idle,
billed per second. Suited to bursty or intermittent use.
:::

:::{grid-item-card} Lambda Cloud
:link: lambda
:link-type: doc
:shadow: none

**A plain on-demand GPU VM**, billed per minute until you terminate it. No orchestration
layer; the instance runs until you stop it.
:::
```

## Which one should I pick?

| If you want… | Use | Description |
|---|---|---|
| To scale to zero when idle | **[Modal](modal.md)** | Serverless; scale-to-zero, per-second billing, managed proxy and volumes. |
| A single dedicated GPU instance | **[Lambda Cloud](lambda.md)** | On-demand, billed per minute until you terminate; no orchestration layer, no preemption. |

## Keeping the GPU busy

A single laptop-driven training run leaves the rented GPU **under-utilized**: the client
tokenizes data, builds batches, and waits on HTTP round-trips between GPU bursts, so the GPU
sits idle for much of the run. Because TuFT is **multi-tenant**, one deployed server can host
**several concurrent jobs or users** on the same GPU — give each their own key under
`authorized_users`, and raise `max_loras` so multiple LoRA adapters can train at once. Sharing
this way improves utilization and splits the cost. It matters most on **Lambda** (billed
continuously whether or not the GPU is working) and during active **Modal** sessions (the
container is warm but the GPU is bursty).

Both guides walk through the **same end-to-end example**: configuring the server,
deploying it, training a "talk like Yoda" LoRA on `Qwen/Qwen3-0.6B` from your laptop, and
downloading the trained adapter — all reusing the runnable code in
[`examples/personality_sft/`](https://github.com/agentscope-ai/TuFT/tree/main/examples/personality_sft).

```{toctree}
:maxdepth: 1
:hidden:

modal
lambda
```
