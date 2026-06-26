# Deploy on Modal

[Modal](https://modal.com) is a serverless GPU platform: you deploy a container and it
runs on a GPU only while requests are in flight, scaling to zero when idle. This guide
takes you end to end — configure a TuFT server, deploy it to Modal, train a "talk like
Yoda" LoRA on `Qwen/Qwen3-0.6B` from your laptop, and download the adapter — with **no
local GPU required**.

```{admonition} What you'll build
:class: note

A TuFT server running on a Modal L4 GPU, reachable at a public `…modal.run` URL, that you
fine-tune from your laptop using the runnable code in
[`examples/personality_sft/`](https://github.com/agentscope-ai/TuFT/tree/main/examples/personality_sft).
```

## Prerequisites

1. **A Modal account + CLI.** Install the client and authenticate (this opens a browser to
   create an API token — see Modal's [Getting started](https://modal.com/docs/guide) and
   [`modal token`](https://modal.com/docs/reference/cli/token) docs):

   ```bash
   pip install modal
   modal token new
   ```

2. **The TuFT repo.** The deploy helper and the training example live in the repo; the
   launcher itself only needs `modal` + `pyyaml` locally (the heavy GPU dependencies run
   inside Modal's container, not on your machine):

   ```bash
   git clone https://github.com/agentscope-ai/TuFT
   cd TuFT
   pip install modal pyyaml
   ```

`Qwen/Qwen3-0.6B` is openly downloadable, so no Hugging Face token is needed. For **gated**
models, create a [Modal Secret](https://modal.com/docs/guide/secrets) holding your
`HF_TOKEN` and pass `--hf-secret <secret-name>` to the launcher.

## Step 1 — Configure the server

The deploy helper [`deploy/modal/launch.py`](https://github.com/agentscope-ai/TuFT/tree/main/deploy/modal/launch.py)
is **config-file driven**: you edit a standard `tuft_config.yaml` (the same file
`tuft launch --config` uses) and run the script. Modal infra goes in an optional `modal:`
section that is stripped before the server sees it.

Save this as `yoda_modal.yaml`:

```yaml
checkpoint_dir: ~/.cache/tuft/checkpoints   # on Modal, launch.py pins this to a Volume automatically
model_owner: cloud-user

supported_models:
  - model_name: Qwen/Qwen3-0.6B
    model_path: Qwen/Qwen3-0.6B            # HF id (downloaded on first launch) or a local path
    max_model_len: 4096
    tensor_parallel_size: 1
    colocate: true                         # single GPU: training + vLLM sampling share it
    sampling_memory_fraction: 0.4
    max_lora_rank: 16
    max_loras: 2

authorized_users:
  tml-REPLACE_WITH_A_STRONG_KEY: cloud-user  # clients send this as the X-API-Key header

persistence:
  mode: DISABLE

telemetry:
  enabled: false

# Modal infra for deploy/modal/launch.py (TuFT ignores this; it's stripped before the server sees it):
modal:
  gpu: L4
  name: tuft-yoda
  proxy_auth: false      # the tml- API key is the auth; the Tinker SDK can't send Modal gateway headers
  min_containers: 0      # scale to zero when idle
```

Generate a real API key for `authorized_users` (it **must** start with `tml-`):

```bash
python -c "import secrets; print('tml-' + secrets.token_urlsafe(24))"
```

```{tip}
A ready-made example with all available options is
[`deploy/modal/tuft_config.example.yaml`](https://github.com/agentscope-ai/TuFT/tree/main/deploy/modal/tuft_config.example.yaml),
and the example used below ships its own
[`config.yaml`](https://github.com/agentscope-ai/TuFT/tree/main/examples/personality_sft/config.yaml).
```

## Step 2 — Deploy to Modal

Run the launcher in the **foreground** so the server is torn down automatically when you
press `Ctrl-C`:

```bash
python deploy/modal/launch.py --config yoda_modal.yaml --foreground
```

The script renders a self-contained Modal app (a
[`@modal.web_server`](https://modal.com/docs/guide/webhooks)) and serves it. It prints a
public URL — copy it:

```text
✓ Created web endpoint => https://<your-workspace>--tuft-yoda-tuftserver-serve.modal.run
```

Confirm the server is up (the health route needs no auth; first call cold-starts the image
and loads vLLM, which can take a few minutes):

```bash
curl https://<your-workspace>--tuft-yoda-tuftserver-serve.modal.run/api/v1/healthz
# {"status":"ok"}
```

**Deploy modes.** `--foreground` (used here) wraps `modal serve` — convenient for an
interactive run; `Ctrl-C` stops it. Omit it for a **detached** deploy (`modal deploy`) that
keeps running until you `--down` it:

```bash
python deploy/modal/launch.py --config yoda_modal.yaml          # detached
python deploy/modal/launch.py --down --name tuft-yoda           # stop it later
```

`checkpoint_dir` is automatically pinned to a [Modal Volume](https://modal.com/docs/guide/volumes)
(`tuft-checkpoints`) so your adapters survive container shutdown.

## Step 3 — Train the Yoda LoRA from your laptop

The training script [`examples/personality_sft/train.py`](https://github.com/agentscope-ai/TuFT/tree/main/examples/personality_sft/train.py)
is a **client** that connects to the running server over HTTP via the Tinker SDK — it needs
only CPU-side dependencies:

```bash
pip install tinker transformers
```

The dataset is ~50 hand-authored `(user, assistant-in-Yoda-voice)` pairs in
[`dataset.py`](https://github.com/agentscope-ai/TuFT/tree/main/examples/personality_sft/dataset.py).
Only the **assistant** tokens get loss weight (the prompt is masked), so the model learns
the *voice*, not the questions. A couple of example pairs:

```python
YODA_PAIRS = [
    (
        "How do I stay motivated?",
        "Motivated, you wish to stay? Hmm. Small steps each day, take you must. "
        "Begin, and momentum follows. Do or do not — there is no try.",
    ),
    (
        "What's the best way to learn programming?",
        "Patient you must be, young one. Write code daily, you should. "
        "Fail often, learn faster you will. Much to learn, you still have.",
    ),
    # ... ~50 pairs total
]
```

Point the script at your Modal URL, pass the `tml-` key, and select the model (it must
match the `supported_models` entry on the server):

```bash
python examples/personality_sft/train.py \
    --base-url https://<your-workspace>--tuft-yoda-tuftserver-serve.modal.run \
    --api-key tml-REPLACE_WITH_A_STRONG_KEY \
    --model Qwen/Qwen3-0.6B
```

The script samples the base model **before** training, runs 60 LoRA steps, then samples the
trained adapter **after** so you can see the personality emerge:

```text
[before] sampling base model on held-out prompts
   [base] How should I spend my weekend?
      -> You could relax, catch up on friends, or work on a hobby. Try to balance rest with something fun.
   [base] I'm nervous about starting a new job.
      -> That's normal. Prepare a little, get good sleep, and remember they hired you for a reason.

[train] 60 steps, batch 8, lr 0.0001, rank 16
   step   0  loss=2.4131
   step  10  loss=1.8027
   step  50  loss=1.2563
   step  59  loss=1.1894

[save] sampler=tinker://abcd1234.../yoda-sampler
[save] run_id=abcd1234-...

[after] sampling the trained adapter
   [yoda] How should I spend my weekend?
      -> Rest and adventure, balance them you must, hmm. Outside, go — the Force in nature lives. Restored, you will be.
   [yoda] I'm nervous about starting a new job.
      -> Nervous, you are? Natural, this fear is. Breathe, you must. Chosen, you were — trust yourself, young one.
```

```{tip}
Tunables: `--lora-rank`, `--num-steps`, `--batch-size`, `--learning-rate`, `--max-length`.
Pass `--no-before` to skip the base-model sampling. Edit `dataset.py` to swap in a
different character.
```

## Step 4 — Download the adapter

Training writes a standard [PEFT](https://huggingface.co/docs/peft) LoRA adapter to the
server's `checkpoint_dir`, which on Modal is the `tuft-checkpoints`
[Volume](https://modal.com/docs/guide/volumes). The script prints a `run_id` — use it to
download the adapter with [`modal volume get`](https://modal.com/docs/reference/cli/volume):

```bash
modal volume get tuft-checkpoints <run_id> ./weights/
# -> ./weights/<run_id>/yoda-final/adapter/{adapter_config.json, adapter_model.safetensors}
```

Optionally merge the adapter into full model weights (needs `torch`, `peft`, `transformers`):

```python
import torch
from transformers import AutoModelForCausalLM
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", torch_dtype=torch.bfloat16)
merged = PeftModel.from_pretrained(base, "./weights/<run_id>/yoda-final/adapter").merge_and_unload()
merged.save_pretrained("./yoda-merged")   # standard HF model dir, servable by vLLM
```

## Step 5 — Tear down

If you deployed in the foreground, just press `Ctrl-C`. For a detached deploy:

```bash
python deploy/modal/launch.py --down --name tuft-yoda
```

With `min_containers: 0`, an idle Modal deployment already costs nothing, but `--down`
removes the app entirely. Your adapters remain in the `tuft-checkpoints` Volume until you
delete it.

```{tip}
Even with scale-to-zero, the GPU is **under-utilized** during an active session — the client
waits between bursts. Because TuFT is multi-tenant, you can keep one warm deployment busy by
pointing **several concurrent clients** at it (your own jobs or other users, each with a key
under `authorized_users`; raise `max_loras` for more adapters at once). See
[Keeping the GPU busy](index.md#keeping-the-gpu-busy).
```

## Next steps

- Try a bigger model (e.g. `Qwen/Qwen3-4B`) by changing both names in the config and setting
  `modal.gpu` to `H100` (or pass `--gpu H100`).
- See the [Lambda Cloud guide](lambda.md) for a dedicated on-demand GPU VM instead.
- Browse [Modal's docs](https://modal.com/docs) for GPU types, Volumes, and Secrets.
