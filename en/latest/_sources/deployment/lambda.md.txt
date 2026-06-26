# Deploy on Lambda Cloud

[Lambda Cloud](https://lambda.ai) rents plain on-demand GPU VMs, billed per minute until you
terminate them — no orchestration layer and no preemption. This guide
takes you end to end — configure a TuFT server, launch it on a Lambda GPU, train a "talk
like Yoda" LoRA on `Qwen/Qwen3-0.6B` from your laptop, and download the adapter — with **no
local GPU required**.

```{admonition} What you'll build
:class: note

A TuFT server running on a Lambda GPU VM (auto-provisioned and self-bootstrapped via Docker),
which you reach over an SSH tunnel and fine-tune from your laptop using the runnable code in
[`examples/personality_sft/`](https://github.com/agentscope-ai/TuFT/tree/main/examples/personality_sft).
```

```{admonition} Lambda has no scale-to-zero
:class: warning

Unlike Modal, a Lambda VM **bills continuously until you terminate it** — there is no
idle/auto-stop. Always tear the instance down when you're done (see [Step 5](#step-5-tear-down)).
```

## Prerequisites

1. **A Lambda Cloud account, API key, and SSH key.** In the
   [Lambda Cloud dashboard](https://cloud.lambda.ai), generate an API key
   ([API keys](https://cloud.lambda.ai/api-keys)) and register an SSH public key
   ([SSH keys](https://cloud.lambda.ai/ssh-keys)) — you'll use the matching private key to
   reach the server. See Lambda's [docs](https://docs.lambda.ai) for details. Export the API
   key so the launcher can find it:

   ```bash
   export LAMBDA_API_KEY=secret_...
   ```

2. **The TuFT repo.** The deploy helper talks to Lambda's HTTP API; it only needs `pyyaml`
   locally (all GPU dependencies run inside the container on the VM):

   ```bash
   git clone https://github.com/agentscope-ai/TuFT
   cd TuFT
   pip install pyyaml
   ```

## Step 1 — Configure the server

The deploy helper [`deploy/lambda/launch.py`](https://github.com/agentscope-ai/TuFT/tree/main/deploy/lambda/launch.py)
is **config-file driven** and mirrors the Modal launcher: you edit a standard
`tuft_config.yaml` and run the script. Lambda infra goes in an optional `lambda:` section
that is stripped before the server sees it.

Save this as `yoda_lambda.yaml`:

```yaml
checkpoint_dir: ~/.cache/tuft/checkpoints   # mapped to the VM's /data; see Step 4 about durability
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

# Lambda Cloud infra for deploy/lambda/launch.py (TuFT ignores this; it's stripped before the server sees it):
lambda:
  gpu: a100              # family hint; auto-pick prefers a100
  name: tuft-yoda
  # ssh_key: my-key      # default: your account's sole registered key
  # filesystem: tuft     # Lambda persistent filesystem for durable checkpoints (else ephemeral root disk)
```

Generate a real API key for `authorized_users` (it **must** start with `tml-`):

```bash
python -c "import secrets; print('tml-' + secrets.token_urlsafe(24))"
```

```{admonition} Pick a100, not a10, for training
:class: warning

The cheapest Lambda GPU (`gpu_1x_a10`, sm_86) has a known issue in the current TuFT image:
**serving works, but training returns null logprobs**. Auto-select therefore **prefers
a100** and uses a10 only as a last resort. For this training example, stay on a100.
```

## Step 2 — Launch the GPU server

Run the launcher. With no instance pinned, it auto-selects the cheapest available GPU that
matches your `gpu:` hint, provisions it, and self-bootstraps TuFT in Docker via cloud-init
(no manual SSH needed):

```bash
python deploy/lambda/launch.py --config yoda_lambda.yaml
```

It prints the chosen instance and, once the VM is up, a connect banner with an SSH-tunnel
command and the instance id:

```text
[launch] gpu_1x_a100_sxm4 in us-east-1 (~$1.99/hr), ssh_key=my-key, name=tuft-yoda
[launch] provisioning instance abcd...  (this takes a minute)
[launch] instance abcd1234... is active at 203.0.113.45

Connect securely over an SSH tunnel (recommended; keeps :10610 off the public net):
    ssh -N -L 10610:localhost:10610 ubuntu@203.0.113.45
```

```{tip}
Check or list instances any time with `python deploy/lambda/launch.py --status`. To reuse an
existing instance instead of launching a new one, pass `--instance-id <id>`.
```

## Step 3 — Train the Yoda LoRA from your laptop

Open the SSH tunnel printed above in one terminal (the first boot pulls the image and loads
vLLM, which can take a few minutes):

```bash
ssh -N -L 10610:localhost:10610 ubuntu@203.0.113.45
```

```{note}
The training still runs **on your laptop** — `train.py` is a CPU-only client that drives the
loop over HTTP. The `-L 10610:localhost:10610` flag forwards your laptop's port `10610` to the
server's port `10610` on the VM, so `http://localhost:10610` (used below) is only the *local
end* of the tunnel: every request is carried over SSH to the **remote GPU**, where the training
and sampling actually run. Keep this SSH terminal open for the whole run. (Prefer not to tunnel?
You can point `--base-url` at `http://<vm-ip>:10610` directly instead, but that exposes the API
on the public internet.)
```

In a second terminal, confirm the server is healthy through the tunnel:

```bash
curl http://localhost:10610/api/v1/healthz
# {"status":"ok"}
```

The training script [`examples/personality_sft/train.py`](https://github.com/agentscope-ai/TuFT/tree/main/examples/personality_sft/train.py)
is a **client** that drives the loop over HTTP via the Tinker SDK — it needs only CPU-side
dependencies:

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

Point the script at the tunnel (`http://localhost:10610`), pass the `tml-` key, and select
the model (it must match the `supported_models` entry on the server):

```bash
python examples/personality_sft/train.py \
    --base-url http://localhost:10610 \
    --api-key tml-REPLACE_WITH_A_STRONG_KEY \
    --model Qwen/Qwen3-0.6B
```

The script samples the base model **before** training, runs 60 LoRA steps, then samples the
trained adapter **after** so you can see the personality emerge:

```text
[before] sampling base model on held-out prompts
   [base] Give me advice on becoming a better person.
      -> Focus on small consistent habits: be honest, listen to others, and learn from your mistakes.
   [base] What's the key to writing good software?
      -> Keep it readable and well-tested, prefer small functions, and document your intent.

[train] 60 steps, batch 8, lr 0.0001, rank 16
   step   0  loss=2.3987
   step  10  loss=1.7765
   step  50  loss=1.2410
   step  59  loss=1.1702

[save] sampler=tinker://abcd1234.../yoda-sampler
[save] run_id=abcd1234-...

[after] sampling the trained adapter
   [yoda] Give me advice on becoming a better person.
      -> Small steps each day, take you must. Honest, be. Listen more than speak, you should. Grow, you will, hmm.
   [yoda] What's the key to writing good software?
      -> Simple, keep it, you must. Read aloud your code, you should. Fail often, learn faster you will, young one.
```

```{tip}
Tunables: `--lora-rank`, `--num-steps`, `--batch-size`, `--learning-rate`, `--max-length`.
Pass `--no-before` to skip the base-model sampling. Edit `dataset.py` to swap in a different
character.
```

## Step 4 — Download the adapter

Training writes a standard [PEFT](https://huggingface.co/docs/peft) LoRA adapter to the
server's `checkpoint_dir`, which on the VM lives under `/home/ubuntu/tuft-data/checkpoints`.
The script prints a `run_id` — copy the adapter off the instance with `scp`:

```bash
scp -r ubuntu@203.0.113.45:/home/ubuntu/tuft-data/checkpoints/<run_id> ./weights/
# -> ./weights/<run_id>/yoda-final/adapter/{adapter_config.json, adapter_model.safetensors}
```

```{admonition} Download before you terminate
:class: warning

By default checkpoints live on the instance's **ephemeral root disk** — terminating the VM
(Step 5) destroys them. **Download first**, or launch with a persistent `filesystem:` (a
[Lambda filesystem](https://docs.lambda.ai)) so checkpoints survive termination.
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

(step-5-tear-down)=
## Step 5 — Tear down

Lambda bills until you terminate, so always shut the instance down when finished:

```bash
python deploy/lambda/launch.py --down --instance-id abcd1234...
# or by name:
python deploy/lambda/launch.py --down --name tuft-yoda
```

Verify nothing is left running:

```bash
python deploy/lambda/launch.py --status
```

```{tip}
Lambda bills the GPU continuously, so a single laptop-driven run **pays for a lot of idle
time**. Because TuFT is multi-tenant, you can point **several concurrent jobs or users** at the
same instance (each with a key under `authorized_users`; raise `max_loras` for more adapters at
once) to keep the GPU busy and split the cost. See
[Keeping the GPU busy](index.md#keeping-the-gpu-busy).
```

## Next steps

- Try a bigger model (e.g. `Qwen/Qwen3-4B`) by changing both names in the config; the
  auto-picked a100 (80 GB) has ample room.
- Prefer a scale-to-zero option? See the [Modal guide](modal.md).
- Browse [Lambda's docs](https://docs.lambda.ai) for instance types, regions, and filesystems.
