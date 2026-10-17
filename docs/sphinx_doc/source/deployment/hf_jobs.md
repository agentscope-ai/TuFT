# Deploy on Hugging Face Jobs

[Hugging Face Jobs](https://huggingface.co/docs/hub/jobs) run a Docker image on Hub-managed
hardware, billed per minute, and can expose a container port through an authenticated proxy.
This guide takes you end to end — configure a TuFT server, start it as a Job, train a "talk
like Yoda" LoRA on `Qwen/Qwen3-1.7B` from your laptop, and download the adapter — with **no
local GPU required**.

```{admonition} What you'll build
:class: note

A TuFT server running on an HF Jobs L4 GPU, reachable at a `https://<job_id>--10610.hf.jobs`
URL, that you fine-tune from your laptop using the runnable code in
[`examples/personality_sft/`](https://github.com/agentscope-ai/TuFT/tree/main/examples/personality_sft).
Checkpoints land in a private [Storage Bucket](https://huggingface.co/docs/hub/storage-buckets)
on your account.
```

## Prerequisites

1. **A Hugging Face account that can run Jobs** (see
   [Jobs pricing](https://huggingface.co/docs/hub/jobs-pricing)), logged in with a token that
   can start Jobs:

   ```bash
   pip install -U huggingface_hub pyyaml
   hf auth login
   ```

2. **The TuFT repo.** The launcher only needs `huggingface_hub` + `pyyaml` locally; the GPU
   dependencies are in the stock TuFT image that runs inside the Job:

   ```bash
   git clone https://github.com/agentscope-ai/TuFT
   cd TuFT
   ```

Models given as HF repo ids are **mounted read-only from the Hub** instead of downloaded, so
the server starts without a download step. Gated models mount the same way once your account
has been granted access; without access, the Job is rejected at submission, before it is
billed. To download inside the Job instead, pass `--no-mount-models --hf-token`.

## Step 1 — Configure the server

The deploy helper [`deploy/hf_jobs/launch.py`](https://github.com/agentscope-ai/TuFT/tree/main/deploy/hf_jobs/launch.py)
is **config-file driven**: you edit a standard `tuft_config.yaml` (the same file
`tuft launch --config` uses) and run the script. Jobs infra goes in an optional `hf_jobs:`
section that is stripped before the server sees it.

Save this as `yoda_hf_jobs.yaml`:

```yaml
checkpoint_dir: ~/.cache/tuft/checkpoints   # on HF Jobs, launch.py pins this to a Storage Bucket
model_owner: cloud-user

supported_models:
  - model_name: Qwen/Qwen3-1.7B
    model_path: Qwen/Qwen3-1.7B            # HF id (mounted read-only from the Hub) or a local path
    max_model_len: 4096
    tensor_parallel_size: 1
    colocate: true                         # single GPU: training + vLLM sampling share it
    sampling_memory_fraction: 0.3
    max_lora_rank: 16
    max_loras: 2

authorized_users:
  tml-REPLACE_WITH_A_STRONG_KEY: cloud-user  # clients send this as the X-API-Key header

persistence:
  mode: DISABLE

telemetry:
  enabled: false

# HF Jobs infra for deploy/hf_jobs/launch.py (TuFT ignores this; it's stripped before the server sees it):
hf_jobs:
  flavor: l4x1           # `hf jobs hardware` lists flavors and prices
  name: tuft-yoda
  timeout: 1h            # hard stop: Jobs don't scale to zero
```

Generate a real API key for `authorized_users` (it **must** start with `tml-`):

```bash
python -c "import secrets; print('tml-' + secrets.token_urlsafe(24))"
```

The launcher passes the config to the Job as an encrypted
[secret](https://huggingface.co/docs/hub/jobs-configuration#environment-variables-and-secrets),
so your keys don't show up in the Job's command or in `hf jobs inspect`.

```{tip}
A ready-made example with all available options is
[`deploy/hf_jobs/tuft_config.example.yaml`](https://github.com/agentscope-ai/TuFT/tree/main/deploy/hf_jobs/tuft_config.example.yaml).
```

## Step 2 — Start the Job

```bash
python deploy/hf_jobs/launch.py --config yoda_hf_jobs.yaml
```

The launcher creates the checkpoint bucket if needed, submits the Job with the stock
`ghcr.io/agentscope-ai/tuft` image, waits for the image pull and for `/api/v1/healthz`, then
prints the URL and a client snippet (startup takes about 5 minutes on an L4):

```text
[job] submitted 6ad308a7c662881452265aa5: https://huggingface.co/jobs/<you>/6ad308a7c662881452265aa5
[wait] stage=SCHEDULING
[wait] stage=RUNNING

✅ TuFT is up: https://6ad308a7c662881452265aa5--10610.hf.jobs
```

**Two layers of auth.** The exposed port sits behind the Jobs proxy, which requires an HF
token with read access to the Job's namespace as `Authorization: Bearer`. TuFT's `tml-` key
travels separately as `X-API-Key`, so the stock Tinker SDK only needs one extra header:

```python
import tinker

client = tinker.ServiceClient(
    base_url="https://<job_id>--10610.hf.jobs",
    api_key="tml-REPLACE_WITH_A_STRONG_KEY",
    default_headers={"Authorization": f"Bearer {HF_TOKEN}"},
)
```

Pass `--public` to drop the HF token layer and rely on the `tml-` key alone.

**Storage.** `checkpoint_dir` is pinned to the private bucket `<you>/tuft-checkpoints`
(`--ckpt-bucket` to change it), so adapters outlive the Job and other Jobs can mount them.

**Private networking.** With `--network-group <name>`, other Jobs in the same group reach the
server at `http://${HF_NETWORK_GROUP_PREFIX}tuft:10610` without going through the proxy —
useful when the training loop itself runs as a Job.

## Step 3 — Train the Yoda LoRA from your laptop

The training script [`examples/personality_sft/train.py`](https://github.com/agentscope-ai/TuFT/tree/main/examples/personality_sft/train.py)
is a **client** that connects to the running server over HTTP via the Tinker SDK. When the
URL is a `*.hf.jobs` address it adds your HF token (from `hf auth login` or `HF_TOKEN`)
automatically; a `--public` Job needs only the `tml-` key. `jinja2` is needed for the chat template:

```bash
pip install tinker transformers jinja2

python examples/personality_sft/train.py \
    --base-url https://<job_id>--10610.hf.jobs \
    --api-key tml-REPLACE_WITH_A_STRONG_KEY
```

The script samples the base model **before** training, runs the LoRA steps, then samples the
trained adapter **after** (output from a 30-step run on `l4x1`):

```text
[train] 30 steps, batch 8, lr 0.0001, rank 16
   step   0  loss=2631.0788
   step  29  loss=773.8658
[save] run_id=daedfac5-1336-495e-a82e-93617e7a7e64

[after] sampling the trained adapter
   [yoda] How should I spend my weekend?
      -> Rest. You must. Enjoy it. Go out, you should. Nature, the best gift. Walk, laugh, and be you.
   [yoda] What's the key to writing good software?
      -> Code you must write, read it, you must. Always. Listen, learn. Good it becomes.
```

## Step 4 — Download the adapter

Training writes a standard [PEFT](https://huggingface.co/docs/peft) LoRA adapter to the
checkpoint bucket. Use the printed `run_id` to sync it down:

```bash
hf buckets sync hf://buckets/<you>/tuft-checkpoints/<run_id> ./weights/<run_id>
# -> ./weights/<run_id>/yoda-final/adapter/{adapter_config.json, adapter_model.safetensors}
```

To merge it into full model weights, see
[Step 4 of the Modal guide](modal.md#step-4--download-the-adapter).

## Step 5 — Tear down

HF Jobs **do not scale to zero**: the Job is billed until it reaches its `timeout` or you
cancel it.

```bash
python deploy/hf_jobs/launch.py --status --config yoda_hf_jobs.yaml   # list TuFT Jobs
python deploy/hf_jobs/launch.py --down --config yoda_hf_jobs.yaml     # or --job-id <id> --namespace <ns>
```

With `--config`, `--down` and `--status` use its `hf_jobs.name` and `hf_jobs.namespace`, so Jobs
started in an organization are found there. Your adapters remain in the bucket until you delete them.

## Multi-GPU training (FSDP)

On a multi-GPU flavor, split the GPUs between FSDP training and vLLM sampling instead of
colocating them. This layout uses all four GPUs of `l4x4`:

```yaml
supported_models:
  - model_name: Qwen/Qwen3-4B
    model_path: Qwen/Qwen3-4B
    max_model_len: 4096
    colocate: false
    tensor_parallel_size: 2      # vLLM sampling on 2 GPUs
    training_backend: fsdp
    fsdp_num_gpus: 2             # FSDP training on the other 2
    max_lora_rank: 16
    max_loras: 8
```

```bash
python deploy/hf_jobs/launch.py --config fsdp.yaml --flavor l4x4
```

In this layout vLLM runs on GPUs 0–1 and the FSDP workers on GPUs 2–3. FSDP adapters land in
the same bucket layout and load with stock PEFT. L4s talk over PCIe,
so on `l4x4` an SFT step on `Qwen/Qwen3-4B` takes several seconds; pick a larger flavor for
throughput.

## Next steps

- Try a bigger model (e.g. `Qwen/Qwen3-4B`) by changing both names in the config and setting
  `hf_jobs.flavor` to `a100-large` (or pass `--flavor a100-large`).
- See the [Modal guide](modal.md) for scale-to-zero, or [Lambda Cloud](lambda.md) for a
  dedicated VM.
- Browse the [HF Jobs docs](https://huggingface.co/docs/hub/jobs) for flavors, volumes, and
  network groups.
