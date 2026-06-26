# Personality SFT — "talk like Yoda"

Fine-tune a LoRA adapter so a base model adopts a character's voice, on a small synthetic
dataset. This is a **client-side training script**: it drives the loop over HTTP against a
running TuFT server (it does not start one). The same `train.py` works whether the server
is local or on Modal.

```
dataset.py   synthetic "talk like Yoda" pairs + assistant-only-masked Datum builder
train.py     connects to a TuFT server, samples before/after, trains the LoRA, saves it
```

A ready-to-use [`config.yaml`](./config.yaml) is included (Qwen3-1.7B on a single L4). First
put a real key in its `authorized_users` (replace `tml-CHANGE-ME`):

```bash
python -c "import secrets; print('tml-' + secrets.token_urlsafe(24))"
```

**Local:**
```bash
tuft launch --host 0.0.0.0 --port 10610 --config examples/personality_sft/config.yaml
```

**On Modal** (server on a GPU, this loop on your laptop — see [`deploy/`](../../deploy/)). The
`modal:` section in `config.yaml` supplies the infra (L4), so no extra flags are needed:
```bash
python deploy/modal/launch.py --config examples/personality_sft/config.yaml --foreground
# prints a URL like https://<workspace>--personality-sft-tuftserver-serve.modal.run
```

**On Lambda Cloud** (rent a GPU VM by the minute; auto-picks the cheapest available GPU).
Needs `export LAMBDA_API_KEY=...`:
```bash
python deploy/lambda/launch.py --config examples/personality_sft/config.yaml
# auto-picks a GPU, bootstraps TuFT, and prints an SSH-tunnel command + connect details.
# Open the tunnel it prints, then use --base-url http://localhost:10610 below.
# Lambda has NO scale-to-zero — terminate when done:
#   python deploy/lambda/launch.py --down --instance-id <id>
```

## 2. Run the training

```bash
pip install tinker transformers          # local deps, no GPU

python examples/personality_sft/train.py \
    --base-url http://localhost:10610 \   # or the Modal URL from step 1
    --api-key tml-...                     # the key you put in config.yaml
# --model defaults to Qwen/Qwen3-1.7B (matches config.yaml); pass --model for others
```

You'll see the base model's plain answers, then — after ~60 steps — the same held-out
prompts answered in Yoda's voice (inverted syntax, "young one", …). Tunables: `--lora-rank`,
`--num-steps`, `--batch-size`, `--learning-rate`, `--max-length`. Edit `dataset.py` to swap
in a different character.

## 3. Get the weights

The LoRA adapter + sampler weights are saved on the **server's** `checkpoint_dir`. On Modal
that's a Volume — download the standard PEFT adapter (the `run_id` is printed at the end):

```bash
modal volume get tuft-checkpoints <run_id> ./weights/
# -> ./weights/<run_id>/yoda-final/adapter/{adapter_config.json, adapter_model.safetensors}
```

**Merge into full model weights** (optional; needs `torch`, `peft`, `transformers`):

```python
import torch
from transformers import AutoModelForCausalLM
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B", torch_dtype=torch.bfloat16)
merged = PeftModel.from_pretrained(base, "./weights/yoda-final/adapter").merge_and_unload()
merged.save_pretrained("./yoda-merged")        # standard HF model dir, servable by vLLM
```
