# On-Policy Distillation — internalize a few-shot teacher into the weights

Train a small model with **on-policy distillation (OPD)**: the **student** samples answers to
its *own* prompts, a **teacher** scores those exact tokens, and the student is nudged toward the
teacher's per-token distribution. It is the recipe from Thinking Machines'
[On-Policy Distillation](https://thinkingmachines.ai/blog/on-policy-distillation/) — dense
per-token supervision (unlike RL's one scalar reward), and on the student's own trajectory
(unlike off-policy SFT) — implemented on TuFT with the **same machinery as RL**
(`importance_sampling` loss). This is a **client-side training script**: it drives the loop over
HTTP against a running TuFT server. The same `train.py` works whether the server is local or on Modal.

```
dataset.py   GSM8K loader, teacher (few-shot) vs student (bare) prompts, the importance_sampling Datum builder
train.py     connects to a TuFT server, evals before, runs OPD, evals after, saves the LoRA
```

**The task — math reasoning (GSM8K).** The teacher is the **same base model** but conditioned on a
strong system prompt + four worked examples, so it reasons step by step and ends with `#### <number>`.
The student sees only the bare question. OPD distills that few-shot behavior into the student's LoRA
— so afterward the student reasons in the teacher's style *without the examples in its prompt*
("context distillation"). Because teacher == student base model, the whole thing fits on **one GPU**.

> **The general OPD recipe (any task).** `advantage[t] = teacher_logprob[t] − student_logprob[t]`
> is the single-sample estimator of the negative per-token reverse-KL; feed it to the
> `importance_sampling` loss. Swap in your own task by editing the prompts/data in `dataset.py`.

## 1. Start a server

A ready-to-use [`config.yaml`](./config.yaml) is included (Qwen3-1.7B on a single A100-40GB; a 24GB
L4 also works with lighter settings — see the config comments). First put a real key in its
`authorized_users` (replace `tml-CHANGE-ME`):

```bash
python -c "import secrets; print('tml-' + secrets.token_urlsafe(24))"
```

**Local:**
```bash
tuft launch --host 0.0.0.0 --port 10610 --config examples/on_policy_distillation/config.yaml
```

**On Modal** (server on a GPU, this loop on your laptop — see [`deploy/`](../../deploy/)). The
`modal:` section in `config.yaml` supplies the infra (A100-40GB), so no extra flags are needed:
```bash
python deploy/modal/launch.py --config examples/on_policy_distillation/config.yaml
# prints a URL like https://<workspace>--on-policy-distillation-tuftserver-serve.modal.run
```

**A different, larger teacher (e.g. Qwen3-8B)?** Run it on its own TuFT server and point `train.py` at it
with `--teacher-model` / `--teacher-base-url` / `--teacher-api-key`. It isn't automatically better, though:
OPD transfers *capability*, not just style, so it pays off only with a real per-token capability gap — a much
weaker student or a harder task. (A modest size bump on an easy task, e.g. 8B → 1.7B on GSM8K, barely helps —
two strong same-family models mostly agree token-by-token.)

## 2. Run the training

```bash
pip install tinker transformers datasets torch     # local deps, no GPU

python examples/on_policy_distillation/train.py \
    --base-url http://localhost:10610 \             # or the Modal URL from step 1
    --api-key tml-...                               # the key you put in config.yaml
# --model defaults to Qwen/Qwen3-1.7B (matches config.yaml)
```

You'll see the **bare student** evaluated on held-out GSM8K, the **few-shot teacher** (the ceiling
we distill toward), then — after ~16 steps — the **trained student** answering the same bare prompts.
In our run held-out accuracy jumped **63% → 72%, matching the few-shot teacher**, with the student's
reasoning now noticeably more concise — all from a bare prompt (no examples). OPD improves fast and
reverse-KL can over-train, so the default keeps it short; `--eval-every` prints the trajectory.
Tunables: `--num-steps`, `--batch-size`, `--group-size`, `--learning-rate`, `--lora-rank`,
`--kl-coef`, `--num-train`, `--num-eval`. Edit `dataset.py` to swap in a different task.

See [`sample_outputs.md`](./sample_outputs.md) for real before/after transcripts — including
problems the bare model gets wrong that the distilled student then gets right.

## 3. Get the weights

The LoRA adapter + sampler weights are saved on the **server's** `checkpoint_dir`. On Modal that's a
Volume — download the standard PEFT adapter (the `run_id` is printed at the end):

```bash
modal volume get tuft-checkpoints <run_id> ./weights/
# -> ./weights/<run_id>/opd-final/adapter/{adapter_config.json, adapter_model.safetensors}
```

**Merge into full model weights** (optional; needs `torch`, `peft`, `transformers`):

```python
import torch
from transformers import AutoModelForCausalLM
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B", torch_dtype=torch.bfloat16)
merged = PeftModel.from_pretrained(base, "./weights/opd-final/adapter").merge_and_unload()
merged.save_pretrained("./opd-merged")          # standard HF model dir, servable by vLLM
```
