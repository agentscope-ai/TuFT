# TuFT — Chat Supervised Fine-Tuning (SFT)

This tutorial demonstrates **supervised fine-tuning (SFT)** on **chat-formatted data** using a **running Tuft server**. Full runnable code is in **`examples/chat_sft.ipynb`**.

---

## What You’ll Learn

1. How to load **chat datasets** from HuggingFace and extract multi-turn `messages`  
2. How to format conversations using **model chat templates** (`apply_chat_template`)  
3. How to implement **assistant-only loss masking** and compute **masked NLL** for evaluation  
4. How to construct **Tuft `Datum`** objects and run an end-to-end **LoRA SFT** loop  
5. How to choose and tune **LoRA rank** and **learning rate** based on train/test curves

---

## Table of Contents
1. [When to Use SFT vs. RL](#1-when-to-use-sft-vs-rl)  
2. [Datasets](#2-datasets)  
3. [Minimal Training Example (SFT)](#3-minimal-training-example-sft)  
4. [Key Concepts](#4-key-concepts)  
   - [Chat Formatting & Templates](#41-chat-formatting--templates)  
   - [Loss Masking (Assistant-only)](#42-loss-masking-assistant-only)  
   - [Tuft Datum Format](#43-tuft-datum-format)  
   - [loss_fn and Masked NLL Metric](#44-loss_fn-and-masked-nll-metric)  
6. [Parameter Selection](#5-parameter-selection)  
7. [Q&A](#6-qa)

---

## 1. When to Use SFT vs. RL

### SFT vs. RL (high-level comparison)

| Topic | SFT (Supervised Fine-Tuning) | RL (Reinforcement Learning) |
|---|---|---|
| Training signal | Demonstrations (target responses) | Reward / preferences (scalar or ranking) |
| Best for | Style, format, instruction following, domain behavior from curated answers | Aligning behavior to preferences/constraints, safety policies, multi-objective trade-offs |
| Data required | High-quality assistant responses | Reward model, preference pairs, or evaluators |
| Stability | Typically stable and predictable | More sensitive; requires careful tuning/monitoring |
| Typical workflow | Often the first stage | Often follows SFT (SFT → RL) |

**Rule of thumb**
- Use **SFT** when you can provide good “gold” assistant responses.
- Use **RL** when there is no single correct response, but you can define what is “better” via a reward/preference signal.

---

## 2. Datasets

This notebook uses **`no_robots`**.

| Dataset | Source | Size | Train On | Use Case |
|---|---|---|---|---|
| `no_robots` | `HuggingFaceH4/no_robots` | ~9.5K train + 500 test | All assistant messages (masked) | Quick experiments |

Minimal loader:
```python
from datasets import load_dataset

ds = load_dataset("HuggingFaceH4/no_robots")
train_data = [row["messages"] for row in ds["train"]]
test_data  = [row["messages"] for row in ds["test"]]
```

Each sample is a list of chat messages:
```json
{"role": "user" | "assistant", "content": "..."}
```

---

## 3. Minimal Training Example (SFT)

Key Tuft calls (full code in `examples/chat_sft.ipynb`):
```python
import tinker
from tinker import types

service_client = tinker.ServiceClient(base_url="http://localhost:8080", api_key=TINKER_API_KEY)

training_client = service_client.create_lora_training_client(
    base_model=BASE_MODEL,
    rank=LORA_RANK,
    train_mlp=True,
    train_attn=True,
    train_unembed=True,
)

fwdbwd = training_client.forward_backward(datums, loss_fn="cross_entropy").result()
training_client.optim_step(types.AdamParams(learning_rate=LEARNING_RATE)).result()
```

---

## 4. Key Concepts

### 4.1 Chat Formatting & Templates

We use the base model’s chat template to ensure formatting matches pretraining/instruction tuning conventions:
```python
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=False,
)
tokens = tokenizer.encode(text, add_special_tokens=False)
```

### 4.2 Loss Masking (Assistant-only)

For chat SFT, we usually want the model to learn to produce **assistant responses**, not to predict the user prompt. We therefore build per-token weights:

- tokens from `assistant` turns → weight = `1.0`
- tokens from `user` turns → weight = `0.0`

This mask is aligned to **next-token targets**.

### 4.3 Tuft Datum Format

Each conversation is converted into a next-token-prediction sample:

- `model_input`: tokens `[0..T-2]`
- `target_tokens`: tokens `[1..T-1]`
- `weights`: mask applied on targets (assistant-only)

Example:
```python
from tinker import types

datum = types.Datum(
    model_input=types.ModelInput.from_ints(input_tokens),
    loss_fn_inputs={
        "target_tokens": list(target_tokens),
        "weights": target_weights.tolist(),
    },
)
```

### 4.4 loss_fn and Masked NLL Metric

Training uses:
- `loss_fn="cross_entropy"`

Tuft returns per-token log probabilities (`logprobs`). The notebook computes **masked NLL**:

NLL = (Σ<sub>t</sub> (-log p(y<sub>t</sub>)) · w<sub>t</sub>) / (Σ<sub>t</sub> w<sub>t</sub>)

Minimal computation:
```python
def masked_nll(loss_fn_outputs, datums):
    total_loss, total_w = 0.0, 0.0
    for out, d in zip(loss_fn_outputs, datums):
        for lp, w in zip(out["logprobs"], d.loss_fn_inputs["weights"]):
            total_loss += -lp * w
            total_w += w
    return total_loss / max(total_w, 1.0)
```

---

## 5. Parameter Selection

This section explains how to choose `lora_rank` and `learning_rate`, and summarizes conclusions from the provided experiment results.

### What do `lora_rank` and `learning_rate` do?

**`lora_rank` (LoRA adapter rank)** controls adapter capacity:
- Higher rank = more trainable params → potentially better fit, more compute/memory, higher overfitting risk  
- Lower rank = cheaper, often sufficient for style/small behavior changes  

**`learning_rate`** controls update step size:
- Too high (e.g. `1e-3`): fast but can be unstable/overfit  
- Too low (e.g. `1e-5`): stable but slow  
- Middle (e.g. `1e-4`): common default for LoRA SFT  

### Experimental conclusions from the plots

Based on **Figure 1** (train mean NLL) and **Figure 2** (test NLL):

1) Very low LR (`1e-5`) converges much more slowly  
2) `1e-4` and `1e-3` improve quickly early  
3) Rank has diminishing returns beyond a point  
4) Best test losses often cluster around moderate rank + moderate/high LR  

> Note: exact “best” depends on stopping step and downstream generation quality (not only NLL).

<p align="center">
  <strong>Figure 1. Train mean NLL</strong><br>
  <img src="../assets/train_mean_nll.png" alt="train_mean_nll" style="max-width: 720px; width: 100%; height: auto;">
</p>

<p align="center">
  <strong>Figure 2. Test NLL</strong><br>
  <img src="../assets/test_nll.png" alt="test_nll" style="max-width: 720px; width: 100%; height: auto;">
</p>



### Practical recommendations

- Strong default: `lora_rank = 8 or 32`, `learning_rate = 1e-4`  
- Faster early progress (riskier): `lora_rank = 8 or 32`, `learning_rate = 1e-3`  
- If unstable/overfitting: lower LR (`1e-4 → 5e-5 → 1e-5`) or lower rank (`32 → 8`)  
- If task is harder: try `32` before `128`, keep LR `1e-4`, increase steps before rank if possible  

---

## 6. Q&A

### (1) Dataset download fails due to network issues (`MaxRetryError` / `Network is unreachable`)

If you see an error like:
```
MaxRetryError('HTTPSConnectionPool(host=\'huggingface.co\', port=443): Max retries exceeded ...
(Caused by NewConnectionError(... [Errno 101] Network is unreachable))')
```

Add the following **at the very top of the first cell**:
```python
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
```

Then **Restart Kernel and Clear ALL Outputs**.

---

### (2) `invalid Api_key`

In the Tinker SDK, the environment variable `TINKER_API_KEY` (set via `export TINKER_API_KEY=...`) takes precedence over the `api_key=` argument passed here:
```python
service_client = tinker.ServiceClient(base_url=TINKER_BASE_URL, api_key=TINKER_API_KEY)
```

So if your code is passing the correct key but you still get `invalid api_key`, clear the environment variable and try again:
```bash
unset TINKER_API_KEY
```

---

### (3) Jupyter warning: `TqdmWarning: IProgress not found...`

If you see:
```
TqdmWarning: IProgress not found. Please update jupyter and ipywidgets.
```

**Option A (recommended): install/upgrade Jupyter widgets**
```bash
pip install -U ipywidgets jupyter
```
Then restart the kernel.

**Option B: avoid widget-based tqdm in notebooks**
Use the standard `tqdm` progress bar instead of `tqdm.auto` / `tqdm.notebook`:
```python
from tqdm import tqdm
```

---

### (4) OOM or slow training

If you run into out-of-memory (OOM) errors or training is too slow, reduce one or more of:
- `MAX_LENGTH`
- `BATCH_SIZE`
- `LORA_RANK`

In most cases, lowering `MAX_LENGTH` gives the biggest memory/speed improvement, followed by `BATCH_SIZE`, then `LORA_RANK`.

### (5) Add a virtual environment to Jupyter (register a new kernel)

If you’re working on a remote server, it’s often convenient to add your existing virtual environment (virtualenv/venv) as a selectable Jupyter kernel.

1) **Activate the virtual environment**
```bash
source /path/to/venv/bin/activate
```

2) **Install `ipykernel` inside the environment**
```bash
pip install ipykernel
```

3) **Register the environment as a Jupyter kernel**
```bash
python -m ipykernel install --user --name=myproject --display-name "Python (myproject)"
```

4) **Select the kernel in Jupyter**
- In Jupyter Notebook/Lab: **Kernel → Change Kernel → Python (myproject)**