# On-Policy Distillation (OPD)

This guide demonstrates **on-policy distillation (OPD)** — training a *student* model to match a
*teacher's* per-token distribution **on the student's own samples** — using a **running TuFT server**.
Full runnable code is in [`examples/on_policy_distillation/`](https://github.com/agentscope-ai/TuFT/tree/main/examples/on_policy_distillation)
(`train.py`, `dataset.py`, `config.yaml`). Although this is a general OPD guide, it also documents the
practical details (teacher log-probs, the reverse-KL advantage, single- vs multi-GPU hosting) needed to
run it end-to-end on TuFT.

OPD is the recipe from Thinking Machines' [*On-Policy Distillation*](https://thinkingmachines.ai/blog/on-policy-distillation/).
It combines the **dense, per-token supervision** of distillation (every token gets a signal, unlike RL's
single scalar reward) with **on-policy sampling** (the student learns on its own trajectories, unlike
off-policy SFT, which suffers from compounding error). On TuFT it reuses the **exact same machinery as RL**:
the `importance_sampling` loss, with a per-token advantage set to the teacher–student log-prob gap.

---

## What You'll Learn

1. What **on-policy distillation** is, and when to reach for it over **SFT** or **RL**
2. How to get a **teacher's per-token log-probs** over the student's tokens with `compute_logprobs`
3. How the **per-token reverse-KL advantage** (`teacher_logprob − student_logprob`) maps onto the `importance_sampling` loss
4. How to run an end-to-end OPD loop on TuFT (sample → score → advantage → `forward_backward` → `optim_step`)
5. How to host the teacher and student, and why a real **capability gap** (not just model size) is what makes OPD pay off

---

## Table of Contents
1. [When to Use OPD vs. SFT vs. RL](#when-to-use-opd-vs-sft-vs-rl)
2. [The Task](#the-task)
3. [Minimal Training Example (OPD)](#minimal-training-example-opd)
4. [Key Concepts](#key-concepts)
   - [The OPD Loop](#the-opd-loop)
   - [Teacher Log-probs via `compute_logprobs`](#teacher-log-probs-via-compute_logprobs)
   - [The Per-Token Reverse-KL Advantage](#the-per-token-reverse-kl-advantage)
   - [Datum Format for OPD](#datum-format-for-opd)
   - [Hosting](#hosting)
5. [Results](#results)
6. [Parameter Selection](#parameter-selection)
7. [Q&A](#qa)

---

## When to Use OPD vs. SFT vs. RL

| Topic | SFT | RL | On-Policy Distillation |
|---|---|---|---|
| Training signal | Gold target tokens | One scalar reward per rollout | Teacher's full distribution at **every** token |
| Data sampled from | A fixed dataset (off-policy) | The student (on-policy) | The student (on-policy) |
| You need | Curated answers | A reward / verifier | A stronger (or better-prompted) **teacher** |
| Bits of feedback / sequence | O(N) but off-policy | O(1) | **O(N) and on-policy** |
| Failure mode it avoids | — | Sparse, slow credit assignment | SFT's compounding error on unseen states |

**Rule of thumb.** Reach for OPD when you have a **teacher you trust** (a larger model, or the same model
under a richer prompt) and want the student to internalize its behavior **cheaply** — OPD is reported to
reach RL-level results at a fraction of the compute because every token carries signal. It pairs naturally
*after* SFT and as a cheaper alternative to RL when a teacher is available.

---

## The Task

We use **[GSM8K](https://huggingface.co/datasets/openai/gsm8k)** grade-school math word problems.

| Role | Prompt it sees | Behavior |
|---|---|---|
| **Teacher** | system instruction **+ 4 worked few-shot examples** + the question | Reasons step by step, ends with `#### <number>` |
| **Student** | just the question (bare) | Trained via OPD to match the teacher |

In the default config the teacher **is the same base model** as the student — its only advantage is the
few-shot prompt. OPD distills that *context* into the student's LoRA, so afterward the student reasons with
the teacher's quality and concision **without any examples in its prompt**. This is "context distillation," and
because there is only one set of base weights it fits on **one cheap GPU**.

The only difference between the two prompts is the few-shot block (`dataset.py`):

```python
def student_prompt_ids(tokenizer, question):           # bare
    messages = [{"role": "system", "content": STUDENT_SYSTEM},
                {"role": "user", "content": question}]
    return _render_ids(tokenizer, messages)

def teacher_prompt_ids(tokenizer, question):           # + few-shot worked examples
    messages = [{"role": "system", "content": TEACHER_SYSTEM}]
    for q, a in FEWSHOT:
        messages += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
    messages.append({"role": "user", "content": question})
    return _render_ids(tokenizer, messages)
```

---

## Minimal Training Example (OPD)

The experiments below ran on a **single NVIDIA A100-40GB** on [Modal](../deployment/modal.md)
(teacher == student base model, `colocate: true`; a 24GB L4 also works with lighter batch/group).
Before running, stand up a server with
[`examples/on_policy_distillation/config.yaml`](https://github.com/agentscope-ai/TuFT/blob/main/examples/on_policy_distillation/config.yaml).

The whole loop uses three TuFT calls you already know from the RL guide — `sample`, `compute_logprobs`,
and `forward_backward(..., loss_fn="importance_sampling")`:

```python
import tinker
from tinker import types

client = tinker.ServiceClient(base_url="http://localhost:10610", api_key=TINKER_API_KEY)

# Teacher = base model conditioned on a few-shot prompt (frozen; used only to score).
teacher = client.create_sampling_client(base_model=BASE_MODEL)

# Student = a LoRA we train.
training = client.create_lora_training_client(
    base_model=BASE_MODEL, rank=LORA_RANK,
    train_mlp=True, train_attn=True, train_unembed=True,
)
```

---

## Key Concepts

### The OPD Loop

Each step is fully **on-policy**: sync the latest student weights to a sampler, let the student answer its
own bare prompts, then ask the teacher how it would have scored those exact tokens.

```python
for step in range(NUM_STEPS):
    # 1. On-policy: sample from the CURRENT student.
    student_path = training.save_weights_for_sampler(name=f"opd-{step}").result().path
    student_sampler = client.create_sampling_client(model_path=student_path)

    datums = []
    for problem in batch:
        student_prompt = types.ModelInput.from_ints(student_prompt_ids(tok, problem.question))
        res = student_sampler.sample(prompt=student_prompt, num_samples=GROUP,
                                     sampling_params=types.SamplingParams(max_tokens=256, temperature=1.0)).result()

        teacher_ids = teacher_prompt_ids(tok, problem.question)
        for seq in res.sequences:
            answer, student_lp = list(seq.tokens), list(seq.logprobs)

            # 2. Teacher scores the student's EXACT tokens (see next section).
            teacher_input = types.ModelInput.from_ints(teacher_ids + answer)
            tlp = teacher.compute_logprobs(teacher_input).result()
            p = len(teacher_ids)

            # 3. Per-token reverse-KL advantage.
            adv = [float(tlp[p + j]) - student_lp[j] for j in range(len(answer))]

            datums.append(build_distillation_datum(
                prompt=student_prompt, answer_tokens=answer,
                sampling_logprobs=student_lp, advantages=adv))

    # 4. One on-policy update.
    training.forward_backward(datums, loss_fn="importance_sampling").result()
    training.optim_step(types.AdamParams(learning_rate=1e-4)).result()
```

### Teacher Log-probs via `compute_logprobs`

`compute_logprobs(prompt)` returns one log-prob per prompt token — `lp[i] = log P(token_i | token_0..i-1)`
under that model. To score the **student's** answer under the **teacher**, concatenate the teacher's context
with the student's sampled token IDs (no re-tokenization — the IDs are reused verbatim) and slice out the
answer region:

```text
teacher_input = [ teacher_context (P tokens) ][ student_answer (T tokens) ]
                                                ^
teacher_logprob[j] = compute_logprobs(teacher_input)[P + j]      # aligns with student_logprob[j]
```

Position `P + j` is the teacher's log-prob of the student's `j`-th answer token, given the teacher's
few-shot context — exactly aligned with `seq.logprobs[j]` from sampling, which is the student's own
log-prob of the same token.

### The Per-Token Reverse-KL Advantage

Setting `advantage[t] = teacher_logprob[t] − student_logprob[t]` is the **single-sample estimator of the
negative per-token reverse-KL** `−KL(student ‖ teacher)`: in expectation over the student's samples it equals
`teacher − student` averaged under the student, which is `−D_KL` at that position. Plugged into the
`importance_sampling` loss

```text
loss = − Σ_t  exp(target_logprob_t − sampling_logprob_t) · advantage_t
```

the gradient (the ratio is ≈ 1 right after sampling) is the REINFORCE update
`−Σ_t advantage_t · ∇ log π_student(a_t)`: it **raises** the probability of tokens the teacher likes more than
the student did, and **lowers** the others — pulling the student toward the teacher. This is the same loss the
[Countdown RL guide](countdown-rl.md) uses; only the advantage differs (per-token KL instead of a reward).

### Datum Format for OPD

The `Datum` is built exactly like the RL example, but `advantages` is a **per-token list** (the KL signal)
rather than one scalar per rollout. The prompt region is masked with zero advantage so only the answer
tokens contribute (`dataset.py: build_distillation_datum`):

```python
model_input = prompt.append(types.EncodedTextChunk(tokens=answer_tokens[:-1]))
ob_len = prompt.length - 1                       # prompt region: masked out

loss_fn_inputs = {
    "target_tokens": [0]*ob_len + answer_tokens,           # what the student must predict
    "logprobs":      [0.0]*ob_len + sampling_logprobs,     # student's own sampling log-probs
    "advantages":    [0.0]*ob_len + advantages,            # teacher_lp − student_lp per token
}
```

The server computes the differentiable current-policy log-prob of `target_tokens`; `logprobs` is the
constant sampling log-prob; `advantages` carries the distillation signal.

### Hosting

Because teacher and student are the **same base model**, this runs on **one GPU** with `colocate: true`
(training and vLLM share it). A *different*, larger teacher can run on its own TuFT server — point `train.py`
at it with `--teacher-model` / `--teacher-base-url` / `--teacher-api-key`.

Before reaching for a bigger model, though, it's worth knowing that **OPD transfers capability, not just
style** — so what makes it pay off is a real *per-token* capability gap between teacher and student, more than
raw model size. A modest size bump on an easy task adds little signal (two strong same-family models mostly
agree token-by-token; we found an 8B→1.7B teacher on GSM8K didn't beat this same-model few-shot setup). A
bigger teacher helps when the gap is large and capability-relevant — a much weaker student, or a harder task
like MATH/AIME, where the small model genuinely struggles. The same-model few-shot setup here is a reliable,
cheap way to manufacture that gap.

---

## Results

Qwen3-1.7B student, single A100-40GB, **16 steps** (batch 8 × group 4), `lr=1e-4`, rank 16, evaluated
greedily on 100 held-out GSM8K problems:

| | Accuracy | Prompt sees | Reasoning style |
|---|---|---|---|
| Bare student (before) | **63%** | just the question | verbose, headers, rarely a clean final line |
| After OPD (bare prompt) | **72%** | just the question | concise teacher-style steps |
| Few-shot teacher (ceiling) | 72% | + 4 worked examples | concise steps, `#### <number>` |

The visible change: with **no few-shot examples in its prompt**, the student's held-out accuracy jumps
**63% → 72%, matching the few-shot teacher** — the teacher's reasoning is now baked into the student's
LoRA. Sample transcripts also get noticeably more concise (closer to the teacher's worked style) instead
of the bare model's rambling. The whole run costs roughly **$1 of A100-40GB time**.

Here is a held-out problem the bare student got **wrong** and the trained student got **right** — same
bare prompt, greedy decode:

> *Susan earns $5 every 10 minutes for an online task she does. If she works between 8 a.m. and 11 a.m. and
> pauses in between for half an hour, how much money does she earn?*  (answer: $75)

**Before OPD** — the bare student *lists* the half-hour pause, then forgets to subtract it, and confidently
boxes the wrong total:

```text
### Given:
- Susan earns $5 every 10 minutes.
- She works between 8 a.m. and 11 a.m.
- She pauses for half an hour in between.       ← noted, then ignored

### Step 1: total time = 11 − 8 = 3 hours = 180 minutes
### Step 2: 180 / 10 = 18 intervals
### Step 3: 18 × $5 = $90

Final Answer: 90
```

**After OPD** — it accounts for the pause and lands the right answer, in the teacher's concise style:

```text
From 8 a.m. to 11 a.m. is 3 hours. She pauses half an hour, so she works 3 − 0.5 = 2.5 hours = 150 minutes.
In 10 minutes she earns $5, so 150 / 10 = 15 intervals.
15 × $5 = $75.
```

This isn't a one-off: on the held-out set, far more problems flip wrong → right than the reverse (**13 vs 6**
in this run — the net is the accuracy gain above). The bare model's typical failure is exactly the above —
an ornate setup that drops a constraint or runs out of the token budget mid-derivation — and OPD makes it
commit to the teacher's short, complete arithmetic.

The same tightening happens **even on problems the bare model already gets right** — it just stops rambling.
Same bare prompt, same answer (142), before and after:

> *Ricardo grows tomatoes and eggplants in his garden. Each tomato plant yields 22 tomatoes while each plant
> of eggplant yields 4 eggplants. He planted 5 tomato plants and 8 plants of eggplant. How many fruits can
> Ricardo get from his plants?*  (answer: 142)

**Before OPD** — opens with markdown headers and boilerplate; 256 tokens in, it is still *setting up*:

```text
Let's break down the problem step by step:

### Given:
- Each **tomato plant** yields **22 tomatoes**.
- Each **eggplant plant** yields **4 eggplants**.
- Ricardo planted:
  - 5 **tomato plants**
  - 8 **eggplant plants**

---

### Step 1:                       ← (still just restating the problem here)
```

**After OPD** — straight to the arithmetic, in the teacher's concise style:

```text
Ricardo's tomato plants: 5 plants × 22 tomatoes = 110 tomatoes
Ricardo's eggplant plants: 8 plants × 4 eggplants = 32 eggplants
Total fruits: 110 + 32 = 142

Ricardo can get 142 fruits from his plants.
```

More real transcripts — including the other wrong → right flips — are in
[`examples/on_policy_distillation/sample_outputs.md`](https://github.com/agentscope-ai/TuFT/blob/main/examples/on_policy_distillation/sample_outputs.md).

OPD improves **fast** — most of the gain lands within a handful of steps (dense per-token signal). Reverse-KL
is mode-seeking and can over-train if pushed far (accuracy drifts back down after a few dozen steps), so we
keep the run short; `--num-steps 16` is a good default (see [Parameter Selection](#parameter-selection)).

---

## Parameter Selection

- **`learning_rate`** — `1e-4` is a stable default (same as the RL example). Reverse-KL is touchier than
  cross-entropy; if loss spikes or the student degenerates, halve it.
- **`kl_coef` / `adv_clip`** — scale and clip the per-token advantage. Keep `kl_coef=1.0`; `adv_clip` (default
  `10`) just guards against rare huge log-prob gaps.
- **`group_size`** — student samples per prompt. More samples = a denser, lower-variance signal per step
  (and more teacher scoring calls). `4` is a good balance; drop to `1–2` to go faster.
- **`temperature`** — sample rollouts at `~1.0` so the student explores tokens for the teacher to correct;
  evaluate greedily (`0.0`).
- **`lora_rank`** — `16` is plenty for this behavior transfer; raise toward `32–64` only for harder
  capability gaps.
- **`num_steps`** — OPD improves fast; most of the gain lands in the first handful of steps. Reverse-KL can
  over-train (accuracy drifts back down if you keep going), so keep it short — `16` is a good default, and
  the script prints periodic `--eval-every` accuracies so you can see the peak.
- **teacher** — the gain tracks the *per-token capability gap*, not raw model size (see [Hosting](#hosting)).

---

## Q&A

### (1) Why `importance_sampling` and not a dedicated "distillation" loss?

Because OPD *is* policy-gradient with a per-token KL "reward." `importance_sampling` already computes
`−Σ exp(target_lp − sampling_lp) · advantage`; setting `advantage = teacher_lp − student_lp` makes that the
on-policy distillation update. No new loss function is needed — and you get RL-style importance correction
for free if the sampler and trainer log-probs drift.

### (2) The teacher and student use different prompts — is the KL still valid?

Yes. The student and teacher define distributions over the **same answer tokens**, each conditioned on its
own context. The reverse-KL is between those two conditionals at each answer position; minimizing it makes
the bare-prompted student behave like the few-shot-prompted teacher. That is precisely the goal of context
distillation.

### (3) The accuracy jumps but the student still doesn't emit the teacher's `####` line — why?

On-policy distillation can only reshape the probability of tokens the student **actually samples**. The bare
Qwen3 student reasons well but never explores the literal `####` marker, so the teacher never gets to score
(and reinforce) it on the student's trajectory. What *does* transfer is the teacher's **reasoning quality and
concision** on the tokens the student does produce — which is what lifts accuracy. This is a general property
of on-policy methods: they sharpen and re-weight explored behavior, they don't graft in unexplored tokens.
(To also transfer a surface format, seed it into the student prompt so it gets explored, or warm-start with a
little SFT.)

### (4) `compute_logprobs` returns `None` at some positions

The first token has no preceding context, so its log-prob is `None`; the example treats any `None` as a
zero advantage for that token. Because we always slice the **answer** region (`P + j` with `P ≥ 1`), this only
matters defensively.

### (5) Dataset download fails (huggingface.co unreachable)

Set a mirror before importing `datasets`:
```python
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
```

### (6) OOM or slow on a small GPU

Lower `--max-new-tokens`, `--group-size`, or `--batch-size` (in that order), and/or reduce
`sampling_memory_fraction` / `sampling_max_model_len` in `config.yaml`. A 24GB L4 works with
`--batch-size 4 --group-size 2`; keep the teacher and student on the same base model (the default) so only
one model is resident on the GPU.
