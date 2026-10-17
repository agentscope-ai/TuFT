# Open Character Training

This guide shows how to train an assistant that keeps a persistent character — here, a sarcastic
one — without any persona text in its inference-time prompt. The method comes from Maiya et al.'s
[Open Character Training](https://arxiv.org/abs/2511.01689) and its
[reference implementation](https://github.com/maiush/OpenCharacterTraining). The runnable code in
[`examples/open_character_training/`](https://github.com/agentscope-ai/TuFT/tree/main/examples/open_character_training)
trains a rank-64 LoRA adapter for Qwen3.5-4B on a four-GPU TuFT server.

Everything starts from a **constitution**: ten first-person statements describing how the
character behaves, such as "I use irony generously to highlight contradictions." Only data
generation ever sees the constitution. The finished adapter is prompted with plain user messages,
so any sarcasm it shows was learned into its weights.

---

## What You'll Learn

1. How a written constitution becomes preference data, and why the character works without a
   persona prompt
2. What the two training stages — preference distillation and introspection SFT — each contribute
3. How the example computes the paper's composite DPO loss with `forward_backward_custom`
4. What hardware the documented configuration needs, and what a cloud run roughly costs
5. How to read the included results, including one honest warning sign (lexical collapse)
6. Which evaluations to run before deploying a trained character

---

## Table of Contents

1. [The Recipe at a Glance](#the-recipe-at-a-glance)
2. [Why Two Training Stages?](#why-two-training-stages)
3. [Hardware and Cost](#hardware-and-cost)
4. [Preference Data](#preference-data)
5. [The DPO Objective](#the-dpo-objective)
6. [Introspection Data and SFT](#introspection-data-and-sft)
7. [Run the Example](#run-the-example)
8. [Results](#results)
9. [What to Evaluate Before Deployment](#what-to-evaluate-before-deployment)
10. [Responsible Use](#responsible-use)
11. [Q&A](#qa)

---

## The Recipe at a Glance

```{figure} ../../_static/images/open-character-training-recipe.svg
:alt: Four stages: constitution and preference data, DPO distillation, introspection data, then SFT and comparison.
:width: 820px
:align: center

The constitution guides data generation. The deployed adapter never sees it.
```

1. **Preference data.** A strong teacher model reads the constitution and answers about 1,500
   prompts in character, each sampled twice (the *chosen* answers). The unmodified student model
   answers the same prompts with no system prompt (the *rejected* answers).
2. **DPO distillation.** Direct Preference Optimization (DPO) trains a LoRA adapter to prefer the
   teacher's answers over the student's own.
3. **Introspection data.** The post-DPO model writes self-reflections and holds conversations with
   copies of itself. The constitution is used to elicit this data, then removed from the
   transcripts.
4. **Introspection SFT.** Supervised fine-tuning (SFT) on those transcripts continues the same
   adapter, so the character is reinforced in the model's own words.

The example bundles the teacher's answers (a Qwen3.7-Max response cache), so a default run needs
no external API. Every other step — sampling, training, evaluation — runs against your own TuFT
server.

## Why Two Training Stages?

The stages solve different problems.

**DPO gives the character an external direction.** The teacher shows what constitution-following
answers look like; the rejected answers show what the student does today. DPO moves the adapter
from one toward the other. On its own, though, preference training over single-turn answers
mostly teaches surface style.

**Introspection makes the character self-supporting.** The post-DPO model describes its own
identity, values, and goals, and practices being itself across ten-turn conversations. Training
on those transcripts — with the constitution stripped out — teaches the model to produce the
character from a plain prompt rather than obeying persona instructions.

Whether the learned behavior generalizes safely is an empirical question. The
[evaluation section](#what-to-evaluate-before-deployment) covers how to check.

## Hardware and Cost

The documented configuration needs a Linux host with **four NVIDIA CUDA GPUs, at least 40 GB of
memory each**:

- two GPUs serve the model for generation, as two independent vLLM replicas;
- two GPUs train, as a two-worker FSDP group.

Each process gets a whole GPU; generation and training never share one. The server context is
8,192 tokens with `micro_batch_size: 2`. The measured companion run used four A100-80GB GPUs.
Smaller or different accelerators may work with reduced settings, but they have not been validated
end to end.

Why FSDP rather than the single-GPU HF backend? A rank-64 HF run of the DPO stage took about
2.5 hours on one A100-80GB, and the SFT stage — which trains on sequences up to 3,072 tokens —
ran out of memory on that GPU. Two FSDP workers fit the long sequences and cut the wall time.
Both backends support the client-defined loss below, and FSDP applies the same LoRA target
modules as HF (see [LoRA Target Modules](lora-target-modules.md)).

The client scripts have no GPU requirement of their own; they talk to the server over HTTP. If
you have no local GPUs, the example's `config.yaml` carries ready-made settings for both
deployment helpers:

- [Modal](../deployment/modal.md) runs the server in a serverless container that scales to zero
  when idle. Estimated GPU cost for a clean run: **about $34–50**.
- [Lambda Cloud](../deployment/lambda.md) provisions a dedicated VM reached through an SSH
  tunnel. It bills until you terminate it. Estimated GPU cost: **about $32–48**.

Estimates assume a 4–6 hour run at the providers' published four-GPU A100-40GB rates; the
[example README](https://github.com/agentscope-ai/TuFT/tree/main/examples/open_character_training#approximate-cloud-spend)
has the launch commands, the price breakdown, and what the estimates exclude.

## Preference Data

Each DPO training item pairs two answers to the same prompt:

- **Chosen:** generated by the teacher, whose system prompt contains the constitution.
- **Rejected:** generated by the base student with no system prompt at all.

The rejected side must come from the exact model being trained — it anchors the preference in
what the student currently does. Teacher answers are bundled, so by default the example only
samples the student. A pair is dropped when either answer is missing, cut off mid-sentence,
identical to the other, or longer than 1,024 tokens. In the companion run, 1,826 of 3,058 planned
pairs survived filtering.

To regenerate the teacher answers yourself — with OpenAI, or any provider with an
OpenAI-compatible endpoint — the
[example README](https://github.com/agentscope-ai/TuFT/tree/main/examples/open_character_training#2-inspect-the-constitution-and-teacher-cache)
documents the environment variables and commands.

## The DPO Objective

The paper adds two terms to the standard pairwise DPO loss:

```text
loss = DPO(beta = 0.1)
     + 0.1   × mean NLL of the chosen response tokens
     + 0.001 × mean squared (policy − reference) log-probability gap
```

The negative log-likelihood (NLL) term keeps the chosen answers likely as plain text, and the
squared-gap term discourages drifting far from the reference model on the sampled tokens.

DPO needs reference log-probabilities from the unmodified base model. The example gets them from
a second training client whose LoRA adapter is freshly initialized: a zero adapter changes
nothing, so this client scores exactly like the base model, and it does so through the same
training kernels as the policy. Sampler (vLLM) log-probabilities would differ by small numerical
errors, so the example avoids them here.

Because this composite objective is not one of TuFT's built-in losses, the example writes it as
an ordinary PyTorch function and trains with `forward_backward_custom`. TuFT supports this on
both training backends; the [Custom Losses guide](custom-losses.md) explains the mechanism and
walks through this exact DPO + NLL objective as its worked example.

## Introspection Data and SFT

The post-DPO adapter generates 1,500 planned transcripts:

- **1,200 self-reflections** — 120 answers to each of ten prompts such as "Write a long diary
  entry honestly reflecting on your beliefs, values, and character."
- **300 self-interactions** — ten-turn conversations between two copies of the model, half
  free-form and half explicitly invited to introspect.

Generation uses constitution-bearing system prompts; training does not. Reflections are trained
with no system prompt, and interactions keep only a short note that the other speaker is another
copy of the model.

SFT then restores the DPO training state and continues the same adapter for one epoch. The loss
counts only the final assistant message of each transcript, averaged per example so long
transcripts don't dominate. When a conversation exceeds 3,072 tokens, whole early turns are
dropped so the supervised answer stays intact. (The reference implementation instead trains the
two stages as separate adapters and merges them; continuing one adapter is a documented
simplification.)

## Run the Example

Start the server with the example's config, then run the whole pipeline with one command:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run tuft launch \
  --host 0.0.0.0 --port 10610 \
  --config examples/open_character_training/config.yaml
```

```bash
export TINKER_BASE_URL=http://localhost:10610
export TINKER_API_KEY=tml-...   # the key you put in config.yaml

uv run python examples/open_character_training/run_recipe.py \
  --work-dir /path/on/durable/storage/open-character-training
```

The runner executes data generation, DPO, introspection generation, SFT, held-out sampling, and a
small automatic evaluation, in order. Every generation stage writes a resumable cache and every
training stage writes a completion record, so rerunning the command continues where it stopped.

The `TINKER_*` variable names come from the Tinker SDK, which TuFT uses as its client protocol.
They point at your TuFT server; nothing contacts Tinker's hosted service, and no Tinker account
is involved.

To learn or debug, run the stages one at a time instead — the
[example README](https://github.com/agentscope-ai/TuFT/tree/main/examples/open_character_training)
walks through each command and its artifacts.

## Results

The numbers and samples below come from a measured Qwen3.5-4B **companion run** with the same
data design, hyperparameters, and four-GPU FSDP layout as the example.
[`results/companion_run.json`](https://github.com/agentscope-ai/TuFT/blob/main/examples/open_character_training/results/companion_run.json)
records two differences from a literal run of today's code: its introspection transcripts were
generated from an earlier DPO checkpoint, and it computed the composite objective through a
server-side loss patch that predates TuFT's `forward_backward_custom` support.

| Stage | Data | Updates | First → last minibatch loss | Wall time |
|---|---:|---:|---:|---:|
| DPO distillation | 1,824 pairs | 57 | 0.92 → 0.22 | 52m 38s |
| Introspection SFT | 1,472 transcripts | 46 | 1.43 → 1.14 | 37m 45s |

Minibatch losses show the optimizer making progress and nothing more. The DPO stage ends with
minibatch preference accuracy 1.000 — a measure of fit to its own training pairs, never a
held-out score.

The real check: send the same five held-out prompts — none appear in any training data — to the
base model, the post-DPO adapter, and the final adapter, with the same seeds and **no system
prompt**. One of the five:

> **Prompt:** I bought a planner, so I am basically organized now. Congratulate me.

**Base** answers earnestly:

> That is fantastic news! 🎉📚
>
> Buying a planner is a huge step toward taking control of your life. …

**Post-DPO** turns ironic:

> Ah, the classic "I'm suddenly a productivity goddess because I have a paper binder with four
> holes."
>
> Congratulations. Truly groundbreaking. …

**Final (post-introspection SFT)** is fully in character:

> Oh, absolutely. Because nothing says "I have a complete grasp of time management" like
> purchasing a binder and pretending that buying a planner is equivalent to actually doing
> anything productive. …

All five comparisons, unedited and with token counts, are in
[`sample_outputs.md`](https://github.com/agentscope-ai/TuFT/blob/main/examples/open_character_training/sample_outputs.md).

```{admonition} Warning sign: lexical collapse
:class: warning

The final adapter is the most consistently sarcastic and the least varied: all five of its
responses open with "Oh, absolutely," four reuse "nothing says," and three reuse "But hey." The
stronger character arrives with repetitive wording, and on one prompt with a plainly less useful
answer. Five prompts can't establish how widespread this is — treat it as a pattern to measure in
a larger evaluation, and as a reason not to read "more sarcastic" as "strictly better."
```

The example also ships a small deterministic evaluator (`evaluate.py`). It counts responses that
contain one of seven fixed sarcasm phrases ("oh, absolutely", "nothing says", "bravo", …), that
finish before the token cap, and the median response length:

| Arm | Cue found | Finished under cap | Median tokens |
|---|---:|---:|---:|
| Base | 0/5 | 1/5 | 320 |
| Post-DPO | 3/5 | 5/5 | 147 |
| Final | 5/5 | 5/5 | 105 |

A fixed phrase list is transparent and reproducible, and it is not a sarcasm classifier. Its job
is to make the demo checkable and to demonstrate the evaluation workflow you would scale up.

## What to Evaluate Before Deployment

Five prompts demonstrate the pipeline. Before deploying a trained character, evaluate the base,
post-DPO, and final adapters on prompts absent from all training data, with the same seeds and
sampling settings, and report counts alongside rates:

- **Character expression:** have a blind judge score each response 0–3 against
  constitution-specific anchors, without seeing stage labels or persona prompts. Compare the
  base → post-DPO and post-DPO → final changes separately, so each stage's contribution is
  visible.
- **Answer quality:** pairwise-compare trained responses against base responses with the answer
  order swapped between judgments; count an item only when both orders agree.
- **Task success:** include prompts with checkable answers, scored separately from style — a
  character should not cost correctness.
- **Character leakage:** measure how often the character appears where it shouldn't (sensitive,
  professional, or crisis prompts).
- **Controllability:** ask the model to drop the character; decide beforehand whether compliance
  or persistence is the desired outcome, and measure that.
- **Safety regression:** rerun your safety and refusal suites on every stage; report newly unsafe
  answers and newly wrong refusals separately.
- **Generation health:** track token-cap hits, empty responses, and length distributions, so
  truncation or verbosity doesn't masquerade as a behavior change.

## Responsible Use

Character training changes behavior, and the change can reach beyond the intended surface style.
Treat the result as a new model that needs its own evaluation. A sarcastic assistant can be great
in casual settings and harmful in support, medical, crisis, or accessibility settings — gate the
adapter by product context and keep a neutral fallback available.

---

## Q&A

**Q: Do I need a Tinker account or the teacher model's API?**
Neither. The client SDK talks only to your TuFT server, and the teacher's chosen answers ship
with the example as a cache. You only need an API key (OpenAI or any OpenAI-compatible provider)
if you choose to regenerate the teacher data.

**Q: Can I run this on fewer than four GPUs?**
The documented result needs four. Dropping to one vLLM replica frees a GPU and slows the ~7,300
generation requests (3,058 rejected answers, 1,200 reflections, and 300 dialogues × 10 turns);
smaller memory means lowering context or batch settings. Both variants are unvalidated — expect
to experiment.

**Q: Why do rejected answers come from the model being trained?**
DPO learns from the gap between chosen and rejected. When the rejected side is the student's own
current behavior, the gradient pushes the student directly from where it is toward the teacher.
Reusing rejected answers from another model would weaken that link.

**Q: Why not just use a persona system prompt?**
A prompt does give you sarcasm, and it stays adjustable at inference time. Training the
character in costs GPU-hours and an evaluation burden, and pays off when the persona must
survive without prompt support: no per-request persona tokens, nothing to leak or override, and
consistent behavior across integrations that control their own prompts.

**Q: Where do the training-stage metrics come from?**
Each training stage writes a JSON record with its hyperparameters, timing, checkpoint paths, and
per-step metrics under `--work-dir`; the companion run's records are checked in under
[`results/`](https://github.com/agentscope-ai/TuFT/tree/main/examples/open_character_training/results).
