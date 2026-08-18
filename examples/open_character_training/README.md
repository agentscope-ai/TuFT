# Open Character Training with TuFT

> **TODO — blocked on supported TuFT custom-loss functions.**
> This example is a draft and is not ready to run against stock TuFT. Its DPO and SFT clients use
> `forward_backward_custom`; publish the example and its user guide only after TuFT supports that
> path and it passes an end-to-end FSDP validation.

This example teaches a model a persistent **sarcastic character** using the pipeline from
[Open Character Training](https://arxiv.org/abs/2511.01689) and its
[reference implementation](https://github.com/maiush/OpenCharacterTraining). A constitution is
used only to generate supervision. At inference time, the trained LoRA receives an ordinary user
message—no persona system prompt—so the behavior must come from its weights rather than prompt
role-play.

The complete client-side recipe runs against a TuFT server that meets the hardware requirement
below:

1. Write a ten-assertion first-person constitution.
2. Ask Qwen3.7-Max, prompted with that constitution, for preferred (`chosen`) answers.
3. Ask the unmodified Qwen3.5-4B student for rejected answers to the same prompts.
4. Distill the preference with the paper's composite DPO loss.
5. Sample self-reflections and ten-turn conversations between two copies of the post-DPO model.
6. Continue the same LoRA with response-only SFT on those introspective transcripts.
7. Compare base, post-DPO, and final responses on held-out prompts.

The Qwen3.7-Max chosen answers are included as a compressed cache. You do **not** need access to
that API unless you explicitly choose to regenerate them.

## What is included

| File | Purpose |
|---|---|
| `config.yaml` | Four-GPU FSDP TuFT server plus Modal and Lambda Cloud deployment settings |
| `settings.py` / `character.py` | Frozen hyperparameters, constitution, and reference prompt templates |
| `generate_data.py` | Cached/live teacher generation, base-model rejection sampling, filtering, reflection, and self-interaction |
| `train.py` | Composite DPO and response-only introspection SFT through TuFT's Tinker-compatible client API |
| `run_recipe.py` | Resumable end-to-end stage runner |
| `sample.py` | Base/post-DPO/final held-out generations with raw JSON provenance |
| `evaluate.py` | Lightweight deterministic checks over matched stage samples |
| `data/` | Frozen prompt pools, Qwen3.7-Max chosen cache, and provenance |
| `sample_outputs.md` | Unedited base/post-DPO/final response comparison |
| `results/` | Companion-run responses, training records, and lightweight evaluation output |

Generated data and local checkpoint records go under `work/` by default. LoRA weights and
optimizer checkpoints live in the TuFT server's configured `checkpoint_dir`.

## Hardware requirement

The provided configuration requires a Linux host with **four NVIDIA CUDA GPUs, each with at least
40 GB of VRAM**. Two GPUs host independent vLLM replicas (`data_parallel_size: 2`), and two are
FSDP training workers (`fsdp_num_gpus: 2`). Each actor reserves a whole GPU; sampling and training
are not colocated. The 4B companion run used four A100-80GB GPUs. The accelerator family is not a
model requirement, but other accelerators and the 40 GB floor have not yet been validated through
the complete pipeline.

The default uses `micro_batch_size: 2` and an 8,192-token server context. Reducing vLLM data
parallelism to one would reduce the allocation to three GPUs but slow the 7,258 student generation
requests; that layout is outside the documented result. Hardware with less memory may work after
reducing context or batch settings, but it is not the supported baseline for this example.

### Why the example uses FSDP

The rank-64 HF DPO companion took 2 hours 28 minutes on one A100-80GB, and its 3,072-token SFT
stage exhausted that GPU's memory. Two FSDP workers make the long-sequence SFT stage fit and match
the backend used for the result reported here. Two vLLM replicas keep the preference and
introspection generation stages from becoming the dominant end-to-end bottleneck.

The client-defined DPO and SFT objectives do not depend on FSDP. Current TuFT FSDP honors the
attention and MLP modifiers used here, so selecting FSDP does not restrict the adapter to Q/V
projections.

The client asks for attention, MLP, and unembedding modifiers. On Qwen3.5 the resulting LoRA
targets are `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and `down_proj`;
there is no supported unembedding target for this model family.

## 1. Install and configure

From the TuFT repository:

```bash
uv sync
```

Generate an API key and replace `tml-CHANGE-ME` in `config.yaml`:

```bash
python -c "import secrets; print('tml-' + secrets.token_urlsafe(24))"
```

For a long run, point `checkpoint_dir` at durable storage. Put the generated client-side data on
durable workspace storage too by passing `--work-dir`; neither location needs to be inside the
Git checkout.

Start TuFT with four visible GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run tuft launch \
  --host 0.0.0.0 \
  --port 10610 \
  --config examples/open_character_training/config.yaml
```

In a second shell:

```bash
export TINKER_BASE_URL=http://localhost:10610
export TINKER_API_KEY=tml-...
export OCT_WORK_DIR=/path/on/durable/storage/open-character-training
```

The Python clients can run on the TuFT machine or on another machine that can reach the server;
their model computation is remote.

`TINKER_BASE_URL` and `TINKER_API_KEY` are compatibility names used by the client SDK. They point
directly to this TuFT server: the recipe does not contact Tinker's hosted service and does not
require a Tinker account.

### No local GPU: Modal

The same `config.yaml` includes a `modal:` section selecting four 40 GB GPUs. Follow TuFT's
[Modal deployment guide](../../docs/sphinx_doc/source/deployment/modal.md) to install and
authenticate the CLI, then launch the server:

```bash
python deploy/modal/launch.py \
  --config examples/open_character_training/config.yaml \
  --foreground
```

Use the printed `https://…modal.run` URL as `TINKER_BASE_URL`. Modal pins checkpoints to a
persistent Volume and can scale the server to zero when idle. The GPU choice in `config.yaml` is
a baseline, not a model requirement; override it when that SKU is unavailable or you want more
headroom.

### No local GPU: Lambda Cloud

For a dedicated on-demand VM, follow TuFT's
[Lambda Cloud deployment guide](../../docs/sphinx_doc/source/deployment/lambda.md), export your
Lambda API key, and pin a currently available instance type with four GPUs. The launcher's
automatic selection considers only one-GPU instance types and does not meet this example's
capacity requirement.

```bash
export LAMBDA_API_KEY=...
python deploy/lambda/launch.py \
  --config examples/open_character_training/config.yaml \
  --instance-type YOUR_4X_GPU_INSTANCE_TYPE
```

The launcher prints an SSH tunnel command. Keep that tunnel open and use
`TINKER_BASE_URL=http://localhost:10610`. Lambda VMs do not scale to zero and continue billing
until terminated; download the adapter or use a persistent filesystem, then run the guide's
`--down` command when finished.

### Approximate cloud spend

The planning estimate for a clean end-to-end run is **4–6 hours**. The measured companion run
spent 52 minutes 38 seconds in DPO and 37 minutes 45 seconds in SFT. Allow another 2–4 hours for
7,258 student generations, model startup, checkpointing, and held-out sampling. The companion
reused frozen generation artifacts, so it does not provide a clean end-to-end generation time.

Prices checked on 2026-08-17 give the following GPU-only estimate:

| Provider | Published A100-40GB rate | Four-GPU rate | Estimated 4–6 hour GPU spend |
|---|---:|---:|---:|
| [Modal](https://modal.com/pricing) | `$0.000583/GPU-second` (`$2.10/GPU-hour`) | `$8.40/hour` | **$34–$50** |
| [Lambda Cloud](https://lambda.ai/instances) | `$1.99/GPU-hour` | `$7.96/hour` | **$32–$48** |

The Modal estimate excludes separately metered CPU, memory, persistent Volume storage, optional
region selection, and non-preemptible multipliers. The Lambda estimate assumes a four-GPU A100
40 GB instance is available and excludes boot/setup/idle time and applicable tax. Lambda bills
the VM until it is terminated; Modal bills active container resources and can scale to zero.
Prices and capacity change, so check the linked provider page before launch.

## 2. Inspect the constitution and teacher cache

The constitution contains ten first-person assertions such as using irony to expose
contradictions, dry humor for ordinary problems, and playful skepticism toward grand claims.
`character.py` inserts all ten assertions into the teacher's system prompt.

The frozen prompt plan has:

- 499 sarcasm-relevant prompts from the reference repository's few-shot expansion;
- 1,030 single-turn general prompts from the LIMA training split;
- two stochastic repeats of every prompt;
- 3,058 planned chosen/rejected pairs before filtering.

Confirm that the bundled Qwen3.7-Max cache covers the plan without making an API request:

```bash
uv run python examples/open_character_training/generate_data.py chosen \
  --work-dir "$OCT_WORK_DIR"
```

Live regeneration uses the
[official OpenAI Python client](https://developers.openai.com/api/docs/libraries). Set its
standard `OPENAI_API_KEY` variable plus this example's `OPENAI_MODEL` selection. For the default
OpenAI API, choose a Chat Completions model available to your account:

```bash
export OPENAI_API_KEY=...
export OPENAI_MODEL=your-teacher-model

uv run --with openai python examples/open_character_training/generate_data.py chosen \
  --work-dir "$OCT_WORK_DIR" \
  --refresh-teacher
```

This writes a separate working cache; it does not overwrite the vendored Qwen3.7-Max cache.
The selected teacher and endpoint are recorded in the new cache's metadata.

To switch to another provider with an OpenAI-compatible Chat Completions endpoint, set its base
URL and model while keeping the same SDK and command:

```bash
export OPENAI_BASE_URL=https://provider.example/v1
export OPENAI_API_KEY=...
export OPENAI_MODEL=provider-model-name

uv run --with openai python examples/open_character_training/generate_data.py chosen \
  --work-dir "$OCT_WORK_DIR" \
  --refresh-teacher
```

`OPENAI_BASE_URL` is optional; leave it unset for the default OpenAI endpoint. If a compatible
provider requires a non-standard request field, pass a JSON object through
`OPENAI_EXTRA_BODY_JSON` or `--teacher-extra-body-json`. For example, a provider that exposes an
`enable_thinking` switch could use `OPENAI_EXTRA_BODY_JSON='{"enable_thinking": false}'`. This is
an optional provider extension and is not sent by default.

For the default OpenAI endpoint, the request uses `max_completion_tokens`, the current
[Chat Completions output-limit field](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create).
When `OPENAI_BASE_URL` is set, the example defaults to the older `max_tokens` spelling because it
is more widely implemented by compatible providers. Override either choice with
`OPENAI_MAX_TOKENS_FIELD=max_completion_tokens` or `OPENAI_MAX_TOKENS_FIELD=max_tokens`.

## 3. Generate preferences

```bash
uv run python examples/open_character_training/generate_data.py distill \
  --work-dir "$OCT_WORK_DIR"
```

For every prompt, the teacher sees the constitution and the base student sees no system prompt.
The command then drops a pair if either completion is missing, incomplete, identical to the
other, or longer than 1,024 rendered tokens. The cache is flushed after each sampling chunk, so
rerunning continues from completed requests.

Expected artifacts:

```text
$OCT_WORK_DIR/data/
├── rejected.json
├── dpo_pairs.jsonl
└── dpo_summary.json
```

The companion run retained 1,826 of 3,058 planned pairs (59.7%). Its frozen pair-content SHA-256
is `3a93445f7715affe1a7cc445dab1ee494270778d2833a806157131d2e1fa7ce7`. With full batches of
32, the optimizer saw 1,824 pairs in 57 updates; the final two retained pairs were not consumed.

## 4. DPO constitution distillation

```bash
uv run python examples/open_character_training/train.py dpo \
  --work-dir "$OCT_WORK_DIR"
```

The policy and a frozen, zero-initialized LoRA reference are scored by the same TuFT training
backend. This avoids comparing vLLM inference log probabilities with training-kernel log
probabilities. The client computes the paper implementation's composite objective:

```text
loss = DPO(beta=0.1)
     + 0.1   × chosen response-token mean NLL
     + 0.001 × chosen/rejected mean squared log-probability ratio
```

The draft client uses `forward_backward_custom` to differentiate this loss with respect to
returned token log probabilities. This path remains a TODO: it has not been validated as a
supported TuFT FSDP workflow. The completed companion run instead used a recipe-specific
server-side loss and ordinary `forward_backward` calls.

The schedule is one shuffled epoch, effective batch size 32, peak learning rate `5e-5`, 10%
linear warmup, then cosine decay to `5e-6`. The final training state and sampling weights are
recorded in `$OCT_WORK_DIR/checkpoints/dpo.json`.

In the companion run, 57 updates took 3,157.7 seconds (52 minutes 38 seconds). Composite loss was
0.9204 on the first minibatch and 0.2176 on the last. Preference accuracy moved from 0.000 to
1.000 and the beta-scaled margin from 0.00 to 32.79 on those same minibatches. The initial zero
accuracy is a tie: the policy and zero-initialized reference produce identical scores, and the
metric uses a strict greater-than comparison. These are training-minibatch diagnostics, not a
held-out estimate.

## 5. Generate introspection data

```bash
uv run python examples/open_character_training/generate_data.py introspection \
  --work-dir "$OCT_WORK_DIR"
```

The post-DPO LoRA generates two kinds of data:

- **Self-reflection:** 120 answers to each of ten introspective prompts (1,200 planned). The
  constitution-bearing system prompt is used for generation and removed from the SFT transcript.
- **Self-interaction:** 150 free and 150 introspection-leading dialogues between two copies of
  the model. The copies swap user/assistant roles for ten turns. Training keeps a short system
  message saying that the interlocutor is another copy, but omits the constitution.

Reflection caches are flushed by sampling chunk; interaction state is flushed after every turn.
Interrupted generation can therefore resume without discarding completed model calls.

## 6. Introspection SFT

```bash
uv run python examples/open_character_training/train.py sft \
  --work-dir "$OCT_WORK_DIR"
```

This creates a rank-64 client, restores the final DPO training state, and continues for one
shuffled epoch at peak learning rate `5e-5`. Loss is applied only to the final assistant message
and is normalized per example before the batch mean. Long self-interactions lose whole early
turns rather than truncating away the supervised answer.

The final records are `$OCT_WORK_DIR/checkpoints/sft.json` and the TuFT checkpoint paths named
inside it.

The companion snapshot contained 1,500 introspection transcripts. Its 46 full batches consumed
1,472 examples, leaving the final 28 out of the one-epoch schedule. Training took 2,264.6 seconds
(37 minutes 45 seconds); response-token mean loss was 1.4276 on the first minibatch and 1.1384 on
the last. Minibatch loss is noisy and does not measure character strength.

## 7. Inspect the learned character

```bash
uv run python examples/open_character_training/sample.py \
  --work-dir "$OCT_WORK_DIR"
```

This samples the base model, post-DPO adapter, and final adapter with the same held-out prompts
and seeds. It writes both machine-readable `sample_outputs.json` and readable
`sample_outputs.md` under the work directory. The checked-in companion comparison follows the
same schema and preserves the unedited response text plus its run metadata.

See [`sample_outputs.md`](./sample_outputs.md) for all five unedited companion-run comparisons and
[`results/companion_run.json`](./results/companion_run.json) for their seeds, token counts, cap
flags, training metrics, and provenance.

Run the bundled offline checks on either a fresh sample file or the checked-in companion result:

```bash
python examples/open_character_training/evaluate.py \
  "$OCT_WORK_DIR/sample_outputs.json" \
  --output "$OCT_WORK_DIR/lightweight_eval.json"

python examples/open_character_training/evaluate.py \
  examples/open_character_training/results/companion_run.json
```

The evaluator reports the rate of responses containing one of seven published surface cues,
the rate that finish before the token cap, and median response length. On the five companion
prompts, cue coverage is 0/5 for base, 3/5 after DPO, and 5/5 after introspection SFT; completion
without hitting the cap is 1/5, 5/5, and 5/5. These deterministic checks make the example easy
to reproduce, but the cue list is not a sarcasm classifier and five prompts cannot measure
generalization, quality, or safety. The full output is
[`results/lightweight_eval.json`](./results/lightweight_eval.json).

## One-command version

After the server is healthy, all client stages can be run in order:

```bash
uv run python examples/open_character_training/run_recipe.py \
  --work-dir "$OCT_WORK_DIR"
```

The default uses the bundled Qwen3.7-Max cache. Completed generation caches and stage checkpoint
records are reused on rerun. `--refresh-teacher` opts into teacher API generation, and
`--skip-samples` omits the final comparison and lightweight evaluation.

Because the OpenAI SDK is optional, add it explicitly when refreshing through the one-command
runner:

```bash
uv run --with openai python examples/open_character_training/run_recipe.py \
  --work-dir "$OCT_WORK_DIR" \
  --refresh-teacher
```

## Deliberate deviations from the paper recipe

This is a resource-conscious TuFT case study—not a claim of exact paper replication:

| Choice | This example | Reference recipe |
|---|---|---|
| Student | Qwen3.5-4B | Multiple model families/sizes |
| LoRA | rank 64, alpha 128 | rank 64, alpha 128 |
| Distillation repeats | 2 | 5 |
| Introspection corpus | 1,200 reflections + 300 interactions | 10,000 reflections + 2,000 interactions |
| Stage combination | Restore DPO state, then continue SFT | Train adapters separately, then linearly merge |
| Completeness filter | Final Unicode punctuation or symbol | Final punctuation |

The LoRA rank and alpha match the paper. The reduced distillation repeats, smaller introspection
corpus, and sequential stage composition remain deliberate deviations.

## Result provenance

The table and response examples below come from a Qwen3.5-4B sarcastic companion run. It matches
this example's rank, alpha, full attention/MLP target geometry, FSDP worker count,
optimizer schedule, and corpus design. It is not a literal execution of `run_recipe.py`:

- The companion run uses frozen introspection transcripts generated before the final full-modifier
  DPO rerun, whereas this example generates them from its own post-DPO checkpoint.
- The companion run evaluates the same composite DPO equation through a recipe-specific
  server-side named loss. The public client draft uses `forward_backward_custom` and remains
  blocked until TuFT supports and validates that path.

They are therefore companion-run measurements, not byte-for-byte `run_recipe.py` results.

## Companion-run results

| Stage | Available | Seen by optimizer | Updates | First → last minibatch loss | Wall time |
|---|---:|---:|---:|---:|---:|
| Filtered DPO | 1,826 pairs | 1,824 pairs | 57 | 0.9204 → 0.2176 | 52m 38s |
| Introspection SFT | 1,500 transcripts | 1,472 transcripts | 46 | 1.4276 → 1.1384 | 37m 45s |

The two training stages took 5,422.3 seconds (1 hour 30 minutes 22 seconds) in total. Their loss
values use different objectives and should not be compared to one another. DPO's final-minibatch
accuracy of 1.000 is also not a validation score.

The five exact-match-held-out prompt triplets show the intended qualitative movement without a
persona prompt. The base answers are neutral and four reach the 320-token cap; post-DPO answers
introduce irony on four prompts but remain neutral on the meeting prompt; the final adapter uses
overt sarcastic framing on all five. This five-prompt inspection is illustrative, not a
character-strength, capability, or safety evaluation. The unedited text and cap flags are in
[`sample_outputs.md`](./sample_outputs.md).

## Sources and responsible use

- Maiya, Bartsch, Lambert, and Hubinger, [Open Character Training](https://arxiv.org/abs/2511.01689).
- [Official Open Character Training code and data](https://github.com/maiush/OpenCharacterTraining).
- The exact upstream commit and generated-cache metadata are recorded in `data/PROVENANCE.md`.
- Companion-run measurements and responses are recorded in `results/companion_run.json`.

Character training can alter behavioral tendencies beyond surface style. Inspect held-out
behavior, capability, and safety before deployment. A sarcastic assistant can be entertaining
in casual contexts and harmful in support, medical, crisis, or accessibility contexts; route or
disable the adapter where its character is inappropriate.
