# Open Character Training

Open Character Training teaches an assistant a persistent character without supplying that
character in its inference-time system prompt. The constitution is a data-generation tool: a
teacher sees it, a student learns its behavioral preferences, and the student then generates
introspective SFT data that reinforces the character in its own words.

This guide follows the three-stage method in Maiya et al.'s
[Open Character Training paper](https://arxiv.org/abs/2511.01689) and
[reference repository](https://github.com/maiush/OpenCharacterTraining): a hand-written
constitution, DPO preference distillation, and introspection. The runnable TuFT example trains a
sarcastic rank-64 LoRA on Qwen3.5-4B using a four-GPU FSDP TuFT server.

```{admonition} Result provenance
:class: note
The measurements and response examples come from a completed Qwen3.5-4B companion run. It
matches the public recipe's adapter geometry, data design, and schedules, with the two code-path
differences described under [Rank and result provenance](#rank-and-result-provenance).
```

## Why use two training stages?

The stages solve different problems:

1. **DPO distillation gives an external direction.** A stronger teacher receives a constitution
   and produces preferred answers. The unmodified student produces rejected answers without a
   persona prompt. DPO moves the LoRA toward the teacher's behavior.
2. **Introspection makes the direction self-reinforcing.** The post-DPO model reflects on its
   identity and talks to copies of itself. The constitution prompt used to elicit these examples
   is removed or replaced before SFT, so the final training examples do not merely teach the
   model to obey that prompt.

This distinction matters. Ordinary persona SFT can teach stylistic phrases, while the
introspection stage supplies longer examples about goals, identity, and consistent interaction.
Whether a trained model generalizes that behavior safely remains an empirical question and must
be evaluated.

## Recipe at a glance

- A human writes a ten-assertion constitution for the teacher-generation prompt.
- Qwen3.7-Max sees the constitution and user prompt to produce each chosen DPO response.
- Base Qwen3.5-4B sees only the user prompt to produce each rejected DPO response.
- The post-DPO LoRA sees the constitution and reflection prompts; SFT omits the constitution.
- Two post-DPO copies generate dialogues; SFT retains only their copy-to-copy setup.
- Base, post-DPO, and final deployment samples receive the user prompt only.

The example vendors the Qwen3.7-Max chosen-response cache. Users without that model's API can run
every student sampling and training step through their TuFT server.

## Hardware requirement

The documented configuration requires **four NVIDIA CUDA GPUs with at least 40 GB of VRAM each**.
Two standalone vLLM replicas handle sampling and two FSDP workers handle training. Ray reserves a
whole GPU for each actor, so this layout does not colocate sampling and training. It uses
`micro_batch_size: 2` and an 8,192-token server context.

The measured Qwen3.5-4B companion run used four A100-80GB GPUs. Other CUDA accelerators can satisfy
the configuration, but the 40 GB floor and other accelerator families have not yet been validated
end to end. A three-GPU variant can use one vLLM replica, but it will process the 7,258 student
generation requests more slowly and is outside the documented result.

DPO simultaneously uses a policy and a frozen reference adapter on the FSDP model.
`lora_alpha_ratio: 2` gives alpha 128. Attention and MLP modifiers adapt all supported Qwen3.5
projections.

### Why FSDP?

The measured rank-64 HF DPO stage took 2 hours 28 minutes on one A100-80GB, and the subsequent
3,072-token SFT stage exhausted that GPU's memory. Two FSDP workers make the long-sequence stage
fit and match the backend used by the companion result. Two vLLM replicas reduce the elapsed time
of preference and introspection generation.

FSDP does not reduce the adapter to Q/V projections. Current TuFT honors the attention and MLP
modifiers, yielding Qwen3.5 targets `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`,
and `down_proj`. The client-defined objective is unchanged by the backend choice.

### No local GPU

The client scripts do not need a GPU; they can drive a remote TuFT server over HTTP. The example's
`config.yaml` includes infrastructure sections for both supported deployment helpers:

- [Modal](../deployment/modal.md) runs a serverless GPU container, stores checkpoints on a
  persistent Volume, and can scale to zero when idle.
- [Lambda Cloud](../deployment/lambda.md) provisions a dedicated GPU VM reached through an SSH
  tunnel. It bills until the VM is terminated, so preserve the checkpoints and tear it down after
  the run.

Both launchers strip their infrastructure section before TuFT reads the standard server config.
The example README contains the exact commands and explains how to point `TINKER_BASE_URL` at the
resulting server.

```bash
# Serverless, scale-to-zero:
python deploy/modal/launch.py \
  --config examples/open_character_training/config.yaml \
  --foreground

# Dedicated on-demand VM:
export LAMBDA_API_KEY=...
python deploy/lambda/launch.py \
  --config examples/open_character_training/config.yaml \
  --instance-type YOUR_4X_GPU_INSTANCE_TYPE
```

The Lambda launcher otherwise auto-selects a one-GPU instance, which is insufficient for this
configuration. Choose an available four-GPU instance type in the Lambda console before launch.

### Approximate cloud spend

Budget **4–6 hours** for a clean end-to-end run. The completed companion run measured 52 minutes
38 seconds for DPO and 37 minutes 45 seconds for SFT. The remaining 2–4 hours allow for 7,258
student generations, model startup, checkpointing, and final sampling. Because the companion
reused frozen generation artifacts, it does not provide a clean end-to-end generation time.

At prices checked on 2026-08-17, the GPU-only calculation is:

- [Modal](https://modal.com/pricing): `$0.000583/GPU-second` (`$2.10/GPU-hour`), or
  `$8.40/hour` for four GPUs and approximately **$34–$50** for 4–6 hours.
- [Lambda Cloud](https://lambda.ai/instances): `$1.99/GPU-hour`, or `$7.96/hour` for four GPUs
  and approximately **$32–$48** for 4–6 hours.

Modal also meters CPU, memory, Volume storage, and optional execution features; those charges are
not included. Lambda cost assumes a four-GPU A100 40 GB instance is available and excludes
boot/setup/idle time and tax. A Lambda VM continues billing until termination, whereas the Modal
deployment can scale to zero. Verify both price and capacity immediately before launch.

## Data provenance and the cached teacher

The frozen prompt plan combines 499 sarcasm-relevant prompts from the reference repository with
1,030 single-turn LIMA training prompts, each repeated twice. That yields 3,058 planned pairs
before response and length filters.

The bundled `qwen3.7-max-sarcastic-chosen.json.gz` file contains the chosen responses. A stable key
derived from persona, source, prompt index, and repeat joins each cached answer to the prompt plan.
`generate_data.py chosen` verifies coverage without calling an API.
Regeneration requires an explicit `--refresh-teacher` flag and writes to the work directory,
leaving the bundled artifact unchanged.

### Regenerate with the OpenAI client

Teacher regeneration uses the
[official OpenAI Python SDK](https://developers.openai.com/api/docs/libraries) and its Chat
Completions client. Set the SDK's standard `OPENAI_API_KEY` variable and the example's
`OPENAI_MODEL` selection, leave `OPENAI_BASE_URL` unset for the default OpenAI API, and run the
data command with `--refresh-teacher`.

An OpenAI-compatible provider uses the same code path. Set `OPENAI_BASE_URL` to that provider's
API root and `OPENAI_MODEL` to its model identifier. No provider-specific SDK is required. The
example sends only standard Chat Completions fields by default; provider-specific fields can be
supplied as a JSON object with `OPENAI_EXTRA_BODY_JSON` or `--teacher-extra-body-json`.

The bundled cache remains Qwen3.7-Max data regardless of the live-provider configuration. A live
refresh writes a separate cache and records its model, base URL, and optional extra request body
in metadata, so results from different teachers are not mislabeled.

The default OpenAI endpoint uses the current `max_completion_tokens` Chat Completions field. A
custom `OPENAI_BASE_URL` defaults to the older `max_tokens` spelling for broader provider
compatibility. `OPENAI_MAX_TOKENS_FIELD` can explicitly select either spelling, and the selected
field is recorded in working-cache metadata.

Rejected responses are not cached in Git: they must come from the exact base model served by the
run. Likewise, reflections and self-interactions must come from that run's post-DPO checkpoint.
This division avoids quietly substituting generations from a different student or adapter.

## The DPO objective

The reference implementation adds chosen-response NLL and a squared log-ratio stability term to
the ordinary pairwise DPO loss:

```text
L = L_DPO(beta = 0.1)
  + 0.1   × mean chosen response NLL
  + 0.001 × mean response-token (log π - log π_ref)²
```

The example obtains reference log probabilities from a frozen zero-init LoRA client in the same
TuFT training backend. It does not use sampler log probabilities as the reference, because vLLM
and training kernels can have small numerical differences.

The compatible SDK's `forward_backward_custom` method makes this objective client-defined. It
requests token log probabilities from TuFT, differentiates the scalar custom loss locally, and
sends the resulting token weights for the server-side backward pass. No recipe-specific TuFT
source change is needed.

## Introspection details

The reduced example plans 1,500 transcripts:

- 1,200 self-reflections: 120 samples for each of ten prompts;
- 150 free self-interactions;
- 150 leading self-interactions that explicitly invite introspection.

Each interaction runs for ten turns with two copies swapping conversational roles. The SFT loss
uses only the final assistant message in each rendered example. If a dialogue exceeds 3,072
tokens, the data builder drops whole early turns so the supervised target remains intact.

The final SFT client restores the DPO optimizer/model state and continues training the same LoRA.
This sequential composition is simpler than the reference repository's separately trained,
linearly merged adapters and is documented as a deliberate implementation difference.

## Run the example

The full source, configuration, commands, and artifact descriptions are in
[`examples/open_character_training`](https://github.com/agentscope-ai/TuFT/tree/main/examples/open_character_training).
After starting TuFT with that directory's `config.yaml`, run:

```bash
export TINKER_BASE_URL=http://localhost:10610
export TINKER_API_KEY=tml-...

uv run python examples/open_character_training/run_recipe.py \
  --work-dir /path/on/durable/storage/open-character-training
```

The one-command runner performs cached-teacher distillation data generation, DPO, introspection
generation, SFT, and held-out sampling. Every long generation stage is cached. Completed training
stages have local records containing their TuFT state and sampler paths.

The `TINKER_BASE_URL` and `TINKER_API_KEY` environment-variable names come from the compatible
client SDK. They point directly to TuFT; this recipe neither contacts Tinker's hosted service nor
requires a Tinker account.

For learning or debugging, run each stage separately and inspect its intermediate JSON/JSONL
artifacts. The example README explains every command and the composite loss implementation.

## What to evaluate

Do not judge character training from training loss alone. Compare base, post-DPO, and final LoRA
on the same held-out prompts and seeds, then inspect at least:

- character strength without a persona system prompt;
- whether DPO or introspection contributes the larger behavioral change;
- response quality, factuality, refusal behavior, and instruction following;
- character leakage into contexts where the behavior is inappropriate;
- robustness to requests to drop the persona or use a neutral style.

The bundled `sample.py` records raw base/DPO/final generations in JSON before they are selected
for documentation.

## Rank and result provenance

The reference recipe and this example both use rank 64 with alpha 128. The example still reduces
the number of distillation repeats and introspection transcripts and continues SFT from the DPO
adapter instead of training and linearly merging a separate adapter.

The measurements and response examples come from a Qwen3.5-4B sarcastic companion run. It
matches the rank, alpha, target modules, FSDP worker count, optimizer schedule, and corpus
design, but it is not a literal `run_recipe.py` execution. Its introspection corpus was
generated before the final full-modifier DPO rerun, and it evaluates the same DPO equation through
a recipe-specific server-side named loss. The runnable example instead regenerates introspection
from its own post-DPO adapter and uses the compatible SDK's public `forward_backward_custom` API,
requiring no TuFT source patch. These are companion-run measurements rather than exact code-path
parity.

## Results and examples

- **DPO distillation:** 1,826 pairs available, 1,824 seen by the optimizer in 57 updates;
  first-to-last minibatch loss 0.9204 → 0.2176; wall time 52 minutes 38 seconds.
- **Introspection SFT:** 1,500 transcripts available, 1,472 seen by the optimizer in 46 updates;
  first-to-last minibatch loss 1.4276 → 1.1384; wall time 37 minutes 45 seconds.

The two training stages took 5,422.3 seconds (1 hour 30 minutes 22 seconds). Only complete
32-example minibatches were used, which accounts for the difference between available and seen
examples. DPO's composite loss and SFT's response-token mean NLL are different quantities. The
first/last values are stochastic training-minibatch diagnostics, not held-out scores. DPO
preference accuracy was 0.000 on the initial policy/reference tie and 1.000 on the final
minibatch; that does not establish generalization.

Five prompts were checked for exact matches against both frozen training snapshots, then sampled
from the base, post-DPO, and final adapters with the same seed per prompt and no persona system
prompt. Post-DPO responses use overt irony on four prompts but remain neutral on the meeting
prompt. The final adapter uses overt sarcastic framing on all five. This small qualitative check
shows that the intended behavior is present; it does not measure character strength, capability,
or safety.

The meeting prompt makes the second-stage movement visible. Post-DPO, the adapter answered:

> If you’re stuck in a meeting that could have been an email, ask yourself:
> - Who is the audience?
> - What is the action?
> - What is the deadline?
> - Is anyone going to be surprised if you email instead?
>
> If the answer is “yes” to any of those, **email it**.

After introspection SFT, the same prompt and seed produced:

> Oh, absolutely. Because nothing says “efficient communication” quite like scheduling a video
> conference when the entire conversation could be condensed into a single sentence with a comma.
>
> You know what to do: just send an email. It’s the most revolutionary, forward-thinking, and
> *urgent* way to share information. And if you’re feeling particularly generous, you can even CC
> your entire organization so everyone feels involved in your decision to avoid talking to each
> other.
>
> Or, if you want to really impress everyone with your commitment to “collaborative
> problem-solving,” invite three people to a Zoom call where each of them spends 45 seconds saying
> “I agree” while staring at a screen that’s clearly too small to see anything anyway.
>
> But hey, at least now you’ve proven you’re not just a human, but a *human* who understands that
> sometimes, the best way to communicate is to pretend you can’t type.

The example's
[`sample_outputs.md`](https://github.com/agentscope-ai/TuFT/blob/main/examples/open_character_training/sample_outputs.md)
contains every unedited response and marks generations that reached the token cap. The companion
metrics, seeds, token counts, hashes, and provenance are stored in
[`results/companion_run.json`](https://github.com/agentscope-ai/TuFT/blob/main/examples/open_character_training/results/companion_run.json).

## Responsible use

A learned character can affect decisions and tone beyond the intended surface style. Evaluate it
as a behavioral model change, not a cosmetic theme. Sarcasm may be suitable for entertainment and
unacceptable in support, health, crisis, or accessibility settings. Gate the adapter by product
context, and retain a neutral fallback.
