# Open Character Training

Open Character Training teaches an assistant a persistent character without supplying that
character in its inference-time system prompt. The constitution is a data-generation tool: a
teacher sees it, a student learns its behavioral preferences, and the student then generates
introspective SFT data that reinforces the character in its own words.

This guide follows the three-stage method in Maiya et al.'s
[Open Character Training paper](https://arxiv.org/abs/2511.01689) and
[reference repository](https://github.com/maiush/OpenCharacterTraining): a hand-written
constitution, DPO preference distillation, and introspection. The runnable TuFT example trains a
sarcastic rank-16 LoRA on Qwen3.5-4B using one machine with two A100-80GB GPUs.

```{admonition} Draft results
:class: warning
The recipe code and cached teacher data are complete, but the documented rank-16 run will start
after the current TuFT server workload finishes. Result tables and response examples are marked
pending until they can be populated from recorded artifacts.
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

| Stage | Producer | Conditioning | Consumer |
|---|---|---|---|
| Constitution | Human author | Ten first-person assertions | Teacher generation prompt |
| Chosen responses | Qwen3.7-Max | Constitution + user prompt | DPO preference pair |
| Rejected responses | Base Qwen3.5-4B | User prompt only | DPO preference pair |
| Self-reflections | Post-DPO LoRA | Constitution + reflection prompt | SFT transcript without constitution |
| Self-interactions | Two post-DPO copies | Constitution + copy-to-copy setup | SFT dialogue with only copy-to-copy setup |
| Deployment samples | Final LoRA | User prompt only | Behavior comparison |

The example vendors the Qwen3.7-Max chosen-response cache. Users without that model's API can run
every student sampling and training step locally through TuFT.

## TuFT topology on 2× A100

The provided server configuration assigns one GPU to vLLM sampling and one GPU to a single-rank
FSDP training backend:

```text
A100:0  vLLM sampler  ── base rejection, introspection, evaluation
A100:1  FSDP worker   ── reference/policy forward, backward, optimizer
```

Set `fsdp_num_gpus: 1`. A two-rank FSDP backend plus the standalone sampler would consume three
GPUs. `colocate: false` keeps the long generation stages from competing with training for the
same A100 memory.

The server preallocates four rank-16 adapter slots. DPO simultaneously needs a policy and a
frozen reference; extra slots leave room for stage transitions and inspection. `lora_alpha_ratio:
2` gives alpha 32. Attention and MLP modifiers adapt all supported Qwen3.5 projections.

## Data provenance and the cached teacher

The frozen prompt plan combines 499 sarcasm-relevant prompts from the reference repository with
1,030 single-turn LIMA training prompts, each repeated twice. That yields 3,058 planned pairs
before response and length filters.

`examples/open_character_training/data/qwen3.7-max-sarcastic-chosen.json.gz` contains the chosen
responses. A stable key derived from persona, source, prompt index, and repeat joins each cached
answer to the prompt plan. `generate_data.py chosen` verifies coverage without calling an API.
Regeneration requires an explicit `--refresh-teacher` flag and writes to the work directory,
leaving the bundled artifact unchanged.

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

Tinker's `forward_backward_custom` makes this objective client-defined. The SDK requests token
log probabilities from TuFT, differentiates the scalar custom loss locally, and sends the
resulting token weights for the server-side backward pass. No recipe-specific TuFT source change
is needed.

## Introspection details

The reduced two-GPU case study plans 1,500 transcripts:

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

## Rank 16 versus the paper setting

The reference recipe uses rank 64 with alpha 128. This example defaults to rank 16 with alpha 32
to reduce resource use and make iteration easier on the two-A100 setup. It is a recipe adaptation,
not an exact paper replication. Rank alone does not determine quality; the final comparison must
report the measured behavior of this specific run.

To restore the paper's LoRA size, change the client rank and the TuFT server's `max_lora_rank` and
`fsdp_rank_slots` to 64, then restart the server and train from a new state. Keep the alpha ratio
at 2. Checkpoints cannot be moved between incompatible ranks or target geometries.

## Results and examples

| Stage | Examples | Updates | Loss | Wall time |
|---|---:|---:|---:|---:|
| DPO distillation | Pending | Pending | Pending | Pending |
| Introspection SFT | Pending | Pending | Pending | Pending |

<!-- TODO(final-results): Populate from the rank-16 TuFT run records. -->
<!-- TODO(final-results): Add three or more unedited base/post-DPO/final response triplets. -->

See the example's `sample_outputs.md` for the response-comparison template. The final guide will
use only outputs traceable to `work/sample_outputs.json`.

## Responsible use

A learned character can affect decisions and tone beyond the intended surface style. Evaluate it
as a behavioral model change, not a cosmetic theme. Sarcasm may be suitable for entertainment and
unacceptable in support, health, crisis, or accessibility settings. Gate the adapter by product
context, and retain a neutral fallback.
