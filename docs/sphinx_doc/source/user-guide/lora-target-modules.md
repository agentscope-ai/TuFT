# LoRA Target Modules

How TuFT turns the client LoRA flags (`train_attn`, `train_mlp`,
`train_unembed`) into concrete module names, and the rules it enforces around
them.

## How resolution works

TuFT reads the model's `config.json` to find its model type and maps each flag
to module names. For Qwen and Llama models, `train_attn` covers
`q_proj`/`k_proj`/`v_proj`/`o_proj` and `train_mlp` covers
`gate_proj`/`up_proj`/`down_proj`; on Llama, `train_unembed` covers `lm_head`.
When no local `config.json` exists yet (a Hugging Face model ID that has not
been downloaded), the model name decides.

On Qwen3.5-based models (model type `qwen3_5`, which Qwen3.6 and Qwen3.8 also
use), three of every four text layers use Gated DeltaNet instead of full
attention, so `train_attn` additionally covers `linear_attn.in_proj_qkv`,
`linear_attn.in_proj_z`, and `linear_attn.out_proj`.

## Breaking change: Qwen3.5-based models

The fix for issue #149 added the Tinker-compatible `linear_attn.in_proj_qkv`,
`linear_attn.in_proj_z`, and `linear_attn.out_proj` modules to the LoRA target
list for Qwen3.5-based models. Checkpoints and training runs from before this
change use the old, shorter list, and the two lists must match exactly. After
upgrading:

- Old checkpoints for these models cannot be loaded.
- Old FSDP training runs are marked corrupted when the server restarts.
- A server config that sets `fsdp_target_modules` to the old list stops the
  server at startup. The error says how to fix the config.
- With persistence enabled, the new `qwen_gated_deltanet_full_lora` model
  field changes the stored configuration signature, so startup fails with a
  configuration-mismatch error until you switch to a new namespace or clear
  the old one (see
  [Changing config safely](persistence.md#changing-config-safely)).

Each error message names this change as the cause. To continue training,
create a new training run.

## Full Gated DeltaNet coverage (opt-in)

Set `qwen_gated_deltanet_full_lora: true` on a model to additionally target
`linear_attn.in_proj_a` and `linear_attn.in_proj_b`. This operator opt-in
applies to both HF and FSDP training and creates a different checkpoint
geometry from the default, so changing it also requires a new training run
(and an FSDP restart). The default remains compatible with Tinker's public
`train_attn` behavior.

## MoE models

On the MoE variant (model type `qwen3_5_moe`, e.g. Qwen3.6-35B-A3B), the same
target list applies, but the routed experts are fused parameters without
per-expert `gate_proj`/`up_proj`/`down_proj` modules. `train_mlp` therefore
trains only the shared expert, and the routed experts stay frozen. vLLM
applies LoRA to the same modules when serving, so trained and served modules
match.

## Every target must match a real module

TuFT requires every resolved target module name to match at least one real
module in the loaded model. The HF backend checks this when an adapter is
created; the FSDP backend checks its slot list when the worker starts. A
model whose `config.json` does not match its real architecture is rejected
with the unmatched names, instead of silently training only part of the
intended modules.

## train_unembed on Qwen models

For the same reason, `train_unembed=True` is rejected on Qwen-family models:
TuFT has no Qwen unembed modules to train, while Tinker's hosted service
adapts `embed_tokens` there, so silently accepting the flag would hide a real
behavior difference. The SDK defaults the flag to true, so pass
`train_unembed=False` explicitly when training Qwen models.
