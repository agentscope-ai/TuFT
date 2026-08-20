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

## Full Gated DeltaNet coverage (opt-in)

Set `qwen_gated_deltanet_full_lora: true` on a model to additionally target
`linear_attn.in_proj_a` and `linear_attn.in_proj_b`. This operator opt-in
applies to both HF and FSDP training and creates a different checkpoint
geometry from the default, so changing it also requires a new training run
(and an FSDP restart). The default remains compatible with Tinker's public
`train_attn` behavior.

## MoE models

On the MoE variant (model type `qwen3_5_moe`, e.g. Qwen3.6-35B-A3B), the same
module list applies and covers the shared expert. The routed experts are
fused 3D parameters without per-expert `gate_proj`/`up_proj`/`down_proj`
modules, so `train_mlp` additionally targets `mlp.experts.gate_up_proj` and
`mlp.experts.down_proj` through peft `target_parameters`. This matches
Tinker's documented behavior of `train_mlp` covering MoE layers: attention,
the Gated DeltaNet projections, the shared expert, and every routed expert
all train. vLLM parses the peft-format expert adapter keys when serving, so
trained and served targets match.

The module list and the parameter list together define a checkpoint's
geometry; both are recorded in the training run, checkpoint metadata, and
the peft `adapter_config.json`, and both must match exactly to load a
checkpoint. On FSDP, `fsdp_target_parameters` explicitly overrides the
parameter side the way `fsdp_target_modules` overrides the module side.

## Every target must match a real module

TuFT requires every resolved target module name to match at least one real
module in the loaded model. The HF backend checks this when an adapter is
created; the FSDP backend checks its slot list when the worker starts. A
model whose `config.json` does not match its real architecture is rejected
with the unmatched names, instead of silently training only part of the
intended modules.

## train_unembed on Qwen models

TuFT currently accepts `train_unembed=True` on Qwen-family models but does
not add an embedding or unembedding target. Support for training and serving
`embed_tokens`, matching Tinker's hosted behavior, is tracked in
[issue #153](https://github.com/agentscope-ai/TuFT/issues/153) and depends on
a released vLLM version containing its Qwen3.5 embedding-module support.
