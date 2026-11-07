"""Real hybrid dense/MoE models: layouts must preserve training semantics."""

import copy
from typing import Any

import pytest
import torch
from peft import LoraConfig, get_peft_model
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeForCausalLM

from tuft.backends.fsdp_shared_weights import prepare_fsdp_shared_layout, shared_fsdp_placement
from tuft.backends.shared_weight_layout import canonical_name
from tuft.backends.shared_weights import pack_hf_base_weights


def hybrid_model(moe=False, adapters=True, linear_head_dim=8):
    args: dict[str, Any] = dict(
        vocab_size=128,
        hidden_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        linear_key_head_dim=linear_head_dim,
        linear_value_head_dim=linear_head_dim,
        linear_num_key_heads=2,
        linear_num_value_heads=6,
        layer_types=["linear_attention"] * 3 + ["full_attention"],
        attn_implementation="eager",
    )
    model: Any
    if moe:
        config = Qwen3_5MoeTextConfig(
            **args,
            num_experts=4,
            num_experts_per_tok=2,
            moe_intermediate_size=32,
            shared_expert_intermediate_size=32,
        )
        model = Qwen3_5MoeForCausalLM(config)
    else:
        config = Qwen3_5TextConfig(**args, intermediate_size=64)
        model = Qwen3_5ForCausalLM(config)
    if not adapters:
        return model.to(torch.bfloat16)
    targets = [
        "q_proj",
        "v_proj",
        "in_proj_qkv",
        "in_proj_z",
        "out_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    params = ["mlp.experts.gate_up_proj", "mlp.experts.down_proj"] if moe else None
    model = get_peft_model(
        model.to(torch.bfloat16), LoraConfig(r=2, target_modules=targets, target_parameters=params)
    )
    with torch.no_grad():
        for name, p in model.named_parameters():
            if ".lora_B." in name:
                p.normal_(std=0.01)
    return model


@pytest.mark.parametrize("moe", [False, True])
@pytest.mark.parametrize("fsdp_layout", [False, True])
def test_hybrid_layout_preserves_logits_gradients_and_expert_adapters(moe, fsdp_layout):
    torch.manual_seed(42)
    model = hybrid_model(moe)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable({"use_reentrant": False})
    reference = copy.deepcopy(model)
    original = {
        canonical_name(n): p.detach().clone()
        for n, p in model.named_parameters()
        if ".lora_" not in n
    }
    if fsdp_layout:
        prepare_fsdp_shared_layout(model, 2)
        placements, ignored = shared_fsdp_placement(model, 2)
        for name, p in model.named_parameters():
            name = canonical_name(name)
            if ".lora_" in name:
                continue
            if name.endswith(".experts.gate_up_proj"):
                assert placements[p].dim == 1
            if name.endswith(".experts.down_proj"):
                assert placements[p].dim == 2
            if name.endswith(".linear_attn.A_log"):
                assert placements[p].dim == 0 and p not in ignored
                assert p.dtype == torch.float32
        # Every rank's local fused QKV shard contains Q, K AND V segments.
        linear = model.get_base_model().model.layers[0].linear_attn
        stored = linear.in_proj_qkv.base_layer.parametrizations.weight.original
        logical = original["base_model.model.model.layers.0.linear_attn.in_proj_qkv.weight"]
        for rank in range(2):
            expected = torch.cat([part.chunk(2)[rank] for part in logical.split([16, 16, 48])])
            torch.testing.assert_close(stored.chunk(2)[rank], expected, rtol=0, atol=0)
    else:
        identities = {n: id(p) for n, p in model.named_parameters()}
        weights = pack_hf_base_weights(model.get_base_model())
        assert identities == {n: id(p) for n, p in model.named_parameters()}
        assert "model.layers.0.linear_attn.in_proj_qkvz.weight" in weights
        assert weights["model.layers.0.linear_attn.A_log"].dtype == torch.float32
        if moe:
            assert "model.layers.0.mlp.experts.routed_experts.w13_weight" in weights
            assert "model.layers.0.mlp.experts.routed_experts.w2_weight" in weights
    tokens = torch.tensor([[1, 3, 5, 7]])
    outputs = [m(tokens, use_cache=False).logits for m in (model, reference)]
    torch.testing.assert_close(*outputs, rtol=0, atol=0)
    for output in outputs:
        output.float().square().mean().backward()
    grads = [
        {n: p.grad for n, p in m.named_parameters() if ".lora_" in n} for m in (model, reference)
    ]
    assert grads[0].keys() == grads[1].keys()
    for name in grads[0]:
        torch.testing.assert_close(grads[0][name], grads[1][name], rtol=0, atol=0)
    assert any(g is not None and g.count_nonzero() for g in grads[0].values())
    # PEFT's temporary expert parametrization must leave our layout installed.
    if fsdp_layout and moe:
        keys = [n for n, _ in model.named_parameters()]
        assert any("parametrizations.gate_up_proj.original" in n for n in keys)
