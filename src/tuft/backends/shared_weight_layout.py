"""Explicit Qwen dense/hybrid/MoE storage layouts shared by HF, FSDP and vLLM."""

from __future__ import annotations


def text_config(config):
    config = config.get_text_config() if hasattr(config, "get_text_config") else config
    if config.model_type not in {"qwen3", "qwen3_5_text", "qwen3_5_moe_text"}:
        raise ValueError("Shared weights support Qwen3 and Qwen3.5-based text models only")
    return config


def canonical_name(name: str) -> str:
    while ".base_layer." in name:
        name = name.replace(".base_layer.", ".")
    for parameter in ("weight", "gate_up_proj"):
        name = name.replace(f".parametrizations.{parameter}.original", f".{parameter}")
    return name


def serving_name(name: str) -> str:
    return name.replace(".experts.gate_up_proj", ".experts.routed_experts.w13_weight").replace(
        ".experts.down_proj", ".experts.routed_experts.w2_weight"
    )


def fused_groups(config):
    config = text_config(config)
    for index in range(config.num_hidden_layers):
        layer = f"model.layers.{index}"
        if config.model_type == "qwen3" or config.layer_types[index] == "full_attention":
            yield f"{layer}.self_attn", ("q_proj", "k_proj", "v_proj"), "qkv_proj"
        elif config.layer_types[index] == "linear_attention":
            yield f"{layer}.linear_attn", ("in_proj_qkv", "in_proj_z"), "in_proj_qkvz"
            yield f"{layer}.linear_attn", ("in_proj_b", "in_proj_a"), "in_proj_ba"
        else:
            raise ValueError(f"Unsupported shared attention layer: {config.layer_types[index]}")
        mlp = "mlp.shared_expert" if config.model_type == "qwen3_5_moe_text" else "mlp"
        yield f"{layer}.{mlp}", ("gate_proj", "up_proj"), "gate_up_proj"


def is_recurrent_scale(name: str) -> bool:
    return name.endswith(".linear_attn.A_log")
