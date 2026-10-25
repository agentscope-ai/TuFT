"""Storage and numerical invariants for frozen base-weight sharing."""

import copy

import pytest
import torch
from peft import LoraConfig, get_peft_model
from transformers import Qwen3Config, Qwen3ForCausalLM

from tuft.backends.shared_weights import (
    bind_shared_weights,
    check_shared_weight_versions,
    export_cuda_weights,
    pack_hf_base_weights,
    tensor_fingerprint,
)


def tiny_model():
    config = Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        tie_word_embeddings=False,
        attention_dropout=0,
    )
    model = Qwen3ForCausalLM(config).to(torch.bfloat16)
    return get_peft_model(model, LoraConfig(r=2, target_modules=["q_proj", "v_proj"]))


def test_validation_fingerprint_handles_scalar_and_bfloat16_buffers():
    scalar = torch.tensor(1.0, dtype=torch.bfloat16)
    before = tensor_fingerprint(scalar)
    assert before == tensor_fingerprint(scalar.reshape(1))
    scalar.add_(1)
    assert tensor_fingerprint(scalar) != before


def test_packing_preserves_forward_gradients_and_parameter_identity():
    torch.manual_seed(7)
    model = tiny_model().eval()
    with torch.no_grad():
        for name, param in model.named_parameters():
            if ".lora_B." in name:
                param.normal_(std=0.01)
    reference = copy.deepcopy(model)
    identities = {name: id(p) for name, p in model.named_parameters()}
    weights = pack_hf_base_weights(model.get_base_model())
    assert {name: id(p) for name, p in model.named_parameters()} == identities
    attention = model.get_base_model().model.layers[0].self_attn
    slab = weights["model.layers.0.self_attn.qkv_proj.weight"]
    assert attention.q_proj.base_layer.weight.untyped_storage().data_ptr() == slab.data_ptr()
    assert attention.v_proj.base_layer.weight.untyped_storage().data_ptr() == slab.data_ptr()
    tokens = torch.tensor([[1, 4, 7, 9]])
    left = model(tokens, labels=tokens)
    right = reference(tokens, labels=tokens)
    torch.testing.assert_close(left.logits, right.logits, rtol=0, atol=0)
    left.loss.backward()
    right.loss.backward()
    for (name, param), (_, expected) in zip(
        model.named_parameters(), reference.named_parameters(), strict=True
    ):
        if ".lora_" in name:
            torch.testing.assert_close(param.grad, expected.grad, rtol=0, atol=0)
            assert param.grad is not None and param.grad.count_nonzero() > 0
        else:
            assert param.grad is None

    # Later tenants may have another rank or add a target module. PEFT must
    # preserve the producer's shared base storage through those transitions.
    def base_pointers():
        return {
            name.replace(".base_layer.", "."): p.data_ptr()
            for name, p in model.get_base_model().named_parameters(remove_duplicate=False)
            if ".lora_" not in name
        }

    pointers = base_pointers()
    model.add_adapter("tenant_2", LoraConfig(r=4, target_modules=["q_proj", "v_proj", "down_proj"]))
    model.set_adapter("tenant_2")
    assert base_pointers() == pointers
    model.set_adapter("default")
    assert base_pointers() == pointers
    torch.testing.assert_close(model(tokens).logits, left.logits, rtol=0, atol=0)


def test_binding_aliases_storage_and_only_copies_vocabulary_padding():
    class Consumer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(4, 6, bias=False, dtype=torch.bfloat16)
            self.lm_head = torch.nn.Linear(4, 8, bias=False, dtype=torch.bfloat16)

    model = Consumer()
    weights = {
        "proj.weight": torch.randn(6, 4, dtype=torch.bfloat16),
        "lm_head.weight": torch.randn(7, 4, dtype=torch.bfloat16),
    }
    report = bind_shared_weights(model, weights)
    weights["proj.weight"][0, 0] = 17
    assert model.proj.weight[0, 0] == 17
    torch.testing.assert_close(model.lm_head.weight[:7], weights["lm_head.weight"])
    assert model.lm_head.weight[7].count_nonzero() == 0
    assert report["shared_parameter_bytes"] == 48
    assert report["copied_parameter_bytes"] == 64


def test_layout_failure_does_not_partially_rebind():
    model = torch.nn.Sequential(torch.nn.Linear(4, 6, bias=False, dtype=torch.bfloat16))
    before = model[0].weight.data_ptr()
    with pytest.raises(ValueError, match="shape"):
        bind_shared_weights(model, {"0.weight": torch.ones(5, 4, dtype=torch.bfloat16)})
    assert model[0].weight.data_ptr() == before
    with pytest.raises(ValueError, match="keys"):
        bind_shared_weights(model, {})
    with pytest.raises(ValueError, match="dtype"):
        bind_shared_weights(model, {"0.weight": torch.ones(6, 4)})


def test_binding_resolves_duplicate_module_paths_without_ignoring_missing_weights():
    model = torch.nn.Module()
    model.router = torch.nn.Linear(4, 6, bias=False, dtype=torch.bfloat16)
    model.runner = torch.nn.Module()
    model.runner.gate = model.router
    weight = torch.randn(6, 4, dtype=torch.bfloat16)
    report = bind_shared_weights(model, {"router.weight": weight})
    assert model.runner.gate.weight.data_ptr() == weight.data_ptr()
    assert report["shared_storage_bytes"] == weight.numel() * weight.element_size()
    model.runner.other = torch.nn.Linear(4, 6, bias=False, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="missing"):
        bind_shared_weights(model, {"router.weight": weight})


def test_rejects_unfrozen_base_unknown_model_and_unsupported_versions():
    model = tiny_model().get_base_model()
    model.model.embed_tokens.weight.requires_grad_(True)
    with pytest.raises(ValueError, match="frozen"):
        pack_hf_base_weights(model)
    model.config.model_type = "unknown"
    with pytest.raises(ValueError, match="Qwen3"):
        pack_hf_base_weights(model)
    with pytest.raises(ValueError, match="CUDA"):
        export_cuda_weights({"weight": torch.ones(1)})
    check_shared_weight_versions("2.11.0+cu130", "0.24.0")
    with pytest.raises(RuntimeError, match="tested"):
        check_shared_weight_versions("2.12.0", "0.24.0")
    with pytest.raises(RuntimeError, match="tested"):
        check_shared_weight_versions("2.11.0", "0.24.0.dev1")
