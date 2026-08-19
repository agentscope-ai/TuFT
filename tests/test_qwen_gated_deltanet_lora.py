"""Architecture coverage for Qwen3.5-based Gated DeltaNet LoRA targets."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from peft import LoraConfig as PeftLoraConfig, PeftModel, get_peft_model
from tinker import types
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
from transformers.models.qwen3_5.configuration_qwen3_5 import (
    Qwen3_5Config,
    Qwen3_5TextConfig,
    Qwen3_5VisionConfig,
)
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
    Qwen3_5MoeConfig,
    Qwen3_5MoeTextConfig,
    Qwen3_5MoeVisionConfig,
)
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeForConditionalGeneration,
)

from tuft.backends.fsdp_training_backend import FSDPTrainingBackend, _config_to_worker_dict
from tuft.backends.hf_training_model import HFTrainingModel, build_peft_lora_config
from tuft.backends.lora_modules import (
    MODULE_MAP,
    QWEN3_5_DEFAULT_GATED_DELTANET_TARGET_MODULES,
    QWEN3_5_GATED_DELTANET_TARGET_MODULES,
    find_unmatched_target_modules,
    gated_deltanet_mismatch_hint,
    get_target_modules,
)
from tuft.backends.vllm_lora_compat import resolve_model_architecture, vllm_nests_language_model
from tuft.checkpoints import CheckpointRecord, read_adapter_target_modules
from tuft.config import AppConfig, ModelConfig
from tuft.exceptions import InvalidRequestException
from tuft.training_controller import TrainingController, TrainingRunRecord


QWEN_TEXT_TARGETS = [
    *MODULE_MAP["qwen"]["attn"],
    *QWEN3_5_DEFAULT_GATED_DELTANET_TARGET_MODULES,
    *MODULE_MAP["qwen"]["mlp"],
]

QWEN_FULL_TEXT_TARGETS = [
    *MODULE_MAP["qwen"]["attn"],
    *QWEN3_5_GATED_DELTANET_TARGET_MODULES,
    *MODULE_MAP["qwen"]["mlp"],
]

# The Qwen3.5 target list from before issue #149 added linear_attn.*.
QWEN_OLD_TEXT_TARGETS = [
    *MODULE_MAP["qwen"]["attn"],
    *MODULE_MAP["qwen"]["mlp"],
]


def _write_qwen3_5_config(model_dir: Path) -> None:
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_5",
                "architectures": ["Qwen3_5ForConditionalGeneration"],
            }
        ),
        encoding="utf-8",
    )


def _tiny_qwen3_5(num_hidden_layers: int) -> Qwen3_5ForConditionalGeneration:
    """Build a cheap model with an official three-linear/one-full layer pattern."""

    assert num_hidden_layers % 4 == 0
    text_config = Qwen3_5TextConfig(
        vocab_size=128,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"]
        * (num_hidden_layers // 4),
    )
    vision_config = Qwen3_5VisionConfig(
        depth=1,
        hidden_size=16,
        intermediate_size=32,
        num_heads=2,
        out_hidden_size=16,
        num_position_embeddings=16,
    )
    config = Qwen3_5Config(
        text_config=text_config,
        vision_config=vision_config,
        image_token_id=120,
        video_token_id=121,
        vision_start_token_id=122,
        vision_end_token_id=123,
    )
    return Qwen3_5ForConditionalGeneration(config)


def _wrapped_lora_modules(model: torch.nn.Module) -> list[str]:
    return [name for name, module in model.named_modules() if hasattr(module, "lora_A")]


def test_qwen35_based_model_ids_resolve_to_shared_architecture(tmp_path):
    assert resolve_model_architecture("Qwen/Qwen3.5-4B") == "qwen3_5"
    assert resolve_model_architecture("Qwen/Qwen3.6-27B") == "qwen3_5"
    assert resolve_model_architecture("Qwen/Qwen3.8-27B") == "qwen3_5"
    assert resolve_model_architecture("Qwen/Qwen3-8B") is None

    # A local config works even when its directory has no model-family hint.
    opaque_dir = tmp_path / "checkpoint-1000"
    _write_qwen3_5_config(opaque_dir)
    assert resolve_model_architecture(opaque_dir) == "qwen3_5"

    # An existing config is authoritative over a misleading path.
    qwen_named_llama = tmp_path / "Qwen3.5-llama"
    qwen_named_llama.mkdir()
    (qwen_named_llama / "config.json").write_text(
        json.dumps({"model_type": "llama"}), encoding="utf-8"
    )
    assert resolve_model_architecture(qwen_named_llama) is None


@pytest.mark.parametrize(
    ("model_name", "num_hidden_layers"),
    [("Qwen3.5-4B", 32), ("Qwen3.8-27B", 64)],
)
def test_qwen_gated_deltanet_targets_cover_text_and_exclude_vision(
    tmp_path, model_name, num_hidden_layers
):
    """Pin official layer counts, HF/FSDP parity, and vision exclusion."""

    model_dir = tmp_path / model_name
    _write_qwen3_5_config(model_dir)
    public_config = types.LoraConfig(
        rank=2,
        train_attn=True,
        train_mlp=True,
        train_unembed=False,
    )

    resolved_targets = get_target_modules(str(model_dir), public_config)
    hf_config = build_peft_lora_config(str(model_dir), public_config)
    fsdp_config = ModelConfig(
        model_name=model_name,
        model_path=model_dir,
        max_model_len=1024,
        max_lora_rank=2,
        training_backend="fsdp",
        fsdp_train_attn=True,
        fsdp_train_mlp=True,
        fsdp_train_unembed=False,
    )
    fsdp_targets = _config_to_worker_dict(fsdp_config)["slot_config"]["target_modules"]

    assert resolved_targets == QWEN_TEXT_TARGETS
    assert hf_config.target_modules is not None
    assert set(hf_config.target_modules) == set(fsdp_targets) == set(resolved_targets)

    peft_model = get_peft_model(_tiny_qwen3_5(num_hidden_layers), hf_config)
    wrapped = _wrapped_lora_modules(peft_model)

    full_attention = [name for name in wrapped if ".self_attn." in name]
    linear_attention = [name for name in wrapped if ".linear_attn." in name]
    mlp = [name for name in wrapped if ".mlp." in name]
    vision = [name for name in wrapped if ".visual." in name]

    full_layers = num_hidden_layers // 4
    linear_layers = num_hidden_layers - full_layers
    assert len(full_attention) == full_layers * 4
    assert len(linear_attention) == linear_layers * 3
    assert len(mlp) == num_hidden_layers * 3
    assert vision == []
    assert len(wrapped) == len(full_attention) + len(linear_attention) + len(mlp)
    assert not any(name.endswith((".in_proj_a", ".in_proj_b")) for name in wrapped)


def test_qwen_full_gated_deltanet_lora_is_an_hf_fsdp_aligned_opt_in(tmp_path):
    """Operators can retain A/B gate adapters without changing public defaults."""

    model_dir = tmp_path / "model"
    _write_qwen3_5_config(model_dir)
    public_config = types.LoraConfig(
        rank=2,
        train_attn=True,
        train_mlp=True,
        train_unembed=False,
    )

    hf_config = build_peft_lora_config(
        str(model_dir),
        public_config,
        qwen_gated_deltanet_full_lora=True,
    )
    fsdp_config = ModelConfig(
        model_name="qwen",
        model_path=model_dir,
        max_model_len=1024,
        max_lora_rank=2,
        training_backend="fsdp",
        fsdp_train_attn=True,
        fsdp_train_mlp=True,
        fsdp_train_unembed=False,
        qwen_gated_deltanet_full_lora=True,
    )
    fsdp_targets = _config_to_worker_dict(fsdp_config)["slot_config"]["target_modules"]

    assert hf_config.target_modules is not None
    assert set(hf_config.target_modules) == set(fsdp_targets) == set(QWEN_FULL_TEXT_TARGETS)

    peft_model = get_peft_model(_tiny_qwen3_5(4), hf_config)
    wrapped = _wrapped_lora_modules(peft_model)
    linear_attention = [name for name in wrapped if ".linear_attn." in name]
    vision = [name for name in wrapped if ".visual." in name]

    assert len(linear_attention) == 3 * 5
    assert any(name.endswith(".in_proj_a") for name in linear_attention)
    assert any(name.endswith(".in_proj_b") for name in linear_attention)
    assert vision == []


def test_qwen_gated_deltanet_adapter_metadata_and_weights_round_trip(tmp_path):
    """PEFT metadata and saved weights retain the expanded target geometry."""

    model_dir = tmp_path / "model"
    _write_qwen3_5_config(model_dir)
    lora_config = build_peft_lora_config(
        str(model_dir),
        types.LoraConfig(rank=2, train_attn=True, train_mlp=True, train_unembed=False),
    )

    model_config = ModelConfig(
        model_name="qwen",
        model_path=model_dir,
        max_model_len=1024,
    )
    controller = object.__new__(TrainingController)
    controller.config = AppConfig(supported_models=[model_config])
    effective_targets = controller._effective_target_modules(
        "qwen",
        types.LoraConfig(rank=2, train_attn=True, train_mlp=True, train_unembed=False),
    )
    checkpoint = CheckpointRecord.from_training_run(
        training_run_id="run",
        checkpoint_name="checkpoint-0001",
        owner_name="tester",
        checkpoint_type="training",
        checkpoint_root_dir=tmp_path,
    )
    checkpoint.save_metadata(
        base_model="qwen",
        session_id="session",
        lora_rank=2,
        target_modules=effective_targets,
    )
    assert checkpoint.metadata.target_modules == QWEN_TEXT_TARGETS

    peft_model = get_peft_model(_tiny_qwen3_5(4), lora_config)
    source_module = next(
        module
        for name, module in peft_model.named_modules()
        if name.endswith("layers.0.linear_attn.in_proj_qkv")
    )
    with torch.no_grad():
        source_module.lora_B["default"].weight.fill_(0.25)  # type: ignore[index]

    adapter_dir = tmp_path / "adapter"
    peft_model.save_pretrained(adapter_dir)

    assert set(read_adapter_target_modules(adapter_dir) or []) == set(QWEN_TEXT_TARGETS)

    reloaded = PeftModel.from_pretrained(_tiny_qwen3_5(4), adapter_dir)
    reloaded_module = next(
        module
        for name, module in reloaded.named_modules()
        if name.endswith("layers.0.linear_attn.in_proj_qkv")
    )
    assert torch.all(reloaded_module.lora_B["default"].weight == 0.25)  # type: ignore[index]


def _write_qwen3_5_moe_config(model_dir: Path) -> None:
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_5_moe",
                "architectures": ["Qwen3_5MoeForConditionalGeneration"],
            }
        ),
        encoding="utf-8",
    )


def _tiny_qwen3_5_moe() -> Qwen3_5MoeForConditionalGeneration:
    """Four text layers, four routed experts, one shared expert per layer."""

    text_config = Qwen3_5MoeTextConfig(
        vocab_size=128,
        hidden_size=16,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=16,
    )
    vision_config = Qwen3_5MoeVisionConfig(
        depth=1,
        hidden_size=16,
        intermediate_size=32,
        num_heads=2,
        out_hidden_size=16,
        num_position_embeddings=16,
    )
    config = Qwen3_5MoeConfig(
        text_config=text_config,
        vision_config=vision_config,
        image_token_id=120,
        video_token_id=121,
        vision_start_token_id=122,
        vision_end_token_id=123,
    )
    return Qwen3_5MoeForConditionalGeneration(config)


def test_qwen35_moe_lora_trains_shared_expert_and_skips_routed_experts(tmp_path):
    """The MoE variant resolves the same targets; only the shared expert MLP trains.

    The routed experts are fused parameters with no per-expert Linear modules,
    so the mlp names cannot match them. The router (``mlp.gate``) and
    ``shared_expert_gate`` must stay untouched too. vLLM applies LoRA to the
    same modules when serving, so nothing trained here is dropped there.
    """

    model_dir = tmp_path / "model"
    _write_qwen3_5_moe_config(model_dir)

    assert resolve_model_architecture(model_dir) == "qwen3_5"
    assert vllm_nests_language_model(model_dir) is True

    public_config = types.LoraConfig(rank=2, train_attn=True, train_mlp=True, train_unembed=False)
    assert get_target_modules(str(model_dir), public_config) == QWEN_TEXT_TARGETS

    peft_model = get_peft_model(
        _tiny_qwen3_5_moe(), build_peft_lora_config(str(model_dir), public_config)
    )
    wrapped = _wrapped_lora_modules(peft_model)

    full_attention = [name for name in wrapped if ".self_attn." in name]
    linear_attention = [name for name in wrapped if ".linear_attn." in name]
    shared_expert = [name for name in wrapped if ".shared_expert." in name]

    assert len(full_attention) == 1 * 4
    assert len(linear_attention) == 3 * 3
    assert len(shared_expert) == 4 * 3
    assert len(wrapped) == len(full_attention) + len(linear_attention) + len(shared_expert)
    assert not any(name.endswith(".mlp.gate") for name in wrapped)
    assert not any(".mlp.experts" in name for name in wrapped)
    assert not any(name.endswith("shared_expert_gate") for name in wrapped)
    assert not any(".visual." in name for name in wrapped)


def test_path_fallback_markers_are_anchored(tmp_path):
    """The hub-ID fallback must not misread names that merely contain a marker."""

    # Official-style IDs resolve, with or without a size suffix.
    assert resolve_model_architecture("Qwen/Qwen3.5") == "qwen3_5"
    assert resolve_model_architecture("org/qwen3_5_sft-final") == "qwen3_5"
    # A 'qwen3_8b' checkpoint is Qwen3-8B, and 'qwen3-0.6b' is Qwen3-0.6B.
    assert resolve_model_architecture("someorg/qwen3_8b-sft") is None
    assert resolve_model_architecture(tmp_path / "Qwen3_8B") is None
    assert resolve_model_architecture("Qwen/Qwen3-0.6B") is None
    assert resolve_model_architecture("org/qwen3.6b-sft") is None
    assert resolve_model_architecture("org/qwen3.55-exp") is None


def test_resolve_target_modules_reads_config_json_once(monkeypatch, tmp_path):
    """Series and architecture resolution share a single config.json read."""

    from tuft.backends import vllm_lora_compat

    model_dir = tmp_path / "model"
    _write_qwen3_5_config(model_dir)
    reads: list[object] = []
    original = vllm_lora_compat.load_model_config_json

    def counting_load(path):
        reads.append(path)
        return original(path)

    monkeypatch.setattr(vllm_lora_compat, "load_model_config_json", counting_load)
    modules = get_target_modules(
        str(model_dir),
        types.LoraConfig(rank=2, train_attn=True, train_mlp=True, train_unembed=False),
    )
    assert modules == QWEN_TEXT_TARGETS
    assert len(reads) == 1


def test_gated_deltanet_hint_fires_only_when_linear_attn_modules_differ():
    hint = gated_deltanet_mismatch_hint(QWEN_OLD_TEXT_TARGETS, QWEN_TEXT_TARGETS)
    assert hint is not None
    assert "Gated DeltaNet" in hint
    assert "#149" in hint

    full_hint = gated_deltanet_mismatch_hint(QWEN_TEXT_TARGETS, QWEN_FULL_TEXT_TARGETS)
    assert full_hint is not None
    assert "qwen_gated_deltanet_full_lora" in full_hint

    # Equal sets, and mismatches not caused by linear_attn.*, get no hint.
    assert gated_deltanet_mismatch_hint(QWEN_TEXT_TARGETS, QWEN_TEXT_TARGETS) is None
    assert (
        gated_deltanet_mismatch_hint(QWEN_OLD_TEXT_TARGETS, [*QWEN_OLD_TEXT_TARGETS, "lm_head"])
        is None
    )
    assert gated_deltanet_mismatch_hint(MODULE_MAP["qwen"]["attn"], QWEN_TEXT_TARGETS) is None


def test_old_lora_state_fails_with_a_clear_notice(tmp_path):
    """Loading old LoRA state fails with an error that names the cause."""

    model_dir = tmp_path / "model"
    _write_qwen3_5_config(model_dir)
    checkpoint = CheckpointRecord.from_training_run(
        training_run_id="run",
        checkpoint_name="checkpoint-0001",
        owner_name="tester",
        checkpoint_type="training",
        checkpoint_root_dir=tmp_path,
    )
    checkpoint.save_metadata(
        base_model="qwen",
        session_id="session",
        lora_rank=2,
        target_modules=list(QWEN_OLD_TEXT_TARGETS),
    )

    controller = object.__new__(TrainingController)
    controller.config = AppConfig(
        supported_models=[ModelConfig(model_name="qwen", model_path=model_dir, max_model_len=1024)]
    )
    destination = TrainingRunRecord(
        training_run_id="destination",
        base_model="qwen",
        lora_rank=2,
        train_attn=True,
        train_mlp=True,
        train_unembed=False,
        target_modules=list(QWEN_TEXT_TARGETS),
        session_id="session",
        model_owner="tester",
    )
    with pytest.raises(InvalidRequestException, match="Gated DeltaNet"):
        controller._check_adapter_compatible(
            "checkpoint-0001", checkpoint, checkpoint.metadata, destination
        )

    backend = FSDPTrainingBackend(
        ModelConfig(
            model_name="qwen",
            model_path=model_dir,
            max_model_len=1024,
            max_lora_rank=2,
            training_backend="fsdp",
        )
    )
    with pytest.raises(InvalidRequestException, match="Gated DeltaNet"):
        backend._validate_checkpoint_geometry(checkpoint)


def test_outdated_fsdp_target_modules_fail_at_startup(tmp_path):
    """An old fsdp_target_modules list stops the server at startup."""

    model_dir = tmp_path / "model"
    _write_qwen3_5_config(model_dir)
    stale = ModelConfig(
        model_name="qwen",
        model_path=model_dir,
        max_model_len=1024,
        max_lora_rank=2,
        training_backend="fsdp",
        fsdp_target_modules=list(QWEN_OLD_TEXT_TARGETS),
    )
    with pytest.raises(ValueError) as excinfo:
        FSDPTrainingBackend(stale)
    message = str(excinfo.value)
    assert "cannot be requested by any client" in message
    assert "Gated DeltaNet" in message

    # The Tinker-compatible default list, the full-coverage opt-in, and custom
    # lists for unknown model families still boot.
    FSDPTrainingBackend(stale.model_copy(update={"fsdp_target_modules": list(QWEN_TEXT_TARGETS)}))
    full_without_opt_in = stale.model_copy(
        update={"fsdp_target_modules": list(QWEN_FULL_TEXT_TARGETS)}
    )
    with pytest.raises(ValueError, match="qwen_gated_deltanet_full_lora"):
        FSDPTrainingBackend(full_without_opt_in)

    FSDPTrainingBackend(
        stale.model_copy(
            update={
                "fsdp_target_modules": list(QWEN_FULL_TEXT_TARGETS),
                "qwen_gated_deltanet_full_lora": True,
            }
        )
    )
    FSDPTrainingBackend(
        ModelConfig(
            model_name="custom",
            model_path=Path("/tmp/custom-model"),
            max_model_len=1024,
            max_lora_rank=2,
            training_backend="fsdp",
            fsdp_target_modules=["proj_in", "proj_out"],
        )
    )

    # A mismatch not caused by linear_attn.* fails without the notice.
    with pytest.raises(ValueError) as excinfo:
        FSDPTrainingBackend(stale.model_copy(update={"fsdp_target_modules": ["q_proj"]}))
    assert "Gated DeltaNet" not in str(excinfo.value)


def _tiny_qwen3() -> Qwen3ForCausalLM:
    """A plain Qwen3 model: standard attention, no Gated DeltaNet layers."""

    return Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=128,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
        )
    )


def test_every_resolved_target_must_match_a_real_module(tmp_path):
    """A mislabeled model cannot silently receive partial LoRA coverage."""

    model_dir = tmp_path / "model"
    _write_qwen3_5_config(model_dir)
    targets = get_target_modules(
        str(model_dir),
        types.LoraConfig(rank=2, train_attn=True, train_mlp=True, train_unembed=False),
    )

    # Both real Qwen3.5-based models carry every resolved target.
    for model in (_tiny_qwen3_5(4), _tiny_qwen3_5_moe()):
        names = [name for name, _ in model.named_modules()]
        assert find_unmatched_target_modules(names, targets) == []

    # A plain Qwen3 model has no Gated DeltaNet layers, so the linear_attn
    # targets match nothing and must be reported.
    plain_names = [name for name, _ in _tiny_qwen3().named_modules()]
    assert (
        find_unmatched_target_modules(plain_names, targets)
        == QWEN3_5_DEFAULT_GATED_DELTANET_TARGET_MODULES
    )


@pytest.mark.asyncio
async def test_create_adapter_rejects_targets_missing_from_the_model(tmp_path):
    """HFTrainingModel refuses an adapter whose targets are not all in the model."""

    model_dir = tmp_path / "model"
    _write_qwen3_5_config(model_dir)

    hf_model = HFTrainingModel.__new__(HFTrainingModel)
    hf_model.config = SimpleNamespace(  # type: ignore[assignment]
        model_path=model_dir,
        lora_alpha_ratio=1,
        qwen_gated_deltanet_full_lora=False,
    )
    hf_model.adapter_optimizer = {}
    hf_model._lock = asyncio.Lock()
    hf_model.logger = logging.getLogger(__name__)

    # The config claims Gated DeltaNet, but the real model is a plain Qwen3.
    hf_model.model = _tiny_qwen3()  # type: ignore[assignment]
    lora_config = types.LoraConfig(rank=2, train_attn=True, train_mlp=True, train_unembed=False)
    with pytest.raises(ValueError, match="linear_attn.in_proj_qkv"):
        await hf_model.create_adapter("mislabeled", lora_config)
    assert hf_model.adapter_optimizer == {}

    # With the real Gated DeltaNet model the same request succeeds. Wrap it
    # the way _init_peft_model does, so add_adapter sees a PeftModel.
    hf_model.model = get_peft_model(  # type: ignore[assignment]
        _tiny_qwen3_5(4), PeftLoraConfig(target_modules=["q_proj"]), adapter_name="default"
    )
    await hf_model.create_adapter("run", lora_config)
    assert "run" in hf_model.adapter_optimizer
