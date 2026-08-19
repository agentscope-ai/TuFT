"""Routed-expert LoRA coverage for Qwen3.5-based MoE models (issue #154).

``train_mlp=True`` targets the fused routed-expert parameters
(``mlp.experts.gate_up_proj`` / ``mlp.experts.down_proj``) through peft
``target_parameters``, in both training backends, and the geometry is
recorded and validated end to end: run records, checkpoint metadata, the
peft ``adapter_config.json``, and the FSDP slot pool.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from peft import LoraConfig as PeftLoraConfig, get_peft_model
from safetensors.torch import load_file
from tinker import types
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeForCausalLM

from tuft.backends.fsdp_engine import FSDPModelConfig
from tuft.backends.fsdp_training_backend import (
    AdapterInfo,
    FSDPTrainingBackend,
    MultiAdapterFSDPWorker,
    SlotPoolConfig,
    _config_to_worker_dict,
)
from tuft.backends.hf_training_model import HFTrainingModel, build_peft_lora_config
from tuft.backends.lora_modules import (
    MODULE_MAP,
    QWEN3_5_DEFAULT_GATED_DELTANET_TARGET_MODULES,
    QWEN3_5_MOE_EXPERT_TARGET_PARAMETERS,
    achievable_lora_target_sets,
    resolve_lora_targets,
    routed_expert_mismatch_hint,
)
from tuft.checkpoints import CheckpointRecord, read_adapter_target_parameters
from tuft.config import AppConfig, ModelConfig
from tuft.exceptions import InvalidRequestException
from tuft.training_controller import TrainingController, TrainingRunRecord


MOE_TEXT_TARGETS = [
    *MODULE_MAP["qwen"]["attn"],
    *QWEN3_5_DEFAULT_GATED_DELTANET_TARGET_MODULES,
    *MODULE_MAP["qwen"]["mlp"],
]

ATTN_ONLY_TARGETS = [
    *MODULE_MAP["qwen"]["attn"],
    *QWEN3_5_DEFAULT_GATED_DELTANET_TARGET_MODULES,
]


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


def _tiny_qwen3_5_moe_causal_lm() -> Qwen3_5MoeForCausalLM:
    """The text-only class the HF and FSDP backends load via AutoModelForCausalLM.

    Its backbone lives at ``model.layers.*`` (no ``language_model`` nesting),
    so saved peft keys take the native layout that the vLLM alias export
    rewrites.
    """

    config = Qwen3_5MoeTextConfig(
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
    return Qwen3_5MoeForCausalLM(config)


def _stub_hf_model(model_dir: Path) -> HFTrainingModel:
    """An HFTrainingModel around a tiny in-memory MoE model, like _init_peft_model."""

    hf_model = HFTrainingModel.__new__(HFTrainingModel)
    hf_model.config = SimpleNamespace(  # type: ignore[assignment]
        model_path=model_dir,
        lora_alpha_ratio=1,
        qwen_gated_deltanet_full_lora=False,
    )
    hf_model.adapter_optimizer = {}
    hf_model._lock = asyncio.Lock()
    hf_model.logger = logging.getLogger(__name__)
    hf_model.model = get_peft_model(  # type: ignore[assignment]
        _tiny_qwen3_5_moe_causal_lm(),
        PeftLoraConfig(target_modules=["q_proj"]),
        adapter_name="default",
    )
    return hf_model


def test_moe_target_resolution_covers_expert_parameters(tmp_path):
    model_dir = tmp_path / "model"
    _write_qwen3_5_moe_config(model_dir)

    full = resolve_lora_targets(
        str(model_dir), train_attn=True, train_mlp=True, train_unembed=False
    )
    assert full.modules == MOE_TEXT_TARGETS
    assert full.parameters == QWEN3_5_MOE_EXPERT_TARGET_PARAMETERS

    attn_only = resolve_lora_targets(
        str(model_dir), train_attn=True, train_mlp=False, train_unembed=False
    )
    assert attn_only.modules == ATTN_ONLY_TARGETS
    assert attn_only.parameters == []

    achievable = achievable_lora_target_sets(str(model_dir))
    assert achievable is not None
    assert any(
        set(targets.modules) == set(MOE_TEXT_TARGETS)
        and set(targets.parameters) == set(QWEN3_5_MOE_EXPERT_TARGET_PARAMETERS)
        for targets in achievable
    )
    # Every achievable geometry that trains the mlp also trains the experts.
    for targets in achievable:
        has_mlp = set(MODULE_MAP["qwen"]["mlp"]) <= set(targets.modules)
        assert (set(targets.parameters) == set(QWEN3_5_MOE_EXPERT_TARGET_PARAMETERS)) == has_mlp


def test_routed_expert_mismatch_hint_fires_only_for_expert_parameters():
    hint = routed_expert_mismatch_hint([], QWEN3_5_MOE_EXPERT_TARGET_PARAMETERS)
    assert hint is not None
    assert "#154" in hint

    partial = routed_expert_mismatch_hint(
        ["mlp.experts.gate_up_proj"], QWEN3_5_MOE_EXPERT_TARGET_PARAMETERS
    )
    assert partial is not None

    assert routed_expert_mismatch_hint([], []) is None
    assert (
        routed_expert_mismatch_hint(
            QWEN3_5_MOE_EXPERT_TARGET_PARAMETERS, QWEN3_5_MOE_EXPERT_TARGET_PARAMETERS
        )
        is None
    )
    assert routed_expert_mismatch_hint([], ["some.other_param"]) is None


@pytest.mark.asyncio
async def test_hf_moe_adapter_save_load_round_trip(tmp_path):
    """The real HF backend save/load path preserves the expert adapter.

    ``save_state`` must record ``target_parameters`` in adapter_config.json,
    write the peft-format expert keys plus their ``language_model.`` aliases
    for vLLM, and ``load_state`` must restore the trained weights.
    """

    model_dir = tmp_path / "model"
    _write_qwen3_5_moe_config(model_dir)
    hf_model = _stub_hf_model(model_dir)
    lora_config = types.LoraConfig(rank=2, train_attn=True, train_mlp=True, train_unembed=False)
    await hf_model.create_adapter("run", lora_config)

    expert_lora_names = [
        name
        for name, param in hf_model.model.named_parameters()
        if ".mlp.experts" in name and ".run." in name
    ]
    # layers x fused parameters x (lora_A, lora_B)
    assert len(expert_lora_names) == 4 * 2 * 2
    marker_name = "base_model.model.model.layers.0.mlp.experts.lora_B.run.weight"
    with torch.no_grad():
        dict(hf_model.model.named_parameters())[marker_name].fill_(0.25)

    checkpoint = CheckpointRecord.from_training_run(
        training_run_id="run",
        checkpoint_name="checkpoint-0001",
        owner_name="tester",
        checkpoint_type="training",
        checkpoint_root_dir=tmp_path,
    )
    await hf_model.save_state("run", checkpoint, optimizer=False)

    assert set(read_adapter_target_parameters(checkpoint.adapter_path) or []) == set(
        QWEN3_5_MOE_EXPERT_TARGET_PARAMETERS
    )
    state = load_file(str(checkpoint.adapter_path / "adapter_model.safetensors"))
    native_expert_keys = [
        key for key in state if ".layers.0.mlp.experts" in key and "language_model" not in key
    ]
    alias_expert_keys = [
        key
        for key in state
        if key.startswith("base_model.model.model.language_model.layers.0.mlp.experts")
    ]
    # The nested peft layout vLLM parses: experts (down_proj) and
    # experts.base_layer (gate_up_proj), each with lora_A/lora_B.
    assert sorted(native_expert_keys) == [
        "base_model.model.model.layers.0.mlp.experts.base_layer.lora_A.weight",
        "base_model.model.model.layers.0.mlp.experts.base_layer.lora_B.weight",
        "base_model.model.model.layers.0.mlp.experts.lora_A.weight",
        "base_model.model.model.layers.0.mlp.experts.lora_B.weight",
    ]
    assert len(alias_expert_keys) == 4

    await hf_model.load_state("restored", checkpoint, optimizer=False)
    restored = dict(hf_model.model.named_parameters())[marker_name.replace(".run.", ".restored.")]
    assert bool((restored == 0.25).all())


@pytest.mark.asyncio
async def test_hf_create_adapter_rejects_missing_expert_parameters(tmp_path):
    """A model whose config claims MoE but has no fused experts is rejected."""

    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
    from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

    model_dir = tmp_path / "model"
    _write_qwen3_5_moe_config(model_dir)
    hf_model = _stub_hf_model(model_dir)
    hf_model.model = Qwen3ForCausalLM(  # type: ignore[assignment]
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
    lora_config = types.LoraConfig(rank=2, train_attn=True, train_mlp=True, train_unembed=False)
    with pytest.raises(ValueError, match="target (modules|parameters)"):
        await hf_model.create_adapter("mislabeled", lora_config)
    assert hf_model.adapter_optimizer == {}


def test_fsdp_slot_pool_geometry_covers_expert_parameters(tmp_path):
    model_dir = tmp_path / "model"
    _write_qwen3_5_moe_config(model_dir)

    config = ModelConfig(
        model_name="moe",
        model_path=model_dir,
        max_model_len=1024,
        max_lora_rank=2,
        training_backend="fsdp",
        fsdp_train_attn=True,
        fsdp_train_mlp=True,
        fsdp_train_unembed=False,
    )
    slot_config = _config_to_worker_dict(config)["slot_config"]
    assert slot_config["target_modules"] == MOE_TEXT_TARGETS
    assert slot_config["target_parameters"] == QWEN3_5_MOE_EXPERT_TARGET_PARAMETERS

    attn_only = config.model_copy(update={"fsdp_train_mlp": False})
    slot_config = _config_to_worker_dict(attn_only)["slot_config"]
    assert slot_config["target_modules"] == ATTN_ONLY_TARGETS
    assert slot_config["target_parameters"] == []

    # Explicit geometry is honored verbatim when it is achievable.
    explicit = config.model_copy(
        update={
            "fsdp_target_modules": list(MOE_TEXT_TARGETS),
            "fsdp_target_parameters": list(QWEN3_5_MOE_EXPERT_TARGET_PARAMETERS),
        }
    )
    slot_config = _config_to_worker_dict(explicit)["slot_config"]
    assert slot_config["target_parameters"] == QWEN3_5_MOE_EXPERT_TARGET_PARAMETERS
    FSDPTrainingBackend(explicit)

    # Explicit modules alone resolve the parameter side from the modifiers,
    # so a pre-#154 modules-only config keeps working and gains the experts.
    modules_only = config.model_copy(update={"fsdp_target_modules": list(MOE_TEXT_TARGETS)})
    slot_config = _config_to_worker_dict(modules_only)["slot_config"]
    assert slot_config["target_parameters"] == QWEN3_5_MOE_EXPERT_TARGET_PARAMETERS
    FSDPTrainingBackend(modules_only)


def test_fsdp_explicit_geometry_without_experts_fails_at_startup(tmp_path):
    """Pinning fsdp_target_parameters to [] on an MoE model is unachievable."""

    model_dir = tmp_path / "model"
    _write_qwen3_5_moe_config(model_dir)
    config = ModelConfig(
        model_name="moe",
        model_path=model_dir,
        max_model_len=1024,
        max_lora_rank=2,
        training_backend="fsdp",
        fsdp_train_unembed=False,
        fsdp_target_parameters=[],
    )
    with pytest.raises(ValueError) as excinfo:
        FSDPTrainingBackend(config)
    message = str(excinfo.value)
    assert "cannot be requested by any client" in message
    assert "#154" in message


@pytest.mark.asyncio
async def test_fsdp_validate_lora_config_compares_expert_parameters(tmp_path):
    model_dir = tmp_path / "model"
    _write_qwen3_5_moe_config(model_dir)
    config = ModelConfig(
        model_name="moe",
        model_path=model_dir,
        max_model_len=1024,
        max_lora_rank=2,
        training_backend="fsdp",
        fsdp_train_attn=True,
        fsdp_train_mlp=False,
        fsdp_train_unembed=False,
    )
    backend = FSDPTrainingBackend(config)
    # Matching modifiers pass.
    backend._validate_lora_config(
        types.LoraConfig(rank=2, train_attn=True, train_mlp=False, train_unembed=False)
    )
    # train_mlp=True resolves expert parameters the attention-only pool lacks.
    with pytest.raises(InvalidRequestException, match="parameters.*mlp.experts") as excinfo:
        backend._validate_lora_config(
            types.LoraConfig(rank=2, train_attn=True, train_mlp=True, train_unembed=False)
        )
    assert "mlp.experts.gate_up_proj" in excinfo.value.detail


def test_fsdp_checkpoint_geometry_rejects_pre_154_moe_checkpoints(tmp_path):
    model_dir = tmp_path / "model"
    _write_qwen3_5_moe_config(model_dir)
    backend = FSDPTrainingBackend(
        ModelConfig(
            model_name="moe",
            model_path=model_dir,
            max_model_len=1024,
            max_lora_rank=2,
            training_backend="fsdp",
            fsdp_train_unembed=False,
        )
    )
    checkpoint = CheckpointRecord(
        checkpoint_id="old-checkpoint",
        owner_name="tester",
        checkpoint_type="training",
        training_run_id="source",
        path=tmp_path / "old-checkpoint",
    )
    checkpoint.adapter_path.mkdir(parents=True)
    # A pre-#154 adapter records the module list but no target_parameters.
    (checkpoint.adapter_path / "adapter_config.json").write_text(
        json.dumps({"target_modules": MOE_TEXT_TARGETS}),
        encoding="utf-8",
    )
    with pytest.raises(InvalidRequestException, match="#154"):
        backend._validate_checkpoint_geometry(checkpoint)

    (checkpoint.adapter_path / "adapter_config.json").write_text(
        json.dumps(
            {
                "target_modules": MOE_TEXT_TARGETS,
                "target_parameters": QWEN3_5_MOE_EXPERT_TARGET_PARAMETERS,
            }
        ),
        encoding="utf-8",
    )
    modules, parameters = backend._validate_checkpoint_geometry(checkpoint)
    assert set(modules) == set(MOE_TEXT_TARGETS)
    assert parameters == QWEN3_5_MOE_EXPERT_TARGET_PARAMETERS


def _build_cpu_worker(tmp_path: Path) -> MultiAdapterFSDPWorker:
    """A worker with two rank-2 slots over a tiny MoE model, without FSDP wrap.

    ``initialize()`` requires CUDA; the checkpoint code paths only need the
    peft module and adapter bookkeeping, which mirror what initialize builds.
    """

    model_dir = tmp_path / "model"
    if not model_dir.exists():
        _write_qwen3_5_moe_config(model_dir)
    slot_config = SlotPoolConfig(
        rank_slots={2: 2},
        lora_alpha_ratio=1,
        target_modules=list(MOE_TEXT_TARGETS),
        target_parameters=list(QWEN3_5_MOE_EXPERT_TARGET_PARAMETERS),
    )
    worker = MultiAdapterFSDPWorker(
        model_config=FSDPModelConfig(path=str(model_dir), max_model_len=1024),
        slot_config=slot_config,
    )
    peft_model = None
    for name in ("adapter_r2_0", "adapter_r2_1"):
        lora_config = PeftLoraConfig(
            r=2,
            lora_alpha=2,
            target_modules=list(slot_config.target_modules),
            target_parameters=list(slot_config.target_parameters),
        )
        if peft_model is None:
            peft_model = get_peft_model(
                _tiny_qwen3_5_moe_causal_lm(), lora_config, adapter_name=name
            )
        else:
            peft_model.add_adapter(name, lora_config)
        worker._adapters[name] = AdapterInfo(
            name=name,
            rank=2,
            lora_alpha=2,
            target_modules=list(slot_config.target_modules),
            target_parameters=list(slot_config.target_parameters),
        )
        worker._adapters_by_rank.setdefault(2, []).append(name)
        worker._allocated[name] = False
    worker.module = peft_model
    worker._initialized = True
    return worker


def test_fsdp_worker_checkpoint_round_trip_with_expert_parameters(tmp_path):
    """The FSDP worker saves and reloads fused-expert LoRA state on CPU.

    The saved files must carry the canonical (slot-independent) expert keys,
    a peft config that records target_parameters, and the ``language_model.``
    alias keys vLLM needs; loading into a different slot restores the same
    weights, and a geometry mismatch is rejected before touching weights.
    """

    worker = _build_cpu_worker(tmp_path)
    marker = "base_model.model.model.layers.0.mlp.experts.lora_B.adapter_r2_0.weight"
    with torch.no_grad():
        dict(worker.module.named_parameters())[marker].fill_(0.125)

    path = tmp_path / "checkpoint"
    worker.save_checkpoint("adapter_r2_0", path, optimizer=False)

    state = torch.load(path / "adapter.pt", weights_only=True)
    expert_keys = [key for key in state if ".layers.0.mlp.experts" in key]
    assert sorted(expert_keys) == [
        "base_model.model.model.layers.0.mlp.experts.base_layer.lora_A.weight",
        "base_model.model.model.layers.0.mlp.experts.base_layer.lora_B.weight",
        "base_model.model.model.layers.0.mlp.experts.lora_A.weight",
        "base_model.model.model.layers.0.mlp.experts.lora_B.weight",
    ]

    adapter_config = json.loads((path / "adapter_config.json").read_text(encoding="utf-8"))
    assert set(adapter_config["target_parameters"]) == set(QWEN3_5_MOE_EXPERT_TARGET_PARAMETERS)
    assert set(read_adapter_target_parameters(path) or []) == set(
        QWEN3_5_MOE_EXPERT_TARGET_PARAMETERS
    )

    saved = load_file(str(path / "adapter_model.safetensors"))
    assert any(
        key.startswith("base_model.model.model.language_model.layers.0.mlp.experts")
        for key in saved
    )

    # Loading into the OTHER slot restores the marker weight (canonical keys
    # make checkpoints slot-independent).
    worker.load_checkpoint(
        "adapter_r2_1",
        path,
        checkpoint_modules=list(MOE_TEXT_TARGETS),
        optimizer=False,
        checkpoint_parameters=list(QWEN3_5_MOE_EXPERT_TARGET_PARAMETERS),
    )
    restored = dict(worker.module.named_parameters())[
        marker.replace("adapter_r2_0", "adapter_r2_1")
    ]
    assert bool((restored == 0.125).all())

    with pytest.raises(RuntimeError, match="parameters"):
        worker.load_checkpoint(
            "adapter_r2_1",
            path,
            checkpoint_modules=list(MOE_TEXT_TARGETS),
            optimizer=False,
            checkpoint_parameters=[],
        )


def test_fsdp_worker_slot_reset_covers_expert_parameters(tmp_path):
    """Releasing a slot clears its trained fused-expert weights."""

    worker = _build_cpu_worker(tmp_path)
    marker = "base_model.model.model.layers.0.mlp.experts.lora_B.adapter_r2_0.weight"
    with torch.no_grad():
        dict(worker.module.named_parameters())[marker].fill_(0.125)
    worker.release_slot("adapter_r2_0")
    reset = dict(worker.module.named_parameters())[marker]
    assert bool((reset == 0).all())


def test_controller_records_and_validates_expert_parameters(tmp_path):
    model_dir = tmp_path / "model"
    _write_qwen3_5_moe_config(model_dir)
    model_config = ModelConfig(model_name="moe", model_path=model_dir, max_model_len=1024)
    controller = object.__new__(TrainingController)
    controller.config = AppConfig(supported_models=[model_config])

    lora_config = types.LoraConfig(rank=2, train_attn=True, train_mlp=True, train_unembed=False)
    effective = controller._effective_lora_targets("moe", lora_config)
    assert effective.modules == MOE_TEXT_TARGETS
    assert effective.parameters == QWEN3_5_MOE_EXPERT_TARGET_PARAMETERS

    checkpoint = CheckpointRecord.from_training_run(
        training_run_id="run",
        checkpoint_name="checkpoint-0001",
        owner_name="tester",
        checkpoint_type="training",
        checkpoint_root_dir=tmp_path,
    )
    checkpoint.save_metadata(
        base_model="moe",
        session_id="session",
        lora_rank=2,
        target_modules=effective.modules,
        target_parameters=effective.parameters,
    )
    assert checkpoint.metadata.target_parameters == QWEN3_5_MOE_EXPERT_TARGET_PARAMETERS
    # No adapter_config.json here, so metadata is the fallback source.
    assert checkpoint.saved_target_parameters == QWEN3_5_MOE_EXPERT_TARGET_PARAMETERS

    destination = TrainingRunRecord(
        training_run_id="destination",
        base_model="moe",
        lora_rank=2,
        train_attn=True,
        train_mlp=True,
        train_unembed=False,
        target_modules=list(MOE_TEXT_TARGETS),
        target_parameters=list(QWEN3_5_MOE_EXPERT_TARGET_PARAMETERS),
        session_id="session",
        model_owner="tester",
    )
    controller._check_adapter_compatible(
        "checkpoint-0001", checkpoint, checkpoint.metadata, destination
    )

    # A pre-#154 checkpoint records no parameter targets and must be rejected
    # with a notice naming the change.
    old_checkpoint = CheckpointRecord.from_training_run(
        training_run_id="run",
        checkpoint_name="checkpoint-0002",
        owner_name="tester",
        checkpoint_type="training",
        checkpoint_root_dir=tmp_path,
    )
    old_checkpoint.save_metadata(
        base_model="moe",
        session_id="session",
        lora_rank=2,
        target_modules=list(MOE_TEXT_TARGETS),
    )
    assert old_checkpoint.saved_target_parameters == []
    with pytest.raises(InvalidRequestException, match="#154"):
        controller._check_adapter_compatible(
            "checkpoint-0002", old_checkpoint, old_checkpoint.metadata, destination
        )


def test_saved_target_parameters_prefers_adapter_config(tmp_path):
    checkpoint = CheckpointRecord(
        checkpoint_id="checkpoint",
        owner_name="tester",
        checkpoint_type="training",
        training_run_id="run",
        path=tmp_path / "checkpoint",
    )
    checkpoint.adapter_path.mkdir(parents=True)

    # Nothing recorded anywhere: pre-#154 checkpoints trained none.
    assert checkpoint.saved_target_parameters == []
    assert read_adapter_target_parameters(checkpoint.adapter_path) is None

    (checkpoint.adapter_path / "adapter_config.json").write_text(
        json.dumps({"target_modules": ["q_proj"]}), encoding="utf-8"
    )
    # A readable config without the key means the adapter trained none.
    assert read_adapter_target_parameters(checkpoint.adapter_path) == []
    assert checkpoint.saved_target_parameters == []

    (checkpoint.adapter_path / "adapter_config.json").write_text(
        json.dumps(
            {
                "target_modules": ["q_proj"],
                "target_parameters": QWEN3_5_MOE_EXPERT_TARGET_PARAMETERS,
            }
        ),
        encoding="utf-8",
    )
    assert checkpoint.saved_target_parameters == QWEN3_5_MOE_EXPERT_TARGET_PARAMETERS


def test_dense_qwen35_geometry_is_unchanged(tmp_path):
    """The dense architecture resolves no parameter targets anywhere."""

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"model_type": "qwen3_5", "architectures": ["Qwen3_5ForConditionalGeneration"]}),
        encoding="utf-8",
    )
    targets = resolve_lora_targets(
        str(model_dir), train_attn=True, train_mlp=True, train_unembed=False
    )
    assert targets.parameters == []
    peft_config = build_peft_lora_config(
        str(model_dir),
        types.LoraConfig(rank=2, train_attn=True, train_mlp=True, train_unembed=False),
    )
    assert not peft_config.target_parameters

    config = ModelConfig(
        model_name="dense",
        model_path=model_dir,
        max_model_len=1024,
        max_lora_rank=2,
        training_backend="fsdp",
        fsdp_train_unembed=False,
    )
    assert _config_to_worker_dict(config)["slot_config"]["target_parameters"] == []
