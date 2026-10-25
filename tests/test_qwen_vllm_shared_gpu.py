"""Exercise the real hybrid vLLM model loader with small dense/MoE weights."""

import pickle
from typing import Any

import pytest


pytestmark = [pytest.mark.gpu, pytest.mark.integration]


@pytest.mark.parametrize("moe", [False, True])
def test_hybrid_vllm_aliases_and_recurrent_sleep(tmp_path, monkeypatch, moe):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen3_5Config, Qwen3_5ForCausalLM, Qwen3_5TextConfig
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
        Qwen3_5MoeConfig,
        Qwen3_5MoeTextConfig,
    )
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeForCausalLM
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    from tuft.backends.hf_training_model import _export_vllm_compatible_lora_aliases
    from tuft.backends.shared_weights import export_cuda_weights, pack_hf_base_weights

    if not torch.cuda.is_available():
        pytest.skip("Requires CUDA")
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    torch.manual_seed(31)
    args: dict[str, Any] = dict(
        vocab_size=128,
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=128,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_num_key_heads=2,
        linear_num_value_heads=6,
        layer_types=["linear_attention"] * 3 + ["full_attention"],
        tie_word_embeddings=False,
    )
    model: Any
    if moe:
        config = Qwen3_5MoeTextConfig(
            **args,
            num_experts=4,
            num_experts_per_tok=2,
            moe_intermediate_size=256,
            shared_expert_intermediate_size=256,
        )
        model = Qwen3_5MoeForCausalLM(config)
        wrapper = Qwen3_5MoeConfig(text_config=config.to_dict())
        wrapper.architectures = ["Qwen3_5MoeForConditionalGeneration"]
    else:
        config = Qwen3_5TextConfig(**args, intermediate_size=512)
        model = Qwen3_5ForCausalLM(config)
        wrapper = Qwen3_5Config(text_config=config.to_dict())
        wrapper.architectures = ["Qwen3_5ForConditionalGeneration"]
    model = model.to(device="cuda", dtype=torch.bfloat16).requires_grad_(False)
    weights = pack_hf_base_weights(model)
    wrapper.save_pretrained(tmp_path)
    manifest = tmp_path / "weights.pkl"
    manifest.write_bytes(pickle.dumps([export_cuda_weights(weights)]))
    llm = LLM(
        model=str(tmp_path),
        skip_tokenizer_init=True,
        dtype="bfloat16",
        max_model_len=256,
        max_num_seqs=4,
        enforce_eager=True,
        worker_cls="tuft.backends.vllm_worker.TuFTGPUWorker",
        load_format="tuft_shared",
        model_loader_extra_config={"manifest": str(manifest)},
        language_model_only=True,
        moe_backend="triton",
        gdn_prefill_backend="triton",
        enable_sleep_mode=True,
        enable_lora=True,
        max_lora_rank=8,
        kv_cache_memory_bytes=1024**3,
    )
    try:
        report = llm.collective_rpc("shared_weight_report")[0]
        assert report["copied_parameters"] == 0
        assert report["verified_alias_bytes"] > 0
        prompt = {"prompt_token_ids": [1, 3, 5, 7]}
        params = SamplingParams(temperature=0, max_tokens=8, logprobs=1, ignore_eos=True)
        base_output = llm.generate([prompt], params)[0].outputs[0]
        # Expert-only LoRA proves routed experts are actually used by vLLM;
        # attention-only changes could otherwise hide a broken expert mapping.
        targets = [] if moe else ["in_proj_qkv", "in_proj_z", "out_proj"]
        model = get_peft_model(
            model,
            LoraConfig(
                r=8,
                target_modules=targets,
                target_parameters=["mlp.experts.gate_up_proj", "mlp.experts.down_proj"]
                if moe
                else None,
            ),
        )
        with torch.no_grad():
            for name, param in model.named_parameters():
                if ".lora_B." in name:
                    param.normal_(std=0.1)
        adapter_dir = tmp_path / "adapter"
        model.save_pretrained(adapter_dir)
        _export_vllm_compatible_lora_aliases(adapter_dir, str(tmp_path))
        adapter = LoRARequest("hybrid", 1, str(adapter_dir))
        before = llm.generate([prompt], params, lora_request=adapter)[0].outputs[0]
        assert before.logprobs != base_output.logprobs
        llm.sleep(level=1)
        llm.wake_up()
        after = llm.generate([prompt], params, lora_request=adapter)[0].outputs[0]
        assert before.token_ids == after.token_ids
        assert before.logprobs == after.logprobs
        assert (
            llm.collective_rpc("shared_weight_report")[0]["verified_alias_bytes"]
            == report["verified_alias_bytes"]
        )
    finally:
        llm.llm_engine.engine_core.shutdown()
