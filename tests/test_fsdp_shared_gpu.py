"""Two-rank FSDP storage and numerical equivalence, independent of vLLM."""

from pathlib import Path

import pytest


pytestmark = [pytest.mark.gpu, pytest.mark.integration]


def _verify_rank(rank, rendezvous, output, family="qwen3"):
    import json
    import weakref

    import torch
    import torch.distributed as dist
    from torch.distributed.tensor import DTensor
    from transformers import Qwen3Config, Qwen3ForCausalLM

    import tuft.backends.fsdp_training_backend as backend
    from tuft.backends.fsdp_engine import FSDPModelConfig
    from tuft.backends.shared_weights import tensor_fingerprint

    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method=rendezvous, rank=rank, world_size=2)
    try:
        config = Qwen3Config(
            vocab_size=128,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            tie_word_embeddings=True,
            attention_dropout=0,
            attn_implementation="eager",  # type: ignore[call-arg] - forwarded by config **kwargs
        )
        original_params = []

        def build_model(_):
            if family == "qwen3":
                model = Qwen3ForCausalLM(config).to(torch.bfloat16)
            else:
                from tests.test_qwen_shared_weights import hybrid_model

                # Use the real GDN head width for CUDA kernels. The smaller
                # CPU fixture exercises layout math without FLA autotuning.
                model = hybrid_model(moe=family == "moe", adapters=False, linear_head_dim=128)
            model.enable_input_require_grads()
            model.gradient_checkpointing_enable({"use_reentrant": False})
            original_params.append(weakref.ref(model.model.embed_tokens.weight))
            return model

        backend.build_base_model = build_model
        empty_cache = torch.cuda.empty_cache
        cache_checks = []

        def checked_empty_cache():
            # Placement callbacks must not keep full pre-sharding Parameters
            # alive when startup attempts to release unused allocator memory.
            cache_checks.append(original_params[-1]() is None)
            empty_cache()

        torch.cuda.empty_cache = checked_empty_cache
        workers = []
        for shared in (False, True):
            torch.manual_seed(123)
            worker = backend.MultiAdapterFSDPWorker(
                FSDPModelConfig("tiny-qwen3", 32, share_base_weights=shared),
                backend.SlotPoolConfig(
                    rank_slots={2: 2},
                    target_modules=["q_proj", "v_proj"]
                    if family == "qwen3"
                    else [
                        "q_proj",
                        "v_proj",
                        "in_proj_qkv",
                        "in_proj_z",
                        "out_proj",
                        "gate_proj",
                        "up_proj",
                        "down_proj",
                    ],
                    target_parameters=["mlp.experts.gate_up_proj", "mlp.experts.down_proj"]
                    if family == "moe"
                    else [],
                ),
            )
            worker.initialize()
            workers.append(worker)
        assert cache_checks and all(cache_checks)
        shared = workers[1]
        weights = shared._shared_base_weights
        pointers = {k: v.data_ptr() for k, v in weights.items()}
        hashes = {k: tensor_fingerprint(v) for k, v in weights.items()}
        tokens = torch.tensor([[1, 3 + rank, 5, 7]], device=f"cuda:{rank}")
        errors = []
        for step in range(3):
            logits, gradients = [], []
            for worker in workers:
                adapter = worker.list_adapters()[step % 2]
                worker._activate_adapter(adapter)
                worker._create_optimizer_for_adapter(adapter)
                result = worker.module(input_ids=tokens, use_cache=False).logits
                result.float().square().mean().backward()
                logits.append(result.detach())
                gradients.append(
                    {
                        n: p.grad.to_local().clone()
                        for n, p in worker.module.named_parameters()
                        if p.grad is not None
                    }
                )
                worker.optim_step(adapter)
            torch.testing.assert_close(logits[0], logits[1], rtol=0, atol=0)
            assert gradients[0].keys() == gradients[1].keys()
            for name in gradients[0]:
                torch.testing.assert_close(gradients[0][name], gradients[1][name], rtol=0, atol=0)
            errors.append(float((logits[0] - logits[1]).abs().max()))
            assert hashes == {k: tensor_fingerprint(v) for k, v in weights.items()}
            assert pointers == {k: v.data_ptr() for k, v in weights.items()}
            if family != "qwen3":
                from tuft.backends.shared_weight_layout import canonical_name

                for name, param in shared.module.named_parameters():
                    logical = canonical_name(name)
                    if (
                        ".lora_" in name
                        or param.ndim > 1
                        and not logical.endswith(
                            (".mlp.gate.weight", ".mlp.shared_expert_gate.weight")
                        )
                    ):
                        assert isinstance(param, DTensor), name
                continue
            # Both base matrices and LoRA remain distributed parameters.
            params = dict(shared.module.named_parameters())
            assert isinstance(
                params["base_model.model.model.layers.0.self_attn.o_proj.weight"], DTensor
            )
            assert (
                params["base_model.model.model.layers.0.self_attn.o_proj.weight"].placements[0].dim
                == 1
            )
            assert isinstance(
                params[
                    "base_model.model.model.layers.0.self_attn.q_proj.lora_A.adapter_r2_0.weight"
                ],
                DTensor,
            )
            q = params[
                "base_model.model.model.layers.0.self_attn.q_proj.base_layer.weight"
            ].to_local()
            assert q.data_ptr() == weights["model.layers.0.self_attn.qkv_proj.weight"].data_ptr()
        Path(output, f"rank-{rank}.json").write_text(
            json.dumps(
                {"rank": rank, "exact_logit_errors": errors, "frozen_storage_preserved": True}
            )
        )
    finally:
        dist.destroy_process_group()


def test_fsdp_tp_shards_preserve_training_and_storage(tmp_path):
    import torch
    import torch.multiprocessing as mp

    if torch.cuda.device_count() < 2:
        pytest.skip("Requires two CUDA GPUs")
    mp.spawn(_verify_rank, args=(f"file://{tmp_path}/rendezvous", str(tmp_path)), nprocs=2)
    assert len(list(tmp_path.glob("rank-*.json"))) == 2


@pytest.mark.parametrize("family", ["dense", "moe"])
def test_hybrid_fsdp_tp_shards_preserve_training_and_storage(tmp_path, family):
    import torch
    import torch.multiprocessing as mp

    if torch.cuda.device_count() < 2:
        pytest.skip("Requires two CUDA GPUs")
    mp.spawn(_verify_rank, args=(f"file://{tmp_path}/rendezvous", str(tmp_path), family), nprocs=2)
    assert len(list(tmp_path.glob("rank-*.json"))) == 2
