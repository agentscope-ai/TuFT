"""Precision and memory regressions for selective log-probability extraction."""

from contextlib import nullcontext

import pytest
import torch

from tuft.backends import fsdp_engine


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("equal_logits", [False, True])
@pytest.mark.parametrize("autocast", [False, True])
def test_target_logprobs_values_and_gradients(monkeypatch, dtype, equal_logits, autocast):
    torch.manual_seed(17)
    # Noncontiguous input, several chunks, and a partial final chunk.
    values = torch.full((7, 2, 256), 20.0) if equal_logits else torch.randn(7, 2, 256) * 5
    logits = values.to(dtype).transpose(0, 1).detach().requires_grad_()
    reference = logits.detach().clone().requires_grad_()
    labels = torch.randint(0, 256, (2, 7))
    work_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    upstream = torch.randn(2, 7, dtype=work_dtype)
    monkeypatch.setattr(fsdp_engine, "_LOGPROB_CHUNK_ELEMENTS", 3 * 256)

    with torch.autocast("cpu", dtype=torch.bfloat16) if autocast else nullcontext():
        actual = fsdp_engine._compute_target_logprobs(logits, labels)
        expected = torch.log_softmax(reference, dim=-1, dtype=work_dtype)
        expected = expected.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    actual.backward(upstream)
    expected.backward(upstream)

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(logits.grad, reference.grad)


def test_target_logprobs_gradcheck(monkeypatch):
    monkeypatch.setattr(fsdp_engine, "_LOGPROB_CHUNK_ELEMENTS", 14)
    logits = torch.randn(2, 3, 7, dtype=torch.float64, requires_grad=True)
    labels = torch.randint(0, 7, (2, 3))
    assert torch.autograd.gradcheck(fsdp_engine._compute_target_logprobs, (logits, labels))


def test_target_logprobs_does_not_retain_promoted_vocabulary_tensors(monkeypatch):
    monkeypatch.setattr(fsdp_engine, "_LOGPROB_CHUNK_ELEMENTS", 2 * 256)
    logits = torch.randn(2, 7, 256, dtype=torch.bfloat16, requires_grad=True)
    labels = torch.randint(0, 256, (2, 7))
    saved = []

    def pack(tensor):
        saved.append(tensor)
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        result = fsdp_engine._compute_target_logprobs(logits, labels)
    # Every retained tensor aliases an existing input; no FP32 vocabulary
    # buffers accumulate across chunks until backward.
    input_storage = {tensor.untyped_storage().data_ptr() for tensor in (logits, labels)}
    assert saved
    assert all(tensor.untyped_storage().data_ptr() in input_storage for tensor in saved)
    result.sum().backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


@pytest.mark.gpu
def test_target_logprobs_cuda_peak_memory(request, monkeypatch):
    if not request.config.getoption("--gpu") or not torch.cuda.is_available():
        pytest.skip("Requires --gpu and CUDA")
    if not torch.cuda.is_bf16_supported():
        pytest.skip("Requires CUDA BF16 support")
    monkeypatch.setattr(fsdp_engine, "_LOGPROB_CHUNK_ELEMENTS", 64 * 4096)
    logits = torch.randn(2, 1024, 4096, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    labels = torch.randint(0, 4096, (2, 1024), device="cuda")

    def peak_bytes(compute):
        logits.grad = None
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            result = compute(logits, labels)
        result.sum().backward()
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() - baseline

    actual = peak_bytes(fsdp_engine._compute_target_logprobs)
    reference = peak_bytes(
        lambda x, y: (
            torch.log_softmax(x, dim=-1, dtype=torch.float32)
            .gather(-1, y.unsqueeze(-1))
            .squeeze(-1)
        )
    )
    assert actual < reference
