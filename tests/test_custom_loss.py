"""Server-side contract of the Tinker SDK's client-defined losses.

``TrainingClient.forward_backward_custom(data, loss_fn, loss_type_input="logprobs")``
never ships Python to the server. The pinned SDK (tinker 0.25,
``tinker/lib/public_interfaces/training_client.py``) lowers it onto two plain
requests that TuFT already serves:

1. ``forward`` with ``loss_fn="cross_entropy"`` (``weights`` zero-filled when a
   datum omits them) -> the server returns per-datum ``loss_fn_outputs[i]["logprobs"]``.
2. The client differentiates its own scalar loss against those log-probs and
   sends ``forward_backward`` with ``weights = -dC/dlogprobs``. The server's
   cross-entropy ``L = -(target_logprobs * weights).sum()`` then back-propagates
   exactly ``dC/dtheta``.

These tests pin the server-side contract that lowering relies on, per backend:

- pass 1 returns log-probs aligned datum-by-datum (row i belongs to ``data[i]``,
  trimmed to ``len(model_input)``) and leaves gradients and parameters untouched;
- the two-pass gradients equal direct autograd of the same composite loss
  through the model, including when micro-batching slices the request;
- a response-normalized SFT callback reproduces an equivalent weighted
  ``cross_entropy`` request.

``_client_two_pass`` mirrors the SDK's client logic; keep it in sync with the
pinned tinker version so the tests exercise the real wire contract.
"""

from __future__ import annotations

import asyncio
import copy
import logging
from types import SimpleNamespace
from typing import Awaitable, Callable

import pytest
import torch
from tinker import types

from tuft.backends.fsdp_engine import forward_backward as fsdp_forward_backward
from tuft.backends.hf_training_model import HFTrainingModel


VOCAB = 32

# Variable lengths on purpose: alignment bugs (row shuffling, padding leaks)
# cannot cancel out when every datum has a distinct length.
TOKEN_ROWS = [
    [1, 2, 3, 4, 5],
    [6, 7, 8],
    [9, 10, 11, 12],
    [13, 14],
]
TARGET_ROWS = [[token + 1 for token in row] for row in TOKEN_ROWS]
REFERENCE_LOGPROB_ROWS = [torch.linspace(-3.0, -2.0, steps=len(tokens)) for tokens in TOKEN_ROWS]

RunRequest = Callable[[list[types.Datum], bool], Awaitable[list[torch.Tensor]]]


class TinyCausalLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(VOCAB, 8)
        self.lm_head = torch.nn.Linear(8, VOCAB, bias=False)

    def forward(self, input_ids, **_kwargs):
        return SimpleNamespace(logits=self.lm_head(self.embed(input_ids)))


def _make_data(weights_rows: list[torch.Tensor] | None = None) -> list[types.Datum]:
    """Datums shaped exactly like the SDK's forward_backward_custom requests.

    ``weights_rows=None`` builds pass-1 data (zero-filled weights, as the SDK
    synthesizes for datums without explicit weights); passing tensors builds
    pass-2 data (``weights = -grad``).
    """
    data = []
    for row, (tokens, targets) in enumerate(zip(TOKEN_ROWS, TARGET_ROWS, strict=True)):
        if weights_rows is None:
            weights = types.TensorData(
                data=[0.0] * len(targets), dtype="float32", shape=[len(targets)]
            )
        else:
            weights = types.TensorData.from_torch(weights_rows[row])
        data.append(
            types.Datum(
                model_input=types.ModelInput.from_ints(tokens=list(tokens)),
                loss_fn_inputs={
                    "target_tokens": types.TensorData(
                        data=list(targets), dtype="int64", shape=[len(targets)]
                    ),
                    "weights": weights,
                },
            )
        )
    return data


def _composite_dpo_loss(logprobs_list: list[torch.Tensor]) -> torch.Tensor:
    """DPO + chosen NLL + policy/reference log-ratio regularization.

    Rows are two (chosen, rejected) pairs. The frozen reference log-probs make
    this a true DPO objective, while the response masks and per-row means mirror
    the Open Character Training objective that motivated issue 146.
    """
    dpo_terms = []
    chosen_nll_terms = []
    squared_logratio_terms = []
    for pair_start in range(0, len(logprobs_list), 2):
        chosen = logprobs_list[pair_start]
        rejected = logprobs_list[pair_start + 1]
        chosen_reference = REFERENCE_LOGPROB_ROWS[pair_start]
        rejected_reference = REFERENCE_LOGPROB_ROWS[pair_start + 1]
        chosen_mask = torch.ones_like(chosen)
        rejected_mask = torch.ones_like(rejected)
        chosen_mask[-1] = 0.0
        rejected_mask[-1] = 0.0

        chosen_logratio = chosen - chosen_reference
        rejected_logratio = rejected - rejected_reference
        chosen_sum = (chosen_logratio * chosen_mask).sum()
        rejected_sum = (rejected_logratio * rejected_mask).sum()
        dpo_terms.append(-torch.nn.functional.logsigmoid(0.1 * (chosen_sum - rejected_sum)))
        chosen_nll_terms.append(-(chosen * chosen_mask).sum() / chosen_mask.sum())
        squared_logratio_terms.extend(
            [
                chosen_logratio[chosen_mask.bool()].square().mean(),
                rejected_logratio[rejected_mask.bool()].square().mean(),
            ]
        )

    dpo = torch.stack(dpo_terms).mean()
    chosen_nll = torch.stack(chosen_nll_terms).mean()
    logratio_mse = torch.stack(squared_logratio_terms).mean()
    return dpo + 0.1 * chosen_nll + 0.001 * logratio_mse


def _response_normalized_sft_loss(logprobs_list: list[torch.Tensor]) -> torch.Tensor:
    """Batch mean of per-example response-token mean NLL."""
    per_example_nlls = []
    for lp in logprobs_list:
        weights = torch.ones_like(lp)
        weights[-1] = 0.0
        per_example_nlls.append(-(lp * weights).sum() / weights.sum())
    return torch.stack(per_example_nlls).mean()


async def _client_two_pass(
    run_request: RunRequest,
    loss_fn: Callable[[list[torch.Tensor]], torch.Tensor],
) -> float:
    """Replicate tinker 0.25's forward_backward_custom client logic.

    ``run_request(data, backward)`` stands in for one /forward_backward request
    and must return the per-datum log-prob rows from the response.
    """
    logprobs_list = [
        lp.clone().detach().float().requires_grad_(True)
        for lp in await run_request(_make_data(), False)
    ]
    # Contract: row i is data[i], trimmed to its own length.
    assert [tuple(lp.shape) for lp in logprobs_list] == [(len(tokens),) for tokens in TOKEN_ROWS]
    loss = loss_fn(logprobs_list)
    loss.backward()
    grads = []
    for lp in logprobs_list:
        assert lp.grad is not None
        grads.append(lp.grad)
    await run_request(_make_data(weights_rows=[-grad for grad in grads]), True)
    return loss.item()


def _direct_autograd_reference(
    model: torch.nn.Module, loss_fn: Callable[[list[torch.Tensor]], torch.Tensor]
) -> tuple[list[torch.Tensor], float]:
    """Differentiate the same composite loss straight through a model copy."""
    reference = copy.deepcopy(model)
    for parameter in reference.parameters():
        parameter.grad = None
    logprobs_list = []
    for tokens, targets in zip(TOKEN_ROWS, TARGET_ROWS, strict=True):
        logits = reference(input_ids=torch.tensor([tokens], dtype=torch.long)).logits[0]
        row_logprobs = torch.log_softmax(logits.float(), dim=-1)
        logprobs_list.append(
            row_logprobs.gather(-1, torch.tensor(targets).unsqueeze(-1)).squeeze(-1)
        )
    loss = loss_fn(logprobs_list)
    loss.backward()
    return [parameter.grad.clone() for parameter in reference.parameters()], loss.item()


def _reference_logprobs_row(network: torch.nn.Module, tokens: list[int], targets: list[int]):
    logits = network(input_ids=torch.tensor([tokens], dtype=torch.long)).logits[0]
    return (
        torch.log_softmax(logits.float(), dim=-1)
        .gather(-1, torch.tensor(targets).unsqueeze(-1))
        .squeeze(-1)
    )


def _build_hf_model(monkeypatch, micro_batch_size: int) -> tuple[HFTrainingModel, TinyCausalLM]:
    torch.manual_seed(7)
    network = TinyCausalLM()
    model = HFTrainingModel.__new__(HFTrainingModel)
    model.config = SimpleNamespace(micro_batch_size=micro_batch_size)  # type: ignore[assignment]
    model.model = network  # type: ignore[assignment]
    model._lock = asyncio.Lock()
    model.logger = logging.getLogger(__name__)
    monkeypatch.setattr(model, "_activate_adapter", lambda _lora_id: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda: 0)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda: 0)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)
    return model, network


def _hf_run_request(model: HFTrainingModel) -> RunRequest:
    async def run_request(data: list[types.Datum], backward: bool) -> list[torch.Tensor]:
        output = await model.forward(data, "lora", "cross_entropy", None, backward=backward)
        return [row["logprobs"].to_torch() for row in output.loss_fn_outputs]

    return run_request


def _fsdp_run_request(network: torch.nn.Module) -> RunRequest:
    async def run_request(data: list[types.Datum], backward: bool) -> list[torch.Tensor]:
        output = fsdp_forward_backward(
            network,
            data,
            "cross_entropy",
            None,
            micro_batch_size=2,
            forward_only=not backward,
        )
        return list(output["model_output"]["log_probs"])

    return run_request


@pytest.mark.parametrize(
    "loss_fn",
    [_composite_dpo_loss, _response_normalized_sft_loss],
    ids=["composite_dpo", "response_normalized_sft"],
)
async def test_hf_two_pass_custom_loss_matches_direct_autograd(monkeypatch, loss_fn):
    # micro_batch_size=2 slices the 4-datum request, so the assertion also pins
    # that micro-batching preserves per-datum output order and summed gradients.
    model, network = _build_hf_model(monkeypatch, micro_batch_size=2)

    loss_value = await _client_two_pass(_hf_run_request(model), loss_fn)

    expected_grads, expected_loss = _direct_autograd_reference(network, loss_fn)
    assert loss_value == pytest.approx(expected_loss, rel=1e-5)
    for parameter, expected in zip(network.parameters(), expected_grads, strict=True):
        torch.testing.assert_close(parameter.grad, expected, rtol=1e-4, atol=1e-6)


async def test_hf_forward_only_pass_leaves_model_state_untouched(monkeypatch):
    """Pass 1 of the custom-loss flow must be a pure read on the HF backend."""
    model, network = _build_hf_model(monkeypatch, micro_batch_size=2)
    parameters_before = [parameter.clone() for parameter in network.parameters()]

    output = await model.forward(_make_data(), "lora", "cross_entropy", None, backward=False)

    assert all(parameter.grad is None for parameter in network.parameters())
    for parameter, before in zip(network.parameters(), parameters_before, strict=True):
        torch.testing.assert_close(parameter, before)
    # The returned rows are the exact values the client differentiates against.
    for row_index, (tokens, targets) in enumerate(zip(TOKEN_ROWS, TARGET_ROWS, strict=True)):
        returned = output.loss_fn_outputs[row_index]["logprobs"]
        torch.testing.assert_close(
            returned.to_torch(),
            _reference_logprobs_row(network, tokens, targets),
            rtol=1e-4,
            atol=1e-6,
        )


async def test_hf_forward_only_skips_autograd_graph(monkeypatch):
    """Forward-only must run under no_grad (parity with the FSDP engine).

    The custom-loss workflow doubles forward passes, so pass 1 building an
    autograd graph would silently cost activation memory on every step.
    """
    model, network = _build_hf_model(monkeypatch, micro_batch_size=4)
    saw_grad_mode: list[bool] = []
    original_forward = network.forward

    def recording_forward(*args, **kwargs):
        saw_grad_mode.append(torch.is_grad_enabled())
        return original_forward(*args, **kwargs)

    monkeypatch.setattr(network, "forward", recording_forward)

    await model.forward(_make_data(), "lora", "cross_entropy", None, backward=False)
    assert saw_grad_mode == [False]

    await model.forward(_make_data(), "lora", "cross_entropy", None, backward=True)
    assert saw_grad_mode == [False, True]


@pytest.mark.parametrize(
    "loss_fn",
    [_composite_dpo_loss, _response_normalized_sft_loss],
    ids=["composite_dpo", "response_normalized_sft"],
)
async def test_fsdp_two_pass_custom_loss_matches_direct_autograd(loss_fn):
    torch.manual_seed(7)
    network = TinyCausalLM()

    loss_value = await _client_two_pass(_fsdp_run_request(network), loss_fn)

    expected_grads, expected_loss = _direct_autograd_reference(network, loss_fn)
    assert loss_value == pytest.approx(expected_loss, rel=1e-5)
    for parameter, expected in zip(network.parameters(), expected_grads, strict=True):
        torch.testing.assert_close(parameter.grad, expected, rtol=1e-4, atol=1e-6)


def test_fsdp_forward_only_pass_leaves_model_state_untouched():
    torch.manual_seed(7)
    network = TinyCausalLM()
    parameters_before = [parameter.clone() for parameter in network.parameters()]

    output = fsdp_forward_backward(
        network, _make_data(), "cross_entropy", None, micro_batch_size=2, forward_only=True
    )

    assert all(parameter.grad is None for parameter in network.parameters())
    for parameter, before in zip(network.parameters(), parameters_before, strict=True):
        torch.testing.assert_close(parameter, before)
    for row, (tokens, targets) in zip(
        output["model_output"]["log_probs"],
        zip(TOKEN_ROWS, TARGET_ROWS, strict=True),
        strict=True,
    ):
        torch.testing.assert_close(
            row.float(),
            _reference_logprobs_row(network, tokens, targets),
            rtol=1e-4,
            atol=1e-6,
        )


@pytest.mark.parametrize("backend", ["hf", "fsdp"])
async def test_two_pass_sft_callback_reproduces_builtin_cross_entropy(monkeypatch, backend):
    """Response-normalized SFT must equal an equivalent weighted server loss.

    This is the equivalence users rely on when porting an SFT recipe from
    ``forward_backward(..., "cross_entropy")`` to ``forward_backward_custom``.
    """

    def builtin_weights(targets: list[int]) -> types.TensorData:
        weight = 1.0 / (len(TOKEN_ROWS) * (len(targets) - 1))
        weights = [weight] * len(targets)
        weights[-1] = 0.0
        return types.TensorData(data=weights, dtype="float32", shape=[len(targets)])

    builtin_data = [
        types.Datum(
            model_input=types.ModelInput.from_ints(tokens=list(tokens)),
            loss_fn_inputs={
                "target_tokens": types.TensorData(
                    data=list(targets), dtype="int64", shape=[len(targets)]
                ),
                "weights": builtin_weights(targets),
            },
        )
        for tokens, targets in zip(TOKEN_ROWS, TARGET_ROWS, strict=True)
    ]

    if backend == "hf":
        model, custom_network = _build_hf_model(monkeypatch, micro_batch_size=2)
        run_request = _hf_run_request(model)
    else:
        torch.manual_seed(7)
        custom_network = TinyCausalLM()
        run_request = _fsdp_run_request(custom_network)
    builtin_network = copy.deepcopy(custom_network)

    await _client_two_pass(run_request, _response_normalized_sft_loss)

    if backend == "hf":
        builtin_model, _ = _build_hf_model(monkeypatch, micro_batch_size=2)
        builtin_model.model = builtin_network  # type: ignore[assignment]
        await builtin_model.forward(builtin_data, "lora", "cross_entropy", None, backward=True)
    else:
        fsdp_forward_backward(builtin_network, builtin_data, "cross_entropy", None, 2)

    for custom_parameter, builtin_parameter in zip(
        custom_network.parameters(), builtin_network.parameters(), strict=True
    ):
        torch.testing.assert_close(
            custom_parameter.grad, builtin_parameter.grad, rtol=1e-4, atol=1e-6
        )
