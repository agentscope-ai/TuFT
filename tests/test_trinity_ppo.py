"""CPU checks against an independent per-token PPO reference and both backends."""

import copy
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from tinker import types
from tinker.proto import tinker_public_pb2 as public_pb
from tinker.proto.request_conv import forward_backward_request_to_proto

from tuft.backends.fsdp_engine import forward_backward
from tuft.backends.hf_training_model import HFTrainingModel
from tuft.compat import decode_forward_backward_request
from tuft.exceptions import LossFunctionMissingInputException
from tuft.loss_fn import get_loss_fn


CONFIG = {"clip_range": 0.2, "clip_ratio_c": 3.0, "kl_coef": 0.03, "num_total_datums": 3}


def _reference(current, old, advantages, mask, reference):
    # Scalar branches express PPO's positive/negative-advantage cases without
    # relying on the server's tensor implementation or reduction helper.
    terms = []
    for row in range(current.shape[0]):
        tokens = []
        for col in range(current.shape[1]):
            if not mask[row, col]:
                continue
            advantage = advantages[row, col]
            ratio = (current[row, col] - old[row, col]).clamp(-20, 20).exp()
            if advantage >= 0:
                pg = -advantage * torch.minimum(ratio, ratio.new_tensor(1.2))
            else:
                pg = -advantage * torch.clamp(ratio, 0.8, 3.0)
            kl = 0.015 * (current[row, col] - reference[row, col]).square()
            tokens.append(pg + kl)
        terms.append(torch.stack(tokens).mean() if tokens else current[row].sum() * 0)
    return torch.stack(terms).sum() / CONFIG["num_total_datums"]


def _inputs():
    return {
        "target_logprobs": torch.tensor(
            [[-3.0, -0.1, -1.0, -2.0], [-1.0, -4.0, -0.1, -2.0], [-2.0] * 4],
            dtype=torch.float64,
            requires_grad=True,
        ),
        "logprobs": torch.full((3, 4), -2.0, dtype=torch.float64),
        "advantages": torch.tensor([[0.0, 1.0, -1.0, 2.0], [0.0, -2.0, -1.0, 0.0], [0.0] * 4]),
        "weights": torch.tensor([[0.0, 1.0, 1.0, 1.0], [0.0, 1.0, 1.0, 0.0], [0.0] * 4]),
        "ref_logprobs": torch.full((3, 4), -2.5, dtype=torch.float64),
    }


def test_loss_and_gradients_match_scalar_reference_and_accumulated_chunks():
    inputs = _inputs()
    expected = _reference(
        inputs["target_logprobs"],
        inputs["logprobs"],
        inputs["advantages"],
        inputs["weights"],
        inputs["ref_logprobs"],
    )
    expected_grad = torch.autograd.grad(expected, inputs["target_logprobs"])[0]
    loss_fn = get_loss_fn("trinity_ppo")
    loss, metrics = loss_fn(inputs, CONFIG)
    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(
        torch.autograd.grad(loss, inputs["target_logprobs"])[0], expected_grad
    )
    accumulated = inputs["target_logprobs"].sum() * 0
    partial_metrics = []
    for start, end in [(0, 2), (2, 3)]:
        part, diagnostic = loss_fn({key: value[start:end] for key, value in inputs.items()}, CONFIG)
        accumulated = accumulated + part
        partial_metrics.append(diagnostic)
    torch.testing.assert_close(accumulated, expected)
    torch.testing.assert_close(
        torch.autograd.grad(accumulated, inputs["target_logprobs"])[0], expected_grad
    )
    for key, value in metrics.items():
        assert sum(part[key] for part in partial_metrics) == pytest.approx(value)


def test_mask_overrides_weights_and_all_masked_nonfinite_tokens_are_harmless():
    inputs = _inputs()
    inputs["mask"] = inputs["weights"].clone()
    inputs["weights"] = torch.ones_like(inputs["weights"])
    masked = inputs["mask"] == 0
    with torch.no_grad():
        for key in ["target_logprobs", "logprobs", "advantages", "ref_logprobs"]:
            inputs[key][masked] = float("nan")
    loss, metrics = get_loss_fn("trinity_ppo")(inputs, CONFIG)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
    gradient = inputs["target_logprobs"].grad
    assert gradient is not None
    assert torch.all(gradient[masked] == 0)
    assert metrics["trinity/response_tokens:sum"] == 5


@pytest.mark.parametrize("count", [0, -1, 2, 3.5, float("inf"), float("nan")])
def test_requires_valid_global_batch_denominator(count):
    with pytest.raises(ValueError, match="num_total_datums"):
        get_loss_fn("trinity_ppo")(_inputs(), {**CONFIG, "num_total_datums": count})


def test_requires_response_mask_and_reference_for_kl():
    inputs = _inputs()
    inputs.pop("weights")
    with pytest.raises(LossFunctionMissingInputException):
        get_loss_fn("trinity_ppo")(inputs, CONFIG)
    inputs = _inputs()
    inputs.pop("ref_logprobs")
    with pytest.raises(LossFunctionMissingInputException):
        get_loss_fn("trinity_ppo")(inputs, CONFIG)
    loss, _ = get_loss_fn("trinity_ppo")(inputs, {**CONFIG, "kl_coef": 0})
    assert torch.isfinite(loss)


class _TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.table = torch.nn.Embedding(12, 12)

    def forward(self, input_ids, **_kwargs):
        return SimpleNamespace(logits=self.table(input_ids))


def _datums():
    datums = []
    for tokens in [[1, 2, 3, 4], [5, 6], [7, 8, 9]]:
        length = len(tokens)
        mask = [0] + [1] * (length - 1)
        datums.append(
            types.Datum(
                model_input=types.ModelInput.from_ints(tokens),
                loss_fn_inputs={
                    "target_tokens": types.TensorData(
                        data=[token + 1 for token in tokens], dtype="int64", shape=[length]
                    ),
                    "weights": types.TensorData(data=mask, dtype="float32", shape=[length]),
                    "logprobs": types.TensorData(
                        data=[-2.0] * length, dtype="float32", shape=[length]
                    ),
                    "advantages": types.TensorData(
                        data=[(-1.0) ** i for i in range(length)], dtype="float32", shape=[length]
                    ),
                    "ref_logprobs": types.TensorData(
                        data=[-2.5] * length, dtype="float32", shape=[length]
                    ),
                },
            )
        )
    return datums


@pytest.mark.parametrize("micro_batch_size", [1, 2, 3])
async def test_hf_fsdp_and_direct_model_gradients_match(micro_batch_size):
    torch.manual_seed(13)
    fsdp_model = _TinyModel()
    hf_network = copy.deepcopy(fsdp_model)
    direct = copy.deepcopy(fsdp_model)
    data = _datums()
    terms = []
    for datum in data:
        tokens = torch.tensor(datum.model_input.to_ints())
        targets = datum.loss_fn_inputs["target_tokens"].to_torch()
        logprobs = direct(tokens).logits.log_softmax(-1).gather(-1, targets[:, None]).squeeze(-1)
        fields = {key: value.to_torch()[None] for key, value in datum.loss_fn_inputs.items()}
        terms.append(
            _reference(
                logprobs[None],
                fields["logprobs"],
                fields["advantages"],
                fields["weights"],
                fields["ref_logprobs"],
            )
        )
    expected = torch.stack(terms).sum()
    expected.backward()
    out = forward_backward(fsdp_model, data, "trinity_ppo", CONFIG, micro_batch_size)
    hf = HFTrainingModel.__new__(HFTrainingModel)
    hf.model = cast(Any, hf_network)
    hf_loss = 0.0
    for start in range(0, len(data), micro_batch_size):
        value, _, _ = await hf._forward_micro_batch(
            data[start : start + micro_batch_size], get_loss_fn("trinity_ppo"), CONFIG, True
        )
        hf_loss += value
    assert out["metrics"]["loss:sum"] == pytest.approx(expected.item(), abs=1e-6)
    assert hf_loss == pytest.approx(expected.item(), abs=1e-6)
    torch.testing.assert_close(fsdp_model.table.weight.grad, direct.table.weight.grad)
    torch.testing.assert_close(hf_network.table.weight.grad, direct.table.weight.grad)


def test_fsdp_validates_later_rows_before_accumulating_gradients():
    model = _TinyModel()
    data = _datums()
    del data[-1].loss_fn_inputs["logprobs"]
    with pytest.raises(ValueError, match="logprobs"):
        forward_backward(model, data, "trinity_ppo", CONFIG, 1)
    assert model.table.weight.grad is None


def test_registered_loss_name_passes_protobuf_decoder():
    request = public_pb.ForwardBackwardRequest(model_id="m", seq_id=1, loss_fn="trinity_ppo")
    assert decode_forward_backward_request(request.SerializeToString()).loss_fn == "trinity_ppo"


def test_sdk_encoder_round_trips_registered_loss_and_inputs():
    request = types.ForwardBackwardRequest(
        model_id="m",
        seq_id=1,
        forward_backward_input=types.ForwardBackwardInput(
            data=_datums(),
            loss_fn=cast(types.LossFnType, "trinity_ppo"),
            loss_fn_config=CONFIG,
        ),
    )
    decoded = decode_forward_backward_request(
        forward_backward_request_to_proto(request).SerializeToString()
    )
    assert decoded.loss_fn == "trinity_ppo"
    assert decoded.loss_fn_config == pytest.approx(CONFIG)
    assert len(decoded.data) == 3
    assert decoded.data[0].loss_fn_inputs["weights"].data == [0, 1, 1, 1]
