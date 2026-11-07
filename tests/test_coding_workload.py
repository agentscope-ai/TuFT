"""CPU checks for task contracts and trusted test-harness fixtures."""

import inspect
import json
import subprocess
import sys

import pytest
from coding_tasks import HARNESS, cases, extract_code, policy, reference


@pytest.mark.parametrize("stage", range(4))
def test_reference_program_passes_harness(stage):
    rules = policy(3, 2, 1)
    code = inspect.getsource(reference) + (
        f"\ndef build_report(orders):\n    return reference(orders, {stage}, {rules!r})\n"
    )
    fixtures = cases(3, 2, 1, stage)
    # Only our trusted reference is executed locally; model output runs in Modal Sandboxes.
    result = subprocess.run(
        [sys.executable, "-I", "-c", HARNESS, json.dumps(dict(code=code, cases=fixtures))],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout) == dict(passed=len(fixtures), total=len(fixtures), failures=[])


def test_validation_rounding_and_no_mutation():
    rows = [
        dict(customer=" a ", sku=" x ", qty=3, price_cents=101),
        dict(customer="a", sku="x", qty=True, price_cents=100),
        dict(customer="a", sku="x", qty=1, price_cents=100, status="cancelled"),
    ]
    saved = json.dumps(rows)
    result = reference(rows, 3, dict(discount_qty=3, discount_bps=500, tax_bps=625))
    assert result == dict(
        orders=1,
        units=3,
        gross_cents=303,
        customers=["a"],
        by_sku={"x": 3},
        rejected=1,
        cancelled=1,
        discount_cents=15,
        net_cents=288,
        tax_cents=18,
        total_cents=306,
        by_customer={"a": dict(orders=1, units=3, total_cents=306)},
    )
    assert json.dumps(rows) == saved


def test_code_extraction_and_deterministic_tasks():
    assert (
        extract_code("```python\ndef build_report(x): return {}\n```")
        == "def build_report(x): return {}"
    )
    assert cases(1, 2, 3, 3) == cases(1, 2, 3, 3)
    assert cases(1, 2, 3, 3) != cases(2, 2, 3, 3)


def test_constants_are_defined_and_old_drafts_are_discarded():
    from coding_tasks import MILESTONES, constants, next_messages

    code = "def build_report(x): return DISCOUNT_QTY + DISCOUNT_BPS + TAX_BPS"
    config = constants(2, 1, 0)
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            HARNESS,
            json.dumps(
                dict(
                    code=code,
                    constants=config,
                    cases=[dict(input=[], expected=sum(config.values()))],
                )
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout)["passed"] == 1
    messages = next_messages(2, 1, 0, 2, "latest complete file", {"passed": 10, "total": 12})
    assert len(messages) == 4
    assert messages[2]["content"] == "latest complete file"
    assert all(rule in messages[1]["content"] for rule in MILESTONES[:3])
    assert MILESTONES[3] in messages[3]["content"]


def test_coding_sampler_configuration_matches_but_borrowing_is_bounded():
    from benchmark_shared_weight_agents import scheduling

    fixed, shared = scheduling("round_robin", "coding"), scheduling("shared_weight", "coding")
    assert fixed["max_inflight_per_replica"] == shared["max_inflight_per_replica"] == 32
    assert shared["borrowing"]["max_inflight"] == 4
    assert {k: v for k, v in fixed.items() if k != "borrowing"} == {
        k: v for k, v in shared.items() if k != "borrowing"
    }


def test_engine_metrics_report_real_batch_counts_without_adapter_names(monkeypatch):
    import prometheus_client
    from prometheus_client import CollectorRegistry, Gauge

    from tuft.backends.vllm_engine import VLLMEngine

    registry = CollectorRegistry()
    Gauge("vllm:num_requests_running", "running", registry=registry).set(32)
    info = Gauge(
        "vllm:lora_requests_info", "adapters", ["running_lora_adapters"], registry=registry
    )
    info.labels("old").set(1)
    info.labels("a,b,c").set(2)
    monkeypatch.setattr(prometheus_client, "REGISTRY", registry)
    metrics = VLLMEngine.__new__(VLLMEngine).sampling_metrics()
    assert metrics == {"vllm:num_requests_running": 32, "running_lora_adapters": 3}


def test_agent_repairs_failed_milestones_before_advancing():
    from coding_tasks import CodingProgress, next_messages

    progress = CodingProgress()
    assert progress.record(dict(passed=6, total=6)) == 1
    assert progress.stage == 1
    assert progress.record(dict(passed=9, total=11)) == 9 / 11
    assert progress.stage == 1 and not progress.done
    repair = next_messages(0, 0, 0, 1, "current code", {"passed": 9, "total": 11}, advance=False)
    assert "Fix the failed tests" in repair[-1]["content"]
    assert "Next milestone" not in repair[-1]["content"]
    for total in (11, 12, 12):
        progress.record(dict(passed=total, total=total))
    assert progress.done and progress.stage == 4 and progress.attempts == 5
    assert progress.scores == [1, 1, 1, 1]
    with pytest.raises(ValueError, match="already finished"):
        progress.record(dict(passed=12, total=12))


def test_agent_attempt_budget_bounds_repeated_failures():
    from coding_tasks import MAX_ATTEMPTS, CodingProgress

    progress = CodingProgress()
    for _ in range(MAX_ATTEMPTS):
        progress.record(dict(passed=0, total=6))
    assert progress.done and progress.stage == 0 and progress.attempts == MAX_ATTEMPTS


def test_feedback_identifies_incorrect_fields():
    payload = dict(
        code='def build_report(x): return {"rejected":0,"nested":{"tax":99}}',
        cases=[dict(input=[], expected=dict(rejected=1, nested=dict(tax=12)))],
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", HARNESS, json.dumps(payload)],
        check=True,
        capture_output=True,
        text=True,
    )
    error = json.loads(result.stdout)["failures"][0]["error"]
    assert "result.rejected: expected=1 actual=0" in error
    assert "result.nested.tax: expected=12 actual=99" in error


def test_benchmark_warmup_restores_nonzero_adapters_and_optimizer_state():
    import torch
    from benchmark_shared_weight_agents import (
        capture_benchmark_adapters,
        restore_benchmark_adapters,
    )

    model = torch.nn.Module()
    model.lora_A = torch.nn.ModuleDict({name: torch.nn.Linear(2, 2) for name in ("a", "b")})
    model.lora_B = torch.nn.ModuleDict({name: torch.nn.Linear(2, 2) for name in ("a", "b")})
    optimizers = {
        run: torch.optim.AdamW([p for n, p in model.named_parameters() if f".{run}." in n])
        for run in ("a", "b")
    }
    saved = capture_benchmark_adapters(model, optimizers, ["a", "b"])
    torch.stack([p.square().sum() for p in model.parameters()]).sum().backward()
    for optimizer in optimizers.values():
        optimizer.step()
        assert optimizer.state
    restore_benchmark_adapters(model, optimizers, saved)
    for name, parameter in model.named_parameters():
        assert torch.equal(parameter, saved[0][name])
        assert parameter.grad is None
        if "lora_B." in name:
            assert torch.count_nonzero(parameter) == parameter.numel()
    assert all(not optimizer.state for optimizer in optimizers.values())
