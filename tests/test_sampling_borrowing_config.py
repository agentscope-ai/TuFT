"""Minimal reactive borrowing configuration and independent routing."""

import pytest
from pydantic import ValidationError

from tuft.config import BorrowingConfig, ModelConfig, SamplingConfig
from tuft.sampling_routing import RoundRobinRouter


def test_minimal_config_defaults_and_round_trip():
    assert set(BorrowingConfig.model_fields) == {"enabled", "max_inflight", "max_request_tokens"}
    assert not SamplingConfig().borrowing.enabled
    assert BorrowingConfig().max_inflight == 1
    cfg = SamplingConfig.model_validate(
        {"borrowing": {"enabled": True, "max_request_tokens": 8192}}
    )
    assert SamplingConfig.model_validate_json(cfg.model_dump_json()) == cfg
    assert "policy" not in SamplingConfig.model_json_schema()["properties"]


@pytest.mark.parametrize(
    "field",
    [
        "cost_aware",
        "max_duration_s",
        "safety_factor",
        "probe_tokens",
        "probe_max_inflight",
        "min_backlog_tokens",
        "min_dwell_s",
        "sampling_alignment_s",
        "training_quiet_s",
        "training_priority_burst",
    ],
)
def test_retired_borrowing_options_fail_clearly(field):
    with pytest.raises(ValidationError, match=field):
        SamplingConfig.model_validate({"borrowing": {field: 1}})


@pytest.mark.parametrize(
    "field",
    [
        "merge_zero_adapters",
        "adapter_batch_window_s",
        "adapter_batch_max_requests",
        "locality_slack",
    ],
)
def test_retired_sampling_options_fail_clearly(field):
    with pytest.raises(ValidationError, match=field):
        SamplingConfig.model_validate({field: 1})


def test_legacy_enable_and_limits_normalize_without_enabling_duplicate_weights():
    cfg = SamplingConfig.model_validate(
        {
            "policy": "shared_weight",
            "flex_max_inflight": 2,
            "flex_max_tokens": 4096,
            "share_base_weights": True,
        }
    )
    assert cfg.borrowing == BorrowingConfig(enabled=True, max_inflight=2, max_request_tokens=4096)
    with pytest.raises(ValidationError):
        SamplingConfig.model_validate({"policy": "shared_weight", "share_base_weights": False})
    with pytest.raises(ValidationError):
        SamplingConfig.model_validate({"policy": "borrow", "borrowing": {"enabled": False}})
    with pytest.raises(ValidationError):
        SamplingConfig.model_validate({"policy": "balanced"})


def test_model_rejects_both_sampling_blocks():
    with pytest.raises(ValidationError, match="do not also supply"):
        ModelConfig.model_validate(
            {
                "model_name": "test",
                "model_path": "test",
                "max_model_len": 512,
                "sampling": {},
                "scheduling": {},
            }
        )


def test_router_only_sees_eligible_replicas():
    router = RoundRobinRouter(3)
    assert [router.choose([1, 2]) for _ in range(4)] == [1, 2, 1, 2]
    assert router.choose([0, 1, 2]) == 0
    assert router.choose([]) is None
    assert router.choose([0, 1, 2]) == 1
