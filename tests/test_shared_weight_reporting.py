"""Reports require evidence of physical sharing and complete measurements."""

import pytest

from scripts.summarize_shared_weight import comparison_runs, performance_data


def test_comparison_does_not_mutate_input_data():
    static = {"policy": "round_robin"}
    shared = {"policy": "shared_weight", "shared_weights": {"awake": {}}}
    data = {"runs": [static, shared]}
    selected = comparison_runs(data)
    assert selected == [static, {**shared, "policy": "shared_weight"}]
    assert data["runs"] == [static, shared]
    assert shared["policy"] == "shared_weight"


def test_current_static_comparison_requires_alias_evidence():
    rows = [
        {"policy": "round_robin"},
        {"policy": "shared_weight", "shared_weights": {"awake": {}}},
    ]
    assert comparison_runs({"runs": rows}) == rows
    rows[1]["shared_weights"] = None
    with pytest.raises(ValueError, match="physically shared"):
        comparison_runs({"runs": rows})


def test_bounded_job_reporting_rejects_failed_or_timed_out_partial_results():
    data = {"runs": [{"policy": "round_robin"}]}
    assert performance_data({"returncode": 0, "benchmark": data}) is data
    assert performance_data(data) is data
    for failure in ({"returncode": 1}, {"timed_out": True}):
        with pytest.raises(ValueError, match="Successful"):
            performance_data({**failure, "benchmark": data})


@pytest.mark.parametrize("tenants,samples", [(8, 16), (2, 8)])
async def test_gpu_workload_has_ready_backlog_for_cold_borrowing(tenants, samples):
    import asyncio

    from scripts.benchmark_shared_weight import workload_scheduling
    from tuft.config import SchedulingConfig
    from tuft.scheduling import ResourceScheduler

    scheduler = ResourceScheduler(
        2,
        SchedulingConfig.model_validate(
            workload_scheduling("shared_weight", tenants, samples, 128)
        ),
        flex_index=0,
    )
    release, borrowed = asyncio.Event(), asyncio.Event()
    destinations = []

    async def sample(index):
        destinations.append(index)
        if index == 0:
            borrowed.set()
        await release.wait()

    jobs = [
        asyncio.create_task(scheduler.sample(str(i % tenants), str(i % tenants), 160, True, sample))
        for i in range(tenants * samples)
    ]
    try:
        await asyncio.wait_for(borrowed.wait(), 2)
        assert destinations.count(0) == 1  # A single bounded calibration probe.
        assert scheduler.snapshot()["queued"] > 0
    finally:
        release.set()
        await asyncio.gather(*jobs)
        await scheduler.close()


def test_agent_summary_separates_counters_from_gauges_and_estimates():
    import json
    from pathlib import Path

    from scripts.report_shared_weight_agents import summarize

    data = json.loads(
        (Path(__file__).parents[1] / "benchmarks/shared-weight/agent-telemetry.json").read_text()
    )
    rows = summarize(data)
    for run, row in zip(data["runs"], rows, strict=True):
        assert row["scheduler_delta"]["completed"] == 576
        for key in ("publication_estimate_s", "queued", "peak_queued", "loop_observations"):
            assert key not in row["scheduler_delta"]
            assert row["scheduler_state"][key] == run["scheduler_after"][key]
