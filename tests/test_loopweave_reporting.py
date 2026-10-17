"""Historical duplicate borrowing must never be reported as physical sharing."""

import pytest

from scripts.summarize_loopweave import comparison_runs, performance_data


def test_historical_comparison_excludes_duplicate_borrowing_without_mutating_data():
    static = {"policy": "round_robin"}
    duplicate = {"policy": "loopweave", "shared_weights": None}
    shared = {"policy": "loopweave_shared", "shared_weights": {"awake": {}}}
    data = {"runs": [static, duplicate, shared]}
    selected = comparison_runs(data)
    assert selected == [static, {**shared, "policy": "loopweave"}]
    assert data["runs"] == [static, duplicate, shared]
    assert shared["policy"] == "loopweave_shared"


def test_current_static_comparison_requires_alias_evidence():
    rows = [
        {"policy": "round_robin"},
        {"policy": "loopweave", "shared_weights": {"awake": {}}},
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

    from scripts.benchmark_loopweave import workload_scheduling
    from tuft.config import SchedulingConfig
    from tuft.scheduling import ResourceScheduler

    scheduler = ResourceScheduler(
        2,
        SchedulingConfig.model_validate(workload_scheduling("loopweave", tenants, samples, 128)),
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
