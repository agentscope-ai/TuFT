from __future__ import annotations

from tuft.schedulers.rl_loop_scheduler import (
    CompositionAction,
    CompositionState,
    ContiguousTrainingScheduler,
    FastLoopStatus,
    FixedFlexCompositionController,
    LoopObservation,
    SamplingGroupKind,
    SamplingGroupState,
    SamplingRequest,
    SamplingSideScheduler,
    SlowLoopConfig,
    TenantLoopProfile,
    TenantLoopProfiler,
    apply_composition_action,
)


def test_loop_profiler_infers_fast_loop_profiles_from_observations() -> None:
    profiler = TenantLoopProfiler(smoothing_alpha=0.5)

    profiler.record(
        LoopObservation(
            tenant_id="tenant-a",
            sampling_submitted_s=0.0,
            training_ready_s=10.0,
            training_started_s=10.5,
            training_finished_s=14.5,
            staleness_window_s=6.0,
        )
    )
    assert profiler.profiles() == ()

    profiler.record(
        LoopObservation(
            tenant_id="tenant-a",
            sampling_submitted_s=20.0,
            training_ready_s=30.0,
            training_started_s=31.0,
            training_finished_s=34.0,
            staleness_window_s=4.0,
        )
    )

    profiles = profiler.profiles()
    assert len(profiles) == 1
    profile = profiles[0]
    assert profile.tenant_id == "tenant-a"
    assert profile.period_s == 20.0
    assert profile.training_burst_s == 3.5
    assert profile.ready_time_s == 50.0
    assert profile.staleness_window_s == 5.0


def test_fast_loop_packs_training_bursts_contiguously() -> None:
    scheduler = ContiguousTrainingScheduler()

    plan = scheduler.plan_fast_loop(
        [
            TenantLoopProfile(
                tenant_id="a",
                period_s=20.0,
                training_burst_s=4.0,
                ready_time_s=10.0,
                staleness_window_s=0.0,
            ),
            TenantLoopProfile(
                tenant_id="b",
                period_s=20.0,
                training_burst_s=3.0,
                ready_time_s=11.0,
                staleness_window_s=5.0,
            ),
            TenantLoopProfile(
                tenant_id="c",
                period_s=20.0,
                training_burst_s=2.0,
                ready_time_s=30.0,
                staleness_window_s=0.0,
            ),
        ]
    )

    assert plan.feasible
    got = [
        (item.tenant_id, item.sampling_delay_s, item.train_start_s)
        for item in plan.assignments
    ]
    assert got == [
        ("a", 0.0, 10.0),
        ("b", 3.0, 14.0),
        ("c", 0.0, 30.0),
    ]


def test_fast_loop_rejects_when_delay_exceeds_staleness_window() -> None:
    scheduler = ContiguousTrainingScheduler()

    plan = scheduler.plan_fast_loop(
        [
            TenantLoopProfile(
                tenant_id="a",
                period_s=20.0,
                training_burst_s=5.0,
                ready_time_s=0.0,
                staleness_window_s=0.0,
            ),
            TenantLoopProfile(
                tenant_id="b",
                period_s=20.0,
                training_burst_s=1.0,
                ready_time_s=1.0,
                staleness_window_s=2.0,
            ),
        ]
    )

    assert plan.status == FastLoopStatus.INFEASIBLE
    assert plan.infeasible_tenant_id == "b"
    assert plan.required_delay_s == 4.0
    assert plan.staleness_window_s == 2.0


def test_sampling_side_scheduler_merges_a0_and_reorders_stably_by_adapter() -> None:
    scheduler = SamplingSideScheduler()

    runs = scheduler.build_runs(
        [
            SamplingRequest("r1", "t1", "adapter-a", 1, 0, 1.0),
            SamplingRequest("r2", "t2", "adapter-b", 0, 1, 1.0),
            SamplingRequest("r3", "t3", "adapter-a", 1, 2, 1.0),
            SamplingRequest("r4", "t4", "adapter-c", 0, 3, 1.0),
        ]
    )

    assert [(run.adapter_key, run.request_ids) for run in runs] == [
        ("adapter-a", ("r1", "r3")),
        ("A0", ("r2", "r4")),
    ]


def test_sampling_side_scheduler_routes_runs_by_horizon_and_pin() -> None:
    scheduler = SamplingSideScheduler(adapter_pins={"adapter-a": "flex-0"})
    groups = [
        SamplingGroupState(
            "flex-0",
            SamplingGroupKind.FLEX,
            available_at_s=0.0,
            flex_flip_at_s=5.0,
        ),
        SamplingGroupState("fixed-0", SamplingGroupKind.FIXED, available_at_s=0.0),
    ]
    runs = scheduler.build_runs(
        [
            SamplingRequest("r1", "t1", "adapter-a", 1, 0, 2.0),
            SamplingRequest("r2", "t2", "adapter-b", 1, 1, 4.0),
            SamplingRequest("r3", "t2", "adapter-b", 1, 2, 4.0),
        ]
    )

    schedule = scheduler.route_runs(runs, groups)

    got = [(item.run.adapter_key, item.group_id, item.pinned) for item in schedule.routed_runs]
    assert got == [
        ("adapter-a", "flex-0", True),
        ("adapter-b", "fixed-0", False),
    ]
    assert schedule.rejected_runs == ()
    assert scheduler.adapter_pins["adapter-b"] == "fixed-0"


def test_sampling_side_scheduler_rejects_run_when_no_group_horizon_fits() -> None:
    scheduler = SamplingSideScheduler()
    schedule = scheduler.schedule(
        [SamplingRequest("r1", "t1", "adapter-a", 1, 0, 10.0)],
        [
            SamplingGroupState(
                "flex-0",
                SamplingGroupKind.FLEX,
                available_at_s=0.0,
                flex_flip_at_s=5.0,
            )
        ],
    )

    assert schedule.routed_runs == ()
    assert schedule.rejected_runs[0].request_ids == ("r1",)


def test_slow_loop_uses_hysteresis_and_keeps_one_flex_group() -> None:
    controller = FixedFlexCompositionController(
        SlowLoopConfig(
            high_pressure_threshold=0.75,
            low_pressure_threshold=0.25,
            smoothing_alpha=1.0,
            min_dwell_s=10.0,
            fixed_to_flex_cost_s=3.0,
            flex_to_fixed_cost_s=1.0,
        )
    )
    state = CompositionState(fixed_groups=1, flex_groups=1, last_conversion_s=0.0)

    controller.observe_pressure(training_demand_s=90.0, period_s=100.0)
    blocked_by_dwell = controller.decide(state, now_s=5.0, expected_gain_s=10.0)
    assert blocked_by_dwell.action == CompositionAction.HOLD

    grow = controller.decide(state, now_s=11.0, expected_gain_s=10.0)
    assert grow.action == CompositionAction.FIXED_TO_FLEX
    state = apply_composition_action(state, grow.action, now_s=11.0)
    assert state.fixed_groups == 0
    assert state.flex_groups == 2

    controller.observe_pressure(training_demand_s=10.0, period_s=100.0)
    shrink = controller.decide(state, now_s=30.0, expected_gain_s=10.0)
    assert shrink.action == CompositionAction.FLEX_TO_FIXED
    state = apply_composition_action(state, shrink.action, now_s=30.0)
    assert state.fixed_groups == 1
    assert state.flex_groups == 1

    hold = controller.decide(state, now_s=50.0, expected_gain_s=10.0)
    assert hold.action == CompositionAction.HOLD


def test_fast_loop_pack_training_merges_idle_into_blocks() -> None:
    scheduler = ContiguousTrainingScheduler()
    profiles = [
        TenantLoopProfile(
            tenant_id="a",
            period_s=20.0,
            training_burst_s=2.0,
            ready_time_s=10.0,
            staleness_window_s=3.0,
        ),
        TenantLoopProfile(
            tenant_id="b",
            period_s=20.0,
            training_burst_s=2.0,
            ready_time_s=12.0,
            staleness_window_s=5.0,
        ),
    ]

    unpacked = scheduler.plan_fast_loop(profiles)
    assert [ (p.tenant_id, p.train_start_s) for p in unpacked.assignments ] == [
        ("a", 10.0),
        ("b", 12.0),
    ]

    packed = scheduler.plan_fast_loop(profiles, pack_training=True, pack_horizon_s=2.0)
    # "a" is held until "b" is ready so the two run back-to-back (12->14->16),
    # merging the 10->12 idle into one flex-sampling window.
    got = [(p.tenant_id, p.sampling_delay_s, p.train_start_s) for p in packed.assignments]
    assert got == [("a", 2.0, 12.0), ("b", 2.0, 14.0)]


def test_fast_loop_pack_training_respects_sync_staleness_window() -> None:
    scheduler = ContiguousTrainingScheduler()
    profiles = [
        TenantLoopProfile(
            tenant_id="a",
            period_s=20.0,
            training_burst_s=2.0,
            ready_time_s=10.0,
            staleness_window_s=0.0,  # sync tenant: cannot be held
        ),
        TenantLoopProfile(
            tenant_id="b",
            period_s=20.0,
            training_burst_s=2.0,
            ready_time_s=12.0,
            staleness_window_s=5.0,
        ),
    ]
    packed = scheduler.plan_fast_loop(profiles, pack_training=True, pack_horizon_s=2.0)
    got = [(p.tenant_id, p.train_start_s) for p in packed.assignments]
    assert got == [("a", 10.0), ("b", 12.0)]
