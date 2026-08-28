from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

from tuft.backends.sampling_pipeline import SamplingPipelineBackend


class _FakeInner:
    """Records dispatch order so L2 run contiguity is observable."""

    def __init__(self, exec_s: float = 0.02) -> None:
        self.exec_s = exec_s
        self.order: list[str] = []

    async def async_init(self) -> None:
        return None

    async def sample(self, **kwargs):
        key = kwargs.get("lora_id") or "A0"
        self.order.append(key)
        await asyncio.sleep(self.exec_s)
        return SimpleNamespace(sequences=[])


def _make(inner, hold_s: float) -> SamplingPipelineBackend:
    cfg = SimpleNamespace(model_name="test-model")
    pipe = SamplingPipelineBackend(cfg, inner, l2_coalesce_max_hold_s=hold_s)
    # Non-zero versions so L1 does not merge everything into A0.
    pipe.notify_adapter_version("a", 3)
    pipe.notify_adapter_version("b", 3)
    return pipe


def _params(max_tokens: int = 8):
    return SimpleNamespace(max_tokens=max_tokens)


def test_l2_disabled_keeps_arrival_order_and_counts_runs() -> None:
    async def main() -> None:
        inner = _FakeInner(exec_s=0.0)
        pipe = _make(inner, hold_s=0.0)
        for key in ("a", "b", "a"):
            await pipe.sample(
                prompt=None, num_samples=1, sampling_params=_params(), lora_id=key
            )
        assert inner.order == ["a", "b", "a"]
        # Every key change starts a new dispatch run when coalescing is off.
        assert pipe.stats["l2_dispatch_runs"] == 3

    asyncio.run(main())


def test_l2_holds_a_run_switch_until_the_current_run_drains() -> None:
    async def main() -> None:
        inner = _FakeInner()
        pipe = _make(inner, hold_s=1.0)
        # Pretend one "a" request is executing downstream.
        pipe._run_key = "a"
        pipe._enter_run("a")

        async def drain() -> None:
            await asyncio.sleep(0.05)
            pipe._leave_run("a")

        drainer = asyncio.ensure_future(drain())
        t0 = time.monotonic()
        await pipe._coalesce_run("b")
        elapsed = time.monotonic() - t0
        await drainer
        assert pipe._run_key == "b"
        assert pipe.stats["l2_coalesce_holds"] > 0
        # Switched once "a" drained, well before the 1s bound.
        assert 0.03 <= elapsed < 0.9

    asyncio.run(main())


def test_l2_hold_is_bounded_so_a_stuck_run_cannot_starve_others() -> None:
    async def main() -> None:
        inner = _FakeInner()
        pipe = _make(inner, hold_s=0.05)
        pipe._run_key = "a"
        pipe._enter_run("a")  # never drains

        t0 = time.monotonic()
        await pipe._coalesce_run("b")
        elapsed = time.monotonic() - t0
        assert pipe._run_key == "b"
        assert elapsed < 0.5

    asyncio.run(main())


def test_l2_makes_interleaved_arrivals_dispatch_as_contiguous_runs() -> None:
    async def main() -> None:
        inner = _FakeInner(exec_s=0.03)
        pipe = _make(inner, hold_s=1.0)

        async def one(key: str) -> None:
            await pipe.sample(
                prompt=None, num_samples=1, sampling_params=_params(), lora_id=key
            )

        # Interleaved arrival order a, b, a, b -> should dispatch as a,a,b,b.
        tasks = [asyncio.ensure_future(one(k)) for k in ("a", "b", "a", "b")]
        await asyncio.gather(*tasks)

        runs = 1 + sum(
            1 for prev, cur in zip(inner.order, inner.order[1:], strict=False) if prev != cur
        )
        assert len(inner.order) == 4
        # Without L2 this interleaving yields 4 runs; coalescing must reduce it.
        assert runs < 4, inner.order
        # In-flight bookkeeping must not leak once everything finished.
        assert pipe._inflight_by_key == {}

    asyncio.run(main())


def test_unknown_adapter_keeps_its_lora_instead_of_serving_the_base_model() -> None:
    """Regression: sampling keys are session ids while versions are reported per
    training run. When the two never met, L1 defaulted to "version 0" and
    dropped the LoRA, so every rollout silently came from the base model."""

    async def main() -> None:
        inner = _FakeInner(exec_s=0.0)
        cfg = SimpleNamespace(model_name="test-model")
        pipe = SamplingPipelineBackend(cfg, inner)
        # Version reported under the training run id, request keyed by session id.
        pipe.notify_adapter_version("run-7", 4)
        await pipe.sample(
            prompt=None,
            num_samples=1,
            sampling_params=_params(),
            lora_id="session-xyz",
        )
        assert inner.order == ["session-xyz"], inner.order
        assert pipe.stats["a0_merged_requests"] == 0
        assert pipe.stats["unknown_version_requests"] == 1

    asyncio.run(main())


def test_alias_lets_l1_resolve_session_keys_to_training_versions() -> None:
    async def main() -> None:
        inner = _FakeInner(exec_s=0.0)
        cfg = SimpleNamespace(model_name="test-model")
        pipe = SamplingPipelineBackend(cfg, inner)
        pipe.notify_adapter_alias("session-a", "run-a")
        pipe.notify_adapter_alias("session-b", "run-b")
        # run-a is still untrained -> L1 may merge it into the base path.
        pipe.notify_adapter_version("run-a", 0)
        # run-b has trained -> its LoRA must be used.
        pipe.notify_adapter_version("run-b", 2)

        for key in ("session-a", "session-b"):
            await pipe.sample(
                prompt=None, num_samples=1, sampling_params=_params(), lora_id=key
            )
        assert inner.order == ["A0", "session-b"], inner.order
        assert pipe.stats["a0_merged_requests"] == 1
        assert pipe.stats["unknown_version_requests"] == 0

    asyncio.run(main())
