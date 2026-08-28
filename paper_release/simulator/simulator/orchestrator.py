"""Orchestrator: launches all tenants concurrently and collects results."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any, Dict

from .backend import TrainingBackend
from .config import SimulatorConfig
from .metrics import TenantMetrics, build_output
from .request_logging import JsonlRequestLog
from .tasks import create_task
from .tenant import Tenant

logger = logging.getLogger(__name__)


def _create_backend(backend_type: str) -> TrainingBackend:
    """Factory for creating backend instances."""
    if backend_type == "tinker":
        from .backend.tinker_backend import TinkerBackend
        return TinkerBackend()
    if backend_type == "switch_tinker":
        from .backend.switch_tinker_backend import SwitchScheduledTinkerBackend
        return SwitchScheduledTinkerBackend()
    if backend_type == "mock":
        from .backend.mock_backend import MockBackend
        return MockBackend()
    raise ValueError(f"Unknown backend type: {backend_type}. Available: ['tinker', 'switch_tinker', 'mock']")


class Orchestrator:
    """Manages the full simulation lifecycle."""

    def __init__(self, config: SimulatorConfig):
        self.config = config
        self.backend: TrainingBackend = None
        self.tenants: list[Tenant] = []
        self._logprob_dump_fp = None
        self._request_log: JsonlRequestLog | None = None

    async def run(self) -> Dict[str, Any]:
        """Run the full simulation and return results."""
        logger.info("=" * 60)
        logger.info("Starting Multi-Tenant RL Training Simulator")
        logger.info("=" * 60)
        logger.info(f"Backend: {self.config.backend.type}")
        logger.info(f"Base model: {self.config.backend.base_model}")
        logger.info(f"Tenants: {len(self.config.tenants)}")

        # Initialize backend
        self.backend = _create_backend(self.config.backend.type)
        await self.backend.initialize(self.config.backend.to_dict())

        # Optional per-token logprob dump writer
        lp_cfg = self.config.logprob_collection
        writer = None
        if lp_cfg.enabled and lp_cfg.dump_per_token:
            dump_path = lp_cfg.output_path or (
                os.path.splitext(self.config.output_path)[0] + ".logprobs.jsonl"
            )
            os.makedirs(os.path.dirname(os.path.abspath(dump_path)) or ".", exist_ok=True)
            self._logprob_dump_fp = open(dump_path, "w")
            logger.info(f"Logprob per-token dump -> {dump_path}")

            def _writer(record: Dict[str, Any]) -> None:
                # Synchronous, single-threaded write. Tenants run in the same
                # event loop, so contention is naturally serialized.
                self._logprob_dump_fp.write(json.dumps(record) + "\n")

            writer = _writer

        # Optional request-arrival dump writer for RL-loop periodicity analysis
        req_log_cfg = self.config.request_arrival_logging
        request_log_writer = None
        if req_log_cfg.enabled:
            request_log_path = req_log_cfg.output_path or (
                os.path.splitext(self.config.output_path)[0] + ".requests.jsonl"
            )
            self._request_log = JsonlRequestLog(
                request_log_path,
                flush_every=req_log_cfg.flush_every,
            )
            request_log_writer = self._request_log.write
            logger.info(f"Request-arrival dump -> {request_log_path}")

        # Create tenants
        for tenant_cfg in self.config.tenants:
            task = create_task(tenant_cfg.task, seed=self.config.seed)
            tenant = Tenant(
                tenant_id=tenant_cfg.id,
                backend=self.backend,
                task=task,
                config=tenant_cfg,
                eval_config=self.config.evaluation,
                collect_logprobs=lp_cfg.enabled,
                logprob_dump_writer=writer,
                request_log_writer=request_log_writer,
            )
            self.tenants.append(tenant)
            logger.info(f"  Tenant '{tenant_cfg.id}': task={tenant_cfg.task}, "
                        f"rate={tenant_cfg.request_rate}/s, buffer={tenant_cfg.buffer_size}, "
                        f"steps={tenant_cfg.num_train_steps}")

        # Run all tenants concurrently (or sequentially for serial-style
        # baselines where only one tenant can own the cluster at a time).
        logger.info("-" * 60)
        logger.info("Launching all tenants...")
        t0 = time.time()

        if os.environ.get("SIM_SERIAL_SEQUENTIAL") == "1":
            for tenant in self.tenants:
                try:
                    await tenant.run()
                except Exception as e:
                    logger.error(f"[{tenant.id}] tenant failed: {e}")
        else:
            await asyncio.gather(
                *[tenant.run() for tenant in self.tenants],
                return_exceptions=True,
            )

        wall_clock = time.time() - t0

        # Completion accounting: a run is only comparable across systems when
        # every tenant finished its configured steps. Tenants that returned
        # early (dataset load failure, swallowed exception, ...) are listed so
        # downstream tooling can reject the run instead of silently comparing
        # partial work.
        incomplete = [
            t.id
            for t in self.tenants
            if t.train_steps_completed < t.config.num_train_steps
        ]

        # Cleanup
        logger.info("-" * 60)
        logger.info("Cleaning up resources...")
        for tenant in self.tenants:
            try:
                await self.backend.cleanup(tenant.id)
            except Exception as e:
                logger.warning(f"Cleanup failed for '{tenant.id}': {e}")

        # Close per-token logprob dump if open
        if self._logprob_dump_fp is not None:
            try:
                self._logprob_dump_fp.flush()
                self._logprob_dump_fp.close()
            except Exception as e:
                logger.warning(f"Failed to close logprob dump file: {e}")
            finally:
                self._logprob_dump_fp = None

        # Close request-arrival dump if open
        if self._request_log is not None:
            try:
                self._request_log.close()
            except Exception as e:
                logger.warning(f"Failed to close request-arrival dump file: {e}")
            finally:
                self._request_log = None

        # Build output
        tenant_metrics = [tenant.metrics for tenant in self.tenants]
        output = build_output(
            backend_type=self.config.backend.type,
            base_model=self.config.backend.base_model,
            wall_clock_seconds=wall_clock,
            tenant_metrics=tenant_metrics,
        )
        output["incomplete_tenants"] = incomplete
        output["all_tenants_completed"] = not incomplete
        for tenant in self.tenants:
            tenant_output = output["per_tenant"].setdefault(tenant.id, {})
            tenant_output["staleness_limit"] = tenant.config.staleness_limit
            tenant_output["staleness_pause_count"] = tenant.staleness_pause_count
            tenant_output["staleness_pause_seconds"] = round(tenant.staleness_pause_seconds, 3)
            tenant_output["request_rate"] = tenant.config.request_rate
            tenant_output["buffer_size"] = tenant.config.buffer_size
            tenant_output["async_sampling"] = tenant.config.async_sampling
            tenant_output["sync_mode"] = tenant.config.sync_mode

        logger.info("=" * 60)
        logger.info(f"Simulation complete in {wall_clock:.1f}s")
        for tenant in self.tenants:
            m = tenant.metrics
            logger.info(f"  [{tenant.id}] steps={m.train_steps_completed}, "
                        f"samples={m.total_samples}, "
                        f"accuracy={m.final_accuracy:.4f}, "
                        f"staleness={m.mean_staleness:.2f}, "
                        f"sampling={m.total_sampling_seconds:.1f}s, "
                        f"training={m.total_training_seconds:.1f}s, "
                        f"sync_weights={m.total_sync_weights_seconds:.1f}s")
        logger.info("=" * 60)

        return output

    async def run_and_save(self) -> None:
        """Run simulation and save results to file."""
        output = await self.run()

        output_path = self.config.output_path
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        logger.info(f"Results saved to {output_path}")
        if not output.get("all_tenants_completed", True):
            logger.error(
                "Run is INCOMPLETE: tenants %s did not finish their steps.",
                output.get("incomplete_tenants"),
            )
            raise SystemExit(3)
