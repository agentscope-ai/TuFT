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
from .metrics import build_output
from .tasks import create_task
from .tenant import Tenant


logger = logging.getLogger(__name__)


def _create_backend(backend_type: str) -> TrainingBackend:
    """Factory for creating backend instances."""
    if backend_type == "tinker":
        from .backend.tinker_backend import TinkerBackend

        return TinkerBackend()
    raise ValueError(f"Unknown backend type: {backend_type}. Available: ['tinker']")


class Orchestrator:
    """Manages the full simulation lifecycle."""

    def __init__(self, config: SimulatorConfig):
        self.config = config
        self.backend: TrainingBackend = None
        self.tenants: list[Tenant] = []
        self._logprob_dump_fp = None

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
            )
            self.tenants.append(tenant)
            logger.info(
                f"  Tenant '{tenant_cfg.id}': task={tenant_cfg.task}, "
                f"rate={tenant_cfg.request_rate}/s, buffer={tenant_cfg.buffer_size}, "
                f"steps={tenant_cfg.num_train_steps}"
            )

        # Run all tenants concurrently
        logger.info("-" * 60)
        logger.info("Launching all tenants...")
        t0 = time.time()

        await asyncio.gather(
            *[tenant.run() for tenant in self.tenants],
            return_exceptions=True,
        )

        wall_clock = time.time() - t0

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

        # Build output
        tenant_metrics = [tenant.metrics for tenant in self.tenants]
        output = build_output(
            backend_type=self.config.backend.type,
            base_model=self.config.backend.base_model,
            wall_clock_seconds=wall_clock,
            tenant_metrics=tenant_metrics,
        )

        logger.info("=" * 60)
        logger.info(f"Simulation complete in {wall_clock:.1f}s")
        for tenant in self.tenants:
            m = tenant.metrics
            logger.info(
                f"  [{tenant.id}] steps={m.train_steps_completed}, "
                f"samples={m.total_samples}, "
                f"accuracy={m.final_accuracy:.4f}, "
                f"staleness={m.mean_staleness:.2f}, "
                f"sampling={m.total_sampling_seconds:.1f}s, "
                f"training={m.total_training_seconds:.1f}s, "
                f"sync_weights={m.total_sync_weights_seconds:.1f}s"
            )
        logger.info("=" * 60)

        return output

    async def run_and_save(self) -> None:
        """Run simulation and save results to file."""
        output = await self.run()

        output_path = self.config.output_path
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        logger.info(f"Results saved to {output_path}")
