"""FSDPWorkerGroup — manages multiple FSDPTrainingWorker Ray actors."""

from __future__ import annotations

import logging
import socket
from typing import Any

import ray
from ray.util.placement_group import (
    PlacementGroupSchedulingStrategy,
    placement_group,
)
from tinker import types

from tuft.config import ModelConfig
from tuft.loss_fn import metrics_reduction


logger = logging.getLogger(__name__)


class FSDPWorkerGroup:
    """Create and coordinate a group of ``FSDPTrainingWorker`` Ray actors."""

    def __init__(
        self,
        config: ModelConfig,
        num_gpus_per_node: int,
        num_nodes: int = 1,
    ) -> None:
        self.config = config
        self.total_gpus = num_gpus_per_node * num_nodes
        self.num_gpus_per_node = num_gpus_per_node
        self.num_nodes = num_nodes
        self.workers: list[Any] = []

        master_addr, master_port = self._get_master_addr_port()

        # --- placement groups ---
        pgs = []
        for _ in range(num_nodes):
            bundles = [{"GPU": 1, "CPU": 1}] * num_gpus_per_node
            pg = placement_group(bundles, strategy="STRICT_PACK")
            ray.get(pg.ready())
            pgs.append(pg)

        pgs = self._sort_pgs_by_ip(pgs)

        # --- create worker actors ---
        from tuft.backends.fsdp_training_worker import FSDPTrainingWorker

        RemoteWorker = ray.remote(num_gpus=1)(FSDPTrainingWorker)

        rank = 0
        for pg in pgs:
            for bundle_idx in range(num_gpus_per_node):
                worker = RemoteWorker.options(
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=pg,
                        placement_group_bundle_index=bundle_idx,
                    ),
                ).remote(config, rank, self.total_gpus, master_addr, master_port)
                self.workers.append(worker)
                rank += 1

        logger.info(
            "FSDPWorkerGroup created: %d workers across %d nodes",
            self.total_gpus,
            num_nodes,
        )

    # ------------------------------------------------------------------
    # Collective operations — call every worker and wait
    # ------------------------------------------------------------------
    def forward_all(
        self,
        data: list[types.Datum],
        lora_id: str,
        loss_fn: types.LossFnType,
        loss_fn_config: dict[str, float] | None,
        backward: bool,
    ) -> types.ForwardBackwardOutput:
        actual_sizes = self._actual_shard_sizes(len(data), self.total_gpus)
        shards = self._split_data(data, self.total_gpus)
        futures = [
            w.forward.remote(shard, lora_id, loss_fn, loss_fn_config, backward)
            for w, shard in zip(self.workers, shards, strict=True)
        ]
        results: list[types.ForwardBackwardOutput] = ray.get(futures)
        return self._merge_forward_results(results, actual_sizes)

    def optim_step_all(
        self,
        adam_params: types.AdamParams,
        lora_id: str,
    ) -> types.OptimStepResponse:
        futures = [w.optim_step.remote(adam_params, lora_id) for w in self.workers]
        ray.get(futures)
        return types.OptimStepResponse()

    def create_adapter_all(
        self,
        lora_id: str,
        lora_config: types.LoraConfig,
    ) -> None:
        futures = [w.create_adapter.remote(lora_id, lora_config) for w in self.workers]
        ray.get(futures)

    def remove_adapter_all(self, lora_id: str) -> None:
        futures = [w.remove_adapter.remote(lora_id) for w in self.workers]
        ray.get(futures)

    def save_state_all(
        self,
        lora_id: str,
        checkpoint_record: Any,
        optimizer: bool,
    ) -> None:
        futures = [w.save_state.remote(lora_id, checkpoint_record, optimizer) for w in self.workers]
        ray.get(futures)

    def load_state_all(
        self,
        lora_id: str,
        checkpoint_record: Any,
        optimizer: bool,
    ) -> None:
        futures = [w.load_state.remote(lora_id, checkpoint_record, optimizer) for w in self.workers]
        ray.get(futures)

    # ------------------------------------------------------------------
    # Data splitting / result merging
    # ------------------------------------------------------------------
    def _split_data(self, data: list[types.Datum], num_shards: int) -> list[list[types.Datum]]:
        """Distribute datums evenly across shards (DP split).

        All shards are padded to the same size (by repeating the last item)
        so that every rank performs the same number of model forward calls —
        a hard requirement for FSDP2 collective operations.
        """
        shard_size = max(1, -(-len(data) // num_shards))  # ceil division
        shards: list[list[types.Datum]] = []
        for i in range(num_shards):
            start = i * shard_size
            end = min(start + shard_size, len(data))
            shard = list(data[start:end])
            while len(shard) < shard_size and shard:
                shard.append(shard[-1])
            if not shard:
                shard = [data[-1]]
            shards.append(shard)
        return shards

    def _merge_forward_results(
        self,
        results: list[types.ForwardBackwardOutput],
        actual_sizes: list[int] | None = None,
    ) -> types.ForwardBackwardOutput:
        all_outputs: list[dict] = []
        all_metrics: list[dict] = []
        weights: list[float] = []
        for idx, result in enumerate(results):
            n = actual_sizes[idx] if actual_sizes else len(result.loss_fn_outputs)
            all_outputs.extend(result.loss_fn_outputs[:n])
            all_metrics.append(result.metrics)
            weights.append(n)

        merged_metrics = metrics_reduction(all_metrics, weights) if all_metrics else {}

        return types.ForwardBackwardOutput(
            loss_fn_output_type=results[0].loss_fn_output_type,
            loss_fn_outputs=all_outputs,
            metrics=merged_metrics,
        )

    @staticmethod
    def _actual_shard_sizes(total: int, num_shards: int) -> list[int]:
        """Return the real (unpadded) item count for each shard."""
        shard_size = max(1, -(-total // num_shards))
        sizes: list[int] = []
        remaining = total
        for _ in range(num_shards):
            n = min(shard_size, remaining)
            sizes.append(max(n, 0))
            remaining -= shard_size
        return sizes

    def get_adapter_param_fingerprint_all(self, lora_id: str) -> list[dict]:
        """Collect adapter parameter fingerprints from every worker."""
        futures = [w.get_adapter_param_fingerprint.remote(lora_id) for w in self.workers]
        return ray.get(futures)

    def get_memory_stats_all(self) -> list[dict]:
        """Collect GPU memory stats from every worker."""
        futures = [w.get_memory_stats.remote() for w in self.workers]
        return ray.get(futures)

    def shutdown(self) -> None:
        """Gracefully clean up workers, then kill the Ray actors.

        Calls ``cleanup()`` on every worker first to destroy the NCCL
        process group, allowing subsequent worker groups to initialise
        new process groups on the same GPUs.
        """
        cleanup_futures = []
        for worker in self.workers:
            try:
                cleanup_futures.append(worker.cleanup.remote())
            except Exception:
                logger.debug("Failed to schedule cleanup on worker", exc_info=True)
        if cleanup_futures:
            try:
                ray.get(cleanup_futures, timeout=30)
            except Exception:
                logger.warning("Timed out waiting for worker cleanup", exc_info=True)

        for worker in self.workers:
            try:
                ray.kill(worker)
            except Exception:
                logger.debug("Failed to kill worker actor", exc_info=True)
        self.workers.clear()
        logger.info("FSDPWorkerGroup shut down — all workers killed")

    # ------------------------------------------------------------------
    # Infrastructure helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _get_master_addr_port() -> tuple[str, int]:
        master_addr = ray.util.get_node_ip_address()
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            master_port = s.getsockname()[1]
        return master_addr, master_port

    @staticmethod
    def _sort_pgs_by_ip(pgs: list) -> list:
        """Sort placement groups by node IP for deterministic RANK assignment.

        Ensures checkpoint resume works after Ray cluster restarts.
        """
        if len(pgs) <= 1:
            return pgs

        node_info = {n["NodeID"]: n["NodeManagerAddress"] for n in ray.nodes()}

        def _get_pg_ip(pg):
            try:
                table = ray._private.state.state.placement_group_table(pg.id)
                bundles_to_node = table.get("bundles_to_node_id", {})
                if bundles_to_node:
                    node_id = next(iter(bundles_to_node.values()))
                    return node_info.get(node_id, "")
            except Exception:
                logger.debug("Could not resolve PG IP", exc_info=True)
            return ""

        return sorted(pgs, key=_get_pg_ip)
