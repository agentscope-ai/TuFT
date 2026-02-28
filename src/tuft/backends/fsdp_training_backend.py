"""FSDP2 multi-GPU training backend for TuFT.

Implements :class:`BaseTrainingBackend` by delegating all work to an
:class:`FSDPWorkerGroup`.  The ``TrainingController`` sees this as an
ordinary backend — adapter management, forward/backward, optimizer
steps, and checkpoint operations are all routed transparently.
"""

from __future__ import annotations

import logging

from tinker import types

from tuft.backends.base_backend import BaseTrainingBackend
from tuft.backends.fsdp_worker_group import FSDPWorkerGroup
from tuft.checkpoints import CheckpointRecord
from tuft.config import ModelConfig
from tuft.telemetry.tracing import get_tracer


_get_tracer = lambda: get_tracer("tuft.fsdp_training_backend")  # noqa: E731
logger = logging.getLogger(__name__)


class FSDPTrainingBackend(BaseTrainingBackend):
    """Multi-GPU FSDP2 training backend."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        logger.info(
            "Creating FSDPTrainingBackend: %s (%d GPUs × %d nodes)",
            config.model_name,
            config.num_gpus_per_node,
            config.num_nodes,
        )
        self.worker_group = FSDPWorkerGroup(
            config,
            num_gpus_per_node=config.num_gpus_per_node,
            num_nodes=config.num_nodes,
        )

    async def async_init(self) -> None:
        pass

    async def create_adapter(self, lora_id: str, lora_config: types.LoraConfig) -> None:
        with _get_tracer().start_as_current_span("fsdp_backend.create_adapter") as span:
            span.set_attribute("tuft.lora_id", lora_id)
            span.set_attribute("tuft.lora_rank", lora_config.rank)
            self.worker_group.create_adapter_all(lora_id, lora_config)

    async def remove_adapter(self, lora_id: str) -> None:
        with _get_tracer().start_as_current_span("fsdp_backend.remove_adapter"):
            self.worker_group.remove_adapter_all(lora_id)

    async def forward(
        self,
        data: list[types.Datum],
        lora_id: str,
        loss_fn: types.LossFnType,
        loss_fn_config: dict[str, float] | None,
        backward: bool = False,
    ) -> types.ForwardBackwardOutput:
        span_name = "fsdp_backend.forward_backward" if backward else "fsdp_backend.forward"
        with _get_tracer().start_as_current_span(span_name) as span:
            span.set_attribute("tuft.lora_id", lora_id)
            span.set_attribute("tuft.backward", backward)
            span.set_attribute("tuft.data_count", len(data))
            return self.worker_group.forward_all(data, lora_id, loss_fn, loss_fn_config, backward)

    async def optim_step(
        self,
        adam_params: types.AdamParams,
        lora_id: str,
    ) -> types.OptimStepResponse:
        with _get_tracer().start_as_current_span("fsdp_backend.optim_step") as span:
            span.set_attribute("tuft.lora_id", lora_id)
            return self.worker_group.optim_step_all(adam_params, lora_id)

    async def save_state(
        self,
        lora_id: str,
        checkpoint_record: CheckpointRecord,
        optimizer: bool,
    ) -> None:
        with _get_tracer().start_as_current_span("fsdp_backend.save_state") as span:
            span.set_attribute("tuft.lora_id", lora_id)
            span.set_attribute("tuft.optimizer", optimizer)
            self.worker_group.save_state_all(lora_id, checkpoint_record, optimizer)

    async def load_state(
        self,
        lora_id: str,
        checkpoint_record: CheckpointRecord,
        optimizer: bool,
    ) -> None:
        with _get_tracer().start_as_current_span("fsdp_backend.load_state") as span:
            span.set_attribute("tuft.lora_id", lora_id)
            span.set_attribute("tuft.optimizer", optimizer)
            self.worker_group.load_state_all(lora_id, checkpoint_record, optimizer)

    async def get_memory_stats(self) -> list[dict]:
        """Collect GPU memory statistics from all FSDP workers."""
        return self.worker_group.get_memory_stats_all()

    def shutdown(self) -> None:
        """Shut down the FSDP backend and release GPU resources."""
        self.worker_group.shutdown()
