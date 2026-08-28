from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Optional

from tinker import types

from tuft.backends.flex.flex_backend import FlexBackend, FlexBackendMode
from tuft.backends.flex.torchtp_zero_copy import (
    TensorDescriptorFactory,
    call_collective_rpc,
    create_fused_vllm_state_dict,
    flex_partial_sleep_vllm_worker,
    flex_partial_wake_vllm_worker,
    flex_sleep_vllm_worker,
    flex_wake_vllm_worker,
    get_pre_capture_alias_result,
    inject_cuda_ipc_alias,
    make_cuda_ipc_descriptor_dict,
    summarize_injection_results,
    tensor_to_cuda_ipc_descriptor,
)
from tuft.checkpoints import CheckpointRecord
from tuft.config import ModelConfig


def _rmtree_quietly(path: Path) -> None:
    """Best-effort recursive delete; never raise from a cleanup path."""
    import shutil

    shutil.rmtree(path, ignore_errors=True)


StateDictBuilder = Callable[[Any], dict[str, Any]]
VLLMEngineFactory = Callable[[ModelConfig], Any]
DescriptorGatherer = Callable[[dict[str, dict[str, Any]]], list[dict[str, dict[str, Any]]]]
TrainingRuntimeFactory = Callable[
    [ModelConfig, list[Any], list[dict[str, dict[str, Any]]] | None],
    Any,
]
RuntimeReleaser = Callable[[Any], None]


class FusedTorchTPVLLMFlexBackend(FlexBackend):
    """FlexBackend for fused PyTorch-TP training weights and vLLM sampling aliasing.

    The training model is expected to already use the vLLM-compatible fused layout:
    qkv_proj and gate_up_proj are present in each decoder layer and each weight is a
    PyTorch-TP local shard. The transform path creates CUDA IPC descriptors from
    those local shards and injects them into vLLM workers via collective_rpc.
    """

    def __init__(
        self,
        config: ModelConfig,
        *,
        training_model: Any | None = None,
        vllm_engine: Any | None = None,
        state_dict_builder: StateDictBuilder = create_fused_vllm_state_dict,
        vllm_engine_factory: VLLMEngineFactory | None = None,
        descriptor_gatherer: DescriptorGatherer | None = None,
        training_runtime_factory: TrainingRuntimeFactory | None = None,
        training_runtime_releaser: RuntimeReleaser | None = None,
        sampling_runtime_releaser: RuntimeReleaser | None = None,
        rank: int | None = None,
        world_size: int | None = None,
        vocab_size: int | None = None,
        verify_inject: bool = False,
        require_cuda_ipc: bool = True,
        descriptor_factory: TensorDescriptorFactory = tensor_to_cuda_ipc_descriptor,
    ) -> None:
        super().__init__(config)
        self.training_model = training_model
        self.vllm_engine = vllm_engine
        self._tp_group: Any | None = None
        self._tp_mesh: Any | None = None
        # Optional physical GPU pin for the whole in-process flex runtime
        # (training model + vLLM sampling engine). Set via pin_device() when a
        # Ray placeholder actor reserves a device for us (optimal deployment).
        self._device_index: int | None = None
        self._adapter_params: dict[str, list[Any]] = {}
        self._adapter_optimizers: dict[str, Any] = {}
        self._adapter_configs: dict[str, types.LoraConfig] = {}
        self._active_lora_id: str | None = None
        self._training_step = 0
        self._training_lock = asyncio.Lock()
        self.state_dict_builder = state_dict_builder
        self.vllm_engine_factory = vllm_engine_factory
        self.descriptor_gatherer = descriptor_gatherer
        self.training_runtime_factory = training_runtime_factory
        self.training_runtime_releaser = training_runtime_releaser
        self.sampling_runtime_releaser = sampling_runtime_releaser
        self.rank = rank
        self.world_size = world_size
        self.vocab_size = vocab_size
        self.verify_inject = verify_inject
        self.require_cuda_ipc = require_cuda_ipc
        self.descriptor_factory = descriptor_factory
        self._base_storage_keepalive: list[Any] = []
        self._base_storage_keepalive_by_key: dict[str, Any] = {}
        self._last_ipc_descriptors: dict[str, dict[str, Any]] | None = None
        self._last_all_rank_descriptors: list[dict[str, dict[str, Any]]] | None = None
        self._last_injection_results: list[dict[str, Any]] | None = None
        self._pre_capture_descriptor_path: str | None = None
        self._sampling_runtime_asleep = False
        self._worker_mode = False
        self._training_worker_pool: Any | None = None
        # Adapter state saved before training->sampling transform so it can be
        # automatically restored when transforming back to training mode.
        self._saved_adapter_state: dict[str, dict[str, Any]] | None = None
        # vLLM-native multi-LoRA serving: registered LoRARequests by lora_id and
        # snapshots of in-memory training adapters taken at t->s switch time so
        # they can be exported to PEFT format and registered with vLLM.
        self._sampling_lora_requests: dict[str, Any] = {}
        self._sampling_adapter_snapshots: dict[str, dict[str, Any]] = {}
        # lora_id -> training step last exported+registered into the sampling
        # engine. Lets a wake skip adapters whose weights have not changed since
        # the previous flip (dominant flip cost at high tenant counts).
        self._sampling_adapter_registered_step: dict[str, int] = {}
        # Per-adapter update counter: the global _training_step is bumped by every
        # tenant, so it cannot tell whether THIS adapter changed.
        self._adapter_update_step: dict[str, int] = {}
        # Adapters already exported to a PEFT dir off the flip critical path, plus
        # the in-flight export tasks a flip must wait for.
        self._sampling_adapter_exported_step: dict[str, int] = {}
        self._pending_adapter_exports: dict[str, Any] = {}
        self._sampling_adapter_dirs: dict[str, Path] = {}
        # Parent of every per-adapter export dir. Kept as one directory so a
        # crashed server leaks a single tree instead of one 64 MB dir per
        # adapter, and so the atexit hook has a single thing to remove.
        self._sampling_adapter_root: Path | None = None
        self._pending_sampling_adapter: dict[str, Path] = {}
        self._lora_int_id_counter = 100
        # Whether the sampling runtime has been pre-warmed during async_init.
        self._warmup_done = False

    # ------------------------------------------------------------------
    # Adapter state save/restore across mode switches
    # ------------------------------------------------------------------

    def _ordered_named_adapter_params(self, params: list[Any]) -> list[tuple[str, Any]]:
        """Return adapter params with stable model parameter names in ``params`` order."""
        id_to_name: dict[int, str] = {}
        if self.training_model is not None and hasattr(self.training_model, "named_parameters"):
            id_to_name = {id(param): name for name, param in self.training_model.named_parameters()}
        ordered: list[tuple[str, Any]] = []
        for index, param in enumerate(params):
            ordered.append((id_to_name.get(id(param), f"__adapter_param_{index}"), param))
        return ordered

    def _save_adapter_state_for_switch(self) -> None:
        """Save adapter params, optimizer states, and configs before releasing training runtime.

        Called automatically during training->sampling transform so that adapter
        state can be restored when transforming back to training mode.
        """
        # Only skippable when the workers own the state and the parent holds no
        # training model: _restore_adapter_state_after_switch returns immediately
        # in that case, so the offload would be pure cost. The fused TorchTP flex
        # deployment does keep a parent training model and releases it on every
        # flip, so it still pays this offload (measured 2.8-12.7s at 16 tenants,
        # which is what makes the duty-cycle round trip ~16s rather than the
        # sub-second transform the wake path achieves) and the restore on the way
        # back needs it.
        if self._has_training_workers() and self.training_model is None:
            self._saved_adapter_state = None
            return
        if not self._adapter_params:
            self._saved_adapter_state = None
            return
        saved: dict[str, dict[str, Any]] = {}
        for lora_id, params in self._adapter_params.items():
            optimizer = self._adapter_optimizers.get(lora_id)
            named_params = self._ordered_named_adapter_params(params)
            param_names = [name for name, _param in named_params]
            saved[lora_id] = {
                "param_names": param_names,
                "named_params": self._offload_named_tensors_to_cpu(
                    [(name, param.detach()) for name, param in named_params]
                ),
                # Preserve any gradients already accumulated (forward ran, optim not
                # yet) so a concurrent tenant's sampling switch doesn't drop them.
                "named_grads": self._offload_named_tensors_to_cpu(
                    [
                        (name, param.grad.detach() if param.grad is not None else None)
                        for name, param in named_params
                    ]
                ),
                # state_dict() hands back the live exp_avg/exp_avg_sq tensors,
                # which for AdamW are two more copies of every LoRA parameter and
                # stayed resident on the device: releasing the training runtime
                # then could not actually reclaim them.
                "optimizer_state": self._offload_optimizer_state_to_cpu(optimizer),
                "config": self._adapter_configs.get(lora_id),
            }
        self._saved_adapter_state = saved

    @staticmethod
    def _offload_optimizer_state_to_cpu(optimizer: Any) -> Optional[dict[str, Any]]:
        """Move an optimizer's tensor state to host memory.

        ``state_dict()`` returns the live tensors, so AdamW's exp_avg and
        exp_avg_sq (two more copies of every LoRA parameter) stayed on the device
        and releasing the training runtime could not reclaim them.
        ``load_state_dict`` moves them back onto the parameters' device on
        restore, so host-resident state round-trips unchanged.
        """
        if optimizer is None:
            return None
        import torch

        state = optimizer.state_dict()

        def to_host(value: Any) -> Any:
            if isinstance(value, torch.Tensor):
                return value.detach().to("cpu", copy=True) if value.is_cuda else value.detach()
            if isinstance(value, dict):
                return {k: to_host(v) for k, v in value.items()}
            if isinstance(value, list):
                return [to_host(v) for v in value]
            return value

        return to_host(state)

    @staticmethod
    def _offload_named_tensors_to_cpu(
        named_tensors: list[tuple[str, Any]],
    ) -> dict[str, Any]:
        """Copy adapter tensors to host memory using one D2H transfer per dtype.

        A per-tensor ``.cpu()`` blocks on its own device sync, and one adapter
        holds hundreds of LoRA tensors. With the tens of adapters a multi-tenant
        run keeps resident this dominated the training->sampling flip: measured
        save_state was 3.5-7.8s of a 7.45s mean round trip, against 162-244ms for
        the transform itself, which left the duty cycle unable to amortize a flip
        against a ~15s training gap. Staging into one flat buffer per dtype keeps
        the saved state host-resident (so releasing the training runtime still
        frees the adapter params) while paying one sync per dtype group.
        """
        import torch

        out: dict[str, Any] = {}
        by_dtype: dict[Any, list[tuple[str, Any]]] = {}
        for name, tensor in named_tensors:
            if tensor is None:
                out[name] = None
                continue
            by_dtype.setdefault(tensor.dtype, []).append((name, tensor))

        for dtype, group in by_dtype.items():
            flat = torch.cat([tensor.reshape(-1) for _name, tensor in group])
            # Pinning is only legal (and only useful) when staging off a device.
            pin = flat.is_cuda and torch.cuda.is_available()
            staging = torch.empty(flat.numel(), dtype=dtype, device="cpu", pin_memory=pin)
            staging.copy_(flat)
            # One bulk copy off the pinned staging buffer, then views into it.
            # Cloning per tensor did ~500 small host allocations per adapter per
            # call, so the batched D2H bought nothing: measured save_state stayed
            # at ~0.9s per resident adapter (66 MB of params+grads), an effective
            # 73 MB/s against the ~20 GB/s the transfer itself runs at. Views are
            # safe because `resident` is a fresh unpinned tensor that owns its
            # storage; the pinned staging buffer is freed on return.
            resident = staging.to("cpu", copy=True) if pin else staging
            offset = 0
            for name, tensor in group:
                count = tensor.numel()
                out[name] = resident[offset : offset + count].view(tensor.shape)
                offset += count
        return out

    def _restore_adapter_state_after_switch(self) -> None:
        """Restore previously saved adapter state after rebuilding the training model.

        Re-applies LoRA adapters to the new training model, restores parameter
        values and optimizer states so RL training can resume seamlessly.
        """
        import torch

        if not self._saved_adapter_state or self.training_model is None:
            return
        from tuft.backends.flex.torchtp_training import (
            apply_fused_torchtp_lora,
            set_fused_torchtp_lora_adapter,
        )

        for lora_id, state in self._saved_adapter_state.items():
            config = state["config"]
            opts = self._effective_lora_options(config)
            seed = getattr(config, "seed", None) if config is not None else None
            if seed is not None:
                torch.manual_seed(int(seed))
            params = apply_fused_torchtp_lora(
                self.training_model,
                lora_rank=opts["rank"],
                lora_alpha=opts["alpha"],
                tp_group=self._tp_group,
                lora_id=lora_id,
                train_attn=opts["train_attn"],
                train_mlp=opts["train_mlp"],
                train_unembed=opts["train_unembed"],
            )
            fresh_named_params = dict(self._ordered_named_adapter_params(params))
            saved_named_params: dict[str, Any] = state.get("named_params", {})
            saved_param_names: list[str] = list(state.get("param_names", saved_named_params.keys()))
            missing = [name for name in saved_param_names if name not in fresh_named_params]
            if missing:
                raise RuntimeError(
                    f"Cannot restore LoRA adapter {lora_id}: missing parameters after rebuild: "
                    f"{missing[:5]}"
                )
            # Restore saved parameter values by stable model parameter name.  A
            # positional restore is unsafe after mode switches because module
            # traversal can differ once LoRA A/B are no longer at initialization.
            for name in saved_param_names:
                param = fresh_named_params[name]
                saved = saved_named_params[name]
                param.data.copy_(saved.to(device=param.device, dtype=param.dtype))
            # Restore accumulated gradients so optim_step can apply them after the
            # switch (they were saved alongside the params in _save_adapter_state).
            saved_grads = state.get("named_grads") or {}
            for name in saved_param_names:
                grad = saved_grads.get(name)
                if grad is not None:
                    target = fresh_named_params[name]
                    target.grad = grad.to(device=target.device, dtype=target.dtype)
            ordered_params = [fresh_named_params[name] for name in saved_param_names]
            optimizer = torch.optim.AdamW(ordered_params)
            if state["optimizer_state"] is not None:
                try:
                    optimizer.load_state_dict(state["optimizer_state"])
                    # Move optimizer tensors to the correct device
                    for opt_param_group in optimizer.param_groups:
                        for opt_param in opt_param_group.get("params", []):
                            if hasattr(opt_param, "data"):
                                opt_param.data = opt_param.data.to(device=opt_param.device)
                    for opt_state in optimizer.state.values():
                        for key, val in opt_state.items():
                            if isinstance(val, torch.Tensor):
                                param_device = ordered_params[0].device if ordered_params else val.device
                                opt_state[key] = val.to(device=param_device)
                except Exception:
                    pass  # Optimizer state restore is best-effort
            set_fused_torchtp_lora_adapter(self.training_model, lora_id)
            self._adapter_params[lora_id] = ordered_params
            self._adapter_optimizers[lora_id] = optimizer
            if config is not None:
                self._adapter_configs[lora_id] = config
            self._active_lora_id = lora_id
        self._saved_adapter_state = None

    def _effective_lora_options(self, config: types.LoraConfig | None) -> dict[str, Any]:
        rank = int(getattr(config, "rank", 8) or 8) if config is not None else 8
        train_attn = bool(getattr(config, "train_attn", True)) if config is not None else True
        train_mlp = bool(getattr(config, "train_mlp", True)) if config is not None else True
        train_unembed = bool(getattr(config, "train_unembed", True)) if config is not None else True
        # Match HFTrainingModel semantics: MODULE_MAP["qwen"]["unembed"] is
        # empty because PEFT warns / behaves inconsistently for tied Qwen heads.
        if "qwen" in str(self.config.model_path).lower():
            train_unembed = False
        return {
            "rank": rank,
            "alpha": rank,
            "train_attn": train_attn,
            "train_mlp": train_mlp,
            "train_unembed": train_unembed,
        }

    def _snapshot_adapters_for_sampling(self) -> None:
        """Snapshot in-memory adapters as PEFT tensors before the training runtime
        is released, so they can be registered with vLLM's native multi-LoRA path
        after the sampling engine is ready. Base weights are never modified.
        """
        snapshots: dict[str, dict[str, Any]] = {}
        if self.training_model is not None and self._adapter_configs and self._rank() == 0:
            from tuft.backends.flex.torchtp_training import fused_adapter_peft_state_dict

            for lora_id, config in self._adapter_configs.items():
                update_step = self._adapter_update_step.get(lora_id, 0)
                # Snapshotted off the flip path right after this adapter's
                # optimizer step: reuse it instead of paying the GPU->CPU copy
                # again while the flip blocks the training lane.
                prev = self._sampling_adapter_snapshots.get(lora_id)
                if prev is not None and int(prev.get("step", -1)) == update_step:
                    snapshots[lora_id] = prev
                    continue
                opts = self._effective_lora_options(config)
                tensors = fused_adapter_peft_state_dict(self.training_model, lora_id)
                if tensors:
                    snapshots[lora_id] = {
                        "rank": opts["rank"],
                        "alpha": opts["alpha"],
                        "tensors": tensors,
                        "step": update_step,
                    }
        self._sampling_adapter_snapshots = snapshots

    def _adapter_export_dir(self, lora_id: str) -> Path:
        """Stable PEFT export directory for one adapter (created on first use)."""
        adapter_dir = self._sampling_adapter_dirs.get(lora_id)
        if adapter_dir is None:
            import tempfile

            adapter_dir = Path(
                tempfile.mkdtemp(prefix=f"{lora_id}_", dir=self._adapter_export_root())
            )
            self._sampling_adapter_dirs[lora_id] = adapter_dir
        return adapter_dir

    def schedule_adapter_export(self, lora_id: str) -> None:
        """Snapshot one adapter now, write its PEFT dir in the background.

        The GPU->CPU copy stays synchronous so it captures a correct point in
        time (the caller still holds the training lock), but the safetensors
        write is pure host work and is the expensive half, so it overlaps with
        the next training step instead of being paid while a flip blocks.
        """
        if self.training_model is None or self._rank() != 0:
            return
        config = self._adapter_configs.get(lora_id)
        if config is None:
            return
        from tuft.backends.flex.torchtp_training import (
            fused_adapter_peft_state_dict,
            write_peft_adapter_dir,
        )

        tensors = fused_adapter_peft_state_dict(self.training_model, lora_id)
        if not tensors:
            return
        opts = self._effective_lora_options(config)
        step = self._adapter_update_step.get(lora_id, 0)
        rank, alpha = int(opts["rank"]), int(opts["alpha"])
        self._sampling_adapter_snapshots[lora_id] = {
            "rank": rank,
            "alpha": alpha,
            "tensors": tensors,
            "step": step,
        }
        adapter_dir = self._adapter_export_dir(lora_id)
        base = str(self.config.model_path)

        async def _export() -> None:
            try:
                await asyncio.to_thread(
                    write_peft_adapter_dir,
                    adapter_dir,
                    tensors,
                    rank=rank,
                    alpha=alpha,
                    base_model_name_or_path=base,
                )
                self._sampling_adapter_exported_step[lora_id] = step
            except Exception:
                logging.getLogger("tuft").exception(
                    "Background PEFT export failed for %s; the next flip exports "
                    "it inline instead.",
                    lora_id,
                )
            finally:
                self._pending_adapter_exports.pop(lora_id, None)

        try:
            self._pending_adapter_exports[lora_id] = asyncio.ensure_future(_export())
        except RuntimeError:  # no running loop (unit tests, worker processes)
            self._pending_adapter_exports.pop(lora_id, None)

    async def drain_adapter_exports(self) -> float:
        """Wait for background PEFT exports so a flip sees complete dirs."""
        pending = list(self._pending_adapter_exports.values())
        if not pending:
            return 0.0
        t0 = time.perf_counter()
        await asyncio.gather(*pending, return_exceptions=True)
        return time.perf_counter() - t0

    async def _inplace_refresh_registered_adapters(self) -> set[str]:
        """Update already-registered vLLM LoRAs in place (no add_lora rebuild).

        This is the fast path that keeps the training->sampling flip under a
        second: instead of reloading each changed adapter through ``add_lora``,
        copy the new weights into the resident vLLM LoRA tensors via a worker
        RPC. Adapters that are not resident or whose export is not ready are
        left for the normal ``add_lora`` fallback.
        """
        if self.vllm_engine is None or self._rank() != 0:
            return set()
        from tuft.backends.flex.torchtp_zero_copy import (
            update_registered_lora_inplace,
        )

        refreshed: set[str] = set()
        for lora_id, snapshot in self._sampling_adapter_snapshots.items():
            snap_step = int(snapshot.get("step", -1))
            if self._sampling_adapter_registered_step.get(lora_id) == snap_step:
                continue
            request = self._sampling_lora_requests.get(lora_id)
            if request is None:
                continue
            if self._sampling_adapter_exported_step.get(lora_id) != snap_step:
                continue  # export not ready; fall back to inline export+add_lora
            path = str(
                self._adapter_export_dir(lora_id) / "adapter_model.safetensors"
            )
            try:
                results = await call_collective_rpc(
                    self.vllm_engine,
                    update_registered_lora_inplace,
                    args=(int(request.lora_int_id), path),
                )
            except Exception:
                continue
            items = results if isinstance(results, list) else [results]
            if items and all(bool(r.get("ok")) for r in items):
                self._sampling_adapter_registered_step[lora_id] = snap_step
                refreshed.add(lora_id)
        return refreshed

    def _register_lora_with_engine(self, lora_id: str, adapter_path: Any) -> bool:
        """Register a PEFT-format adapter directory with the vLLM engine.

        Reuses the existing LoRARequest (stable lora_int_id) for a given
        lora_id so vLLM's adapter cache hits across engine rebuilds/wakes
        instead of reloading the adapter files on every mode switch.
        """
        engine = self.vllm_engine
        if engine is None:
            return False
        from vllm.lora.request import LoRARequest

        self._lora_int_id_counter += 1
        request = LoRARequest(
            lora_name=f"{lora_id}__snapshot_{self._lora_int_id_counter}",
            lora_int_id=self._lora_int_id_counter,
            lora_path=str(adapter_path),
        )
        llm_engine = getattr(engine, "llm_engine", engine)
        if not hasattr(llm_engine, "add_lora"):
            return False
        try:
            llm_engine.add_lora(request)
        except Exception as exc:

            logging.getLogger("tuft").warning(
                "Failed to register LoRA adapter %s with vLLM: %r", lora_id, exc
            )
            return False
        self._sampling_lora_requests[lora_id] = request
        return True

    def _register_all_sampling_adapters(self, skip: set | None = None) -> int:
        """Export snapshotted adapters to PEFT dirs and register pending adapters."""
        if self.vllm_engine is None or self._rank() != 0:
            return 0

        from tuft.backends.flex.torchtp_training import write_peft_adapter_dir

        registered = 0
        skipped = 0
        exported_offpath = 0
        skip = skip or set()
        for lora_id, snapshot in self._sampling_adapter_snapshots.items():
            if lora_id in skip:
                skipped += 1
                self._pending_sampling_adapter.pop(lora_id, None)
                continue
            # Incremental registration: if this adapter's weights have not
            # changed since we last exported+registered it (same training step)
            # and it is already known to the engine, skip the costly PEFT export
            # and re-registration. This is the dominant per-flip cost when many
            # tenants are resident.
            snap_step = int(snapshot.get("step", -1))
            prev_step = self._sampling_adapter_registered_step.get(lora_id)
            if (
                prev_step is not None
                and prev_step == snap_step
                and lora_id in self._sampling_adapter_dirs
            ):
                skipped += 1
                self._pending_sampling_adapter.pop(lora_id, None)
                continue
            # Reuse the existing export directory for a given lora_id so the
            # vLLM adapter cache can hit on repeated switches; the directory
            # content is overwritten with the latest snapshot values.
            adapter_dir = self._adapter_export_dir(lora_id)
            if self._sampling_adapter_exported_step.get(lora_id) != snap_step:
                # Never exported off the flip path (or that export failed): pay
                # for it inline so the engine never loads stale adapter files.
                write_peft_adapter_dir(
                    adapter_dir,
                    snapshot["tensors"],
                    rank=int(snapshot["rank"]),
                    alpha=int(snapshot["alpha"]),
                    base_model_name_or_path=str(self.config.model_path),
                )
                self._sampling_adapter_exported_step[lora_id] = snap_step
            else:
                exported_offpath += 1
            if self._register_lora_with_engine(lora_id, adapter_dir):
                registered += 1
                self._sampling_adapter_registered_step[lora_id] = snap_step
            # The in-memory snapshot supersedes any pending path registration.
            self._pending_sampling_adapter.pop(lora_id, None)
        for lora_id, pending in list(self._pending_sampling_adapter.items()):
            if self._register_lora_with_engine(lora_id, pending):
                del self._pending_sampling_adapter[lora_id]
                registered += 1
        if skipped:
            logging.getLogger("tuft").info(
                "sampling adapter registration: %d re-registered "
                "(%d reused a background export), %d skipped (unchanged)",
                registered,
                exported_offpath,
                skipped,
            )
        return registered

    def _adapter_export_root(self) -> Path:
        """Lazily create the parent dir for PEFT adapter exports.

        Each export is ~64 MB and /tmp is small; the dirs used to be created
        directly under /tmp and never removed, so a few campaigns filled the
        filesystem and every later run died during server init. The root is
        registered with atexit so a clean shutdown leaves nothing behind.
        """
        if self._sampling_adapter_root is None:
            import atexit
            import tempfile

            export_parent = os.environ.get("TUFT_ADAPTER_EXPORT_DIR")
            parent = (
                Path(export_parent).expanduser().resolve()
                if export_parent
                else Path("/dev/shm/tuft_adapter_exports").resolve()
            )
            parent.mkdir(parents=True, exist_ok=True)
            root = Path(tempfile.mkdtemp(prefix="tuft_flex_lora_", dir=parent))
            self._sampling_adapter_root = root
            atexit.register(_rmtree_quietly, root)
        return self._sampling_adapter_root

    def _discard_adapter_export_dir(self, lora_id: str) -> None:
        """Drop one adapter's export dir once the adapter is gone."""
        adapter_dir = self._sampling_adapter_dirs.pop(lora_id, None)
        if adapter_dir is not None:
            _rmtree_quietly(adapter_dir)

    def _release_training_runtime(self) -> None:
        if self.training_model is not None and self.training_runtime_releaser is not None:
            self.training_runtime_releaser(self.training_model)
        self.training_model = None
        self._tp_group = None
        self._tp_mesh = None
        # In worker-pool mode the real adapter state lives in the worker
        # processes; the parent only keeps bookkeeping (_adapter_configs) which
        # must survive mode switches so callers can detect existing adapters.
        if not self._has_training_workers():
            self._adapter_params.clear()
            self._adapter_optimizers.clear()
            self._adapter_configs.clear()
            self._active_lora_id = None
        # One-time cleanup at mode-switch boundary (not in training hot paths):
        # return cached allocator blocks so the sampling runtime can use them.
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _release_sampling_runtime(self) -> None:
        if self.vllm_engine is not None and self.sampling_runtime_releaser is not None:
            self.sampling_runtime_releaser(self.vllm_engine)
        elif self.vllm_engine is not None and hasattr(self.vllm_engine, "shutdown"):
            self.vllm_engine.shutdown()
        self.vllm_engine = None
        self._sampling_runtime_asleep = False

    def _is_rank0(self) -> bool:
        """True for rank 0 (or single-process)."""
        return self._rank() == 0

    def pin_device(self, index: int) -> None:
        """Pin the in-process flex runtime (training model + vLLM engine) to one GPU.

        Must be called before any model/engine construction. The caller (server
        state) obtains ``index`` from a Ray placeholder actor so Ray-managed
        sampling replicas avoid this device.
        """
        import torch

        self._device_index = int(index)
        torch.cuda.set_device(self._device_index)

    def _dist_world_size(self) -> int:
        try:
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                return int(dist.get_world_size())
        except Exception:
            pass
        return 1

    def _broadcast_worker_command(self, method: str, *args: Any, **kwargs: Any) -> None:
        """Ask TP worker ranks to execute the same backend method.

        Rank 0 runs the HTTP server. For distributed training TP, worker ranks
        wait in the CLI worker loop and receive commands through
        broadcast_object_list. This keeps the external startup command unchanged
        while still executing training forward/backward/optim on all TP ranks.
        """
        if self._worker_mode:
            return
        if not self._is_rank0() or self._dist_world_size() <= 1:
            return
        import torch.distributed as dist

        command = {
            "op": "backend_call",
            "model_name": self.config.model_name,
            "method": method,
            "args": args,
            "kwargs": kwargs,
        }
        import os

        import torch

        logging.getLogger("tuft").info("Broadcasting Flex worker command: %s", method)
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        dist.broadcast_object_list([command], src=0, device=torch.device(f"cuda:{local_rank}"))

    def _should_use_training_workers(self) -> bool:
        if self._worker_mode:
            return False
        if self._training_worker_pool is not None:
            return False
        if self.training_model is not None:
            return False
        try:
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                return False
        except Exception:
            pass
        return self._world_size() > 1

    def _has_training_workers(self) -> bool:
        return self._training_worker_pool is not None

    async def async_init(self) -> None:
        if self._should_use_training_workers():
            from tuft.backends.flex.training_worker_pool import FlexTrainingWorkerPool

            self._training_worker_pool = FlexTrainingWorkerPool(self.config, self._world_size())
            self._training_worker_pool.start()
            print("[FlexBackend] training worker pool started", flush=True)
        elif self.training_model is None and not self._has_training_workers():
            self.training_model = await asyncio.to_thread(self._load_default_training_model)
        # Warm up the sampling runtime at startup so the first user-facing
        # transform_to_sampling hits the fast wake path instead of the ~100s
        # cold start (vLLM dummy load + torch.compile + CUDA Graph capture).
        # Only active for the optimal path and when no fake engine was injected.
        if (
            self.config.sampling_warmup_at_init
            and self.config.sampling_pre_capture_alias
            and self.config.sampling_keep_runtime_on_training
            and not self._warmup_done
            and self.vllm_engine is None
            and self.vllm_engine_factory is None
            and self._is_rank0()
            and not self._worker_mode
            and not self._has_training_workers()
        ):
            print("[FlexBackend] sampling warmup starting", flush=True)
            await self._warmup_sampling_runtime()

    async def _warmup_sampling_runtime(self) -> None:
        """Pre-create the vLLM sampling runtime during async_init.

        Does a cold transform_to_sampling (creates vLLM with dummy load, captures
        CUDA Graph, injects IPC alias) followed by transform_to_training (sleeps
        the runtime via Flex sleep). After warmup the backend is in TRAINING mode
        with the vLLM runtime slept, so the first user-facing transform_to_sampling
        hits the wake path (~100ms) instead of the cold start (~100s).

        Warmup is best-effort: if it fails the first real transform_to_sampling
        will do the cold start. The backend is always left in TRAINING mode.
        """

        logger = logging.getLogger(__name__)
        try:
            await self.transform_to_sampling()
            await self.transform_to_training()
            self._warmup_done = True
            logger.info("FlexBackend warmup complete; sampling runtime is slept and ready")
        except Exception:
            logger.warning(
                "FlexBackend warmup failed; first transform_to_sampling will cold-start",
                exc_info=True,
            )
            # Ensure we are back in training mode even if warmup partially failed.
            if self.mode != FlexBackendMode.TRAINING:
                try:
                    await self.transform_to_training()
                except Exception:
                    logger.warning("Failed to restore training mode after warmup error")

    def _load_default_training_model(self) -> Any:
        from tuft.backends.flex.torchtp_training import (
            load_fused_torchtp_model,
            resolve_flex_model_spec,
        )

        if self._device_index is not None:
            import torch

            torch.cuda.set_device(self._device_index)
        spec = resolve_flex_model_spec(self.config.model_name)
        model, self._tp_group, self._tp_mesh = load_fused_torchtp_model(
            spec,
            self._rank(),
            self._training_world_size(),
            attn_implementation=self.config.attn_implementation or "sdpa",
        )
        return model

    def _vllm_compilation_config(self) -> dict[str, Any] | None:
        if self.config.sampling_disable_cudagraph:
            return {"cudagraph_mode": 0}
        if not self.config.sampling_enforce_eager and not self.config.sampling_pre_capture_alias:
            return {"cudagraph_mode": 0}
        return None

    def _prepare_pre_capture_alias_file(
        self,
        all_rank_descriptors: list[dict[str, dict[str, Any]]] | None,
    ) -> dict[str, Any]:
        if not self.config.sampling_pre_capture_alias:
            return {}
        if all_rank_descriptors is None:
            raise RuntimeError("pre-capture alias requires all-rank IPC descriptors")
        import os
        import pickle
        import tempfile

        descriptor_file = tempfile.NamedTemporaryFile(
            prefix="tuft_flex_ipc_",
            suffix=".pkl",
            delete=False,
        )
        self._pre_capture_descriptor_path = descriptor_file.name
        with descriptor_file:
            pickle.dump(all_rank_descriptors, descriptor_file)
        os.environ["TUFT_FLEX_PRECAPTURE_IPC_PATH"] = self._pre_capture_descriptor_path
        os.environ["TUFT_FLEX_PRECAPTURE_VERIFY"] = "1" if self.verify_inject else "0"
        os.environ["TUFT_FLEX_PRECAPTURE_LOG_PATH"] = self._pre_capture_descriptor_path + ".log"
        return {"worker_cls": "tuft.backends.flex.vllm_worker.TuftFlexGPUWorker"}

    def _create_default_vllm_engine(
        self,
        all_rank_descriptors: list[dict[str, dict[str, Any]]] | None = None,
    ) -> Any:
        from vllm import LLM

        kwargs: dict[str, Any] = {}
        kwargs.update(self._prepare_pre_capture_alias_file(all_rank_descriptors))
        # Native multi-LoRA serving: base weights stay frozen/shared via IPC
        # alias and adapters are served concurrently through LoRARequest.
        kwargs["enable_lora"] = True
        kwargs["max_lora_rank"] = int(getattr(self.config, "max_lora_rank", 16) or 16)
        kwargs["max_loras"] = max(2, int(getattr(self.config, "max_loras", 1) or 1))
        compilation_config = self._vllm_compilation_config()
        if compilation_config is not None:
            kwargs["compilation_config"] = compilation_config

        # If this process is already a torch.distributed training rank, host vLLM
        # in a separate child process so vLLM can create its own TP process group
        # without inheriting/conflicting with the training group.
        try:
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized() and self._world_size() > 1:
                from tuft.backends.flex.vllm_runtime_proxy import VLLMRuntimeProxy

                return VLLMRuntimeProxy(self.config, all_rank_descriptors, self.verify_inject)
        except Exception:
            raise

        # vLLM TP workers must not inherit the internal torchrun environment
        # used by the training TP process group; vLLM creates its own process
        # group and rendezvous. Restore env after LLM construction.
        import os

        os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        dist_env_keys = [
            "RANK",
            "WORLD_SIZE",
            "LOCAL_RANK",
            "LOCAL_WORLD_SIZE",
            "GROUP_RANK",
            "ROLE_RANK",
            "ROLE_WORLD_SIZE",
            "MASTER_ADDR",
            "MASTER_PORT",
            "TORCHELASTIC_RUN_ID",
            "TORCHELASTIC_RESTART_COUNT",
            "TORCHELASTIC_MAX_RESTARTS",
        ]
        saved_env = {key: os.environ.get(key) for key in dist_env_keys}
        for key in dist_env_keys:
            os.environ.pop(key, None)
        # Pin spawned vLLM workers to the reserved device (optimal deployment);
        # without this they default to GPU0 and collide with other runtimes.
        saved_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        if self._device_index is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self._device_index)
        try:
            return LLM(
                model=str(self.config.model_path),
                dtype="bfloat16",
                tensor_parallel_size=self._world_size(),
                gpu_memory_utilization=self.config.sampling_memory_fraction,
                trust_remote_code=True,
                enforce_eager=self.config.sampling_enforce_eager,
                disable_custom_all_reduce=self.config.sampling_disable_custom_all_reduce,
                max_model_len=self.config.sampling_max_model_len or self.config.max_model_len,
                load_format="dummy",
                enable_sleep_mode=self.config.sampling_enable_sleep_mode,
                **kwargs,
            )
        finally:
            if saved_cvd is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = saved_cvd
            for key, value in saved_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def _rank(self) -> int:
        if self.rank is not None:
            return self.rank
        try:
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                return int(dist.get_rank())
        except Exception:
            pass
        return 0

    def _world_size(self) -> int:
        if self.world_size is not None:
            return self.world_size
        try:
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                return int(dist.get_world_size())
        except Exception:
            pass
        return int(self.config.tensor_parallel_size or 1)

    def _single_process_tp(self) -> bool:
        """True when vLLM uses TP > 1 but the training model runs TP=1 (single process)."""
        import os

        # If launched by torch.distributed.run (including internal relaunch),
        # this process will initialize dist during training model load.
        if os.environ.get("RANK") is not None and os.environ.get("WORLD_SIZE") is not None:
            return False
        try:
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                return False
        except Exception:
            pass
        return self._world_size() > 1

    def _training_world_size(self) -> int:
        """TP size for the training model. 1 in single-process mode."""
        return 1 if self._single_process_tp() else self._world_size()

    def _vocab_size(self) -> int | None:
        if self.vocab_size is not None:
            return self.vocab_size
        model = self.training_model
        config = getattr(model, "config", None)
        vocab_size = getattr(config, "vocab_size", None)
        return int(vocab_size) if vocab_size is not None else None

    def _gather_descriptors(
        self, descriptors: dict[str, dict[str, Any]]
    ) -> list[dict[str, dict[str, Any]]]:
        if self.descriptor_gatherer is not None:
            return self.descriptor_gatherer(descriptors)
        world_size = self._world_size()
        if world_size == 1:
            return [descriptors]
        import torch.distributed as dist

        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("Cannot gather descriptors: torch.distributed is not initialized")
        # Only rank 0 needs the full descriptor matrix for vLLM injection.
        # Do not send large CUDA IPC descriptor objects through NCCL object
        # collectives; use a same-node file rendezvous instead.
        import os
        import pickle
        import time
        from pathlib import Path

        desc_dir = Path(os.environ.get("TUFT_FLEX_DESCRIPTOR_DIR", "/tmp/tuft_flex_desc"))
        desc_dir.mkdir(parents=True, exist_ok=True)
        round_id = "latest"
        rank_path = desc_dir / f"rank{self._rank()}_{round_id}.pkl"
        with rank_path.open("wb") as f:
            pickle.dump(descriptors, f)
        if not self._is_rank0():
            return []
        gathered: list[dict[str, dict[str, Any]]] = []
        deadline = time.time() + 300
        paths = [desc_dir / f"rank{rank}_{round_id}.pkl" for rank in range(world_size)]
        while time.time() < deadline:
            if all(path.exists() for path in paths):
                break
            time.sleep(0.1)
        missing = [str(path) for path in paths if not path.exists()]
        if missing:
            raise RuntimeError(f"Descriptor file gather timed out; missing={missing}")
        for path in paths:
            with path.open("rb") as f:
                gathered.append(pickle.load(f))
        return gathered

    def _ensure_vllm_engine(
        self,
        all_rank_descriptors: list[dict[str, dict[str, Any]]] | None = None,
    ) -> Any:
        if self.vllm_engine is not None:
            return self.vllm_engine
        if self.vllm_engine_factory is not None:
            self.vllm_engine = self.vllm_engine_factory(self.config)
        else:
            self.vllm_engine = self._create_default_vllm_engine(all_rank_descriptors)
        return self.vllm_engine

    async def _sleep_sampling_runtime(self) -> dict[str, float]:
        if self.vllm_engine is None:
            return {}
        if self.config.sampling_flex_kv_cache_only_sleep:
            if self.config.sampling_partial_kv_cache_gb > 0:
                results = await call_collective_rpc(
                    self.vllm_engine,
                    flex_partial_sleep_vllm_worker,
                    args=(self.config.sampling_partial_kv_cache_gb,),
                )
            else:
                results = await call_collective_rpc(
                    self.vllm_engine,
                    flex_sleep_vllm_worker,
                    args=(),
                )
            if not isinstance(results, list):
                results = list(results)
            self._sampling_runtime_asleep = True
            return {
                "sampling_runtime_slept:sum": 1.0,
                "sampling_sleep_ms:max": max(float(item["sleep_ms"]) for item in results),
                "sampling_sleep_freed_gb:max": max(float(item["freed_gb"]) for item in results),
            }
        import time

        start = time.perf_counter()
        self.vllm_engine.sleep(level=1)
        self._sampling_runtime_asleep = True
        return {
            "sampling_runtime_slept:sum": 1.0,
            "sampling_sleep_ms:max": (time.perf_counter() - start) * 1000,
        }

    async def _wake_sampling_runtime(self) -> dict[str, float]:
        if self.vllm_engine is None or not self._sampling_runtime_asleep:
            return {}
        tags = list(self.config.sampling_sleep_wake_tags)
        if self.config.sampling_flex_kv_cache_only_sleep:
            if self.config.sampling_partial_kv_cache_gb > 0:
                results = await call_collective_rpc(
                    self.vllm_engine,
                    flex_partial_wake_vllm_worker,
                    args=(),
                )
            else:
                results = await call_collective_rpc(
                    self.vllm_engine,
                    flex_wake_vllm_worker,
                    args=(tags,),
                )
            if not isinstance(results, list):
                results = list(results)
            self._sampling_runtime_asleep = False
            return {
                "sampling_runtime_woke:sum": 1.0,
                "sampling_wake_ms:max": max(float(item["wake_ms"]) for item in results),
            }
        import time

        start = time.perf_counter()
        self.vllm_engine.wake_up(tags=tags)
        self._sampling_runtime_asleep = False
        return {
            "sampling_runtime_woke:sum": 1.0,
            "sampling_wake_ms:max": (time.perf_counter() - start) * 1000,
        }

    def _shard_full_state_dict_for_vllm_tp(
        self, full_state_dict: dict[str, Any], world_size: int
    ) -> list[dict[str, Any]]:
        """Shard a full (TP=1) state dict into per-rank dicts for vLLM TP.

        ColwiseParallel weights (qkv_proj, gate_up_proj): split along dim 0,
        respecting the Q/K/V and gate/up sub-structure so each rank gets the
        correct sub-shard.
        RowwiseParallel weights (o_proj, down_proj): split along dim 1.
        Vocab-parallel weights (embed_tokens, lm_head): split along dim 0.
        Layernorm weights: replicated.
        """
        import torch

        model = self.training_model
        config = getattr(model, "config", None)
        hidden = int(getattr(config, "hidden_size", 0))
        intermediate = int(getattr(config, "intermediate_size", 0))
        num_heads = int(getattr(config, "num_attention_heads", 0))
        num_kv_heads = int(getattr(config, "num_key_value_heads", num_heads))
        head_dim = hidden // num_heads if num_heads else 0
        q_size = num_heads * head_dim
        kv_size = num_kv_heads * head_dim
        int_size = intermediate
        vocab = self._vocab_size() or 0
        vocab_chunk = vocab // world_size if vocab else 0

        rank_dicts: list[dict[str, Any]] = [{} for _ in range(world_size)]
        for key, tensor in full_state_dict.items():
            t = tensor
            if hasattr(t, "contiguous"):
                t = t.contiguous()
            for rank in range(world_size):
                if key.endswith("qkv_proj.weight"):
                    q_part = t[:q_size]
                    k_part = t[q_size : q_size + kv_size]
                    v_part = t[q_size + kv_size :]
                    qi = q_part[rank * q_size // world_size : (rank + 1) * q_size // world_size]
                    ki = k_part[rank * kv_size // world_size : (rank + 1) * kv_size // world_size]
                    vi = v_part[rank * kv_size // world_size : (rank + 1) * kv_size // world_size]
                    shard = torch.cat([qi, ki, vi], dim=0)
                elif key.endswith("gate_up_proj.weight"):
                    g_part = t[:int_size]
                    u_part = t[int_size:]
                    gi = g_part[rank * int_size // world_size : (rank + 1) * int_size // world_size]
                    ui = u_part[rank * int_size // world_size : (rank + 1) * int_size // world_size]
                    shard = torch.cat([gi, ui], dim=0)
                elif key.endswith("o_proj.weight"):
                    shard = t[:, rank * hidden // world_size : (rank + 1) * hidden // world_size]
                elif key.endswith("down_proj.weight"):
                    shard = t[:, rank * int_size // world_size : (rank + 1) * int_size // world_size]
                elif key in ("model.embed_tokens.weight", "lm_head.weight") and vocab_chunk:
                    shard = t[rank * vocab_chunk : (rank + 1) * vocab_chunk]
                else:
                    # layernorm and other replicated weights
                    shard = t
                rank_dicts[rank][key] = shard.contiguous() if hasattr(shard, "contiguous") else shard
        return rank_dicts

    async def _transform_to_sampling_impl(self, *, force: bool = False) -> dict[str, float]:
        self._broadcast_worker_command("transform_to_sampling", force=force)
        # Save adapter state before any release so it can be restored on the
        # way back (sampling -> training).
        _ts0 = time.perf_counter()
        self._save_adapter_state_for_switch()
        _ts1 = time.perf_counter()
        _drain_s = await self.drain_adapter_exports()
        _t_snap = time.perf_counter()
        # Snapshot in-memory adapters BEFORE releasing the training runtime so
        # they can be exported to PEFT format and served by vLLM's native
        # multi-LoRA path. Base weights stay frozen and shared via IPC alias;
        # LoRA deltas are never merged into base.
        self._snapshot_adapters_for_sampling()
        logging.getLogger("tuft").info(
            "flip prep: save_state=%.3fs drain_exports=%.3fs snapshot=%.3fs",
            _ts1 - _ts0,
            _drain_s,
            time.perf_counter() - _t_snap,
        )

        if (
            self._sampling_runtime_asleep
            and self.vllm_engine is not None
            and self._sampling_adapter_snapshots
            and self.config.sampling_rebuild_runtime_for_lora
        ):
            # vLLM sleep/wake does not reliably preserve or refresh native LoRA
            # serving state after adapter files are overwritten.  Rebuild the
            # sampling runtime for LoRA-bearing RL steps; this matches the
            # HF/FSDP save_weights_for_sampler semantics and avoids stale policy
            # logprobs that explode importance ratios.
            self._release_sampling_runtime()

        if self._sampling_runtime_asleep and self.vllm_engine is not None:
            _tw0 = time.perf_counter()
            wake_metrics = await self._wake_sampling_runtime()
            _tw1 = time.perf_counter()
            if self._last_all_rank_descriptors is None:
                raise RuntimeError("Cannot wake sampling runtime without cached IPC descriptors.")
            injection_results = await call_collective_rpc(
                self.vllm_engine,
                inject_cuda_ipc_alias,
                args=(self._last_all_rank_descriptors, self.verify_inject),
            )
            _tw2 = time.perf_counter()
            if not isinstance(injection_results, list):
                injection_results = list(injection_results)
            self._last_injection_results = injection_results
            # LoRA adapter weights are not part of the CuMem pool that sleep/wake
            # manages, so re-export and re-register adapters after every wake.
            if self._has_training_workers():
                replies = self._training_worker_pool.call_all({"op": "gather_peft_adapters"})
                peft_snapshots = next(
                    (
                        reply.get("peft_snapshots")
                        for reply in replies
                        if reply.get("peft_snapshots")
                    ),
                    None,
                )
                if peft_snapshots is not None:
                    self._sampling_adapter_snapshots = peft_snapshots
            refreshed = await self._inplace_refresh_registered_adapters()
            self._register_all_sampling_adapters(skip=refreshed)
            _tw3 = time.perf_counter()
            if self.training_model is not None:
                self._release_training_runtime()
            _tw4 = time.perf_counter()
            logging.getLogger("tuft").info(
                "t2s breakdown: wake=%.3fs inject=%.3fs register=%.3fs release_train=%.3fs wake_ms=%.1f",
                _tw1 - _tw0,
                _tw2 - _tw1,
                _tw3 - _tw2,
                _tw4 - _tw3,
                float(wake_metrics.get("sampling_wake_ms:max", -1)),
            )
            metrics = {
                "zero_copy": 1.0,
                "base_transform_supported": 1.0,
                "source_released": 1.0,
                "training_runtime_released:sum": 1.0,
                "force:sum": float(force),
            }
            metrics.update(wake_metrics)
            return metrics

        if self.training_model is None and not self._has_training_workers():
            await self.async_init()
        if self.training_model is None and not self._has_training_workers():
            raise RuntimeError("FusedTorchTPVLLMFlexBackend has no training_model")

        # Steady-state fast path: base storage is kept alive and stable across
        # switches, so previously built IPC descriptors remain valid and the
        # per-switch state_dict/descriptor rebuild can be skipped entirely.
        descriptor_reused = (
            self.config.sampling_reuse_ipc_descriptors
            and not force
            and self._last_ipc_descriptors is not None
            and self._last_all_rank_descriptors is not None
            and bool(self._base_storage_keepalive)
        )
        if descriptor_reused:
            descriptors = self._last_ipc_descriptors
            all_rank_descriptors = self._last_all_rank_descriptors
        elif self._has_training_workers():
            replies = self._training_worker_pool.call_all({"op": "transform_to_sampling"})
            all_rank_descriptors = [reply["descriptors"] for reply in replies]
            descriptors = all_rank_descriptors[0]
            self._base_storage_keepalive = [object()]
            self._base_storage_keepalive_by_key = {}
            # Workers all-gathered full PEFT adapter matrices across TP ranks;
            # rank0 hands them over for vLLM multi-LoRA registration.
            peft_snapshots = next(
                (reply.get("peft_snapshots") for reply in replies if reply.get("peft_snapshots")),
                None,
            )
            if peft_snapshots is not None:
                self._sampling_adapter_snapshots = peft_snapshots
        elif self._single_process_tp():
            # Single-process TP: training model is full (TP=1) on one GPU.
            # Manually shard the full state dict per vLLM rank and create
            # per-rank IPC descriptors on each rank's GPU.
            import torch

            if hasattr(self.training_model, "eval"):
                self.training_model.eval()
            full_state_dict = self.state_dict_builder(self.training_model)
            ws = self._world_size()
            rank_state_dicts = self._shard_full_state_dict_for_vllm_tp(full_state_dict, ws)
            all_rank_descriptors = []
            all_keepalive: list[Any] = []
            all_keepalive_by_key: dict[str, Any] = {}
            descriptors = {}  # rank-0 descriptors for metrics
            for rank_idx, rank_sd in enumerate(rank_state_dicts):
                # Move shards to the corresponding GPU
                device = torch.device(f"cuda:{rank_idx}")
                for k in rank_sd:
                    rank_sd[k] = rank_sd[k].to(device)
                rank_desc, rank_keepalive = make_cuda_ipc_descriptor_dict(
                    rank_sd,
                    rank=rank_idx,
                    world_size=ws,
                    vocab_size=self._vocab_size(),
                    descriptor_factory=self.descriptor_factory,
                    require_cuda=self.require_cuda_ipc,
                )
                all_rank_descriptors.append(rank_desc)
                all_keepalive.extend(rank_keepalive)
                if rank_idx == 0:
                    descriptors = rank_desc
                    # Keep full-model weight refs (rank-0's full state dict) for rebuild
                    all_keepalive_by_key = dict(zip(full_state_dict.keys(), full_state_dict.values(), strict=False))
            self._base_storage_keepalive = all_keepalive
            self._base_storage_keepalive_by_key = all_keepalive_by_key
        else:
            if hasattr(self.training_model, "eval"):
                self.training_model.eval()

            state_dict = self.state_dict_builder(self.training_model)

            logging.getLogger("tuft").info(
                "Rank %s building IPC descriptors for %d tensors",
                self._rank(),
                len(state_dict),
            )
            descriptors, keepalive = make_cuda_ipc_descriptor_dict(
                state_dict,
                rank=self._rank(),
                world_size=self._world_size(),
                vocab_size=self._vocab_size(),
                descriptor_factory=self.descriptor_factory,
                require_cuda=self.require_cuda_ipc,
            )
            logging.getLogger("tuft").info("Rank %s gathering IPC descriptors", self._rank())
            all_rank_descriptors = self._gather_descriptors(descriptors)
            logging.getLogger("tuft").info("Rank %s gathered IPC descriptors", self._rank())
            self._base_storage_keepalive = keepalive
            self._base_storage_keepalive_by_key = dict(
                zip(state_dict.keys(), keepalive, strict=False)
            )

        # Worker ranks participate in descriptor creation/all_gather above, then
        # release their training runtime. Only rank 0 owns the vLLM engine.
        if not self._is_rank0():
            if self.training_model is not None:
                self._release_training_runtime()
            return {
                "zero_copy": 1.0,
                "base_transform_supported": 1.0,
                "source_released": 1.0,
                "training_runtime_released:sum": 1.0,
                "force:sum": float(force),
            }
        engine = self._ensure_vllm_engine(all_rank_descriptors)
        if self.config.sampling_pre_capture_alias:
            injection_results = await call_collective_rpc(
                engine,
                get_pre_capture_alias_result,
                args=(),
            )
        else:
            injection_results = await call_collective_rpc(
                engine,
                inject_cuda_ipc_alias,
                args=(all_rank_descriptors, self.verify_inject),
            )
        if not isinstance(injection_results, list):
            injection_results = list(injection_results)

        self._release_training_runtime()
        self._last_ipc_descriptors = descriptors
        self._last_all_rank_descriptors = all_rank_descriptors
        self._last_injection_results = injection_results
        # Serve the snapshotted adapters through vLLM's native multi-LoRA path.
        self._register_all_sampling_adapters()

        metrics = summarize_injection_results(injection_results)
        metrics["ipc_descriptors:sum"] = float(len(descriptors))
        metrics["ipc_ranks:sum"] = float(len(all_rank_descriptors))
        metrics["descriptor_reused:sum"] = float(descriptor_reused)
        metrics["source_released"] = 1.0
        metrics["training_runtime_released:sum"] = 1.0
        metrics["force:sum"] = float(force)
        return metrics

    def _build_default_training_runtime_from_keepalive(self) -> Any:
        from tuft.backends.flex.torchtp_training import (
            build_training_runtime_from_keepalive,
            resolve_flex_model_spec,
        )

        if not self._base_storage_keepalive_by_key:
            raise RuntimeError("No base storage keepalive is available for sampling-to-training")
        spec = resolve_flex_model_spec(self.config.model_name)
        model_tuple = build_training_runtime_from_keepalive(
            spec,
            self._base_storage_keepalive_by_key,
            self._training_world_size(),
            apply_lora=False,
        )
        model, self._tp_group, self._tp_mesh, _lora_params, _alias_result = model_tuple
        return model

    async def _transform_to_training_impl(self, *, force: bool = False) -> dict[str, float]:
        self._broadcast_worker_command("transform_to_training", force=force)
        if self._has_training_workers():
            self._training_worker_pool.call_all({"op": "transform_to_training", "force": force})
            if self.config.sampling_keep_runtime_on_training and self.vllm_engine is not None:
                sleep_metrics = await self._sleep_sampling_runtime()
            else:
                sleep_metrics = {}
                self._release_sampling_runtime()
            self._vllm_base_alias_ready = False
            self._base_model_transformed_to_sampling = False
            return {
                "base_transform_supported": 1.0,
                "zero_copy": 1.0,
                "source_released": 1.0,
                "sampling_runtime_released:sum": 0.0 if sleep_metrics else 1.0,
                "force:sum": float(force),
                **sleep_metrics,
            }
        if self.training_runtime_factory is None:
            if self._worker_mode:
                print(
                    f"[FlexBackend] rank={self._rank()} s2t rebuild starting, "
                    f"keepalive_keys={len(self._base_storage_keepalive_by_key)}",
                    flush=True,
                )
            try:
                self.training_model = self._build_default_training_runtime_from_keepalive()
                if self._worker_mode:
                    print(f"[FlexBackend] rank={self._rank()} s2t keepalive rebuild OK", flush=True)
            except Exception as rebuild_exc:
                # Keepalive rebuild failed (e.g. descriptors stale or storage
                # released). Fall back to loading the model from disk so the
                # worker can still participate in training. NOTE: this fallback
                # contains NCCL collectives; under TP>1 all ranks must take the
                # same path, otherwise collectives desync and hang.
                print(
                    f"[FlexBackend] rank={self._rank()} keepalive rebuild failed, "
                    f"falling back to disk reload: {rebuild_exc!r}",
                    flush=True,
                )
                try:
                    self.training_model = await asyncio.to_thread(self._load_default_training_model)
                except Exception:
                    return {
                        "base_transform_supported": 0.0,
                        "zero_copy": float(self._vllm_base_alias_ready),
                        "force:sum": float(force),
                    }
        else:
            self.training_model = self.training_runtime_factory(
                self.config,
                self._base_storage_keepalive,
                self._last_all_rank_descriptors,
            )
        # Worker ranks skip sleep/wake of the vLLM runtime. They only rebuild
        # training model state and restore adapters.
        _s2t_start = time.perf_counter()
        if not self._is_rank0():
            self._vllm_base_alias_ready = False
            self._base_model_transformed_to_sampling = False
            self._restore_adapter_state_after_switch()
            return {
                "base_transform_supported": 1.0,
                "zero_copy": 1.0,
                "source_released": 1.0,
                "force:sum": float(force),
            }
        if self.config.sampling_keep_runtime_on_training and self.vllm_engine is not None:
            sleep_metrics = await self._sleep_sampling_runtime()
        else:
            sleep_metrics = {}
            self._release_sampling_runtime()
        sleep_s = time.perf_counter() - _s2t_start
        self._vllm_base_alias_ready = False
        self._base_model_transformed_to_sampling = False
        # Restore adapter state that was saved before the training->sampling switch
        _restore_start = time.perf_counter()
        self._restore_adapter_state_after_switch()
        restore_s = time.perf_counter() - _restore_start
        # The return path had no timing at all, which left most of the round trip
        # unattributed: with save_state down to ~1.3s the measured one-way flip is
        # 1.6s while mean_switch_round_trip_s is 8.1s, so ~6.4s lives here.
        # `logger` in this module is a local inside the warmup path only, so a
        # module-level reference raised NameError on every return flip and took
        # four arms down with it. This file logs via logging.getLogger("tuft").
        logging.getLogger("tuft").info(
            "s2t return breakdown: sleep_or_release=%.3fs restore_state=%.3fs total=%.3fs",
            sleep_s,
            restore_s,
            time.perf_counter() - _s2t_start,
        )
        if self._worker_mode:
            print(
                f"[FlexBackend] rank={self._rank()} s2t restored adapters="
                f"{list(self._adapter_optimizers.keys())}",
                flush=True,
            )
        result = {
            "base_transform_supported": 1.0,
            "zero_copy": 1.0,
            "source_released": 1.0,
            "sampling_runtime_released:sum": 0.0 if sleep_metrics else 1.0,
            "force:sum": float(force),
        }
        result.update(sleep_metrics)
        return result

    async def sample(
        self,
        prompt: types.ModelInput,
        num_samples: int,
        sampling_params: types.SamplingParams,
        include_prompt_logprobs: bool = False,
        topk_prompt_logprobs: int = 0,
        lora_id: Optional[str] = None,
    ) -> types.SampleResponse:
        if self.mode != FlexBackendMode.SAMPLING:
            await self.transform_to_sampling()
        engine = self._ensure_vllm_engine()
        if not hasattr(engine, "generate"):
            raise NotImplementedError("Sampling requires a local vLLM LLM-compatible engine")
        from vllm import SamplingParams

        lora_request = None
        if lora_id is not None:
            lora_request = self._sampling_lora_requests.get(lora_id)
            if lora_request is None:
                raise ValueError(f"LoRA adapter {lora_id} not found in FlexBackend.")

        params = SamplingParams(
            max_tokens=sampling_params.max_tokens or self.config.default_max_tokens or 16,
            temperature=sampling_params.temperature,
            top_p=sampling_params.top_p,
            top_k=sampling_params.top_k,
            seed=sampling_params.seed,
            n=num_samples,
            logprobs=0,
        )
        generate_kwargs: dict[str, Any] = {}
        if lora_request is not None:
            generate_kwargs["lora_request"] = lora_request
        # Called synchronously on purpose. LLM.generate() does hold the event loop
        # for the whole decode, which is why flex presents one prompt at a time and
        # measures 5.09 req/s against 15.63 for a fixed replica serving over an
        # OpenAI API with continuous batching - but asyncio.to_thread is not the
        # fix. vLLM's LLM wrapper is not thread safe, and dispatching concurrent
        # generate() calls to worker threads collapsed throughput by three orders
        # of magnitude in a measured run: 0.01 req/s, decode rate 3.7 tok/s, the
        # flex GPU pinned in sampling mode for 6292.8s, 47 drain timeouts, and the
        # arm timed out at 7200s having finished 38 of 40 steps (against 830s and
        # 40 steps for this version). Closing the gap needs an engine that accepts
        # concurrent submissions - AsyncLLMEngine or the OpenAI server path the
        # fixed replicas already use - not a thread around the offline API.
        outputs = engine.generate(
            [{"prompt_token_ids": prompt.to_ints()}], params, **generate_kwargs
        )
        sequences: list[types.SampledSequence] = []
        for output in outputs[0].outputs:
            token_ids = list(output.token_ids)
            token_logprobs = getattr(output, "logprobs", None)
            if token_logprobs is None:
                raise RuntimeError("vLLM sampling did not return token logprobs.")
            logprobs: list[float] = []
            for idx, token_id in enumerate(token_ids):
                if idx >= len(token_logprobs) or token_logprobs[idx] is None:
                    raise RuntimeError(
                        f"Missing logprob for sampled token at position {idx}: token={token_id}"
                    )
                entry = token_logprobs[idx]
                lp = entry.get(token_id)
                if lp is None:
                    # Match VLLMSamplingBackend fallback: with logprobs=0 vLLM
                    # should only return the sampled token's logprob. If key
                    # lookup fails due to tokenizer/int wrapper differences,
                    # use the sole returned value but fail if the entry is empty.
                    if len(entry) != 1:
                        raise RuntimeError(
                            f"Sampled token {token_id} absent from logprob entry keys={list(entry.keys())}"
                        )
                    lp = next(iter(entry.values()))
                logprobs.append(float(lp.logprob))
            sequences.append(
                types.SampledSequence(
                    stop_reason=getattr(output, "finish_reason", "length") or "length",
                    _tokens_list=token_ids,
                    _logprobs_list=logprobs,
                )
            )
        return types.SampleResponse(sequences=sequences)

    def _ensure_training_model(self) -> Any:
        if self.training_model is None:
            raise RuntimeError("FusedTorchTPVLLMFlexBackend has no training_model")
        return self.training_model

    async def _ensure_training_mode(self) -> Any:
        if self.mode != FlexBackendMode.TRAINING:
            result = await self.transform_to_training()
            if not result.supported:
                raise RuntimeError("Cannot transform FlexBackend to training mode")
        if self.training_model is None:
            await self.async_init()
        return self._ensure_training_model()

    def _prepare_loss_fn_inputs(self, data: list[types.Datum], device: Any) -> dict[str, Any]:
        import torch
        from torch.nn.utils.rnn import pad_sequence

        loss_fn_input_dict: dict[str, Any] = {}
        if not data or not data[0].loss_fn_inputs:
            return loss_fn_input_dict
        for key in data[0].loss_fn_inputs.keys():
            tensors = [datum.loss_fn_inputs[key].to_torch() for datum in data]
            if all(tensor.dim() == 1 for tensor in tensors):
                loss_fn_input_dict[key] = pad_sequence(
                    tensors,
                    batch_first=True,
                    padding_value=0,
                ).to(device)
            else:
                try:
                    loss_fn_input_dict[key] = torch.stack(tensors).to(device)
                except Exception:
                    max_shape = list(tensors[0].shape)
                    for tensor in tensors:
                        for index, size in enumerate(tensor.shape):
                            if size > max_shape[index]:
                                max_shape[index] = size
                    padded_tensors = []
                    for tensor in tensors:
                        pad_width = [
                            (0, max_size - size)
                            for size, max_size in zip(tensor.shape, max_shape, strict=False)
                        ]
                        pad_args: list[int] = []
                        for pad in reversed(pad_width):
                            pad_args.extend(pad)
                        padded_tensors.append(torch.nn.functional.pad(tensor, pad_args, value=0))
                    loss_fn_input_dict[key] = torch.stack(padded_tensors).to(device)
        return loss_fn_input_dict

    def _compute_logprobs_from_target_tokens(self, logits: Any, target_tokens: Any) -> Any:
        import torch

        if logits.dtype in [torch.float32, torch.float64]:
            logits_labels = torch.gather(logits, dim=-1, index=target_tokens.unsqueeze(-1)).squeeze(
                -1
            )
            logsumexp_values = torch.stack([torch.logsumexp(logit, dim=-1) for logit in logits])
            return logits_labels - logsumexp_values
        log_probs_labels = []
        for row_logits, row_labels in zip(logits, target_tokens, strict=True):
            row_log_probs = torch.nn.functional.log_softmax(row_logits, dim=-1)
            log_probs_labels.append(
                row_log_probs.gather(dim=-1, index=row_labels.unsqueeze(-1)).squeeze(-1)
            )
        return torch.stack(log_probs_labels)

    def _unpad_tensor(self, padded_tensor: Any, original_lengths: list[int]) -> list[Any]:
        return [padded_tensor[index, :length] for index, length in enumerate(original_lengths)]

    def _activate_training_adapter(self, lora_id: str) -> None:
        if lora_id not in self._adapter_optimizers:
            raise ValueError(f"Adapter {lora_id} not found.")
        from tuft.backends.flex.torchtp_training import set_fused_torchtp_lora_adapter

        set_fused_torchtp_lora_adapter(self._ensure_training_model(), lora_id)
        self._active_lora_id = lora_id

    def _run_forward_micro_batches(
        self,
        model: Any,
        data: list[types.Datum],
        micro_batch_size: int,
        num_micro_batches: int,
        loss_fn_callable: Any,
        loss_fn_config: dict[str, float] | None,
        backward: bool,
    ) -> tuple[list[dict[str, Any]], list[dict[str, float]], list[float]]:
        """Run the micro-batch forward/backward loop synchronously in a worker thread.

        Called from ``forward`` under ``self._training_lock``, so only one of these
        runs at a time. The thread does not inherit the main thread's CUDA device,
        so the device is re-pinned before any kernel is launched.
        """
        import torch
        from torch.nn.utils.rnn import pad_sequence

        device = next(model.parameters()).device
        if device.type == "cuda":
            torch.cuda.set_device(device)

        all_outputs: list[dict[str, Any]] = []
        metric_list: list[dict[str, float]] = []
        micro_batch_weights: list[float] = []
        if hasattr(model, "train"):
            model.train()
        for micro_idx in range(num_micro_batches):
            start_idx = micro_idx * micro_batch_size
            end_idx = min(start_idx + micro_batch_size, len(data))
            micro_data = data[start_idx:end_idx]
            input_ids = [
                torch.tensor(datum.model_input.to_ints(), dtype=torch.long)
                for datum in micro_data
            ]
            input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=0)
            attention_mask = (input_ids_padded != 0).long()
            position_ids = (
                torch.arange(input_ids_padded.size(1), dtype=torch.long)
                .unsqueeze(0)
                .expand(input_ids_padded.size(0), -1)
            )
            device = next(model.parameters()).device
            input_ids_padded = input_ids_padded.to(device)
            attention_mask = attention_mask.to(device)
            position_ids = position_ids.to(device)
            outputs = model(
                input_ids=input_ids_padded,
                attention_mask=attention_mask,
                position_ids=position_ids,
                return_dict=True,
            )
            logits = outputs.logits
            if loss_fn_config and "temperature" in loss_fn_config:
                logits = logits / loss_fn_config["temperature"]
            loss_fn_inputs = self._prepare_loss_fn_inputs(micro_data, device)
            target_tokens = loss_fn_inputs["target_tokens"]
            target_logprobs = self._compute_logprobs_from_target_tokens(logits, target_tokens)
            loss_fn_inputs["target_logprobs"] = target_logprobs
            loss, metrics = loss_fn_callable(loss_fn_inputs, loss_fn_config or {})
            if backward:
                loss.backward(retain_graph=False)
            target_lengths = [
                len(datum.loss_fn_inputs["target_tokens"].tolist()) for datum in micro_data
            ]
            unpadded = self._unpad_tensor(target_logprobs.detach(), target_lengths)
            all_outputs.extend(
                {"logprobs": types.TensorData.from_torch(logprobs.cpu().clone())}
                for logprobs in unpadded
            )
            metric_list.append(metrics)
            micro_batch_weights.append(float(len(micro_data)))
            del outputs, logits, target_logprobs, loss_fn_inputs, loss
        # Release cached blocks once per forward call instead of per
        # micro-batch: empty_cache is synchronous and expensive, and
        # per-micro-batch calls dominate training latency for small batches.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return all_outputs, metric_list, micro_batch_weights

    async def forward(
        self,
        data: list[types.Datum],
        lora_id: str,
        loss_fn: types.LossFnType,
        loss_fn_config: dict[str, float] | None,
        backward: bool = False,
    ) -> types.ForwardBackwardOutput:
        if self._has_training_workers():
            # call_all blocks on a pipe read for the whole forward; keep it off the
            # event loop so the server can still serve sampling while flex trains.
            replies = await asyncio.to_thread(
                self._training_worker_pool.call_all,
                {
                    "op": "forward",
                    "data": data,
                    "lora_id": lora_id,
                    "loss_fn": loss_fn,
                    "loss_fn_config": loss_fn_config,
                    "backward": backward,
                },
            )
            return next(reply["result"] for reply in replies if reply.get("result") is not None)
        self._broadcast_worker_command(
            "forward",
            data,
            lora_id,
            loss_fn,
            loss_fn_config,
            backward,
        )
        from tuft.loss_fn import get_loss_fn, metrics_reduction

        model = await self._ensure_training_mode()
        if lora_id not in self._adapter_optimizers:
            raise ValueError(f"Adapter {lora_id} not found.")
        self._activate_training_adapter(lora_id)
        loss_fn_callable = get_loss_fn(loss_fn)
        micro_batch_size = max(int(self.config.micro_batch_size or 1), 1)
        num_micro_batches = (len(data) + micro_batch_size - 1) // micro_batch_size
        async with self._training_lock:
            # The forward/backward, the .cpu() logprob copies and empty_cache are
            # all blocking CUDA calls. Running them inline froze the server event
            # loop for the whole step, which stalled every concurrent sampling
            # request (the fixed DP replicas went idle whenever flex trained).
            all_outputs, metric_list, micro_batch_weights = await asyncio.to_thread(
                self._run_forward_micro_batches,
                model,
                data,
                micro_batch_size,
                num_micro_batches,
                loss_fn_callable,
                loss_fn_config,
                backward,
            )
        metrics = metrics_reduction(metric_list, micro_batch_weights)
        metrics["step:max"] = float(self._training_step)
        return types.ForwardBackwardOutput(
            loss_fn_output_type=loss_fn,
            loss_fn_outputs=all_outputs,
            metrics=metrics,
        )

    async def create_adapter(self, lora_id: str, lora_config: types.LoraConfig) -> None:
        if self._has_training_workers():
            self._training_worker_pool.call_all(
                {"op": "create_adapter", "lora_id": lora_id, "lora_config": lora_config}
            )
            # Track adapter on the parent for bookkeeping (actual state is on workers).
            self._adapter_configs[lora_id] = lora_config
            self._active_lora_id = lora_id
            return
        self._broadcast_worker_command("create_adapter", lora_id, lora_config)
        import torch

        from tuft.backends.flex.torchtp_training import apply_fused_torchtp_lora

        model = await self._ensure_training_mode()
        if lora_id in self._adapter_optimizers:
            raise ValueError(f"Adapter {lora_id} already exists.")
        seed = getattr(lora_config, "seed", None)
        if seed is not None:
            torch.manual_seed(int(seed))
        opts = self._effective_lora_options(lora_config)
        params = apply_fused_torchtp_lora(
            model,
            lora_rank=opts["rank"],
            lora_alpha=opts["alpha"],
            tp_group=self._tp_group,
            lora_id=lora_id,
            train_attn=opts["train_attn"],
            train_mlp=opts["train_mlp"],
            train_unembed=opts["train_unembed"],
        )
        optimizer = torch.optim.AdamW(params)
        self._adapter_params[lora_id] = params
        self._adapter_optimizers[lora_id] = optimizer
        self._adapter_configs[lora_id] = lora_config
        self._active_lora_id = lora_id

    async def remove_adapter(self, lora_id: str) -> None:
        # Remove from the sampling engine / registries regardless of whether the
        # adapter is currently loaded on the training side.
        request = self._sampling_lora_requests.pop(lora_id, None)
        if request is not None and self.vllm_engine is not None:
            try:
                llm_engine = getattr(self.vllm_engine, "llm_engine", self.vllm_engine)
                if hasattr(llm_engine, "remove_lora"):
                    llm_engine.remove_lora(request.lora_int_id)
            except Exception:
                pass
        self._pending_sampling_adapter.pop(lora_id, None)
        self._sampling_adapter_snapshots.pop(lora_id, None)
        self._sampling_adapter_registered_step.pop(lora_id, None)
        self._discard_adapter_export_dir(lora_id)
        if lora_id not in self._adapter_optimizers:
            return
        from tuft.backends.flex.torchtp_training import remove_fused_torchtp_lora_adapter

        if self.training_model is not None:
            remove_fused_torchtp_lora_adapter(self.training_model, lora_id)
        optimizer = self._adapter_optimizers.pop(lora_id)
        del optimizer
        self._adapter_params.pop(lora_id, None)
        self._adapter_configs.pop(lora_id, None)
        if self._active_lora_id == lora_id:
            self._active_lora_id = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _run_optim_step(
        self,
        optimizer: Any,
        params: list[Any],
        adam_params: types.AdamParams,
    ) -> float:
        """Apply the optimizer step synchronously in a worker thread.

        Called from ``optim_step`` under ``self._training_lock``. Returns the
        grad norm (reading it syncs the device, hence the thread).
        """
        import torch

        if params:
            device = params[0].device
            if device.type == "cuda":
                torch.cuda.set_device(device)
        for param_group in optimizer.param_groups:
            param_group["lr"] = adam_params.learning_rate
            param_group["betas"] = (adam_params.beta1, adam_params.beta2)
            param_group["eps"] = adam_params.eps
            param_group["weight_decay"] = adam_params.weight_decay
        grad_norm = 0.0
        if adam_params.grad_clip_norm > 0 and params:
            grad_norm = float(torch.nn.utils.clip_grad_norm_(params, adam_params.grad_clip_norm))
        elif params:
            grad_norm = float(
                torch.linalg.vector_norm(
                    torch.stack([
                        param.grad.detach().float().norm()
                        for param in params
                        if param.grad is not None
                    ])
                ).item()
            ) if any(param.grad is not None for param in params) else 0.0
        optimizer.step()
        # set_to_none frees the grad tensors instead of zeroing them. Keeping
        # zeroed grads made every train->sample flip offload a full copy of
        # all-zero gradients (33 MB per adapter at rank 8 on qwen3-4b), which is
        # pure cost: _save_adapter_state_for_switch preserves grads only so a
        # flip mid-accumulation does not drop them, and None carries that just as
        # well.
        optimizer.zero_grad(set_to_none=True)
        return grad_norm

    async def optim_step(
        self,
        adam_params: types.AdamParams,
        lora_id: str,
    ) -> types.OptimStepResponse:
        if self._has_training_workers():
            replies = await asyncio.to_thread(
                self._training_worker_pool.call_all,
                {"op": "optim_step", "adam_params": adam_params, "lora_id": lora_id},
            )
            return next(reply["result"] for reply in replies if reply.get("result") is not None)
        self._broadcast_worker_command("optim_step", adam_params, lora_id)
        # A concurrent tenant's sampling may have switched this backend to
        # sampling mode (releasing the training runtime + adapters) since this
        # tenant's forward ran. Re-enter training mode so the saved adapter state
        # is restored before we look the adapter up (mirrors forward()).
        await self._ensure_training_mode()
        if lora_id not in self._adapter_optimizers:
            raise ValueError(f"Adapter {lora_id} not found.")
        self._activate_training_adapter(lora_id)
        optimizer = self._adapter_optimizers[lora_id]
        params = self._adapter_params.get(lora_id, [])
        async with self._training_lock:
            # clip_grad_norm_/step/zero_grad are blocking CUDA work plus a device
            # sync on the grad-norm read: keep them off the event loop too.
            grad_norm = await asyncio.to_thread(
                self._run_optim_step, optimizer, params, adam_params
            )
            self._training_step += 1
            self._adapter_update_step[lora_id] = (
                self._adapter_update_step.get(lora_id, 0) + 1
            )
            # Export for sampling now, off the flip critical path.
            self.schedule_adapter_export(lora_id)
        return types.OptimStepResponse(
            metrics={
                "learning_rate:mean": adam_params.learning_rate,
                "step:max": float(self._training_step),
                "grad_norm:mean": grad_norm,
            }
        )

    async def save_state(
        self,
        lora_id: str,
        checkpoint_record: CheckpointRecord,
        optimizer: bool,
    ) -> None:
        import torch

        if self.mode != FlexBackendMode.TRAINING:
            result = await self.transform_to_training()
            if not result.supported:
                raise RuntimeError("Cannot transform FlexBackend to training mode before save_state")

        if self._has_training_workers():
            checkpoint_record.adapter_path.mkdir(parents=True, exist_ok=True)
            if optimizer:
                checkpoint_record.optimizer_path.mkdir(parents=True, exist_ok=True)
            self._training_worker_pool.call_all(
                {
                    "op": "save_state",
                    "lora_id": lora_id,
                    "adapter_path": checkpoint_record.adapter_path,
                    "optimizer_path": checkpoint_record.optimizer_path,
                    "optimizer": optimizer,
                }
            )
            return

        if lora_id not in self._adapter_params:
            raise ValueError(f"Adapter {lora_id} not found.")
        self._activate_training_adapter(lora_id)
        checkpoint_record.adapter_path.mkdir(parents=True, exist_ok=True)
        adapter_path = checkpoint_record.adapter_path / "flex_torchtp_adapter.pt"
        params = self._adapter_params[lora_id]
        config = self._adapter_configs.get(lora_id)
        torch.save(
            {
                "lora_id": lora_id,
                "rank": int(getattr(config, "rank", 8) or 8),
                "step": self._training_step,
                "params": [param.detach().cpu() for param in params],
            },
            adapter_path,
        )
        # Also write PEFT-format files so the same checkpoint can be served by
        # vLLM's native multi-LoRA path (add_adapter / create_sampling_session).
        if self._rank() == 0 and self.training_model is not None:
            from tuft.backends.flex.torchtp_training import (
                fused_adapter_peft_state_dict,
                write_peft_adapter_dir,
            )

            peft_tensors = fused_adapter_peft_state_dict(self.training_model, lora_id)
            if peft_tensors:
                opts = self._effective_lora_options(config)
                write_peft_adapter_dir(
                    checkpoint_record.adapter_path,
                    peft_tensors,
                    rank=opts["rank"],
                    alpha=opts["alpha"],
                    base_model_name_or_path=str(self.config.model_path),
                )
        if optimizer:
            checkpoint_record.optimizer_path.mkdir(parents=True, exist_ok=True)
            opt_path = checkpoint_record.optimizer_path / f"{lora_id}.pt"
            torch.save(self._adapter_optimizers[lora_id].state_dict(), opt_path)

    async def load_state(
        self,
        lora_id: str,
        checkpoint_record: CheckpointRecord,
        optimizer: bool,
    ) -> None:
        import torch

        if self._has_training_workers():
            self._training_worker_pool.call_all(
                {"op": "load_adapter", "lora_id": lora_id, "adapter_path": checkpoint_record.adapter_path}
            )
            rank_path = checkpoint_record.adapter_path / "flex_torchtp_adapter_rank0.pt"
            legacy_path = checkpoint_record.adapter_path / "flex_torchtp_adapter.pt"
            adapter_path = rank_path if rank_path.exists() else legacy_path
            state = torch.load(adapter_path, map_location="cpu")
            self._adapter_configs[lora_id] = types.LoraConfig(rank=int(state.get("rank", 8)))
            self._active_lora_id = lora_id
            self._last_ipc_descriptors = None
            self._last_all_rank_descriptors = None
            self._base_model_transformed_to_sampling = False
            self._vllm_base_alias_ready = False
            return

        rank_path = checkpoint_record.adapter_path / f"flex_torchtp_adapter_rank{self._rank()}.pt"
        legacy_path = checkpoint_record.adapter_path / "flex_torchtp_adapter.pt"
        adapter_path = rank_path if rank_path.exists() else legacy_path
        state = torch.load(adapter_path, map_location="cpu")
        if lora_id not in self._adapter_params:
            await self.create_adapter(lora_id, types.LoraConfig(rank=int(state.get("rank", 8))))
        params = self._adapter_params[lora_id]
        for param, value in zip(params, state["params"], strict=True):
            param.data.copy_(value.to(device=param.device, dtype=param.dtype))
        self._activate_training_adapter(lora_id)
        self._training_step = int(state.get("step", self._training_step))
        if optimizer:
            opt_path = checkpoint_record.optimizer_path / f"{lora_id}_rank{self._rank()}.pt"
            if not opt_path.exists():
                opt_path = checkpoint_record.optimizer_path / f"{lora_id}.pt"
            if opt_path.exists():
                self._adapter_optimizers[lora_id].load_state_dict(torch.load(opt_path))

    async def add_adapter(self, lora_id: str, adapter_path: Path) -> None:
        import torch

        # PEFT-format adapter (adapter_model.safetensors): serve it through
        # vLLM's native multi-LoRA path. Base weights stay untouched.
        if (adapter_path / "adapter_model.safetensors").exists():
            if self.vllm_engine is not None:
                if not self._register_lora_with_engine(lora_id, adapter_path):
                    self._pending_sampling_adapter[lora_id] = adapter_path
            else:
                self._pending_sampling_adapter[lora_id] = adapter_path
            return

        rank_path = adapter_path / f"flex_torchtp_adapter_rank{self._rank()}.pt"
        flex_adapter_path = rank_path if rank_path.exists() else adapter_path / "flex_torchtp_adapter.pt"
        if not flex_adapter_path.exists():
            raise ValueError(f"Flex LoRA adapter file {flex_adapter_path} does not exist.")
        state = torch.load(flex_adapter_path, map_location="cpu")
        rank = int(state.get("rank", 8))
        if self.mode != FlexBackendMode.TRAINING:
            result = await self.transform_to_training()
            if not result.supported:
                raise RuntimeError("Cannot transform FlexBackend to training mode before add_adapter")
        if self._has_training_workers():
            self._training_worker_pool.call_all(
                {"op": "load_adapter", "lora_id": lora_id, "adapter_path": adapter_path}
            )
            self._adapter_configs[lora_id] = types.LoraConfig(rank=rank)
            self._active_lora_id = lora_id
        else:
            if lora_id not in self._adapter_configs:
                await self.create_adapter(lora_id, types.LoraConfig(rank=rank))
            params = self._adapter_params[lora_id]
            for param, value in zip(params, state["params"], strict=True):
                param.data.copy_(value.to(device=param.device, dtype=param.dtype))
            self._activate_training_adapter(lora_id)
            self._training_step = int(state.get("step", self._training_step))
        # Adapter state changed; force the next training->sampling transform to
        # rebuild descriptors instead of reusing pre-adapter base descriptors.
        self._last_ipc_descriptors = None
        self._last_all_rank_descriptors = None
        self._base_model_transformed_to_sampling = False
        self._vllm_base_alias_ready = False
