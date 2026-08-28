from __future__ import annotations

import multiprocessing as mp
import os
from typing import Any

from tinker import types

from tuft.backends.flex.flex_backend import FlexBackendMode
from tuft.backends.flex.torchtp_zero_copy import (
    create_fused_vllm_state_dict,
    make_cuda_ipc_descriptor_dict,
    tensor_to_cuda_ipc_descriptor,
)
from tuft.config import ModelConfig


def _gather_worker_peft_snapshots(backend: Any, rank: int, world_size: int) -> dict[str, Any]:
    """Collectively gather full PEFT adapter tensors across TP worker ranks.

    All ranks must call this together (same adapter set) because it performs
    NCCL all-gather for column/row-sharded LoRA matrices. Returns a non-empty
    dict on every rank (identical content), empty when no adapters exist.
    """
    snapshots: dict[str, Any] = {}
    if not backend._adapter_configs or backend.training_model is None:
        return snapshots
    from tuft.backends.flex.torchtp_training import gather_fused_adapter_peft_state_dict

    for lora_id, cfg in backend._adapter_configs.items():
        tensors = gather_fused_adapter_peft_state_dict(
            backend.training_model,
            lora_id,
            tp_group=backend._tp_group,
            world_size=world_size,
        )
        if tensors:
            adapter_rank = int(getattr(cfg, "rank", 8) or 8)
            snapshots[lora_id] = {
                "rank": adapter_rank,
                "alpha": adapter_rank,
                "tensors": tensors,
                "step": backend._training_step,
            }
    return snapshots


def _worker_main(rank: int, world_size: int, config_data: dict[str, Any], conn, master_port: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    import asyncio
    import traceback

    import torch
    import torch.distributed as dist

    torch.cuda.set_device(rank)
    config = ModelConfig.model_validate(config_data)
    config = config.model_copy(update={"sampling_warmup_at_init": False})

    from tuft.backends.flex.torchtp import FusedTorchTPVLLMFlexBackend

    backend = FusedTorchTPVLLMFlexBackend(config, rank=rank, world_size=world_size)
    backend._worker_mode = True

    try:
        asyncio.run(backend.async_init())
        print(f"[FlexTrainingWorker] rank={rank} READY", flush=True)
        conn.send({"ok": True, "rank": rank, "type": "READY"})
        while True:
            try:
                msg = conn.recv()
            except EOFError:
                print(f"[FlexTrainingWorker] rank={rank} parent pipe closed; exiting", flush=True)
                break
            op = msg.get("op")
            try:
                if op == "shutdown":
                    conn.send({"ok": True})
                    break
                elif op == "create_adapter":
                    print(f"[FlexTrainingWorker] rank={rank} create_adapter start", flush=True)
                    asyncio.run(backend.create_adapter(msg["lora_id"], msg["lora_config"]))
                    print(f"[FlexTrainingWorker] rank={rank} create_adapter done", flush=True)
                    conn.send({"ok": True})
                elif op == "load_adapter":
                    import torch

                    rank_path = msg["adapter_path"] / f"flex_torchtp_adapter_rank{rank}.pt"
                    legacy_path = msg["adapter_path"] / "flex_torchtp_adapter.pt"
                    adapter_path = rank_path if rank_path.exists() else legacy_path
                    state = torch.load(adapter_path, map_location="cpu")
                    rank_value = int(state.get("rank", 8))
                    if msg["lora_id"] not in backend._adapter_configs:
                        asyncio.run(
                            backend.create_adapter(
                                msg["lora_id"], types.LoraConfig(rank=rank_value)
                            )
                        )
                    params = backend._adapter_params[msg["lora_id"]]
                    for param, value in zip(params, state["params"], strict=True):
                        param.data.copy_(value.to(device=param.device, dtype=param.dtype))
                    backend._activate_training_adapter(msg["lora_id"])
                    backend._training_step = int(state.get("step", backend._training_step))
                    conn.send({"ok": True})
                elif op == "save_state":
                    import torch

                    lora_id = msg["lora_id"]
                    if lora_id not in backend._adapter_params:
                        raise ValueError(f"Adapter {lora_id} not found.")
                    backend._activate_training_adapter(lora_id)
                    adapter_dir = msg["adapter_path"]
                    adapter_dir.mkdir(parents=True, exist_ok=True)
                    config = backend._adapter_configs.get(lora_id)
                    torch.save(
                        {
                            "lora_id": lora_id,
                            "rank": int(getattr(config, "rank", 8) or 8),
                            "step": backend._training_step,
                            "params": [param.detach().cpu() for param in backend._adapter_params[lora_id]],
                        },
                        adapter_dir / f"flex_torchtp_adapter_rank{rank}.pt",
                    )
                    if msg.get("optimizer"):
                        opt_dir = msg["optimizer_path"]
                        opt_dir.mkdir(parents=True, exist_ok=True)
                        torch.save(
                            backend._adapter_optimizers[lora_id].state_dict(),
                            opt_dir / f"{lora_id}_rank{rank}.pt",
                        )
                    conn.send({"ok": True})
                elif op == "forward":
                    result = asyncio.run(
                        backend.forward(
                            msg["data"],
                            msg["lora_id"],
                            msg["loss_fn"],
                            msg["loss_fn_config"],
                            msg["backward"],
                        )
                    )
                    # Return rank0 output to server; other ranks just acknowledge.
                    conn.send({"ok": True, "result": result if rank == 0 else None})
                elif op == "optim_step":
                    result = asyncio.run(backend.optim_step(msg["adam_params"], msg["lora_id"]))
                    conn.send({"ok": True, "result": result if rank == 0 else None})
                elif op == "transform_to_sampling":
                    print(f"[FlexTrainingWorker] rank={rank} transform_to_sampling", flush=True)
                    backend._save_adapter_state_for_switch()
                    print(
                        f"[FlexTrainingWorker] rank={rank} saved_adapter_state="
                        f"{list((backend._saved_adapter_state or {}).keys())}",
                        flush=True,
                    )
                    if backend.training_model is None:
                        asyncio.run(backend.async_init())
                    # Base weights stay frozen and shared via IPC alias; LoRA
                    # deltas are served by vLLM's native multi-LoRA path and are
                    # never merged into base. Gather full PEFT adapter matrices
                    # (all-gather over TP shards) so rank0 can hand them to the
                    # parent for vLLM LoRA registration.
                    peft_snapshots = _gather_worker_peft_snapshots(backend, rank, world_size)
                    if hasattr(backend.training_model, "eval"):
                        backend.training_model.eval()
                    state_dict = create_fused_vllm_state_dict(backend.training_model)
                    descriptors, keepalive = make_cuda_ipc_descriptor_dict(
                        state_dict,
                        rank=rank,
                        world_size=world_size,
                        vocab_size=backend._vocab_size(),
                        descriptor_factory=tensor_to_cuda_ipc_descriptor,
                        require_cuda=True,
                    )
                    backend._base_storage_keepalive = keepalive
                    backend._base_storage_keepalive_by_key = dict(
                        zip(state_dict.keys(), keepalive, strict=False)
                    )
                    backend._release_training_runtime()
                    # The op manually performed the t->s work; sync the mode so
                    # the subsequent transform_to_training does not early-return
                    # and actually rebuilds/restores the training runtime.
                    backend._mode = FlexBackendMode.SAMPLING
                    backend._base_model_transformed_to_sampling = False
                    conn.send(
                        {
                            "ok": True,
                            "descriptors": descriptors,
                            "peft_snapshots": peft_snapshots if rank == 0 else None,
                        }
                    )
                elif op == "gather_peft_adapters":
                    # Collective PEFT gather used by the keep-runtime wake path.
                    snapshots = _gather_worker_peft_snapshots(backend, rank, world_size)
                    conn.send(
                        {"ok": True, "peft_snapshots": snapshots if rank == 0 else None}
                    )
                elif op == "transform_to_training":
                    print(f"[FlexTrainingWorker] rank={rank} transform_to_training start", flush=True)
                    result = asyncio.run(backend.transform_to_training(force=msg.get("force", False)))
                    print(f"[FlexTrainingWorker] rank={rank} transform_to_training done", flush=True)
                    if not result.supported:
                        conn.send({"ok": False, "error": f"transform_to_training not supported: {result.message}"})
                    else:
                        conn.send({"ok": True, "result": result.metrics if rank == 0 else None})
                else:
                    raise ValueError(f"Unknown worker op: {op}")
            except Exception as exc:
                conn.send({"ok": False, "error": repr(exc), "traceback": traceback.format_exc()})
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


class FlexTrainingWorkerPool:
    def __init__(self, config: ModelConfig, world_size: int) -> None:
        self.config = config
        self.world_size = world_size
        self._ctx = mp.get_context("spawn")
        self._conns = []
        self._procs = []
        self._master_port = self._free_port()

    def _free_port(self) -> int:
        import socket

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            return int(s.getsockname()[1])

    def start(self) -> None:
        if self._procs:
            return
        for rank in range(self.world_size):
            parent, child = self._ctx.Pipe()
            proc = self._ctx.Process(
                target=_worker_main,
                args=(rank, self.world_size, self.config.model_dump(mode="python"), child, self._master_port),
            )
            proc.start()
            self._conns.append(parent)
            self._procs.append(proc)
        for index, conn in enumerate(self._conns):
            if not conn.poll(300):
                exitcodes = [proc.exitcode for proc in self._procs]
                raise RuntimeError(
                    f"Timed out waiting for Flex training worker {index} READY; exitcodes={exitcodes}"
                )
            try:
                msg = conn.recv()
            except EOFError as exc:
                exitcodes = [proc.exitcode for proc in self._procs]
                raise RuntimeError(
                    f"Flex training worker {index} pipe closed before READY; exitcodes={exitcodes}"
                ) from exc
            if not msg.get("ok"):
                raise RuntimeError(f"Flex training worker failed: {msg}")
            print(f"[FlexTrainingWorkerPool] received READY from rank={msg.get('rank')}", flush=True)

    def call_all(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        for conn in self._conns:
            conn.send(payload)
        replies = []
        for index, conn in enumerate(self._conns):
            if not conn.poll(300):
                exitcodes = [proc.exitcode for proc in self._procs]
                raise RuntimeError(
                    f"Timed out waiting for Flex training worker {index} reply to {payload.get('op')}; "
                    f"exitcodes={exitcodes}"
                )
            try:
                replies.append(conn.recv())
            except EOFError as exc:
                exitcodes = [proc.exitcode for proc in self._procs]
                raise RuntimeError(
                    f"Flex training worker {index} pipe closed during {payload.get('op')}; "
                    f"exitcodes={exitcodes}"
                ) from exc
        for reply in replies:
            if not reply.get("ok"):
                raise RuntimeError(f"Flex training worker call failed: {reply}")
        return replies

    def shutdown(self) -> None:
        if not self._procs:
            return
        print("[FlexTrainingWorkerPool] shutting down", flush=True)
        try:
            self.call_all({"op": "shutdown"})
        except Exception:
            pass
        for proc in self._procs:
            proc.join(timeout=5)
            if proc.is_alive():
                proc.terminate()
        self._procs.clear()
        self._conns.clear()
