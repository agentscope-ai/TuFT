from __future__ import annotations

import multiprocessing as mp
from types import SimpleNamespace
from typing import Any

from tuft.config import ModelConfig


def _clear_torchrun_env() -> None:
    import os

    for key in (
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
    ):
        os.environ.pop(key, None)


def _function_by_name(name: str):
    from tuft.backends.flex import torchtp_zero_copy as zc

    return getattr(zc, name)


def _prepare_pre_capture_env(
    config: ModelConfig,
    all_rank_descriptors: list[dict[str, dict[str, Any]]] | None,
    verify_inject: bool,
) -> dict[str, Any]:
    if not config.sampling_pre_capture_alias:
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
    with descriptor_file:
        pickle.dump(all_rank_descriptors, descriptor_file)
    os.environ["TUFT_FLEX_PRECAPTURE_IPC_PATH"] = descriptor_file.name
    os.environ["TUFT_FLEX_PRECAPTURE_VERIFY"] = "1" if verify_inject else "0"
    os.environ["TUFT_FLEX_PRECAPTURE_LOG_PATH"] = descriptor_file.name + ".log"
    return {"worker_cls": "tuft.backends.flex.vllm_worker.TuftFlexGPUWorker"}


def _vllm_compilation_config(config: ModelConfig) -> dict[str, Any] | None:
    if config.sampling_disable_cudagraph:
        return {"cudagraph_mode": 0}
    if not config.sampling_enforce_eager and not config.sampling_pre_capture_alias:
        return {"cudagraph_mode": 0}
    return None


def _vllm_runtime_main(conn, config_data: dict[str, Any], all_rank_descriptors, verify_inject: bool) -> None:
    _clear_torchrun_env()
    import os

    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    config = ModelConfig.model_validate(config_data)
    try:
        from vllm import LLM, SamplingParams

        kwargs: dict[str, Any] = {}
        kwargs.update(_prepare_pre_capture_env(config, all_rank_descriptors, verify_inject))
        compilation_config = _vllm_compilation_config(config)
        if compilation_config is not None:
            kwargs["compilation_config"] = compilation_config
        llm = LLM(
            model=str(config.model_path),
            dtype="bfloat16",
            tensor_parallel_size=int(config.tensor_parallel_size or 1),
            gpu_memory_utilization=config.sampling_memory_fraction,
            trust_remote_code=True,
            enforce_eager=config.sampling_enforce_eager,
            disable_custom_all_reduce=config.sampling_disable_custom_all_reduce,
            max_model_len=config.sampling_max_model_len or config.max_model_len,
            load_format="dummy",
            enable_sleep_mode=config.sampling_enable_sleep_mode,
            **kwargs,
        )
        conn.send({"ok": True})
    except Exception as exc:
        conn.send({"ok": False, "error": repr(exc)})
        raise

    while True:
        msg = conn.recv()
        op = msg.get("op")
        try:
            if op == "shutdown":
                conn.send({"ok": True})
                break
            if op == "collective_rpc":
                fn = _function_by_name(msg["fn_name"])
                result = llm.collective_rpc(fn, args=tuple(msg.get("args", ())))
                conn.send({"ok": True, "result": result})
            elif op == "generate":
                params = SamplingParams(**msg["sampling_params"])
                outputs = llm.generate([{"prompt_token_ids": msg["prompt_token_ids"]}], params)
                result = [
                    {
                        "token_ids": list(output.token_ids),
                        "finish_reason": getattr(output, "finish_reason", "length") or "length",
                    }
                    for output in outputs[0].outputs
                ]
                conn.send({"ok": True, "result": result})
            elif op == "sleep":
                llm.sleep(level=msg.get("level", 1))
                conn.send({"ok": True, "result": None})
            elif op == "wake_up":
                llm.wake_up(tags=msg.get("tags"))
                conn.send({"ok": True, "result": None})
            else:
                raise ValueError(f"Unknown vLLM runtime op: {op}")
        except Exception as exc:
            conn.send({"ok": False, "error": repr(exc)})


class VLLMRuntimeProxy:
    """Small synchronous proxy for a vLLM LLM object hosted in a child process."""

    def __init__(
        self,
        config: ModelConfig,
        all_rank_descriptors: list[dict[str, dict[str, Any]]] | None,
        verify_inject: bool,
    ) -> None:
        ctx = mp.get_context("spawn")
        parent_conn, child_conn = ctx.Pipe()
        self._conn = parent_conn
        self._proc = ctx.Process(
            target=_vllm_runtime_main,
            args=(child_conn, config.model_dump(mode="python"), all_rank_descriptors, verify_inject),
        )
        self._proc.start()
        init = self._conn.recv()
        if not init.get("ok"):
            raise RuntimeError(f"vLLM runtime init failed: {init.get('error')}")

    def _request(self, payload: dict[str, Any]) -> Any:
        self._conn.send(payload)
        response = self._conn.recv()
        if not response.get("ok"):
            raise RuntimeError(response.get("error"))
        return response.get("result")

    def collective_rpc(self, fn: Any, args: tuple[Any, ...] = ()) -> Any:
        return self._request(
            {"op": "collective_rpc", "fn_name": getattr(fn, "__name__"), "args": args}
        )

    def generate(self, prompts: list[dict[str, Any]], params: Any) -> list[Any]:
        result = self._request(
            {
                "op": "generate",
                "prompt_token_ids": prompts[0]["prompt_token_ids"],
                "sampling_params": {
                    "max_tokens": params.max_tokens,
                    "temperature": params.temperature,
                    "top_p": params.top_p,
                    "top_k": params.top_k,
                    "seed": params.seed,
                    "n": params.n,
                },
            }
        )
        return [
            SimpleNamespace(
                outputs=[
                    SimpleNamespace(token_ids=item["token_ids"], finish_reason=item["finish_reason"])
                    for item in result
                ]
            )
        ]

    def sleep(self, level: int = 1) -> None:
        self._request({"op": "sleep", "level": level})

    def wake_up(self, tags: list[str] | None = None) -> None:
        self._request({"op": "wake_up", "tags": tags})

    def shutdown(self) -> None:
        if self._proc.is_alive():
            try:
                self._request({"op": "shutdown"})
            except Exception:
                pass
            self._proc.join(timeout=5)
        if self._proc.is_alive():
            self._proc.terminate()
