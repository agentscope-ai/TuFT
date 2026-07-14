"""Custom vLLM GPU worker for TuFT.

Adapted from Trinity-RFT (``trinity/common/models/vllm_worker.py`` and
``trinity/common/models/vllm_patch/worker_patch.py``, Apache-2.0,
https://github.com/agentscope-ai/Trinity-RFT), with the weight-sync pieces
(verl MoE weight-loader patch, checkpoint weight-transfer engine) removed --
TuFT delivers trained weights to the sampler as on-disk LoRA adapters and
never syncs full model weights into vLLM.

WHY THIS PATCH EXISTS:
  Upstream vLLM does not apply temperature scaling when computing *prompt*
  logprobs (sampled-token logprobs are temperature-scaled natively via the
  ``logprobs_mode="processed_logprobs"`` engine argument). TuFT's
  Tinker-compatible sampling API exposes prompt logprobs, which are consumed
  by importance-sampling-style losses and must therefore be consistent with
  the actual sampling distribution when temperature != 1.0.
  ``patch_vllm_prompt_logprobs`` replaces
  ``GPUModelRunner._get_prompt_logprobs_dict`` with a copy of the upstream
  implementation plus the missing ``logits /= temperature`` step (the blocks
  marked ``PATCH START``/``PATCH END`` below are the only deltas).

This module is imported inside vLLM worker processes (via the
``worker_cls`` engine argument set in ``vllm_engine.py``), so it may import
vllm at module level; nothing in TuFT's server process imports it.
"""

from types import MethodType
from typing import Any, Optional

import torch
import vllm
from packaging.version import InvalidVersion, parse as parse_version
from vllm.v1.outputs import LogprobsTensors
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm.v1.worker.gpu_worker import Worker as VLLMGPUWorker


def get_vllm_version():
    """Parsed vLLM version (self-compiled builds fall back to the lowest supported)."""
    try:
        return parse_version(vllm.__version__)
    except InvalidVersion:
        return parse_version("0.19.1")


def patch_vllm_prompt_logprobs(model_runner: GPUModelRunner):  # noqa: C901
    """Patch vLLM model runner to apply temperature scaling to prompt logprobs."""
    version = get_vllm_version()

    def _get_prompt_logprobs_dict_v21(
        self,
        hidden_states: torch.Tensor,
        num_scheduled_tokens: dict[str, int],
    ) -> dict[str, LogprobsTensors | None]:
        """Patched version of _get_prompt_logprobs_dict for vLLM v0.21.0."""
        num_prompt_logprobs_dict = self.num_prompt_logprobs
        if not num_prompt_logprobs_dict:
            return {}

        prompt_logprobs_dict: dict[str, LogprobsTensors | None] = {}

        # Since prompt logprobs are a rare feature, prioritize simple,
        # maintainable loop over optimal performance.
        completed_prefill_reqs = []
        for req_id, num_prompt_logprobs in num_prompt_logprobs_dict.items():
            num_tokens = num_scheduled_tokens.get(req_id)
            if num_tokens is None:
                # This can happen if the request was preempted in prefill stage.
                continue

            # Get metadata for this request.
            request = self.requests[req_id]
            if request.prompt_token_ids is None:
                # Prompt logprobs is incompatible with prompt embeddings
                continue

            num_prompt_tokens = len(request.prompt_token_ids)
            prompt_token_ids = torch.tensor(request.prompt_token_ids).to(
                self.device, non_blocking=True
            )

            # Set up target LogprobsTensors object.
            logprobs_tensors = request.in_progress_prompt_logprobs_cpu
            if logprobs_tensors is None:
                # Create empty logprobs CPU tensors for the entire prompt.
                # If chunked, we'll copy in slice by slice.
                logprobs_tensors = LogprobsTensors.empty_cpu(
                    num_prompt_tokens - 1, num_prompt_logprobs + 1
                )
                request.in_progress_prompt_logprobs_cpu = logprobs_tensors

            # Determine number of logits to retrieve.
            start_idx = request.num_computed_tokens
            start_tok = start_idx + 1
            num_remaining_tokens = num_prompt_tokens - start_tok
            if num_tokens <= num_remaining_tokens:
                # This is a chunk, more tokens remain.
                # In the == case, there are no more prompt logprobs to produce
                # but we want to defer returning them to the next step where we
                # have new generated tokens to return.
                num_logits = num_tokens
            else:
                # This is the last chunk of prompt tokens to return.
                num_logits = num_remaining_tokens
                completed_prefill_reqs.append(req_id)
                prompt_logprobs_dict[req_id] = logprobs_tensors

            if num_logits <= 0:
                # This can happen for the final chunk if we prefilled exactly
                # (num_prompt_tokens - 1) tokens for this request in the prior
                # step. There are no more prompt logprobs to produce.
                continue

            # Get the logits corresponding to this req's prompt tokens.
            # If this is a partial request (i.e. chunked prefill),
            # then there is prompt logprob generated for each index.
            req_idx = self.input_batch.req_id_to_index[req_id]
            offset = self.query_start_loc.np[req_idx].item()
            prompt_hidden_states = hidden_states[offset : offset + num_logits]
            logits = self.model.compute_logits(prompt_hidden_states)

            # PATCH START
            temp = request.sampling_params.temperature
            if temp >= 1e-5:
                logits.div_(temp)
            # PATCH END

            # Get the "target" tokens for each index. For prompt at index i,
            # the token at prompt index i+1 is the "sampled" token we want
            # to gather the logprob for.
            tgt_token_ids = prompt_token_ids[start_tok : start_tok + num_logits]

            # Compute prompt logprobs.
            logprobs = self.sampler.compute_logprobs(logits)
            token_ids, logprobs, ranks, _ = self.sampler.gather_logprobs(
                logprobs, num_prompt_logprobs, tgt_token_ids
            )

            # Transfer GPU->CPU async.
            chunk_slice = slice(start_idx, start_idx + num_logits)
            logprobs_tensors.logprob_token_ids[chunk_slice].copy_(token_ids, non_blocking=True)
            logprobs_tensors.logprobs[chunk_slice].copy_(logprobs, non_blocking=True)
            logprobs_tensors.selected_token_ranks[chunk_slice].copy_(ranks, non_blocking=True)

        # Remove requests that have completed prefill from the batch
        # num_prompt_logprobs_dict.
        for req_id in completed_prefill_reqs:
            del num_prompt_logprobs_dict[req_id]
            self.requests[req_id].in_progress_prompt_logprobs_cpu = None

        # Must synchronize the non-blocking GPU->CPU transfers.
        if prompt_logprobs_dict:
            self._sync_device()

        return prompt_logprobs_dict

    def _get_prompt_logprobs_dict_v12(
        self,
        hidden_states: torch.Tensor,
        num_scheduled_tokens: dict[str, int],
    ) -> dict[str, Optional[LogprobsTensors]]:
        """Patched version of _get_prompt_logprobs_dict.

        This is a monkey-patched version of `_get_prompt_logprobs_dict` from
        `vllm.v1.worker.gpu_model_runner.GPUModelRunner` (vLLM versions
        0.12.0 to 0.15.1).

        The original function does not apply temperature scaling to logits when
        calculating prompt logprobs, which can lead to incorrect logprob values
        when the temperature is not 1.0. This patch adds the missing
        temperature scaling.
        """
        num_prompt_logprobs_dict = self.num_prompt_logprobs
        if not num_prompt_logprobs_dict:
            return {}

        in_progress_dict = self.input_batch.in_progress_prompt_logprobs_cpu
        prompt_logprobs_dict: dict[str, Optional[LogprobsTensors]] = {}

        # Since prompt logprobs are a rare feature, prioritize simple,
        # maintainable loop over optimal performance.
        completed_prefill_reqs = []
        for req_id, num_prompt_logprobs in num_prompt_logprobs_dict.items():
            num_tokens = num_scheduled_tokens.get(req_id)
            if num_tokens is None:
                # This can happen if the request was preempted in prefill stage.
                continue

            # Get metadata for this request.
            request = self.requests[req_id]
            if request.prompt_token_ids is None:
                # Prompt logprobs is incompatible with prompt embeddings
                continue

            num_prompt_tokens = len(request.prompt_token_ids)
            prompt_token_ids = torch.tensor(request.prompt_token_ids).to(
                self.device, non_blocking=True
            )

            # Set up target LogprobsTensors object.
            logprobs_tensors = in_progress_dict.get(req_id)
            if not logprobs_tensors:
                # Create empty logprobs CPU tensors for the entire prompt.
                # If chunked, we'll copy in slice by slice.
                logprobs_tensors = LogprobsTensors.empty_cpu(
                    num_prompt_tokens - 1, num_prompt_logprobs + 1
                )
                in_progress_dict[req_id] = logprobs_tensors

            # Determine number of logits to retrieve.
            start_idx = request.num_computed_tokens
            start_tok = start_idx + 1
            num_remaining_tokens = num_prompt_tokens - start_tok
            if num_tokens <= num_remaining_tokens:
                # This is a chunk, more tokens remain.
                # In the == case, there are no more prompt logprobs to produce
                # but we want to defer returning them to the next step where we
                # have new generated tokens to return.
                num_logits = num_tokens
            else:
                # This is the last chunk of prompt tokens to return.
                num_logits = num_remaining_tokens
                completed_prefill_reqs.append(req_id)
                prompt_logprobs_dict[req_id] = logprobs_tensors

            if num_logits <= 0:
                # This can happen for the final chunk if we prefilled exactly
                # (num_prompt_tokens - 1) tokens for this request in the prior
                # step. There are no more prompt logprobs to produce.
                continue

            # Get the logits corresponding to this req's prompt tokens.
            # If this is a partial request (i.e. chunked prefill),
            # then there is prompt logprob generated for each index.
            req_idx = self.input_batch.req_id_to_index[req_id]
            offset = self.query_start_loc.np[req_idx].item()
            prompt_hidden_states = hidden_states[offset : offset + num_logits]
            logits = self.model.compute_logits(prompt_hidden_states)

            # PATCH START
            temp = request.sampling_params.temperature
            if temp >= 1e-5:
                logits.div_(temp)
            # PATCH END

            # Get the "target" tokens for each index. For prompt at index i,
            # the token at prompt index i+1 is the "sampled" token we want
            # to gather the logprob for.
            tgt_token_ids = prompt_token_ids[start_tok : start_tok + num_logits]

            # Compute prompt logprobs.
            logprobs = self.sampler.compute_logprobs(logits)
            logprob_tensors = self.sampler.gather_logprobs(
                logprobs, num_prompt_logprobs, tgt_token_ids
            )

            # Transfer GPU->CPU async.
            chunk_slice = slice(start_idx, start_idx + num_logits)
            logprobs_tensors.logprob_token_ids[chunk_slice].copy_(
                logprob_tensors.logprob_token_ids, non_blocking=True
            )
            logprobs_tensors.logprobs[chunk_slice].copy_(
                logprob_tensors.logprobs, non_blocking=True
            )
            logprobs_tensors.selected_token_ranks[chunk_slice].copy_(
                logprob_tensors.selected_token_ranks, non_blocking=True
            )

        # Remove requests that have completed prefill from the batch
        # num_prompt_logprobs_dict.
        for req_id in completed_prefill_reqs:
            del num_prompt_logprobs_dict[req_id]
            del in_progress_dict[req_id]

        # Must synchronize the non-blocking GPU->CPU transfers.
        if prompt_logprobs_dict:
            self._sync_device()

        return prompt_logprobs_dict

    if version >= parse_version("0.12.0") and version < parse_version("0.21.0"):
        model_runner._get_prompt_logprobs_dict = MethodType(
            _get_prompt_logprobs_dict_v12, model_runner
        )
    elif version >= parse_version("0.21.0"):
        model_runner._get_prompt_logprobs_dict = MethodType(
            _get_prompt_logprobs_dict_v21, model_runner
        )
    else:
        raise ValueError(f"Unsupported vllm version for patching: {vllm.__version__}.")


class TuFTGPUWorker(VLLMGPUWorker):
    """vLLM GPU worker with TuFT patches applied.

    ``apply_patches`` is invoked on every worker right after engine creation
    via ``collective_rpc`` (see ``VLLMEngine.prepare`` in ``vllm_engine.py``).
    """

    def apply_patches(self, *args: Any, **kwargs: Any) -> None:
        """Apply necessary patches to vLLM inside the worker process."""
        patch_vllm_prompt_logprobs(self.model_runner)
