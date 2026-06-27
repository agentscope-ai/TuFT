"""Tinker SDK backend implementation."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List

import tinker
import torch
from tinker import types
from tinker.types.tensor_data import TensorData

from .base import AdapterConfig, SampleResult, TrainingBackend, TrainStepResult


logger = logging.getLogger(__name__)


class TinkerBackend(TrainingBackend):
    """Backend that calls the Tinker SDK for training and sampling."""

    def __init__(self):
        self._service_client: tinker.ServiceClient = None
        self._training_clients: Dict[str, Any] = {}
        self._sampling_clients: Dict[str, Any] = {}
        self._adapter_configs: Dict[str, AdapterConfig] = {}
        self._weight_paths: Dict[str, List[str]] = {}
        self._weight_versions: Dict[str, int] = {}
        self._base_model: str = ""
        self._tokenizer: Any = None

    async def initialize(self, config: Dict[str, Any]) -> None:
        base_url = config.get("base_url", "http://localhost:10610")
        api_key = config.get("api_key")
        self._base_model = config.get("base_model", "Qwen/Qwen3-4B")

        logger.info(f"Connecting to Tinker service at {base_url}")
        self._service_client = await asyncio.to_thread(
            tinker.ServiceClient,
            base_url=base_url,
            api_key=api_key,
        )
        logger.info("Tinker service connected")

    async def create_adapter(self, adapter_id: str, adapter_config: AdapterConfig) -> None:
        logger.info(f"Creating adapter '{adapter_id}' (rank={adapter_config.lora_rank})")

        tc = await asyncio.to_thread(
            self._service_client.create_lora_training_client,
            base_model=self._base_model,
            rank=adapter_config.lora_rank,
            train_mlp=True,
            train_attn=True,
            train_unembed=True,
        )

        self._training_clients[adapter_id] = tc
        self._adapter_configs[adapter_id] = adapter_config
        self._weight_paths[adapter_id] = []
        self._weight_versions[adapter_id] = 0

        # Initialize tokenizer from first adapter created
        if self._tokenizer is None:
            self._tokenizer = await asyncio.to_thread(tc.get_tokenizer)
            logger.info(f"Tokenizer initialized: {type(self._tokenizer).__name__}")

    async def sync_weights(self, adapter_id: str) -> int:
        tc = self._training_clients[adapter_id]
        version = self._weight_versions[adapter_id]
        name = f"{adapter_id}_v{version:04d}"

        save_result = await asyncio.to_thread(
            lambda: tc.save_weights_for_sampler(name=name).result()
        )

        sc = await asyncio.to_thread(
            self._service_client.create_sampling_client,
            model_path=save_result.path,
        )

        self._sampling_clients[adapter_id] = sc
        self._weight_paths[adapter_id].append(save_result.path)
        self._weight_versions[adapter_id] = version + 1

        # Clean up old checkpoints (keep only the last 2)
        paths = self._weight_paths[adapter_id]
        if len(paths) > 2:
            old_path = paths.pop(0)
            asyncio.create_task(self._delete_checkpoint(old_path))

        return version

    async def sample(
        self,
        adapter_id: str,
        prompt_tokens: List[int],
        num_samples: int,
        max_tokens: int,
        temperature: float,
    ) -> List[SampleResult]:
        sc = self._sampling_clients[adapter_id]
        prompt = types.ModelInput(chunks=[types.EncodedTextChunk(tokens=prompt_tokens)])
        sp = types.SamplingParams(max_tokens=max_tokens, temperature=temperature)

        result = await asyncio.to_thread(
            lambda: sc.sample(
                prompt=prompt,
                num_samples=num_samples,
                sampling_params=sp,
            ).result()
        )

        samples = []
        for seq in result.sequences:
            tokens = list(seq.tokens)
            logprobs = list(seq.logprobs) if seq.logprobs else []
            text = self._tokenizer.decode(tokens, skip_special_tokens=True)
            samples.append(SampleResult(tokens=tokens, logprobs=logprobs, text=text))

        return samples

    async def train_step(
        self,
        adapter_id: str,
        training_datums: List[Dict[str, Any]],
    ) -> TrainStepResult:
        tc = self._training_clients[adapter_id]
        adapter_config = self._adapter_configs[adapter_id]

        # Build Tinker-specific datums
        tinker_datums = [self._build_tinker_datum(d) for d in training_datums]

        # Forward-backward pass
        await asyncio.to_thread(
            lambda: tc.forward_backward(
                tinker_datums,
                loss_fn="importance_sampling",
            ).result()
        )

        # Optimizer step
        adam_params = types.AdamParams(
            learning_rate=adapter_config.learning_rate,
            beta1=0.9,
            beta2=0.95,
            eps=1e-8,
        )
        await asyncio.to_thread(lambda: tc.optim_step(adam_params).result())

        version = self._weight_versions[adapter_id]
        return TrainStepResult(loss=None, weight_version=version)

    def get_tokenizer(self) -> Any:
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer not initialized. Call create_adapter first.")
        return self._tokenizer

    async def compute_training_logprobs(
        self,
        adapter_id: str,
        items: List[Dict[str, Any]],
        temperature: float = 1.0,
    ) -> List[List[float]]:
        """Compute training-side per-response-token logprobs via forward(cross_entropy).

        We build one Datum per item, exactly aligned with how `_build_tinker_datum`
        builds the training input, but with `cross_entropy` loss and zero `weights`
        so no gradient/loss flows back. The returned logprobs are extracted at
        the response-token positions.

        Args:
            temperature: Temperature applied to logits before log_softmax.
                Passed via loss_fn_config to align with sampling-side logprobs
                (which are computed after temperature scaling in vLLM V0).
        """
        import numpy as np

        if not items:
            return []

        tc = self._training_clients[adapter_id]

        datums: List[types.Datum] = []
        slices: List[tuple] = []  # (start_pos, end_pos) per item
        for it in items:
            prompt_tokens = it["prompt_tokens"]
            response_tokens = it["response_tokens"]

            ob_len = len(prompt_tokens) - 1
            model_input_tokens = prompt_tokens + response_tokens[:-1]
            model_input = types.ModelInput(
                chunks=[types.EncodedTextChunk(tokens=model_input_tokens)]
            )

            target_tokens = [0] * ob_len + response_tokens
            weights = [0.0] * len(target_tokens)  # no loss contribution

            datum = types.Datum(
                model_input=model_input,
                loss_fn_inputs={
                    "target_tokens": TensorData.from_torch(
                        torch.tensor(target_tokens, dtype=torch.long)
                    ),
                    "weights": TensorData.from_torch(torch.tensor(weights, dtype=torch.float32)),
                },
            )
            datums.append(datum)
            slices.append((ob_len, ob_len + len(response_tokens)))

        # Pass temperature via loss_fn_config so the training backend applies
        # logits /= temperature before computing log_softmax.  This aligns with
        # the sampling-side logprobs which are computed after temperature scaling.
        loss_fn_config = {}
        if temperature != 1.0:
            loss_fn_config["temperature"] = temperature

        fwd_result = await asyncio.to_thread(
            lambda: tc.forward(
                datums,
                loss_fn="cross_entropy",
                loss_fn_config=loss_fn_config if loss_fn_config else None,
            ).result()
        )

        out: List[List[float]] = []
        for (start, end), loss_out in zip(slices, fwd_result.loss_fn_outputs):
            lp_td = loss_out["logprobs"]
            arr = np.array(lp_td.data)
            if lp_td.shape is not None:
                arr = arr.reshape(lp_td.shape)
            out.append(arr[start:end].tolist())
        return out

    async def cleanup(self, adapter_id: str) -> None:
        """Delete all saved checkpoints for an adapter."""
        paths = self._weight_paths.get(adapter_id, [])
        for path in paths:
            await self._delete_checkpoint(path)
        self._weight_paths[adapter_id] = []
        logger.info(f"Cleaned up {len(paths)} checkpoints for adapter '{adapter_id}'")

    async def _delete_checkpoint(self, path: str) -> None:
        """Delete a single checkpoint from Tinker storage."""
        try:
            rest_client = self._service_client.create_rest_client()
            await asyncio.to_thread(
                lambda: rest_client.delete_checkpoint_from_tinker_path(path).result()
            )
        except Exception as e:
            logger.warning(f"Failed to delete checkpoint {path}: {e}")

    def _build_tinker_datum(self, datum_dict: Dict[str, Any]) -> types.Datum:
        """Convert a generic datum dict to a Tinker types.Datum."""
        prompt_tokens = datum_dict["prompt_tokens"]
        response_tokens = datum_dict["response_tokens"]
        sampling_logprobs = datum_dict["sampling_logprobs"]
        advantage = datum_dict["advantage"]

        # model_input = prompt + response[:-1] (predict next token)
        ob_len = len(prompt_tokens) - 1
        model_input_tokens = prompt_tokens + response_tokens[:-1]
        model_input = types.ModelInput(chunks=[types.EncodedTextChunk(tokens=model_input_tokens)])

        # Targets and features aligned with model_input length
        target_tokens = [0] * ob_len + response_tokens
        padded_logprobs = [0.0] * ob_len + sampling_logprobs
        padded_advantages = [0.0] * ob_len + [advantage] * len(response_tokens)

        # Ensure length consistency
        expected_len = len(model_input_tokens)
        assert len(target_tokens) == expected_len, (
            f"target_tokens length {len(target_tokens)} != model_input length {expected_len}"
        )
        assert len(padded_logprobs) == expected_len, (
            f"padded_logprobs length {len(padded_logprobs)} != model_input length {expected_len}"
        )
        assert len(padded_advantages) == expected_len, (
            f"padded_advantages length {len(padded_advantages)} != model_input length {expected_len}"
        )

        return types.Datum(
            model_input=model_input,
            loss_fn_inputs={
                "target_tokens": TensorData.from_torch(
                    torch.tensor(target_tokens, dtype=torch.long)
                ),
                "logprobs": TensorData.from_torch(
                    torch.tensor(padded_logprobs, dtype=torch.float32)
                ),
                "advantages": TensorData.from_torch(
                    torch.tensor(padded_advantages, dtype=torch.float32)
                ),
            },
        )
