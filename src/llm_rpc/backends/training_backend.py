import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from llm_rpc.backends.base_backend import BaseTrainingBackend
from llm_rpc.config import ModelConfig
from tinker import types


class HFTrainingBackend(BaseTrainingBackend):
    """A training backend using Hugging Face transformers."""

    def __init__(self, config: ModelConfig) -> None:
        from .hf_training_model import HFTrainingModel

        self.config = config
        self.model = HFTrainingModel.get_actor(config)

    async def async_init(self) -> None:
        await self.model.async_init.remote()

    async def create_lora_adapter(self, lora_id: str, lora_config: types.LoraConfig) -> None:
        """Create a LoRA adapter with the given ID and configuration."""
        await self.model.create_adapter.remote(lora_id, lora_config)

    async def forward(
        self,
        data: list[types.Datum],
        lora_id: str,
        loss_fn: types.LossFnType,
        loss_fn_config: dict[str, float] | None,
        backward: bool = False,
    ) -> types.ForwardBackwardOutput:
        """Forward pass (and backward if specified).

        Args:
            data: List of Datum objects containing input data.
            lora_id: The LoRA adapter ID to use.
            loss_fn: The loss function to apply.
            loss_fn_config: Optional configuration for the loss function.
            backward: Whether to perform backward pass.

        Returns:
            ForwardBackwardOutput: The output of the forward (and backward) pass.
        """
        return await self.model.forward.remote(
            data=data,
            lora_id=lora_id,
            loss_fn=loss_fn,
            loss_fn_config=loss_fn_config,
            backward=backward,
        )

    async def optim_step(
        self, adam_params: types.AdamParams, lora_id: str
    ) -> types.OptimStepResponse:
        """Perform an optimization step using Adam optimizer.

        Args:
            adam_params: Parameters for the Adam optimizer.

        Returns:
            OptimStepResponse: The response containing optimization metrics.
        """
        return await self.model.optim_step.remote(adam_params, lora_id)

    async def save_state(self, lora_id: str, lora_path: Path, optimizer: bool) -> None:
        """Save the state of the specified LoRA adapter."""
        await self.model.save_state.remote(
            lora_id=lora_id, save_path=lora_path, optimizer=optimizer
        )

    async def load_state(self, lora_id: str, lora_path: Path, optimizer: bool) -> None:
        """Load the state of the specified LoRA adapter from the given path."""
        await self.model.load_state.remote(
            lora_id=lora_id, load_path=lora_path, optimizer=optimizer
        )


@dataclass
class DummyTrainingBackend(BaseTrainingBackend):
    """A dummy training backend for testing purposes."""

    _lock: asyncio.Lock = field(init=False, repr=False)
    _weights: np.ndarray = field(init=False, repr=False)
    _adam_m: np.ndarray = field(init=False, repr=False)
    _adam_v: np.ndarray = field(init=False, repr=False)
    _beta1_power: float = field(init=False, default=1.0, repr=False)
    _beta2_power: float = field(init=False, default=1.0, repr=False)
    _pending_grad: np.ndarray | None = field(init=False, default=None, repr=False)
    _pending_examples: int = field(init=False, default=0, repr=False)
    _embedding_cache: dict[int, np.ndarray] = field(init=False, default_factory=dict, repr=False)
    step: int = field(init=False, default=0)

    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self.seed = config.seed
        self.hidden_dim = 16
        rng = np.random.default_rng(self.seed or 0)
        self._lock = asyncio.Lock()
        self._weights = rng.standard_normal(self.hidden_dim, dtype=np.float32)
        self._adam_m = np.zeros_like(self._weights)
        self._adam_v = np.zeros_like(self._weights)
        self._embedding_cache = {}

    async def async_init(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Forward / backward helpers
    # ------------------------------------------------------------------
    async def forward(
        self,
        data: list[types.Datum],
        lora_id: str,
        loss_fn: types.LossFnType,
        loss_fn_config: dict[str, float] | None,
        backward: bool = False,
    ) -> types.ForwardBackwardOutput:
        return await self._run_step(data, backward=backward)

    async def _run_step(
        self, data: list[types.Datum], *, backward: bool
    ) -> types.ForwardBackwardOutput:
        outputs: list[types.LossFnOutput] = []
        total_loss = 0.0
        grad_accum = np.zeros_like(self._weights)
        for datum in data:
            prompt_tokens = datum.model_input.to_ints()
            target_tokens = self._target_tokens(datum)
            prompt_vec = self._vectorize(prompt_tokens)
            target_scalar = self._target_scalar(target_tokens)
            prediction = float(np.dot(self._weights, prompt_vec))
            loss = (prediction - target_scalar) ** 2
            total_loss += loss
            if backward:
                grad = 2 * (prediction - target_scalar) * prompt_vec
                grad_accum += grad
            logprob_tensor = types.TensorData(
                data=[float(-abs(prediction - target_scalar))] * max(len(target_tokens), 1),
                dtype="float32",
                shape=[max(len(target_tokens), 1)],
            )
            outputs.append({"logprobs": logprob_tensor})

        metrics = {
            "loss:mean": total_loss / max(len(data), 1),
            "step:max": float(self.step),
        }
        if backward:
            grad_norm = float(np.linalg.norm(grad_accum) / max(len(data), 1))
            metrics["grad_norm:mean"] = grad_norm
            async with self._lock:
                if self._pending_grad is None:
                    self._pending_grad = grad_accum
                else:
                    self._pending_grad += grad_accum
                self._pending_examples += len(data)

        return types.ForwardBackwardOutput(
            loss_fn_output_type="ToyLoss",
            loss_fn_outputs=outputs,
            metrics=metrics,
        )

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------
    async def optim_step(
        self, adam_params: types.AdamParams, lora_id: str
    ) -> types.OptimStepResponse:
        async with self._lock:
            grad = self._pending_grad
            examples = self._pending_examples
            self._pending_grad = None
            self._pending_examples = 0

        if grad is None or not np.any(grad):
            return types.OptimStepResponse(
                metrics={
                    "learning_rate:mean": adam_params.learning_rate,
                    "step:max": float(self.step),
                }
            )

        grad = grad / max(examples, 1)
        if adam_params.grad_clip_norm > 0:
            norm = np.linalg.norm(grad)
            if norm > adam_params.grad_clip_norm:
                grad *= adam_params.grad_clip_norm / max(norm, 1e-12)

        if adam_params.weight_decay:
            grad += adam_params.weight_decay * self._weights

        beta1 = adam_params.beta1
        beta2 = adam_params.beta2

        self._adam_m = beta1 * self._adam_m + (1 - beta1) * grad
        self._adam_v = beta2 * self._adam_v + (1 - beta2) * (grad**2)
        self._beta1_power *= beta1
        self._beta2_power *= beta2
        m_hat = self._adam_m / (1 - self._beta1_power + 1e-12)
        v_hat = self._adam_v / (1 - self._beta2_power + 1e-12)

        update = adam_params.learning_rate * m_hat / (np.sqrt(v_hat) + adam_params.eps)
        self._weights -= update
        self.step += 1

        metrics = {
            "learning_rate:mean": adam_params.learning_rate,
            "step:max": float(self.step),
            "update_norm:mean": float(np.linalg.norm(update)),
        }
        return types.OptimStepResponse(metrics=metrics)

    async def save_state(self, lora_id: str, lora_path: Path, optimizer: bool) -> None:
        pass

    async def load_state(self, lora_id: str, lora_path: Path, optimizer: bool) -> None:
        pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _target_tokens(self, datum: types.Datum) -> list[int]:
        if not datum.loss_fn_inputs:
            return datum.model_input.to_ints()
        tensor = datum.loss_fn_inputs.get("target_tokens")
        if tensor is None:
            return datum.model_input.to_ints()
        return [int(value) for value in tensor.data]

    def _vectorize(self, tokens: Sequence[int]) -> np.ndarray:
        if not tokens:
            return np.zeros(self.hidden_dim, dtype=np.float32)
        vecs = [self._token_embedding(token) for token in tokens]
        return np.mean(vecs, axis=0)

    def _token_embedding(self, token_id: int) -> np.ndarray:
        cached = self._embedding_cache.get(token_id)
        if cached is None:
            rng = np.random.default_rng(self.seed + token_id)
            cached = rng.standard_normal(self.hidden_dim, dtype=np.float32)
            self._embedding_cache[token_id] = cached
        return cached

    def _safe_mean(self, values: Sequence[int]) -> float:
        if not values:
            return 0.0
        return float(sum(values) / len(values))

    def _target_scalar(self, tokens: Sequence[int]) -> float:
        if not tokens:
            return 0.0
        return np.tanh(self._safe_mean(tokens) / 100.0)
