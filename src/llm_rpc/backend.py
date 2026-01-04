"""Toy backend implementations used by the local LLM-RPC server."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from tinker import types


def _safe_mean(values: Sequence[int]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


@dataclass
class ModelBackend:
    """Deterministic toy backend with lightweight gradient/optimizer tracking."""

    base_model: str
    lora_rank: int
    seed: int = 0
    hidden_dim: int = 16

    _lock: threading.Lock = field(init=False, repr=False)
    _weights: np.ndarray = field(init=False, repr=False)
    _adam_m: np.ndarray = field(init=False, repr=False)
    _adam_v: np.ndarray = field(init=False, repr=False)
    _beta1_power: float = field(init=False, default=1.0, repr=False)
    _beta2_power: float = field(init=False, default=1.0, repr=False)
    _pending_grad: np.ndarray | None = field(init=False, default=None, repr=False)
    _pending_examples: int = field(init=False, default=0, repr=False)
    _embedding_cache: dict[int, np.ndarray] = field(init=False, default_factory=dict, repr=False)
    step: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed or 0)
        self._lock = threading.Lock()
        self._weights = rng.standard_normal(self.hidden_dim, dtype=np.float32)
        self._adam_m = np.zeros_like(self._weights)
        self._adam_v = np.zeros_like(self._weights)

    # ------------------------------------------------------------------
    # Forward / backward helpers
    # ------------------------------------------------------------------
    def forward(
        self, data: list[types.Datum], _: types.LossFnType, __: dict[str, float] | None
    ) -> types.ForwardBackwardOutput:
        return self._run_step(data, backward=False)

    def forward_backward(
        self,
        data: list[types.Datum],
        _: types.LossFnType,
        __: dict[str, float] | None,
    ) -> types.ForwardBackwardOutput:
        return self._run_step(data, backward=True)

    def _run_step(self, data: list[types.Datum], *, backward: bool) -> types.ForwardBackwardOutput:
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
            with self._lock:
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
    def optim_step(self, adam_params: types.AdamParams) -> types.OptimStepResponse:
        with self._lock:
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

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------
    def sample(
        self,
        prompt: types.ModelInput,
        num_samples: int,
        sampling_params: types.SamplingParams,
        include_prompt_logprobs: bool,
        topk_prompt_logprobs: int,
    ) -> types.SampleResponse:
        prompt_tokens = prompt.to_ints()
        max_tokens = sampling_params.max_tokens or 16
        sequences: list[types.SampledSequence] = []
        for _ in range(num_samples):
            generated = self._generate_tokens(prompt_tokens, max_tokens)
            seq = types.SampledSequence(
                stop_reason="length",
                tokens=generated,
                logprobs=[-0.3 for _ in generated],
            )
            sequences.append(seq)
        prompt_logprobs = None
        topk_prompt = None
        if include_prompt_logprobs:
            prompt_logprobs = [-0.1 if tok is not None else None for tok in prompt_tokens]
        if topk_prompt_logprobs > 0:
            topk_prompt = [
                [
                    (token, round(-0.05 - idx * 0.01, 4))
                    for idx, token in enumerate(prompt_tokens[:topk_prompt_logprobs])
                ]
                if token is not None
                else None
                for token in prompt_tokens
            ]
        return types.SampleResponse(
            sequences=sequences,
            prompt_logprobs=prompt_logprobs,
            topk_prompt_logprobs=topk_prompt,
        )

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------
    def snapshot_state(self) -> dict[str, Any]:
        return {
            "weights": self._weights.astype(float).tolist(),
            "adam_m": self._adam_m.astype(float).tolist(),
            "adam_v": self._adam_v.astype(float).tolist(),
            "step": self.step,
            "beta1_power": self._beta1_power,
            "beta2_power": self._beta2_power,
        }

    def load_state(self, state: dict[str, Any]) -> None:
        with self._lock:
            self._weights = np.array(state.get("weights", self._weights.tolist()), dtype=np.float32)
            self._adam_m = np.array(state.get("adam_m", self._adam_m.tolist()), dtype=np.float32)
            self._adam_v = np.array(state.get("adam_v", self._adam_v.tolist()), dtype=np.float32)
            self.step = int(state.get("step", self.step))
            self._beta1_power = float(state.get("beta1_power", self._beta1_power))
            self._beta2_power = float(state.get("beta2_power", self._beta2_power))
            self._pending_grad = None
            self._pending_examples = 0

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

    def _target_scalar(self, tokens: Sequence[int]) -> float:
        if not tokens:
            return 0.0
        return np.tanh(_safe_mean(tokens) / 100.0)

    def _generate_tokens(self, prompt_tokens: list[int], max_tokens: int) -> list[int]:
        start = prompt_tokens[-1] if prompt_tokens else (abs(self.seed) % 32000) + 1
        return [(start + i) % 32000 for i in range(1, max_tokens + 1)]


def build_backend(base_model: str, lora_rank: int, seed: int | None = None) -> ModelBackend:
    return ModelBackend(base_model=base_model, lora_rank=lora_rank, seed=seed or 0)
