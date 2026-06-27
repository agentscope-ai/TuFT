"""Per-tenant metrics collection and output formatting."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple


@dataclass
class TenantMetrics:
    """Collects metrics for a single tenant during simulation."""

    tenant_id: str

    # Reward tracking
    _sample_rewards: List[float] = field(default_factory=list)
    _sample_latencies_ms: List[float] = field(default_factory=list)

    # Training step tracking: (step, mean_reward_of_batch)
    _reward_curve: List[Tuple[int, float]] = field(default_factory=list)

    # Eval tracking: (step, accuracy)
    _eval_curve: List[Tuple[int, float]] = field(default_factory=list)

    # Staleness tracking
    _stalenesses: List[float] = field(default_factory=list)

    # Per-step staleness distribution: list of (step, {version_gap: count})
    _staleness_distributions: List[Tuple[int, Dict[int, int]]] = field(default_factory=list)

    # Per-step logprob mismatch summaries (sampling vs training).
    # Each entry is a flat dict of summary statistics for that train step.
    _logprob_mismatch: List[Dict[str, Any]] = field(default_factory=list)

    # Counters
    train_steps_completed: int = 0
    total_samples: int = 0

    # Cumulative wall-clock time spent in sampling, training, and sync_weights (seconds)
    total_sampling_seconds: float = 0.0
    total_training_seconds: float = 0.0
    total_sync_weights_seconds: float = 0.0

    def record_sample(self, reward: float, latency_s: float) -> None:
        """Record a single sampling event."""
        self._sample_rewards.append(reward)
        self._sample_latencies_ms.append(latency_s * 1000)
        self.total_samples += 1
        self.total_sampling_seconds += latency_s

    def record_train_step(
        self,
        step: int,
        mean_batch_reward: float,
        train_duration_s: float = 0.0,
        sync_duration_s: float = 0.0,
    ) -> None:
        """Record a completed training step."""
        self._reward_curve.append((step, mean_batch_reward))
        self.train_steps_completed = step
        self.total_training_seconds += train_duration_s
        self.total_sync_weights_seconds += sync_duration_s

    def record_eval(self, step: int, accuracy: float) -> None:
        """Record an evaluation result."""
        self._eval_curve.append((step, accuracy))

    def record_staleness(self, step: int, stalenesses: List[float]) -> None:
        """Record staleness values from a training batch.

        Also records per-step distribution: how many samples have gap=0, gap=1, etc.
        """
        self._stalenesses.extend(stalenesses)

        # Build distribution: version_gap -> count
        distribution: Dict[int, int] = {}
        for s in stalenesses:
            gap = int(s)
            distribution[gap] = distribution.get(gap, 0) + 1
        self._staleness_distributions.append((step, distribution))

    def record_logprob_mismatch(
        self,
        step: int,
        weight_version: int,
        sampling_logprobs: List[List[float]],
        training_logprobs: List[List[float]],
        stalenesses: List[float],
    ) -> Dict[str, Any]:
        """Record per-step sampling-vs-training logprob mismatch summary.

        Stores summary stats (returned and appended to internal buffer).
        Per-token raw arrays are NOT kept here to keep memory bounded; callers
        that want them should dump separately to JSONL.
        """
        import math

        # Per-item cumulative logprobs (used for IS-weight = exp(sum diff))
        seq_diff_sum: List[float] = []
        seq_samp_sum: List[float] = []
        seq_train_sum: List[float] = []
        all_diffs: List[float] = []
        all_abs: List[float] = []
        n_tokens = 0

        for samp, train in zip(sampling_logprobs, training_logprobs, strict=False):
            # Defensive: sometimes lengths can differ by 1 due to truncation; align.
            n = min(len(samp), len(train))
            if n == 0:
                seq_diff_sum.append(0.0)
                seq_samp_sum.append(0.0)
                seq_train_sum.append(0.0)
                continue
            ssum = 0.0
            tsum = 0.0
            dsum = 0.0
            for i in range(n):
                d = samp[i] - train[i]
                ssum += samp[i]
                tsum += train[i]
                dsum += d
                all_diffs.append(d)
                all_abs.append(abs(d))
                n_tokens += 1
            seq_diff_sum.append(dsum)
            seq_samp_sum.append(ssum)
            seq_train_sum.append(tsum)

        def _mean(xs):
            return float(sum(xs) / len(xs)) if xs else 0.0

        def _std(xs, mu):
            return float((sum((x - mu) ** 2 for x in xs) / len(xs)) ** 0.5) if xs else 0.0

        def _max_abs(xs):
            return float(max(xs)) if xs else 0.0

        mean_diff = _mean(all_diffs)
        mean_abs_diff = _mean(all_abs)
        max_abs_diff = _max_abs(all_abs)
        std_diff = _std(all_diffs, mean_diff)

        # IS weights per-sequence = exp(cum_diff). Clamp to avoid overflow.
        is_weights: List[float] = []
        for s in seq_diff_sum:
            try:
                is_weights.append(math.exp(max(min(s, 50.0), -50.0)))
            except OverflowError:
                is_weights.append(float("inf"))

        mean_is = _mean(is_weights)
        std_is = _std(is_weights, mean_is)

        def _quantile(xs, q):
            if not xs:
                return 0.0
            ys = sorted(xs)
            k = max(0, min(len(ys) - 1, int(q * (len(ys) - 1))))
            return float(ys[k])

        clip_01 = (
            sum(1 for w in is_weights if abs(w - 1.0) > 0.1) / len(is_weights)
            if is_weights
            else 0.0
        )
        clip_02 = (
            sum(1 for w in is_weights if abs(w - 1.0) > 0.2) / len(is_weights)
            if is_weights
            else 0.0
        )

        mean_staleness = _mean(stalenesses)

        summary = {
            "step": step,
            "weight_version": weight_version,
            "n_items": len(sampling_logprobs),
            "n_tokens": n_tokens,
            "mean_sampling_logprob": _mean(seq_samp_sum),
            "mean_training_logprob": _mean(seq_train_sum),
            "mean_diff": mean_diff,
            "mean_abs_diff": mean_abs_diff,
            "max_abs_diff": max_abs_diff,
            "std_diff": std_diff,
            "mean_cum_diff": _mean(seq_diff_sum),
            "std_cum_diff": _std(seq_diff_sum, _mean(seq_diff_sum)),
            "mean_is_weight": mean_is,
            "std_is_weight": std_is,
            "min_is_weight": float(min(is_weights)) if is_weights else 0.0,
            "max_is_weight": float(max(is_weights)) if is_weights else 0.0,
            "p99_is_weight": _quantile(is_weights, 0.99),
            "p_out_clip_01": float(clip_01),
            "p_out_clip_02": float(clip_02),
            "mean_staleness": mean_staleness,
        }
        self._logprob_mismatch.append(summary)
        return summary

    @property
    def final_accuracy(self) -> float:
        if not self._eval_curve:
            return 0.0
        return self._eval_curve[-1][1]

    @property
    def mean_staleness(self) -> float:
        if not self._stalenesses:
            return 0.0
        return sum(self._stalenesses) / len(self._stalenesses)

    @property
    def mean_sample_latency_ms(self) -> float:
        if not self._sample_latencies_ms:
            return 0.0
        return sum(self._sample_latencies_ms) / len(self._sample_latencies_ms)

    def to_dict(self) -> Dict[str, Any]:
        """Export metrics as a dictionary."""
        # Convert staleness distributions to serializable format
        # Each entry: {"step": N, "distribution": {"0": count, "1": count, ...}}
        staleness_per_step = [
            {"step": step, "distribution": {str(k): v for k, v in dist.items()}}
            for step, dist in self._staleness_distributions
        ]

        return {
            "final_accuracy": self.final_accuracy,
            "reward_curve": self._reward_curve,
            "eval_curve": self._eval_curve,
            "train_steps_completed": self.train_steps_completed,
            "total_samples": self.total_samples,
            "mean_staleness": self.mean_staleness,
            "staleness_per_step": staleness_per_step,
            "logprob_mismatch_per_step": list(self._logprob_mismatch),
            "mean_sample_latency_ms": self.mean_sample_latency_ms,
            "total_sampling_seconds": round(self.total_sampling_seconds, 2),
            "total_training_seconds": round(self.total_training_seconds, 2),
            "total_sync_weights_seconds": round(self.total_sync_weights_seconds, 2),
        }


def build_output(
    backend_type: str,
    base_model: str,
    wall_clock_seconds: float,
    tenant_metrics: List[TenantMetrics],
) -> Dict[str, Any]:
    """Build the final output JSON structure."""
    per_tenant = {}
    for m in tenant_metrics:
        per_tenant[m.tenant_id] = m.to_dict()

    return {
        "backend": backend_type,
        "base_model": base_model,
        "total_wall_clock_seconds": round(wall_clock_seconds, 2),
        "per_tenant": per_tenant,
    }
