"""Configuration dataclasses and YAML loading for the simulator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class BackendConfig:
    type: str = "tinker"
    base_url: str = "http://localhost:10610"
    api_key: Optional[str] = None
    base_model: str = "Qwen/Qwen3-4B"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "base_url": self.base_url,
            "api_key": self.api_key,
            "base_model": self.base_model,
        }


@dataclass
class EvaluationConfig:
    eval_interval_steps: int = 50
    eval_sample_size: int = 200
    eval_temperature: float = 0.1


@dataclass
class LogprobCollectionConfig:
    """Controls collection of sampling/training logprob mismatch data.

    enabled: turn on/off the whole feature.
    dump_per_token: also dump per-token (sampling_lp, training_lp) pairs to a
        sidecar JSONL file (one record per training step per item). Off by
        default since the file can grow large.
    output_path: path to the JSONL file (only used when dump_per_token=True).
        Defaults to <output_path>.logprobs.jsonl alongside the main results.
    """

    enabled: bool = True
    dump_per_token: bool = False
    output_path: Optional[str] = None


@dataclass
class TenantConfig:
    id: str
    task: str
    request_rate: float = 2.0
    buffer_size: int = 64
    num_train_steps: int = 500
    lora_rank: int = 8
    learning_rate: float = 1e-4
    max_tokens: int = 512
    temperature: float = 0.7
    # Agent RL specific: max turns per episode (0 = single-turn task)
    max_turns: int = 0
    # If True, sampling pauses while a training step is in flight.
    # Effect: every trained batch has staleness == 0 for all items, which
    # isolates pure framework mismatch (sampling vs training code path on
    # the same weights). Default False = original async behaviour.
    sync_mode: bool = False
    # If True, the tenant issues up to `async_sampling_concurrency` sample
    # requests concurrently without waiting for the previous one to finish.
    # This models a "shared" tenant with multiple concurrent users, which
    # increases queue depth at the scheduler and makes batch-sorting
    # optimization observable.  Default False = original serial behaviour.
    async_sampling: bool = False
    async_sampling_concurrency: int = 4


@dataclass
class SimulatorConfig:
    backend: BackendConfig = field(default_factory=BackendConfig)
    tenants: List[TenantConfig] = field(default_factory=list)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    logprob_collection: LogprobCollectionConfig = field(default_factory=LogprobCollectionConfig)
    output_path: str = "results.json"
    seed: int = 42


def load_config(path: str) -> SimulatorConfig:
    """Load simulator configuration from a YAML file."""
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    backend_raw = raw.get("backend", {})
    backend = BackendConfig(
        type=backend_raw.get("type", "tinker"),
        base_url=backend_raw.get("base_url", "http://localhost:10610"),
        api_key=backend_raw.get("api_key"),
        base_model=backend_raw.get("base_model", "Qwen/Qwen3-4B"),
    )

    eval_raw = raw.get("evaluation", {})
    evaluation = EvaluationConfig(
        eval_interval_steps=eval_raw.get("eval_interval_steps", 50),
        eval_sample_size=eval_raw.get("eval_sample_size", 200),
        eval_temperature=eval_raw.get("eval_temperature", 0.1),
    )

    lp_raw = raw.get("logprob_collection", {}) or {}
    logprob_collection = LogprobCollectionConfig(
        enabled=lp_raw.get("enabled", True),
        dump_per_token=lp_raw.get("dump_per_token", False),
        output_path=lp_raw.get("output_path"),
    )

    tenants = []
    for t in raw.get("tenants", []):
        tenants.append(
            TenantConfig(
                id=t["id"],
                task=t["task"],
                request_rate=t.get("request_rate", 2.0),
                buffer_size=t.get("buffer_size", 64),
                num_train_steps=t.get("num_train_steps", 500),
                lora_rank=t.get("lora_rank", 8),
                learning_rate=t.get("learning_rate", 1e-4),
                max_tokens=t.get("max_tokens", 512),
                temperature=t.get("temperature", 0.7),
                max_turns=t.get("max_turns", 0),
                sync_mode=t.get("sync_mode", False),
                async_sampling=t.get("async_sampling", False),
                async_sampling_concurrency=t.get("async_sampling_concurrency", 4),
            )
        )

    return SimulatorConfig(
        backend=backend,
        tenants=tenants,
        evaluation=evaluation,
        logprob_collection=logprob_collection,
        output_path=raw.get("output_path", "results.json"),
        seed=raw.get("seed", 42),
    )
