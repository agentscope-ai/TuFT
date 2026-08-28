#!/usr/bin/env python
"""Generate the fixed-seed multi-tenant evaluation workload for the simulator.

Reads the tenant list (tasks, standard generation lengths, batch sizes) from an
existing simulator config and re-assigns each tenant a staleness tolerance in
{0, 1, 2, 3} with a fixed seed. All baselines must be driven with the exact
same generated file so comparisons are fair. The simulator itself is not
modified; this script only emits YAML.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import yaml


DEFAULT_SOURCE = "/mnt/nas/hanzhang.yhz/evaluation/simulator/configs/config.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", default=DEFAULT_SOURCE, help="Simulator config to copy tenants from"
    )
    parser.add_argument(
        "--output",
        default="eval_workload_seed42.yaml",
        help="Output workload YAML path",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for staleness assignment")
    parser.add_argument("--base-url", default="http://localhost:10610")
    parser.add_argument("--api-key", default="tml-tuft-dev-key")
    # Must match model_name in the TuFT eval configs (not the HF repo id).
    parser.add_argument("--base-model", default="qwen3-4b")
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=None,
        help="Override buffer_size (training batch) for every tenant",
    )
    parser.add_argument(
        "--eval-sample-size",
        type=int,
        default=None,
        help="Override evaluation.eval_sample_size (smoke runs)",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=None,
        help="Override evaluation.eval_interval_steps (smoke runs)",
    )
    parser.add_argument(
        "--num-train-steps",
        type=int,
        default=None,
        help="Override num_train_steps for every tenant (defaults to source values)",
    )
    parser.add_argument(
        "--max-tenants",
        type=int,
        default=None,
        help="Keep only the first N tenants (smoke runs)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_path = Path(args.source)
    if not source_path.exists():
        print(f"source config not found: {source_path}", file=sys.stderr)
        return 1

    with open(source_path) as f:
        source = yaml.safe_load(f)

    rng = random.Random(args.seed)
    tenants = []
    source_tenants = source.get("tenants", [])
    if args.max_tenants is not None:
        source_tenants = source_tenants[: args.max_tenants]
    for tenant in source_tenants:
        staleness = rng.randint(0, 3)
        new_tenant = {
            "id": tenant["id"],
            "task": tenant["task"],
            "request_rate": tenant.get("request_rate", 2.0),
            "buffer_size": args.buffer_size or tenant.get("buffer_size", 64),
            "num_train_steps": args.num_train_steps or tenant.get("num_train_steps", 50),
            "lora_rank": tenant.get("lora_rank", 8),
            "learning_rate": tenant.get("learning_rate", 1e-4),
            # Standard task generation length comes from the source tenant.
            "max_tokens": tenant.get("max_tokens", 512),
            "temperature": tenant.get("temperature", 0.7),
            "staleness_limit": staleness,
            # staleness_limit=0 means fully synchronous sampling for this tenant.
            "sync_mode": staleness == 0,
        }
        if tenant.get("max_turns"):
            new_tenant["max_turns"] = tenant["max_turns"]
        if tenant.get("async_sampling"):
            new_tenant["async_sampling"] = True
            new_tenant["async_sampling_concurrency"] = tenant.get("async_sampling_concurrency", 4)
        tenants.append(new_tenant)

    evaluation = dict(source.get("evaluation", {}) or {})
    if args.eval_sample_size is not None:
        evaluation["eval_sample_size"] = args.eval_sample_size
    if args.eval_interval is not None:
        evaluation["eval_interval_steps"] = args.eval_interval
    doc = {
        "backend": {
            "type": "tinker",
            "base_url": args.base_url,
            "api_key": args.api_key,
            "base_model": args.base_model,
        },
        "tenants": tenants,
        "evaluation": evaluation or {"eval_interval_steps": 50, "eval_sample_size": 200},
        # Scheduling evaluation does not need per-token logprob dumps; the real
        # Corrector calibration is collected separately.
        "logprob_collection": {"enabled": False},
        "request_arrival_logging": {"enabled": True},
        "output_path": "results.json",
        "seed": args.seed,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        f"# Auto-generated evaluation workload (seed={args.seed}).\n"
        f"# Tenants: {len(tenants)}; staleness_limit drawn from {{0,1,2,3}} with the fixed seed.\n"
        "# Use this exact file for every baseline; do not edit by hand.\n"
    )
    with open(out_path, "w") as f:
        f.write(header)
        yaml.safe_dump(doc, f, sort_keys=False, allow_unicode=True)
    print(f"wrote {out_path} ({len(tenants)} tenants, seed={args.seed})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
