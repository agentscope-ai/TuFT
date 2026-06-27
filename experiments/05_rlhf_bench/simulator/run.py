"""Entry point for the multi-tenant RL training simulator."""

from __future__ import annotations

import argparse
import asyncio
import logging

from simulator.config import load_config
from simulator.orchestrator import Orchestrator


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-Tenant Online RL Training Simulator")
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="config.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Override output path for results JSON",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose/debug logging",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    setup_logging(verbose=args.verbose)

    config = load_config(args.config)

    if args.output:
        config.output_path = args.output

    orchestrator = Orchestrator(config)
    await orchestrator.run_and_save()


if __name__ == "__main__":
    asyncio.run(main())
