"""Run the complete cached-teacher Open Character Training recipe in order."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import settings as C


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url", default=os.getenv("TINKER_BASE_URL", "http://localhost:10610")
    )
    parser.add_argument("--api-key", default=os.getenv("TINKER_API_KEY"))
    parser.add_argument("--model", default=C.MODEL)
    parser.add_argument("--work-dir", type=Path, default=C.HERE / "work")
    parser.add_argument(
        "--refresh-teacher",
        action="store_true",
        help="regenerate chosen responses with the configured OpenAI client",
    )
    parser.add_argument("--skip-samples", action="store_true")
    args = parser.parse_args()
    if not args.api_key:
        raise SystemExit("set TINKER_API_KEY or pass --api-key")

    common = [
        "--base-url",
        args.base_url,
        "--api-key",
        args.api_key,
        "--model",
        args.model,
        "--work-dir",
        str(args.work_dir),
    ]
    commands = [
        [sys.executable, str(C.HERE / "generate_data.py"), "distill", *common],
        [sys.executable, str(C.HERE / "train.py"), "dpo", *common],
        [sys.executable, str(C.HERE / "generate_data.py"), "introspection", *common],
        [sys.executable, str(C.HERE / "train.py"), "sft", *common],
    ]
    if args.refresh_teacher:
        commands[0].append("--refresh-teacher")
    if not args.skip_samples:
        commands.append([sys.executable, str(C.HERE / "sample.py"), *common])
        commands.append(
            [
                sys.executable,
                str(C.HERE / "evaluate.py"),
                str(args.work_dir / "sample_outputs.json"),
                "--output",
                str(args.work_dir / "lightweight_eval.json"),
            ]
        )

    for index, command in enumerate(commands, 1):
        print(f"\n[pipeline] stage {index}/{len(commands)}: {' '.join(command[:3])}", flush=True)
        subprocess.run(command, check=True)
    print(f"\n[pipeline] complete; artifacts are in {args.work_dir}", flush=True)


if __name__ == "__main__":
    main()
