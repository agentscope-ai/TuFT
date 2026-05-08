"""Embedded TuFT quickstart — demonstrates auto-init (embedded mode).

This example shows how to use TuFT in embedded mode, where the service
is automatically started and managed within your training script's lifecycle.

No manual `tuft launch` or configuration files needed!

Usage:
    python train.py --model /path/to/Qwen2.5-0.5B-Instruct

The script will:
1. Auto-detect the model and GPU configuration
2. Start a TuFT server in the background
3. Connect and run a minimal training loop
4. Automatically shut down the server on exit
"""

from __future__ import annotations

import argparse

from tinker import types

import tuft


def main():
    parser = argparse.ArgumentParser(description="Embedded TuFT quickstart")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the base model (e.g., /path/to/Qwen2.5-0.5B-Instruct)",
    )
    parser.add_argument("--rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--steps", type=int, default=5, help="Number of training steps")
    args = parser.parse_args()

    # =========================================================================
    # Step 1: Initialize TuFT in embedded mode
    # This will auto-detect GPUs, generate a minimal config, and start the server.
    # If a TuFT server is already running, it will connect to it instead.
    # =========================================================================
    print(f"[1/4] Initializing TuFT with model: {args.model}")
    tuft.init(model=args.model)
    print("      TuFT initialized (mode: embedded)")

    # =========================================================================
    # Step 2: Create a training client
    # =========================================================================
    print(f"[2/4] Creating LoRA training client (rank={args.rank})")
    training_client = tuft.create_training_client(
        base_model=args.model,
        rank=args.rank,
        train_mlp=True,
        train_attn=True,
    )

    # =========================================================================
    # Step 3: Run a minimal training loop
    # =========================================================================
    print(f"[3/4] Running {args.steps} training steps (with fake data)")
    for step in range(args.steps):
        # Create a fake training datum (in practice, use real tokenized data)
        datum = types.Datum(
            model_input=types.ModelInput.from_ints([101, 42, 37, 102]),
            loss_fn_inputs={
                "target_tokens": types.TensorData(
                    data=[101, 99, 73, 102], dtype="int64", shape=[4]
                ),
                "weights": types.TensorData(data=[1.0, 1.0, 1.0, 1.0], dtype="float32", shape=[4]),
            },
        )
        training_client.forward_backward([datum], loss_fn="cross_entropy").result()
        training_client.optim_step(types.AdamParams(learning_rate=1e-4)).result()
        print(f"      Step {step + 1}/{args.steps} complete")

    # =========================================================================
    # Step 4: Clean up (optional — atexit handles this automatically)
    # =========================================================================
    print("[4/4] Shutting down TuFT")
    tuft.shutdown()
    print("      Done!")


if __name__ == "__main__":
    main()
