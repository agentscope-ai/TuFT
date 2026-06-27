"""Evaluate a trained predictor checkpoint on arbitrary JSONL data.

Loads a trained predictor (best.pt) and runs it on a new dataset,
reporting baseline vs corrected metrics — without needing to retrain.

Usage:
    predictor/.venv/bin/python predictor/predict.py \
        --checkpoint predictor/checkpoints/transformer_22agents_v1/best.pt \
        --data simulator/22_agents_50_step_results.logprobs.jsonl

    # optionally limit to first N records (useful for quick sanity check)
    predictor/.venv/bin/python predictor/predict.py \
        --checkpoint predictor/checkpoints/transformer_22agents_v1/best.pt \
        --data simulator/new_data.logprobs.jsonl \
        --max_records 5000

    # save results to file
    predictor/.venv/bin/python predictor/predict.py \
        --checkpoint ... --data ... --output results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import torch
from data import (
    LogprobMismatchDataset,
    collate,
    filter_clean_records,
    load_jsonl,
)
from losses import evaluate
from model import build_model
from torch.utils.data import DataLoader


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run predictor inference on JSONL data")
    p.add_argument("--checkpoint", required=True, help="path to best.pt checkpoint")
    p.add_argument("--data", required=True, help="path to .logprobs.jsonl file to evaluate")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument(
        "--max_records", type=int, default=None, help="limit number of records (for quick checks)"
    )
    p.add_argument("--max_seq_len", type=int, default=2048)
    p.add_argument("--output", default=None, help="optional: save EvalSummary as JSON to this path")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def load_checkpoint(ckpt_path: str, device: str) -> Dict[str, Any]:
    """Load checkpoint and rebuild model."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Determine model type from saved args
    args = ckpt.get("args", {})
    model_type = args.get("model", "transformer")

    # Rebuild model
    model_kwargs: Dict[str, Any] = {
        "vocab_size": ckpt["vocab_size"],
    }
    if model_type == "transformer":
        model_kwargs.update(
            {
                "d_model": args.get("d_model", 128),
                "token_emb_dim": args.get("token_emb_dim", 32),
                "n_heads": args.get("n_heads", 4),
                "n_layers": args.get("n_layers", 2),
            }
        )
    else:
        model_kwargs.update(
            {
                "token_emb_dim": args.get("token_emb_dim", 32),
                "hidden": args.get("hidden", 128),
            }
        )

    model = build_model(model_type, **model_kwargs)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device).eval()

    return {
        "model": model,
        "ckpt": ckpt,
        "model_type": model_type,
    }


def main() -> None:
    args = parse_args()

    # ── 1. Load model ──
    print(f"[predict] loading checkpoint: {args.checkpoint}")
    loaded = load_checkpoint(args.checkpoint, args.device)
    model = loaded["model"]
    ckpt = loaded["ckpt"]

    ckpt.get("args", {})
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[predict] model: {loaded['model_type']}  params: {n_params:,}")
    print(f"[predict] trained at epoch {ckpt.get('epoch', '?')}")
    if "val_summary" in ckpt:
        vs = ckpt["val_summary"]
        print(
            f"[predict] train-time val: MAE={vs.get('corrected_token_mae', '?'):.5f}  "
            f"R²={vs.get('delta_r2', '?'):.4f}"
        )

    # ── 2. Load data ──
    print(f"\n[predict] loading data: {args.data}")
    records = load_jsonl(args.data)
    records = filter_clean_records(records)
    print(f"[predict] {len(records)} clean records (staleness == 0)")

    if not records:
        print("[predict] ERROR: no clean records found. Exiting.")
        sys.exit(1)

    if args.max_records and len(records) > args.max_records:
        records = records[: args.max_records]
        print(f"[predict] limited to first {args.max_records} records")

    # ── 3. Build DataLoader ──
    ds = LogprobMismatchDataset(records, max_seq_len=args.max_seq_len)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=0,
        drop_last=False,
    )
    print(f"[predict] {len(ds)} sequences, {len(loader)} batches")

    # ── 4. Evaluate ──
    print(f"\n[predict] running inference on {args.device} ...")
    summ = evaluate(model, loader, args.device)

    # ── 5. Output ──
    print("\n" + "=" * 60)
    print("  PREDICTION RESULTS")
    print("=" * 60)
    print(summ.pretty())
    print("=" * 60)

    # Summary comparison
    mae_impr = (1 - summ.corrected_token_mae / summ.baseline_token_mae) * 100
    bias_impr = (1 - summ.corrected_token_bias / max(summ.baseline_token_bias, 1e-12)) * 100
    print("\n[summary]")
    print(
        f"  Token MAE:    {summ.baseline_token_mae:.5f} → {summ.corrected_token_mae:.5f}  ({mae_impr:+.1f}% improvement)"  # noqa: E501
    )
    print(
        f"  Token bias:   {summ.baseline_token_bias:.5f} → {summ.corrected_token_bias:.5f}  ({bias_impr:+.1f}% improvement)"  # noqa: E501
    )
    print(f"  mean_log_IS:  {summ.baseline_mean_log_is:.4f} → {summ.corrected_mean_log_is:.4f}")
    print(f"  clip01:       {summ.baseline_clip01:.4f} → {summ.corrected_clip01:.4f}")
    print(f"  Δ R²:         {summ.delta_r2:.4f}")

    # Save if requested
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(summ.__dict__, f, indent=2)
        print(f"\n[predict] results saved to {out_path}")


if __name__ == "__main__":
    main()
