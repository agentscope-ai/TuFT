"""Train + evaluate the framework-mismatch predictor.

Usage
─────
    # 1. collect data first (sync_mode tenant in simulator/config.yaml,
    #    then  python simulator/run.py)
    # 2. train
    python predictor/train.py \
        --data simulator/results.logprobs.jsonl \
        --output predictor/checkpoints/v1 \
        --model transformer

    # quick smoke test on the same machine
    python predictor/train.py --data ... --epochs 1 --batch_size 4 --model mlp
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Dict

import torch
from data import (
    LogprobMismatchDataset,
    collate,
    filter_clean_records,
    infer_vocab_size,
    load_jsonl,
    split_by_tenant,
    split_by_weight_version,
)
from losses import evaluate, predictor_loss, scalar_predictor_loss
from model import build_model
from torch.utils.data import DataLoader


# ──────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="path to results.logprobs.jsonl")
    p.add_argument("--output", default="predictor/checkpoints/run", help="checkpoint + log dir")
    p.add_argument(
        "--model", choices=["mlp", "transformer", "scalar_transformer"], default="transformer"
    )

    # data split
    p.add_argument(
        "--split_mode",
        choices=["weight_version", "tenant"],
        default="weight_version",
        help="how to split train/val/test: by weight_version (chronological) or by tenant",
    )
    p.add_argument("--train_frac", type=float, default=0.7)
    p.add_argument("--val_frac", type=float, default=0.15)
    p.add_argument("--max_seq_len", type=int, default=2048)

    # training
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--lambda_seq", type=float, default=1.0)
    p.add_argument("--lambda_bias", type=float, default=5.0)
    p.add_argument("--warmup_epochs", type=int, default=3, help="linear warmup epochs")
    p.add_argument("--seed", type=int, default=0)

    # model hparams (transformer)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--n_layers", type=int, default=2)
    p.add_argument("--token_emb_dim", type=int, default=32)

    # model hparams (mlp)
    p.add_argument("--hidden", type=int, default=128)

    # eval
    p.add_argument("--eval_every", type=int, default=1, help="run val evaluation every N epochs")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────
def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(records, args, shuffle: bool) -> DataLoader:
    ds = LogprobMismatchDataset(records, max_seq_len=args.max_seq_len)
    return DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=shuffle,
        collate_fn=collate,
        num_workers=0,
        drop_last=False,
    )


def build(args, vocab_size: int):
    if args.model == "transformer":
        return build_model(
            "transformer",
            vocab_size=vocab_size,
            d_model=args.d_model,
            token_emb_dim=args.token_emb_dim,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
        )
    return build_model(
        "mlp",
        vocab_size=vocab_size,
        token_emb_dim=args.token_emb_dim,
        hidden=args.hidden,
    )


# ──────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output, exist_ok=True)

    # ── 1. Load + filter + split ──
    print(f"[load] reading {args.data}")
    records = load_jsonl(args.data)
    records = filter_clean_records(records)
    print(f"[load] {len(records)} clean records (staleness == 0)")
    if not records:
        raise RuntimeError("no clean records — check that simulator was run with sync_mode=true")

    train_r, val_r, test_r = (
        split_by_weight_version(records, args.train_frac, args.val_frac)
        if args.split_mode == "weight_version"
        else split_by_tenant(records, args.train_frac, args.val_frac)
    )
    print(
        f"[split] mode={args.split_mode}  train={len(train_r)}  val={len(val_r)}  test={len(test_r)}"  # noqa: E501
    )

    # ── 2. Vocab size (fit on TRAIN only) ──
    vocab_size = infer_vocab_size(train_r)
    print(f"[vocab] vocab_size={vocab_size}")

    # ── 3. Loaders ──
    train_loader = make_loader(train_r, args, shuffle=True)
    val_loader = make_loader(val_r, args, shuffle=False) if val_r else None
    test_loader = make_loader(test_r, args, shuffle=False) if test_r else None

    # ── 4. Model ──
    model = build(args, vocab_size).to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] {args.model}  params={n_params:,}")

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # LR schedule: linear warmup → cosine decay to 0
    total_steps = args.epochs * len(train_loader)
    warmup_steps = args.warmup_epochs * len(train_loader)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)

    # ── 5. Train ──
    log_path = Path(args.output) / "training_log.jsonl"
    log_fp = open(log_path, "w")

    # initial baseline on val (epoch=0, no training has happened yet)
    if val_loader is not None:
        summ = evaluate(model, val_loader, args.device)
        print("[val/init]\n" + summ.pretty())
        log_fp.write(json.dumps({"epoch": 0, "val": summ.__dict__}) + "\n")
        log_fp.flush()

    # scalar model: optimize sequence-level bias; per-token model: optimize token MAE
    best_metric = (
        "corrected_token_bias" if args.model == "scalar_transformer" else "corrected_token_mae"
    )
    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_start = time.time()
        running = {"L_total": 0.0, "L_token": 0.0, "L_seq": 0.0, "L_bias": 0.0, "n": 0}

        for batch in train_loader:
            for k in batch:
                batch[k] = batch[k].to(args.device)

            target = batch["sampling_lps"] - batch["training_lps"]
            pred = model(
                batch["token_ids"],
                batch["sampling_lps"],
                batch["mask"],
                batch["n_prompt_tokens"],
                batch["temperature"],
                batch["lora_rank"],
            )
            is_scalar = args.model == "scalar_transformer"
            if is_scalar:
                loss, parts = scalar_predictor_loss(
                    pred,
                    target,
                    batch["mask"],
                    lambda_bias=args.lambda_bias,
                )
            else:
                loss, parts = predictor_loss(
                    pred,
                    target,
                    batch["mask"],
                    lambda_seq=args.lambda_seq,
                    lambda_bias=args.lambda_bias,
                )

            optim.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optim.step()
            scheduler.step()

            for k in ["L_total", "L_token", "L_seq", "L_bias"]:
                running[k] += parts[k]
            running["n"] += 1

        n = max(running["n"], 1)
        train_summary = {k: running[k] / n for k in ["L_total", "L_token", "L_seq", "L_bias"]}
        elapsed = time.time() - epoch_start
        current_lr = scheduler.get_last_lr()[0]
        print(
            f"[epoch {epoch:>3}/{args.epochs}] "
            f"L_total={train_summary['L_total']:.5f}  "
            f"L_token={train_summary['L_token']:.5f}  "
            f"L_seq={train_summary['L_seq']:.5f}  "
            f"L_bias={train_summary['L_bias']:.5f}  "
            f"lr={current_lr:.2e}  "
            f"({elapsed:.1f}s)"
        )

        log_record: Dict[str, Any] = {"epoch": epoch, "train": train_summary, "elapsed": elapsed}

        if val_loader is not None and (epoch % args.eval_every == 0 or epoch == args.epochs):
            summ = evaluate(model, val_loader, args.device)
            print("[val]\n" + summ.pretty())
            log_record["val"] = summ.__dict__

            # checkpoint best by appropriate metric
            val_metric = getattr(summ, best_metric)
            if val_metric < best_val:
                best_val = val_metric
                ckpt_path = Path(args.output) / "best.pt"
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "args": vars(args),
                        "vocab_size": vocab_size,
                        "epoch": epoch,
                        "val_summary": summ.__dict__,
                    },
                    ckpt_path,
                )
                print(f"[ckpt] saved {ckpt_path}  ({best_metric}={best_val:.5f})")

        log_fp.write(json.dumps(log_record) + "\n")
        log_fp.flush()

    log_fp.close()

    # ── 6. Final evaluation on test ──
    if test_loader is not None:
        # reload best
        ckpt_path = Path(args.output) / "best.pt"
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=args.device, weights_only=False)
            model.load_state_dict(ckpt["model_state"])
            print(f"[test] loaded {ckpt_path} (epoch {ckpt['epoch']})")
        summ = evaluate(model, test_loader, args.device)
        print("[test]\n" + summ.pretty())
        with open(Path(args.output) / "test_summary.json", "w") as f:
            json.dump(summ.__dict__, f, indent=2)


if __name__ == "__main__":
    main()
