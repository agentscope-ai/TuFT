"""
多任务 v2 数据集 bias 校正离线对比
────────────────────────────────────────────────────────────────────────────────
对比方案（7种）：
  1. baseline       : 不校正
  2. per_token      : TransformerPredictor → 每 token 减 delta_hat[t]
  3. seq_mean       : TransformerPredictor → 减序列均值（当前 RL 实现）
  4. scalar_pred    : ScalarTransformerPredictor → 直接预测标量偏移
  5. mlp_per_token  : MLPPredictor → 每 token 减 delta_hat[t]
  6. mlp_seq_mean   : MLPPredictor → 减序列均值
  7. global_mean    : 减全局均值常数（最简基线）

使用方式：
  python correction_comparison_v2.py \
      --data_train  data/v2_train.jsonl \
      --data_holdout data/v2_holdout.jsonl \
      --transformer_ckpt checkpoints/multitask_transformer_v1/best.pt \
      --scalar_ckpt      checkpoints/multitask_scalar_v1/best.pt \
      --mlp_ckpt         checkpoints/multitask_mlp_v1/best.pt
"""

import argparse
import os
import sys


sys.path.insert(0, os.path.dirname(__file__))

import json
from collections import defaultdict

import numpy as np
import torch
from model import build_model


MAX_SEQ_LEN = 2048
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_train", required=True, help="v2_train.jsonl")
    p.add_argument("--data_holdout", required=True, help="v2_holdout.jsonl")
    p.add_argument("--transformer_ckpt", required=True)
    p.add_argument("--scalar_ckpt", required=True)
    p.add_argument("--mlp_ckpt", required=True)
    p.add_argument("--mlp_scalar_ckpt", required=True)
    p.add_argument(
        "--test_frac",
        type=float,
        default=0.2,
        help="fraction of train data to use as test split (by weight_version)",
    )
    return p.parse_args()


def load_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if "sampling_logprobs" in r and "training_logprobs" in r:
                records.append(r)
    return records


def load_model(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    a = ckpt["args"]
    mtype = a.get("model", "transformer")
    kwargs = dict(vocab_size=ckpt["vocab_size"], token_emb_dim=a.get("token_emb_dim", 32))
    if mtype in ("transformer", "scalar_transformer"):
        kwargs.update(
            d_model=a.get("d_model", 128),
            n_heads=a.get("n_heads", 4),
            n_layers=a.get("n_layers", 2),
        )
    else:
        kwargs.update(hidden=a.get("hidden", 128))
    m = build_model(mtype, **kwargs)
    m.load_state_dict(ckpt["model_state"])
    m.eval().to(DEVICE)
    print(f"  [{mtype}] loaded {os.path.basename(ckpt_path)} (epoch={ckpt['epoch']})")
    return m


@torch.no_grad()
def predict(model, r):
    resp = r["response_tokens"]
    slp = r["sampling_logprobs"]
    T = min(len(resp), len(slp), MAX_SEQ_LEN)
    out = model(
        torch.tensor([resp[:T]], dtype=torch.long, device=DEVICE),
        torch.tensor([slp[:T]], dtype=torch.float32, device=DEVICE),
        torch.ones(1, T, dtype=torch.bool, device=DEVICE),
        torch.tensor([float(r["n_prompt_tokens"])], device=DEVICE),
        torch.tensor([r["temperature"]], device=DEVICE),
        torch.tensor([r["lora_rank"] / 64.0], device=DEVICE),
    )
    return out[0].cpu().numpy()[:T]


def run_schemes(records, model_tok, model_scalar, model_mlp, model_mlp_scalar, global_mean):
    results = defaultdict(list)
    for r in records:
        slp = np.array(r["sampling_logprobs"])
        tlp = np.array(r["training_logprobs"])
        T = min(len(slp), len(tlp), MAX_SEQ_LEN)
        slp, tlp = slp[:T], tlp[:T]

        dh_tok = predict(model_tok, r)
        dh_scalar = predict(model_scalar, r)
        dh_mlp = predict(model_mlp, r)
        dh_mlp_scalar = predict(model_mlp_scalar, r)

        schemes = {
            "baseline": slp,
            "per_token": slp - dh_tok,
            "seq_mean": slp - dh_tok.mean(),
            "scalar_pred": slp - float(dh_scalar[0]),
            "mlp_per_token": slp - dh_mlp,
            "mlp_seq_mean": slp - dh_mlp.mean(),
            "mlp_scalar_pred": slp - float(dh_mlp_scalar[0]),
            "global_mean": slp - global_mean,
        }
        for name, clp in schemes.items():
            delta = clp - tlp
            results[name].append(
                {
                    "token_deltas": delta.tolist(),
                    "cum_log_is": float(delta.sum()),
                }
            )
    return results


def metrics(seqs):
    all_tok = np.concatenate([s["token_deltas"] for s in seqs])
    cum_is = np.array([s["cum_log_is"] for s in seqs])
    seq_b = np.array([np.mean(s["token_deltas"]) for s in seqs])
    return dict(
        token_bias=abs(float(np.mean(all_tok))),
        token_mae=float(np.mean(np.abs(all_tok))),
        seq_bias_mae=float(np.mean(np.abs(seq_b))),
        clip02=float(np.mean(np.abs(cum_is) > 0.2)),
        clip01=float(np.mean(np.abs(cum_is) > 0.1)),
        mean_log_is=float(np.mean(cum_is)),
        med_abs_log_is=float(np.median(np.abs(cum_is))),
    )


def print_table(title, mets, n_seqs):
    ORDER = [
        "baseline",
        "per_token",
        "seq_mean",
        "scalar_pred",
        "mlp_per_token",
        "mlp_seq_mean",
        "mlp_scalar_pred",
        "global_mean",
    ]
    MODEL_LABEL = {
        "baseline": "—",
        "per_token": "Transformer",
        "seq_mean": "Transformer",
        "scalar_pred": "ScalarTransf.",
        "mlp_per_token": "MLP",
        "mlp_seq_mean": "MLP",
        "mlp_scalar_pred": "MLP-Scalar",
        "global_mean": "—",
    }
    W = 104
    print(f"\n{'=' * W}")
    print(f"  {title}  (n={n_seqs} seqs)")
    print(f"{'=' * W}")
    hdr = (
        f"{'Scheme':<16} {'Model':>14}  {'token_bias':>10} {'token_mae':>10} "
        f"{'seq_bias_mae':>12} {'clip02':>7} {'clip01':>7} "
        f"{'mean_logIS':>10} {'med|logIS|':>10}"
    )
    print(hdr)
    print("-" * W)
    for n in ORDER:
        if n not in mets:
            continue
        m = mets[n]
        print(
            f"{n:<16} {MODEL_LABEL[n]:>14}  "
            f"{m['token_bias']:>10.6f} {m['token_mae']:>10.6f} {m['seq_bias_mae']:>12.6f} "
            f"{m['clip02']:>7.4f} {m['clip01']:>7.4f} "
            f"{m['mean_log_is']:>10.4f} {m['med_abs_log_is']:>10.4f}"
        )
    print("-" * W)
    base = mets["baseline"]
    print("\n  Δ vs baseline (negative = improvement):")
    print(
        f"  {'Scheme':<16} {'Δtoken_bias':>12} {'Δseq_bias_mae':>14} {'Δclip02':>9} {'Δmed|logIS|':>12}"
    )
    for n in ORDER[1:]:
        if n not in mets:
            continue
        m = mets[n]
        pct_bias = (m["token_bias"] - base["token_bias"]) / max(base["token_bias"], 1e-9) * 100
        print(
            f"  {n:<16} "
            f"{m['token_bias'] - base['token_bias']:>+12.6f} ({pct_bias:>+6.1f}%) "
            f"{m['seq_bias_mae'] - base['seq_bias_mae']:>+14.6f} "
            f"{m['clip02'] - base['clip02']:>+9.4f} "
            f"{m['med_abs_log_is'] - base['med_abs_log_is']:>+12.4f}"
        )


def main():
    args = parse_args()

    # ── 1. Load data ──────────────────────────────────────────────────────
    print(f"Loading train data: {args.data_train}")
    train_all = load_jsonl(args.data_train)
    train_all.sort(key=lambda x: x.get("train_weight_version", x.get("step", 0)))
    n_test = max(1, int(len(train_all) * args.test_frac))
    test_in = train_all[-n_test:]
    print(f"  Total train+val: {len(train_all)}, In-dist test: {len(test_in)}")

    print(f"Loading holdout data: {args.data_holdout}")
    holdout_recs = load_jsonl(args.data_holdout)
    print(f"  Holdout records: {len(holdout_recs)}")

    # ── 2. Load models ────────────────────────────────────────────────────
    print("\nLoading models...")
    model_tok = load_model(args.transformer_ckpt)
    model_scalar = load_model(args.scalar_ckpt)
    model_mlp = load_model(args.mlp_ckpt)
    model_mlp_scalar = load_model(args.mlp_scalar_ckpt)

    # ── 3. Global mean (from train data) ─────────────────────────────────
    all_deltas = []
    for r in train_all:
        slp = np.array(r["sampling_logprobs"])
        tlp = np.array(r["training_logprobs"])
        T = min(len(slp), len(tlp), MAX_SEQ_LEN)
        all_deltas.extend((slp[:T] - tlp[:T]).tolist())
    global_mean = float(np.mean(all_deltas))
    print(f"\nGlobal mean bias (train): {global_mean:.6f}")

    # ── 4. In-dist evaluation ─────────────────────────────────────────────
    print(f"\nRunning inference on in-dist test ({len(test_in)} seqs)...")
    res_in = run_schemes(test_in, model_tok, model_scalar, model_mlp, model_mlp_scalar, global_mean)
    mets_in = {n: metrics(res_in[n]) for n in res_in}
    print_table(
        "In-Distribution (train tasks: gsm8k/math/countdown/humaneval/mbpp)", mets_in, len(test_in)
    )

    # ── 5. Holdout evaluation ──────────────────────────────────────────────
    if holdout_recs:
        print(f"\nRunning inference on holdout ({len(holdout_recs)} seqs)...")
        res_ho = run_schemes(
            holdout_recs, model_tok, model_scalar, model_mlp, model_mlp_scalar, global_mean
        )
        mets_ho = {n: metrics(res_ho[n]) for n in res_ho}
        print_table("Out-of-Distribution Holdout (ifeval)", mets_ho, len(holdout_recs))

    print("\nDone.")


if __name__ == "__main__":
    main()
