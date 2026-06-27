"""
六种 bias 校正方案的离线效果对比
─────────────────────────────────────────────────────────────────────────────
方案：
  1. baseline          : 不校正
  2. per_token         : per-token transformer → 每 token 减 delta_hat[t]
  3. seq_mean (current): per-token transformer → 减序列均值（当前实现）
  4. scalar_pred       : scalar_transformer → 直接输出标量，每 token 均减之
  5. mlp_per_token     : MLP per-token → 每 token 减 delta_hat[t]
  6. mlp_seq_mean      : MLP per-token → 减序列均值
  7. global_mean       : 减全局均值常数
"""

import os
import sys


sys.path.insert(0, os.path.dirname(__file__))

import json
from collections import defaultdict

import numpy as np
import torch
from model import build_model


DATA_FILE = "/mnt/nas/hanzhang.yhz/evaluation/simulator/logprobs/bench_merged_r8_t07.jsonl"
CKPT_TOKEN = "/mnt/nas/hanzhang.yhz/evaluation/predictor/checkpoints/bench_v1/best.pt"
CKPT_SCALAR = "/mnt/nas/hanzhang.yhz/evaluation/predictor/checkpoints/scalar_bench_v1/best.pt"
CKPT_MLP = "/mnt/nas/hanzhang.yhz/evaluation/predictor/checkpoints/mlp_bench_v1/best.pt"
MAX_SEQ_LEN = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── 加载数据 ───────────────────────────────────────────────────────────────────
print("Loading data...")
records = []
with open(DATA_FILE) as f:
    for line in f:
        records.append(json.loads(line))
records.sort(key=lambda x: x["train_weight_version"])
n_test = max(1, int(len(records) * 0.15))
test_records = records[-n_test:]
print(f"Total: {len(records)}, Test: {len(test_records)}")


# ── 加载模型 ───────────────────────────────────────────────────────────────────
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
    print(f"  [{mtype}] {os.path.basename(ckpt_path)} (epoch={ckpt['epoch']})")
    return m


print("Loading models...")
model_token = load_model(CKPT_TOKEN)
model_scalar = load_model(CKPT_SCALAR)
model_mlp = load_model(CKPT_MLP)

# ── 全局均值 ───────────────────────────────────────────────────────────────────
all_deltas = []
for r in records:
    slp = np.array(r["sampling_logprobs"])
    tlp = np.array(r["training_logprobs"])
    all_deltas.extend((slp - tlp).tolist())
GLOBAL_MEAN = float(np.mean(all_deltas))
print(f"Global mean bias: {GLOBAL_MEAN:.6f}")


# ── 推理 ───────────────────────────────────────────────────────────────────────
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


# ── 跑所有方案 ─────────────────────────────────────────────────────────────────
print("Running inference...")
results = defaultdict(list)

for r in test_records:
    slp = np.array(r["sampling_logprobs"])
    tlp = np.array(r["training_logprobs"])
    T = min(len(slp), len(tlp), MAX_SEQ_LEN)
    slp, tlp = slp[:T], tlp[:T]

    dh_tok = predict(model_token, r)
    dh_scalar = predict(model_scalar, r)
    dh_mlp = predict(model_mlp, r)

    schemes = {
        "baseline": slp,
        "per_token": slp - dh_tok,
        "seq_mean": slp - dh_tok.mean(),
        "scalar_pred": slp - float(dh_scalar[0]),
        "mlp_per_token": slp - dh_mlp,
        "mlp_seq_mean": slp - dh_mlp.mean(),
        "global_mean": slp - GLOBAL_MEAN,
    }

    for name, clp in schemes.items():
        delta = clp - tlp
        results[name].append({"token_deltas": delta.tolist(), "cum_log_is": float(delta.sum())})


# ── 指标 ───────────────────────────────────────────────────────────────────────
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


ORDER = [
    "baseline",
    "per_token",
    "seq_mean",
    "scalar_pred",
    "mlp_per_token",
    "mlp_seq_mean",
    "global_mean",
]
mets = {n: metrics(results[n]) for n in ORDER}

# ── 打印 ───────────────────────────────────────────────────────────────────────
W = 96
print("\n" + "=" * W)
print(f"  Offline Bias Correction — Seven Schemes  (test n={len(test_records)} seqs, staleness=0)")
print("=" * W)
hdr = f"{'Scheme':<18} {'Model':>16}  {'token_bias':>10} {'token_mae':>10} {'seq_bias_mae':>12} {'clip02':>7} {'clip01':>7} {'mean_logIS':>10} {'med|logIS|':>10}"
print(hdr)
print("-" * W)

MODEL_LABEL = {
    "baseline": "—",
    "per_token": "Transformer",
    "seq_mean": "Transformer",
    "scalar_pred": "Transformer",
    "mlp_per_token": "MLP",
    "mlp_seq_mean": "MLP",
    "global_mean": "—",
}

for n in ORDER:
    m = mets[n]
    print(
        f"{n:<18} {MODEL_LABEL[n]:>16}  "
        f"{m['token_bias']:>10.6f} {m['token_mae']:>10.6f} {m['seq_bias_mae']:>12.6f} "
        f"{m['clip02']:>7.4f} {m['clip01']:>7.4f} "
        f"{m['mean_log_is']:>10.4f} {m['med_abs_log_is']:>10.4f}"
    )

print("-" * W)
base = mets["baseline"]
print("\n  Δ vs baseline  (negative = improvement):")
print(
    f"  {'Scheme':<18} {'Δtoken_bias':>12} {'Δclip02':>9} {'Δseq_bias_mae':>14} {'Δmed|logIS|':>12}"
)
for n in ORDER[1:]:
    m = mets[n]
    print(
        f"  {n:<18} "
        f"{m['token_bias'] - base['token_bias']:>+12.6f} "
        f"{m['clip02'] - base['clip02']:>+9.4f} "
        f"{m['seq_bias_mae'] - base['seq_bias_mae']:>+14.6f} "
        f"{m['med_abs_log_is'] - base['med_abs_log_is']:>+12.4f}"
    )
print("\nDone.")
