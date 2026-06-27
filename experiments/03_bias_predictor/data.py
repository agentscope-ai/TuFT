"""Data loading + splitting + batching for the framework-mismatch predictor.

Input: JSONL file produced by simulator's logprob dump
       (one record per (tenant, step, item)).
Each record contains:
    response_tokens, sampling_logprobs, training_logprobs,
    n_prompt_tokens, temperature, lora_rank, task,
    sample_weight_version, staleness, ...

We:
  1. filter staleness == 0 (pure framework mismatch only)
  2. split chronologically by sample_weight_version
     so the test set evaluates *cross-weight* generalisation
  3. dynamically pad each batch to the longest sequence inside it

Note: task_id / tenant_id are intentionally NOT used as features.
      lora_rank is normalized (÷ 64) as a continuous scalar.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import Dataset


# Maximum LoRA rank used for normalization.
LORA_RANK_MAX = 64.0


# ──────────────────────────────────────────────────────────────────────
# Loading
# ──────────────────────────────────────────────────────────────────────
def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def filter_clean_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep only staleness == 0 records that look well-formed."""
    out = []
    for r in records:
        if r.get("staleness", 1.0) != 0.0:
            continue
        n = len(r.get("response_tokens", []))
        if n == 0:
            continue
        if len(r.get("sampling_logprobs", [])) != n:
            continue
        if len(r.get("training_logprobs", [])) != n:
            continue
        out.append(r)
    return out


# ──────────────────────────────────────────────────────────────────────
# Splitting (chronological by sample_weight_version)
# ──────────────────────────────────────────────────────────────────────
def split_by_weight_version(
    records: List[Dict[str, Any]],
    train_frac: float = 0.7,
    val_frac: float = 0.15,
) -> Tuple[List, List, List]:
    """Sort by (sample_weight_version, tenant_id) then take prefixes.

    This deliberately leaks NO future weight versions into the past, so the
    test set measures the predictor's ability to generalise to weights it
    has never seen — the exact use-case during real RL training.
    """
    rs = sorted(
        records,
        key=lambda r: (r["sample_weight_version"], r.get("tenant_id", ""), r.get("item_idx", 0)),
    )
    n = len(rs)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    return rs[:n_train], rs[n_train : n_train + n_val], rs[n_train + n_val :]


def split_by_tenant(
    records: List[Dict[str, Any]],
    train_frac: float = 0.7,
    val_frac: float = 0.15,
) -> Tuple[List, List, List]:
    """Split by tenant_id: all records of a tenant go into the same split.

    Tests cross-tenant generalisation — can the predictor correct mismatch
    for tenants (tasks / configs) it has never seen during training?
    """
    # Get unique tenants, sorted for reproducibility
    tenants = sorted(set(r.get("tenant_id", "unknown") for r in records))
    n_t = len(tenants)
    n_train = max(1, int(n_t * train_frac))
    n_val = max(1, int(n_t * val_frac))

    train_tenants = set(tenants[:n_train])
    val_tenants = set(tenants[n_train : n_train + n_val])
    test_tenants = set(tenants[n_train + n_val :])

    print(f"[split_by_tenant] train tenants ({len(train_tenants)}): {sorted(train_tenants)}")
    print(f"[split_by_tenant] val tenants   ({len(val_tenants)}): {sorted(val_tenants)}")
    print(f"[split_by_tenant] test tenants  ({len(test_tenants)}): {sorted(test_tenants)}")

    train_r, val_r, test_r = [], [], []
    for r in records:
        tid = r.get("tenant_id", "unknown")
        if tid in train_tenants:
            train_r.append(r)
        elif tid in val_tenants:
            val_r.append(r)
        else:
            test_r.append(r)

    return train_r, val_r, test_r


# ──────────────────────────────────────────────────────────────────────
# Dataset / collate
# ──────────────────────────────────────────────────────────────────────
class LogprobMismatchDataset(Dataset):
    def __init__(
        self,
        records: List[Dict[str, Any]],
        max_seq_len: int = 2048,
    ) -> None:
        self.records = records
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        r = self.records[idx]
        T = min(len(r["response_tokens"]), self.max_seq_len)
        return {
            "token_ids": torch.tensor(r["response_tokens"][:T], dtype=torch.long),
            "sampling_lps": torch.tensor(r["sampling_logprobs"][:T], dtype=torch.float32),
            "training_lps": torch.tensor(r["training_logprobs"][:T], dtype=torch.float32),
            "n_prompt_tokens": torch.tensor(float(r["n_prompt_tokens"]), dtype=torch.float32),
            "temperature": torch.tensor(float(r["temperature"]), dtype=torch.float32),
            "lora_rank": torch.tensor(float(r["lora_rank"]) / LORA_RANK_MAX, dtype=torch.float32),
        }


def collate(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    B = len(batch)
    Ts = [b["token_ids"].numel() for b in batch]
    max_T = max(Ts)

    token_ids = torch.zeros(B, max_T, dtype=torch.long)
    sampling_lps = torch.zeros(B, max_T, dtype=torch.float32)
    training_lps = torch.zeros(B, max_T, dtype=torch.float32)
    mask = torch.zeros(B, max_T, dtype=torch.bool)
    n_prompt_tokens = torch.zeros(B, dtype=torch.float32)
    temperature = torch.zeros(B, dtype=torch.float32)
    lora_rank = torch.zeros(B, dtype=torch.float32)

    for i, b in enumerate(batch):
        T = Ts[i]
        token_ids[i, :T] = b["token_ids"]
        sampling_lps[i, :T] = b["sampling_lps"]
        training_lps[i, :T] = b["training_lps"]
        mask[i, :T] = True
        n_prompt_tokens[i] = b["n_prompt_tokens"]
        temperature[i] = b["temperature"]
        lora_rank[i] = b["lora_rank"]

    return {
        "token_ids": token_ids,
        "sampling_lps": sampling_lps,
        "training_lps": training_lps,
        "mask": mask,
        "n_prompt_tokens": n_prompt_tokens,
        "temperature": temperature,
        "lora_rank": lora_rank,
    }


# ──────────────────────────────────────────────────────────────────────
# Helper to infer vocab size from data
# ──────────────────────────────────────────────────────────────────────
def infer_vocab_size(records: List[Dict[str, Any]], default: int = 152064) -> int:
    """Vocab size = max token id observed (+1).  Fall back to default."""
    mx = 0
    for r in records:
        for tid in r["response_tokens"]:
            if tid > mx:
                mx = tid
        for tid in r.get("prompt_tokens", []) or []:
            if tid > mx:
                mx = tid
    return max(mx + 1, default)
