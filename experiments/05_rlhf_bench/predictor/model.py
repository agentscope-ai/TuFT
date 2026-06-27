"""Predictor architectures for token-level framework mismatch.

Both models output Δ̂_t  (≈ sampling_lp_t − training_lp_t).
Corrected logprob = sampling_lp_t − Δ̂_t.

Tier-0  MLPPredictor          : per-token MLP, no context (fast baseline).
Tier-1  TransformerPredictor  : small encoder over the response sequence
                                (recommended main model).

─── Input Features ───────────────────────────────────────────────────────
The model input is user-agnostic by design (mismatch depends on adapter
configuration, not user identity). Final feature set:

  token_ids      [B, T]   int     Response token IDs. Embedded to capture
                                   per-token logprob sensitivity.
  sampling_lps   [B, T]   float   Log-probability under the *sampling* policy
                                   at generation time (observed, not predicted).
  mask           [B, T]   bool    True = valid position; False = padding.
  n_prompt_tokens [B]     float   Number of prompt tokens (context length).
                                   Affects KV-cache depth and attention pattern,
                                   thus indirectly modulates mismatch magnitude.
  temperature    [B]      float   Sampling temperature used at generation.
                                   Higher T → flatter dist → smaller raw
                                   logprobs but different mismatch profile.
  lora_rank      [B]      float   Normalized LoRA rank (rank / 64). Controls
                                   adapter expressiveness; larger rank generally
                                   means larger potential divergence from base
                                   weights between sampling and training steps.

Design notes:
  • No task_id / tenant_id: mismatch is user-agnostic; the above continuous
    features already capture all relevant adapter-config information.
  • lora_rank is normalized (÷ 64) so the model sees a [0, 1]-range scalar;
    this avoids OOV issues and generalises to unseen rank values.
  • vocab_size only determines the token_emb table size.
──────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


# ──────────────────────────────────────────────────────────────────────
# Sinusoidal encoding (works for both scalars like n_prompt_tokens and
# integer positions like t = 0..T-1).
# ──────────────────────────────────────────────────────────────────────
def sinusoidal_encode(values: torch.Tensor, dim: int) -> torch.Tensor:
    """values: arbitrary float tensor.  Output: same shape + (dim,) at the end."""
    half = dim // 2
    freqs = torch.exp(
        torch.arange(half, dtype=torch.float32, device=values.device)
        * -(math.log(10000.0) / max(half, 1))
    )
    args = values.unsqueeze(-1) * freqs  # [..., half]
    out = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if out.shape[-1] < dim:
        # pad if dim is odd
        pad = torch.zeros(*out.shape[:-1], dim - out.shape[-1], device=values.device)
        out = torch.cat([out, pad], dim=-1)
    return out


# ──────────────────────────────────────────────────────────────────────
# Tier-0: per-token MLP
# ──────────────────────────────────────────────────────────────────────
class MLPPredictor(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        token_emb_dim: int = 32,
        pos_dim: int = 16,
        hidden: int = 128,
    ) -> None:
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, token_emb_dim)
        self.pos_dim = pos_dim

        # token_emb + position + prefix_len + sampling_lp + temperature + lora_rank
        in_dim = token_emb_dim + pos_dim + pos_dim + 1 + 1 + 1
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        # Init last layer near-zero so the predictor starts at "no correction".
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(
        self,
        token_ids: torch.Tensor,  # [B, T]
        sampling_lps: torch.Tensor,  # [B, T]
        mask: torch.Tensor,  # [B, T]
        n_prompt_tokens: torch.Tensor,  # [B]
        temperature: torch.Tensor,  # [B]
        lora_rank: torch.Tensor,  # [B]  normalized (rank / 64)
    ) -> torch.Tensor:  # [B, T]
        B, T = token_ids.shape

        tok = self.token_emb(token_ids)  # [B, T, E]

        positions = torch.arange(T, device=token_ids.device, dtype=torch.float32)
        pos_enc = (
            sinusoidal_encode(positions, self.pos_dim).unsqueeze(0).expand(B, -1, -1)
        )  # [B, T, P]

        prefix_enc = (
            sinusoidal_encode(n_prompt_tokens, self.pos_dim).unsqueeze(1).expand(-1, T, -1)
        )  # [B, T, P]

        lp = sampling_lps.unsqueeze(-1)  # [B, T, 1]
        temp = temperature.unsqueeze(-1).unsqueeze(1).expand(-1, T, -1)  # [B, T, 1]
        lr = lora_rank.unsqueeze(-1).unsqueeze(1).expand(-1, T, -1)  # [B, T, 1]

        x = torch.cat([tok, pos_enc, prefix_enc, lp, temp, lr], dim=-1)
        delta = self.mlp(x).squeeze(-1)
        return delta


# ──────────────────────────────────────────────────────────────────────
# Tier-1: small Transformer over the response sequence
# ──────────────────────────────────────────────────────────────────────
class ScalarMLPPredictor(nn.Module):
    """MLP variant that predicts a single scalar bias per sequence."""

    def __init__(
        self,
        vocab_size: int,
        token_emb_dim: int = 32,
        pos_dim: int = 16,
        hidden: int = 128,
    ) -> None:
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, token_emb_dim)
        self.pos_dim = pos_dim

        in_dim = token_emb_dim + pos_dim + pos_dim + 1 + 1 + 1
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(
        self,
        token_ids: torch.Tensor,
        sampling_lps: torch.Tensor,
        mask: torch.Tensor,
        n_prompt_tokens: torch.Tensor,
        temperature: torch.Tensor,
        lora_rank: torch.Tensor,
    ) -> torch.Tensor:
        B, T = token_ids.shape

        tok = self.token_emb(token_ids)
        positions = torch.arange(T, device=token_ids.device, dtype=torch.float32)
        pos_enc = sinusoidal_encode(positions, self.pos_dim).unsqueeze(0).expand(B, -1, -1)
        prefix_enc = sinusoidal_encode(n_prompt_tokens, self.pos_dim).unsqueeze(1).expand(-1, T, -1)
        lp = sampling_lps.unsqueeze(-1)
        temp = temperature.unsqueeze(-1).unsqueeze(1).expand(-1, T, -1)
        lr = lora_rank.unsqueeze(-1).unsqueeze(1).expand(-1, T, -1)

        x = torch.cat([tok, pos_enc, prefix_enc, lp, temp, lr], dim=-1)
        delta = self.mlp(x).squeeze(-1)  # [B, T]

        # mean pool over valid positions to get a single scalar per sequence
        m = mask.float()
        scalar = (delta * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)  # [B]
        return scalar.unsqueeze(1).expand(-1, T) * m


class TransformerPredictor(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        token_emb_dim: int = 32,
        n_heads: int = 4,
        n_layers: int = 2,
        ffn_mult: int = 4,
        max_position: int = 4096,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model

        # Compact token embedding projected up — keeps vocab table small.
        self.token_emb = nn.Embedding(vocab_size, token_emb_dim)
        self.token_proj = nn.Linear(token_emb_dim, d_model)

        self.lp_proj = nn.Linear(1, d_model)
        self.temp_proj = nn.Linear(1, d_model)
        self.prefix_proj = nn.Linear(d_model, d_model)
        self.lora_proj = nn.Linear(1, d_model)  # normalized lora_rank → d_model

        positions = torch.arange(max_position, dtype=torch.float32)
        self.register_buffer(
            "position_table",
            sinusoidal_encode(positions, d_model),
            persistent=False,
        )

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * ffn_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

        self.head = nn.Linear(d_model, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(
        self,
        token_ids: torch.Tensor,  # [B, T]
        sampling_lps: torch.Tensor,  # [B, T]
        mask: torch.Tensor,  # [B, T]  True = valid
        n_prompt_tokens: torch.Tensor,  # [B]
        temperature: torch.Tensor,  # [B]
        lora_rank: torch.Tensor,  # [B]  normalized (rank / 64)
    ) -> torch.Tensor:  # [B, T]
        B, T = token_ids.shape

        x = self.token_proj(self.token_emb(token_ids))  # [B, T, D]
        x = x + self.position_table[:T].unsqueeze(0)
        x = x + self.lp_proj(sampling_lps.unsqueeze(-1))

        prefix_enc = sinusoidal_encode(n_prompt_tokens, self.d_model)  # [B, D]
        seq_feat = (
            self.prefix_proj(prefix_enc)
            + self.temp_proj(temperature.unsqueeze(-1))
            + self.lora_proj(lora_rank.unsqueeze(-1))
        ).unsqueeze(1)  # [B, 1, D]
        x = x + seq_feat

        # nn.TransformerEncoder: True in src_key_padding_mask = position is padding
        x = self.encoder(x, src_key_padding_mask=~mask)
        delta = self.head(x).squeeze(-1)
        # Zero out masked positions for safety (loss already masks, but cheap).
        return delta * mask.float()


# ──────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────
# Tier-2: per-sequence scalar predictor
#   Outputs a single scalar per sequence (mean mismatch to subtract from
#   sampling logprobs).  Forward broadcasts it back to [B, T] so the rest
#   of the training/evaluation code stays unchanged.
# ──────────────────────────────────────────────────────────────────────
class ScalarTransformerPredictor(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        token_emb_dim: int = 32,
        n_heads: int = 4,
        n_layers: int = 2,
        ffn_mult: int = 4,
        max_position: int = 4096,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model

        self.token_emb = nn.Embedding(vocab_size, token_emb_dim)
        self.token_proj = nn.Linear(token_emb_dim, d_model)

        self.lp_proj = nn.Linear(1, d_model)
        self.temp_proj = nn.Linear(1, d_model)
        self.prefix_proj = nn.Linear(d_model, d_model)
        self.lora_proj = nn.Linear(1, d_model)

        positions = torch.arange(max_position, dtype=torch.float32)
        self.register_buffer(
            "position_table",
            sinusoidal_encode(positions, d_model),
            persistent=False,
        )

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * ffn_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(
        self,
        token_ids: torch.Tensor,
        sampling_lps: torch.Tensor,
        mask: torch.Tensor,
        n_prompt_tokens: torch.Tensor,
        temperature: torch.Tensor,
        lora_rank: torch.Tensor,
    ) -> torch.Tensor:
        B, T = token_ids.shape

        x = self.token_proj(self.token_emb(token_ids))
        x = x + self.position_table[:T].unsqueeze(0)
        x = x + self.lp_proj(sampling_lps.unsqueeze(-1))

        prefix_enc = sinusoidal_encode(n_prompt_tokens, self.d_model)
        seq_feat = (
            self.prefix_proj(prefix_enc)
            + self.temp_proj(temperature.unsqueeze(-1))
            + self.lora_proj(lora_rank.unsqueeze(-1))
        ).unsqueeze(1)
        x = x + seq_feat

        x = self.encoder(x, src_key_padding_mask=~mask)

        # mean pool over valid positions
        m = mask.unsqueeze(-1).float()
        x_mean = (x * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)
        scalar = self.head(x_mean).squeeze(-1)  # [B]

        # broadcast scalar to per-token shape [B, T] for compatibility
        return scalar.unsqueeze(1).expand(-1, T) * mask.float()


def build_model(name: str, **kwargs) -> nn.Module:
    """Build predictor model.

    Required kwarg: vocab_size.
    Optional kwargs forwarded to the chosen architecture.
    """
    if name == "mlp":
        return MLPPredictor(**kwargs)
    if name == "mlp_scalar":
        return ScalarMLPPredictor(**kwargs)
    if name == "transformer":
        return TransformerPredictor(**kwargs)
    if name == "scalar_transformer":
        return ScalarTransformerPredictor(**kwargs)
    raise ValueError(f"Unknown predictor model: {name}")
