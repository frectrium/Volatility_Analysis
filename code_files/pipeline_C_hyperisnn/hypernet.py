"""HyperNetwork g_θ(Z): reference set → modulation vector ω ∈ R^P.

Architecture (matches plan.md §3.1):

    Z (B, N_ref, 3) ─► Linear(3, d) ─► TransformerEncoder(L layers, h heads)
                                          │
                                          ▼
                                       MeanPool over N (mask-aware)
                                          │
                                          ▼
                                       LayerNorm(d)
                                          │
              ctx (B, 9) ─► Linear(9, d) ─┼─► concatenate (2d)
                                          │
                                          ▼
                                       Linear(2d, 256) ─► GELU ─► Linear(256, P)
                                          │
                                          ▼
                                       ω ∈ R^P
"""

import torch
import torch.nn as nn

from .config import HyperConfig


class HyperNet(nn.Module):
    def __init__(
        self,
        out_dim: int,
        cfg: HyperConfig,
        n_ref_features: int = 3,
        n_ctx_features: int = 9,
    ):
        super().__init__()
        d = cfg.d_hyper
        self.out_dim = out_dim

        # Encoders
        self.q_enc = nn.Linear(n_ref_features, d)
        self.c_enc = nn.Linear(n_ctx_features, d)

        # Transformer over the reference set
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=cfg.n_heads,
            dim_feedforward=4 * d,
            dropout=cfg.transformer_dropout,
            activation=cfg.head_activation,
            batch_first=True,
            norm_first=True,        # pre-norm: more stable for deep transformers
        )
        self.tr = nn.TransformerEncoder(enc_layer, num_layers=cfg.n_layers)

        # Pool normalisation
        self.pool_norm = nn.LayerNorm(d)
        self.ctx_norm  = nn.LayerNorm(d)

        # 2-layer MLP head
        self.head = nn.Sequential(
            nn.Linear(2 * d, cfg.head_hidden),
            nn.GELU() if cfg.head_activation.lower() == "gelu" else nn.ReLU(),
            nn.Linear(cfg.head_hidden, out_dim),
        )

        # Init: keep ω small at start so W_raw ≈ W_base, but not so small that
        # the hypernet contribution is invisible to the optimizer.
        with torch.no_grad():
            nn.init.normal_(self.head[-1].weight, std=cfg.head_init_std if hasattr(cfg, "head_init_std") else 0.01)
            nn.init.zeros_(self.head[-1].bias)

    def forward(
        self,
        q: torch.Tensor,           # (B, N, n_ref_features)
        mask: torch.Tensor,        # (B, N) bool, True = valid contract
        ctx: torch.Tensor,         # (B, n_ctx_features)
    ) -> torch.Tensor:
        """Returns ω of shape (B, out_dim)."""
        # Encode reference contracts
        q_emb = self.q_enc(q)                                  # (B, N, d)

        # Transformer expects key_padding_mask = True for PAD positions
        kpm = ~mask
        z = self.tr(q_emb, src_key_padding_mask=kpm)           # (B, N, d)

        # Mask-aware mean pool
        w = mask.unsqueeze(-1).float()
        pooled = (z * w).sum(dim=1) / w.sum(dim=1).clamp(min=1.0)   # (B, d)
        pooled = self.pool_norm(pooled)

        # Context features
        c = self.ctx_norm(self.c_enc(ctx))                     # (B, d)

        # Concatenate and project to output dim
        h = torch.cat([pooled, c], dim=-1)                     # (B, 2d)
        omega = self.head(h)                                   # (B, out_dim)
        return omega
