"""
Output head: multi-head cross-attention chart readout + independent energy MLPs.

Architecture
------------

chart features:
    h_a = [scalar(64), ||vector||(8), level_emb(8)]          → (A, 80)
    H_a = chart_enc(h_a)                                     → (A, token_dim=96)

Shared K and V across all targets:
    K = W_k(H_a).view(A, num_heads, head_dim)
    V = W_v(H_a).view(A, num_heads, head_dim)

Target-specific query (T, num_heads, head_dim):
    q_t^h  ← nn.Parameter

Attention per (target, head), using segment_softmax within each molecule:
    score = q_t^h · K_h / sqrt(head_dim)    shape (A,)
    alpha = segment_softmax(score, chart_batch, B)   → (A,)  sums to 1 per mol
    g_{t,h} = scatter_add(alpha * V_h, chart_batch, B)       → (B, head_dim)

Readout per target:
    g_t = concat_h(g_{t,h})                  → (B, token_dim)

EnergyHeads:
    T independent MLPs (Linear→LN→GELU→Dropout→Linear(→1)), no shared weights
    pred: (B, T)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ed2e.data.stage3_local import Stage3TensorBatch
from ed2e.model.stage3_local import DualStreamState, _MLP, _safe_norm, _scatter_add, _segment_softmax

_EPS = 1e-8


@dataclass
class ReadoutConfig:
    scalar_dim:    int = 64
    vector_dim:    int = 8
    level_emb_dim: int = 8
    num_levels:    int = 4
    token_dim:     int = 96   # must be divisible by num_heads
    num_heads:     int = 4    # head_dim = token_dim / num_heads = 24


class MultiHeadChartReadout(nn.Module):
    """
    Multi-head cross-attention readout pooling chart states into per-molecule
    representations, one per energy target.

    Parameters
    ----------
    cfg         : ReadoutConfig
    num_targets : int  (injected from ED2EConfig to keep a single source of truth)

    forward() takes DualStreamState (A, scalar_dim) + (A, vector_dim, 2)
    and Stage3TensorBatch, returns dict:
      "global_features" : (B, num_targets, token_dim)
      "attn_weights"    : (num_targets, num_heads, A)  — only when return_attn=True
    """

    def __init__(self, cfg: ReadoutConfig, num_targets: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_targets = num_targets

        sd  = cfg.scalar_dim    # 64
        vd  = cfg.vector_dim    # 8
        led = cfg.level_emb_dim # 8
        td  = cfg.token_dim     # 96
        H   = cfg.num_heads     # 4
        T   = num_targets

        assert td % H == 0, f"token_dim={td} must be divisible by num_heads={H}"
        self.head_dim = td // H   # 24

        # Level embedding
        self.level_emb = nn.Embedding(cfg.num_levels, led)

        # h_a encoder: (scalar + ||vector|| + level_emb) → token_dim
        # ||vector|| along last dim → (A, vector_dim)
        self.chart_enc = _MLP(sd + vd + led, td, td)

        # Shared K, V projections
        self.W_k = nn.Linear(td, td, bias=False)
        self.W_v = nn.Linear(td, td, bias=False)

        # Target-specific query parameters: (T, num_heads, head_dim)
        self.target_queries = nn.Parameter(
            torch.empty(T, H, self.head_dim)
        )
        nn.init.normal_(self.target_queries, std=1.0 / math.sqrt(self.head_dim))

    def forward(
        self,
        p_new: DualStreamState,
        batch: Stage3TensorBatch,
        return_attn: bool = False,
    ) -> Dict[str, Any]:
        H  = self.cfg.num_heads
        T  = self.num_targets
        d  = self.head_dim
        td = self.cfg.token_dim

        # ── Build per-chart feature ──────────────────────────────────────────
        level_id = batch.chart_frame_metadata["chart_level_id"]   # (A,) int64
        level_id = level_id.clamp(max=self.cfg.num_levels - 1)
        lev_emb  = self.level_emb(level_id)                       # (A, 8)

        # ||vector|| across last dim (equivariant norm)
        v_norm = _safe_norm(p_new.vector)                        # (A, vector_dim)

        h_a = torch.cat([p_new.scalar, v_norm, lev_emb], dim=-1) # (A, 80)
        H_a = self.chart_enc(h_a)                                 # (A, td)

        # ── Shared K, V ──────────────────────────────────────────────────────
        A = H_a.shape[0]
        B = int(batch.chart_batch.max().item()) + 1

        K = self.W_k(H_a).view(A, H, d)    # (A, H, d)
        V = self.W_v(H_a).view(A, H, d)    # (A, H, d)

        chart_batch = batch.chart_batch     # (A,) long

        # ── Per-target attention & pooling ───────────────────────────────────
        # g: (B, T, td)
        g = H_a.new_zeros(B, T, td)

        if return_attn:
            # (T, H, A) — store all attention weights
            attn_all = H_a.new_zeros(T, H, A)

        scale = math.sqrt(d)

        for t in range(T):
            q_t = self.target_queries[t]      # (H, d)
            # score per head: (A, H)
            scores = (K * q_t.unsqueeze(0)).sum(dim=-1) / scale  # (A, H)

            for h in range(H):
                alpha = _segment_softmax(scores[:, h], chart_batch, B)  # (A,)
                # scatter: (B, d)
                g[:, t, h * d:(h + 1) * d] = _scatter_add(
                    alpha.unsqueeze(-1) * V[:, h, :], chart_batch, B
                )
                if return_attn:
                    attn_all[t, h] = alpha

        out: Dict[str, Any] = {"global_features": g}
        if return_attn:
            out["attn_weights"] = attn_all
        return out


class EnergyHeads(nn.Module):
    """
    num_targets independent output MLPs.

    Each MLP: Linear → LayerNorm → GELU → Dropout → Linear(1)
    Weights are fully independent — no sharing between targets.

    forward(g: (B, T, in_dim)) → (B, T)
    """

    def __init__(self, in_dim: int, hidden_dim: int, num_targets: int, dropout: float) -> None:
        super().__init__()
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
            for _ in range(num_targets)
        ])
        self.num_targets = num_targets

    def forward(self, g: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        g : (B, T, in_dim)

        Returns
        -------
        (B, T)  — predicted (normalised) energies
        """
        return torch.cat(
            [self.heads[t](g[:, t, :]) for t in range(self.num_targets)],
            dim=-1,
        )  # (B, T)


__all__ = [
    "ReadoutConfig",
    "MultiHeadChartReadout",
    "EnergyHeads",
]
