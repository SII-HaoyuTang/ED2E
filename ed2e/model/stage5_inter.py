"""
Stage 5 (T_inter): Inter-level chart-graph message passing.

Propagates information bidirectionally between charts on adjacent manifold
levels (k ↔ k+1) using counting-based directional weights w̃_{a←b} and
7-dimensional edge features encoding the cross-level geometry.

B-block order: T_inter ∘ T_intra ∘ T_chart_encode ∘ (T_local_msg)³ ∘ T_init
                ↑ this module
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import torch
import torch.nn as nn

from ed2e.data.stage3_local import Stage3TensorBatch
from ed2e.model.stage3_local import (
    DualStreamState,
    _MLP,
    _scatter_add,
    _segment_softmax,
)

_EPS = 1e-8


@dataclass
class Stage5InterConfig:
    edge_attr_dim: int = 7
    scalar_dim: int = 64
    vector_dim: int = 8
    token_dim: int = 96


class InterLevelBlock(nn.Module):
    """
    One T_inter step: bidirectional message passing on the inter-level chart graph.

    forward() signature:
        p_bar  — DualStreamState  (A, 64) + (A, 8, 2)  chart states from T_intra
        batch  — Stage3TensorBatch  (for inter_level_* fields via inter_static_bundle)
        inter_static — dict from batch.inter_static_bundle()

    Returns updated DualStreamState (A, 64) + (A, 8, 2).
    """

    def __init__(self, cfg: Stage5InterConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or Stage5InterConfig()
        sd = self.cfg.scalar_dim   # 64
        vd = self.cfg.vector_dim   # 8
        td = self.cfg.token_dim    # 96
        ed = self.cfg.edge_attr_dim  # 7

        # Edge feature encoder
        self.enc_e = _MLP(ed, td, td)                    # 7 → 96 → 96

        # Attention: logit = psi_inter([p̄_dst^s, p̄_src^s, F_e])
        self.psi_inter = _MLP(sd + sd + td, td, 1)       # (64+64+96=224) → 96 → 1

        # Edge modulation
        # z_ab = [p̄_dst^s, p̄_src^s, F_e, p̄_dst^s - p̄_src^s, p̄_dst^s ⊙ p̄_src^s]
        # dim = 64 + 64 + 96 + 64 + 64 = 352
        self.mlp_beta   = _MLP(sd * 4 + td, td, 1)       # 352 → 96 → 1
        self.mlp_gate_s = _MLP(sd * 4 + td, td, sd)      # 352 → 96 → 64
        self.mlp_gate_v = _MLP(sd * 4 + td, td, vd)      # 352 → 96 → 8

        # Sender transforms
        self.phi_s = _MLP(sd,   td, sd)                  # 64 → 96 → 64
        self.phi_v = _MLP(vd*2, td, vd*2)               # 16 → 96 → 16

        # Chart update
        self.mlp_update_s     = _MLP(sd + sd + vd, td, sd)  # 136 → 96 → 64
        self.scalar_update_norm = nn.LayerNorm(sd)
        self.mlp_vgate        = _MLP(sd + sd, td, vd)       # 128 → 96 → 8
        self.scalar_to_v      = _MLP(sd, td, vd * 2)        # 64 → 96 → 16

    def forward(
        self,
        p_bar: DualStreamState,
        batch: Stage3TensorBatch,
        inter_static: Dict[str, Any],
    ) -> DualStreamState:
        # Step 1: empty graph short-circuit
        edge_index = inter_static["inter_level_edge_index"]  # (2, E_inter)
        if edge_index.numel() == 0:
            return p_bar

        A = p_bar.scalar.shape[0]
        vd = self.cfg.vector_dim
        src, dst = edge_index[0], edge_index[1]           # (E,)

        # Step 2: edge feature encoding
        e_ab = inter_static["inter_level_edge_attr"]      # (E, 7)
        F_e  = self.enc_e(e_ab)                           # (E, 96)

        # z_ab: [p̄_dst^s, p̄_src^s, F_e, p̄_dst^s − p̄_src^s, p̄_dst^s ⊙ p̄_src^s]
        p_dst = p_bar.scalar[dst]                         # (E, 64)
        p_src = p_bar.scalar[src]                         # (E, 64)
        z_ab = torch.cat(
            [p_dst, p_src, F_e, p_dst - p_src, p_dst * p_src],
            dim=-1,
        )                                                  # (E, 352)

        # Step 3: attention weights
        beta  = self.mlp_beta(z_ab).squeeze(-1)           # (E,)
        g_s   = torch.sigmoid(self.mlp_gate_s(z_ab))     # (E, 64)
        g_v   = torch.sigmoid(self.mlp_gate_v(z_ab))     # (E, 8)

        logit = self.psi_inter(
            torch.cat([p_dst, p_src, F_e], dim=-1)
        ).squeeze(-1)                                      # (E,)
        log_w = torch.log(inter_static["inter_level_weights"] + _EPS)  # (E,)
        alpha = _segment_softmax(logit + log_w + beta, dst, A)          # (E,)

        # Step 4: messages
        m_s = alpha.unsqueeze(-1) * g_s * self.phi_s(p_src)            # (E, 64)
        m_v = (
            alpha.view(-1, 1, 1) * g_v.unsqueeze(-1)
            * self.phi_v(p_bar.vector[src].flatten(1)).view(-1, vd, 2)
        )                                                  # (E, 8, 2)

        # Step 5: aggregate
        m_a_s = _scatter_add(m_s, dst, A)                # (A, 64)
        m_a_v = _scatter_add(m_v, dst, A)                # (A, 8, 2)

        # Step 6: update
        delta_s = self.mlp_update_s(
            torch.cat([p_bar.scalar, m_a_s, m_a_v.norm(dim=-1)], dim=-1)
        )                                                  # (A, 64)
        p_new_s = self.scalar_update_norm(p_bar.scalar + delta_s)

        gate_v  = torch.sigmoid(
            self.mlp_vgate(torch.cat([p_bar.scalar, m_a_s], dim=-1))
        ).unsqueeze(-1)                                    # (A, 8, 1)
        from_s  = self.scalar_to_v(p_new_s).view(A, vd, 2)
        p_new_v = p_bar.vector + gate_v * m_a_v + 0.1 * from_s   # (A, 8, 2)

        return DualStreamState(scalar=p_new_s, vector=p_new_v)


__all__ = [
    "Stage5InterConfig",
    "InterLevelBlock",
]
