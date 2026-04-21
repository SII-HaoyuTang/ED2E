"""
Stage 4 (T_intra): Intra-level chart-graph message passing.

For each manifold level, passes messages between charts belonging to the same
connected component using:
  1. Explicit-structure modulation (F̃_a) from intra_geom_static
  2. Overlap-context correction using shared membership entries

B-block order: T_inter ∘ T_intra ∘ T_chart_encode ∘ (T_local_msg)³ ∘ T_init
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
    _rotate_vectors,
    _safe_norm,
    _scatter_add,
    _scatter_mean,
    _segment_softmax,
)

_EPS = 1e-8


@dataclass
class Stage4IntraConfig:
    geom_intra_dim: int = 7
    scalar_dim: int = 64
    vector_dim: int = 8
    token_dim: int = 96
    token_heads: int = 4


class ExplicitStructureEncoderIntra(nn.Module):
    """
    Encodes each chart's identity relative to its group's reference chart.

    Input:
      geom_static  (A, 7)   — [d_M, Δx_u, Δx_v, 1-n·n₀, cosθ, sinθ, log_area_ratio]
      scalar_es    (A, 192) — [p_a^s, p_ref^s, p_a^s - p_ref^s]
      vec_es       (A, 72)  — rotated vectors + norms + dot products

    Output: F̃_a (A, token_dim)
    """

    def __init__(self, cfg: Stage4IntraConfig) -> None:
        super().__init__()
        self.enc_g = _MLP(cfg.geom_intra_dim, cfg.token_dim, cfg.token_dim)
        self.enc_s = _MLP(cfg.scalar_dim * 3, cfg.token_dim, cfg.token_dim)
        self.enc_v = _MLP(cfg.vector_dim * 9, cfg.token_dim, cfg.token_dim)
        self.type_embedding = nn.Parameter(torch.randn(3, cfg.token_dim) * 0.02)
        self.attn = nn.MultiheadAttention(cfg.token_dim, cfg.token_heads, batch_first=True)
        self.token_norm = nn.LayerNorm(cfg.token_dim)
        self.fuse_norm = nn.LayerNorm(cfg.token_dim)
        self.F_tilde_head = _MLP(cfg.token_dim, cfg.token_dim, cfg.token_dim)

    def forward(
        self,
        geom_static: torch.Tensor,   # (A, 7)
        scalar_es: torch.Tensor,     # (A, 192)
        vec_es: torch.Tensor,        # (A, 72)
    ) -> torch.Tensor:               # (A, token_dim)
        tok_g = self.enc_g(geom_static)
        tok_s = self.enc_s(scalar_es)
        tok_v = self.enc_v(vec_es)
        tokens = torch.stack([tok_g, tok_s, tok_v], dim=1)   # (A, 3, D)
        tokens = tokens + self.type_embedding.unsqueeze(0)
        # seq=3 and head_dim=24 are incompatible with Flash/mem-efficient CUDA kernels.
        with torch.backends.cuda.sdp_kernel(
            enable_flash=False, enable_math=True, enable_mem_efficient=False
        ):
            attn_out, _ = self.attn(tokens, tokens, tokens, need_weights=False)
        tokens = self.token_norm(tokens + attn_out)
        pooled = self.fuse_norm(tokens.mean(dim=1))           # (A, D)
        return self.F_tilde_head(pooled)                      # (A, D)


class OverlapContextEncoder(nn.Module):
    """
    Aggregates shared membership state for each overlap edge → C_ab.

    Input (from intra_static bundle):
      local_state_final  (N_M, 64+8×2) — final membership-level state from Stage 3
      overlap_shared_membership_index (S, 2) — CSR content
      overlap_shared_ptr  (E_ov+1,)
      overlap_jaccard     (E_ov,)
    """

    def __init__(self, cfg: Stage4IntraConfig) -> None:
        super().__init__()
        # (64 + 8 + 1) → 64
        self.mlp_ctx = _MLP(cfg.scalar_dim + cfg.vector_dim + 1, cfg.token_dim, cfg.scalar_dim)

    def forward(
        self,
        local_state_final: DualStreamState,
        intra_static: Dict[str, Any],
    ) -> torch.Tensor:
        """Returns C_ab of shape (E_ov, scalar_dim)."""
        pairs = intra_static["overlap_shared_membership_index"]   # (S, 2)
        ptr   = intra_static["overlap_shared_ptr"]                # (E_ov+1,)
        jac   = intra_static["overlap_jaccard"]                   # (E_ov,)
        E_ov  = jac.shape[0]
        device = jac.device

        h_s = local_state_final.scalar[pairs[:, 0]]              # (S, 64)
        h_v_norm = _safe_norm(local_state_final.vector[pairs[:, 0]])  # (S, 8)

        counts  = ptr[1:] - ptr[:-1]                              # (E_ov,)
        seg_idx = torch.repeat_interleave(
            torch.arange(E_ov, device=device), counts
        )                                                         # (S,)

        C_mean_s = _scatter_mean(h_s,      seg_idx, E_ov)        # (E_ov, 64)
        C_mean_v = _scatter_mean(h_v_norm, seg_idx, E_ov)        # (E_ov, 8)
        ctx_in   = torch.cat([C_mean_s, C_mean_v, jac.unsqueeze(-1)], dim=-1)  # (E_ov, 73)
        return self.mlp_ctx(ctx_in)                               # (E_ov, 64)


class IntraLevelBlock(nn.Module):
    """
    One T_intra step: message passing on the intra-level chart graph.

    forward() signature:
        p_mid            — (A, 64) + (A, 8, 2)  chart states from T_chart_encode
        local_state_final — (N_M, 64) + (N_M, 8, 2)  final membership state from Stage 3
        batch            — Stage3TensorBatch  (for chart_frame_metadata)
        intra_static     — dict from batch.intra_static_bundle()

    Returns updated DualStreamState (A, 64) + (A, 8, 2).
    """

    def __init__(self, cfg: Stage4IntraConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or Stage4IntraConfig()
        sd = self.cfg.scalar_dim   # 64
        vd = self.cfg.vector_dim   # 8
        td = self.cfg.token_dim    # 96

        self.es_encoder = ExplicitStructureEncoderIntra(self.cfg)
        self.overlap_ctx_enc = OverlapContextEncoder(self.cfg)

        # Edge modulation & attention
        self.psi_intra  = _MLP(sd * 2,   td,  1)        # (128 → 96 → 1)
        self.mlp_beta   = _MLP(td * 4,   td,  1)        # (384 → 96 → 1)
        self.mlp_gate_s = _MLP(td * 4,   td,  sd)       # (384 → 96 → 64)
        self.mlp_gate_v = _MLP(td * 4,   td,  vd)       # (384 → 96 → 8)

        # Sender transforms
        self.phi_s = _MLP(sd,   td, sd)                 # (64 → 96 → 64)
        self.phi_v = _MLP(vd*2, td, vd*2)               # (16 → 96 → 16)

        # Overlap correction
        self.psi_ctx_s = _MLP(sd + sd + 1, td, sd)     # (129 → 96 → 64)

        # Chart update
        self.mlp_update_s     = _MLP(sd + sd + vd, td, sd)  # (136 → 96 → 64)
        self.scalar_update_norm = nn.LayerNorm(sd)
        self.mlp_vgate        = _MLP(sd + sd, td, vd)       # (128 → 96 → 8)
        self.scalar_to_v      = _MLP(sd, td, vd * 2)        # (64 → 96 → 16)

    def forward(
        self,
        p_mid: DualStreamState,
        local_state_final: DualStreamState,
        batch: Stage3TensorBatch,
        intra_static: Dict[str, Any],
    ) -> DualStreamState:
        # Step 1: empty graph short-circuit
        edge_index = intra_static["chart_graph_edge_index"]  # (2, E)
        if edge_index.numel() == 0:
            return p_mid

        A = p_mid.scalar.shape[0]
        device = p_mid.scalar.device

        # Step 2: F̃_a via explicit structure encoder
        ref_idx = intra_static["chart_to_ref"]                    # (A,)
        chart_frame = batch.chart_frame_metadata["chart_frame"]   # (A, 2, 3)

        p_ref_s = p_mid.scalar[ref_idx]                           # (A, 64)
        scalar_es = torch.cat(
            [p_mid.scalar, p_ref_s, p_mid.scalar - p_ref_s],
            dim=-1,
        )                                                          # (A, 192)

        p_a_in_ref   = _rotate_vectors(p_mid.vector, chart_frame, chart_frame[ref_idx])  # (A, 8, 2)
        p_ref_in_ref = p_mid.vector[ref_idx]                      # (A, 8, 2)
        diff_in_ref  = p_a_in_ref - p_ref_in_ref                  # (A, 8, 2)
        vec_es = torch.cat(
            [
                p_a_in_ref.flatten(1),                            # (A, 16)
                p_ref_in_ref.flatten(1),                          # (A, 16)
                diff_in_ref.flatten(1),                           # (A, 16)
                _safe_norm(p_mid.vector),                        # (A,  8)
                _safe_norm(p_mid.vector[ref_idx]),               # (A,  8)
                (p_a_in_ref * p_ref_in_ref).sum(dim=-1),          # (A,  8)
            ],
            dim=-1,
        )                                                          # (A, 72)

        geom_intra = intra_static["intra_geom_static"]
        g_scale = geom_intra.abs().amax(dim=-1, keepdim=True).clamp_min(1.0)
        F_tilde = self.es_encoder(
            geom_intra / g_scale, scalar_es, vec_es
        )                                                          # (A, 96)

        # Step 3: main messages
        src, dst = edge_index[0], edge_index[1]                   # (E,)
        z_ab = torch.cat(
            [F_tilde[dst], F_tilde[src],
             F_tilde[dst] - F_tilde[src],
             F_tilde[dst] * F_tilde[src]],
            dim=-1,
        )                                                          # (E, 384)

        beta  = self.mlp_beta(z_ab).squeeze(-1)                   # (E,)
        g_s   = torch.sigmoid(self.mlp_gate_s(z_ab))             # (E, 64)
        g_v   = torch.sigmoid(self.mlp_gate_v(z_ab))             # (E, 8)

        logit = self.psi_intra(
            torch.cat([p_mid.scalar[dst], p_mid.scalar[src]], dim=-1)
        ).squeeze(-1)                                              # (E,)
        alpha = _segment_softmax(logit + beta, dst, A)             # (E,)

        m_s = alpha.unsqueeze(-1) * g_s * self.phi_s(p_mid.scalar[src])       # (E, 64)
        m_v = (
            alpha.view(-1, 1, 1) * g_v.unsqueeze(-1)
            * self.phi_v(p_mid.vector[src].flatten(1)).view(-1, self.cfg.vector_dim, 2)
        )                                                          # (E, 8, 2)

        # Step 4: overlap-context correction
        ov2ce = intra_static["overlap_edge_to_chart_edge_index"]  # (E_ov,)
        E_ov = ov2ce.shape[0]
        if E_ov > 0:
            C_ab = self.overlap_ctx_enc(local_state_final, intra_static)  # (E_ov, 64)
            delta = self.psi_ctx_s(
                torch.cat(
                    [m_s[ov2ce], C_ab,
                     intra_static["overlap_jaccard"].unsqueeze(-1)],
                    dim=-1,
                )
            )                                                      # (E_ov, 64)
            # autograd-safe scatter correction
            # cast delta to m_s.dtype to survive AMP bf16↔fp32 mixing
            delta_full = torch.zeros_like(m_s)
            delta_full.scatter_add_(
                0,
                ov2ce.unsqueeze(-1).expand_as(delta),
                delta.to(m_s.dtype),
            )
            m_s = m_s + delta_full                                # (E, 64)

        # Step 5: aggregate
        m_a_s = _scatter_add(m_s, dst, A)                        # (A, 64)
        m_a_v = _scatter_add(m_v, dst, A)                        # (A, 8, 2)

        # Step 6: update
        delta_s = self.mlp_update_s(
            torch.cat([p_mid.scalar, m_a_s, _safe_norm(m_a_v)], dim=-1)
        )                                                          # (A, 64)
        p_bar_s = self.scalar_update_norm(p_mid.scalar + delta_s)

        gate_v  = torch.sigmoid(
            self.mlp_vgate(torch.cat([p_mid.scalar, m_a_s], dim=-1))
        ).unsqueeze(-1)                                            # (A, 8, 1)
        from_s  = self.scalar_to_v(p_bar_s).view(A, self.cfg.vector_dim, 2)
        p_bar_v = p_mid.vector + gate_v * m_a_v + 0.1 * from_s   # (A, 8, 2)

        return DualStreamState(scalar=p_bar_s, vector=p_bar_v)


__all__ = [
    "Stage4IntraConfig",
    "ExplicitStructureEncoderIntra",
    "OverlapContextEncoder",
    "IntraLevelBlock",
]
