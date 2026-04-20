"""
Stage 3 local aggregation and Stage 4-ready interface bundle.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from ed2e.data.stage3_local import Stage3TensorBatch

_EPS = 1e-8


def _safe_norm(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """L2 norm along dim with epsilon inside sqrt for stable backpropagation.

    Plain .norm() has gradient x/||x|| which is NaN at ||x||=0.
    This version computes sqrt(sum(x^2) + eps) so gradient is x/sqrt(sum(x^2)+eps),
    finite everywhere.
    """
    return (x.pow(2).sum(dim=dim) + _EPS).sqrt()


@dataclass
class DualStreamState:
    scalar: torch.Tensor
    vector: torch.Tensor


@dataclass
class Stage3LocalConfig:
    scalar_dim: int = 64
    vector_dim: int = 8
    token_dim: int = 96
    token_heads: int = 4
    num_local_steps: int = 2
    raw_scalar_dim: int = 5
    raw_vector_channels: int = 2
    edge_attr_dim: int = 6
    geom_dim: int = 32


def _scatter_add(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    if src.numel() == 0:
        return src.new_zeros((dim_size,) + src.shape[1:])
    out = src.new_zeros((dim_size,) + src.shape[1:])
    expand_index = index.view(-1, *([1] * (src.dim() - 1))).expand_as(src)
    out.scatter_add_(0, expand_index, src)
    return out


def _scatter_mean(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    if src.numel() == 0:
        return src.new_zeros((dim_size,) + src.shape[1:])
    total = _scatter_add(src, index, dim_size)
    ones = src.new_ones((src.shape[0],) + (1,) * (src.dim() - 1))
    count = _scatter_add(ones, index, dim_size).clamp_min_(1.0)
    return total / count


def _scatter_max_1d(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    out = src.new_full((dim_size,), -torch.inf)
    if src.numel() == 0:
        return out
    out.scatter_reduce_(0, index, src, reduce="amax", include_self=True)
    return out


def _segment_softmax(logits: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    if logits.numel() == 0:
        return logits
    max_per_dst = _scatter_max_1d(logits, index, dim_size)
    exp = torch.exp(logits - max_per_dst[index])
    denom = logits.new_zeros((dim_size,))
    denom.scatter_add_(0, index, exp)
    return exp / (denom[index] + _EPS)


def _reconstruct_vectors(vectors_2d: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return torch.einsum("nvc,ncd->nvd", vectors_2d, basis)


def _project_vectors(vectors_3d: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return torch.einsum("nvd,ncd->nvc", vectors_3d, basis)


def _rotate_vectors(vectors_2d: torch.Tensor, src_basis: torch.Tensor, dst_basis: torch.Tensor) -> torch.Tensor:
    vec3 = _reconstruct_vectors(vectors_2d, src_basis)
    return _project_vectors(vec3, dst_basis)


class _MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ExplicitStructureEncoder(nn.Module):
    """Enc_g / Enc_s / Enc_v + one typed-token attention block."""

    def __init__(self, cfg: Stage3LocalConfig, scalar_es_dim: int, vector_es_dim: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.enc_g = _MLP(cfg.geom_dim, cfg.token_dim, cfg.token_dim)
        self.enc_s = _MLP(scalar_es_dim, cfg.token_dim, cfg.token_dim)
        self.enc_v = _MLP(vector_es_dim, cfg.token_dim, cfg.token_dim)
        self.type_embedding = nn.Parameter(torch.randn(3, cfg.token_dim) * 0.02)
        self.attn = nn.MultiheadAttention(cfg.token_dim, cfg.token_heads, batch_first=True)
        self.token_norm = nn.LayerNorm(cfg.token_dim)
        self.fuse_norm = nn.LayerNorm(cfg.token_dim)
        self.vector_seed = _MLP(vector_es_dim, cfg.token_dim, cfg.vector_dim * 2)
        self.mod_scalar_head = _MLP(cfg.token_dim, cfg.token_dim, cfg.scalar_dim)
        self.chart_scalar_head = _MLP(cfg.token_dim, cfg.token_dim, cfg.scalar_dim)
        self.mod_vector_from_tokens = _MLP(cfg.token_dim, cfg.token_dim, cfg.vector_dim * 2)
        self.chart_vector_from_tokens = _MLP(cfg.token_dim, cfg.token_dim, cfg.vector_dim * 2)
        self.mod_vector_gate = nn.Linear(cfg.token_dim, cfg.vector_dim)
        self.chart_vector_gate = nn.Linear(cfg.token_dim, cfg.vector_dim)

    def forward(
        self,
        geom_static: torch.Tensor,
        scalar_es: torch.Tensor,
        vector_es: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        tok_g = self.enc_g(geom_static)
        tok_s = self.enc_s(scalar_es)
        tok_v = self.enc_v(vector_es)
        tokens = torch.stack([tok_g, tok_s, tok_v], dim=1)
        tokens = tokens + self.type_embedding.unsqueeze(0)
        attn_out, _ = self.attn(tokens, tokens, tokens, need_weights=False)
        tokens = self.token_norm(tokens + attn_out)
        pooled = self.fuse_norm(tokens.mean(dim=1))

        vector_seed = self.vector_seed(vector_es).view(-1, self.cfg.vector_dim, 2)
        mod_vec_tok = self.mod_vector_from_tokens(pooled).view(-1, self.cfg.vector_dim, 2)
        chart_vec_tok = self.chart_vector_from_tokens(pooled).view(-1, self.cfg.vector_dim, 2)
        mod_gate = torch.sigmoid(self.mod_vector_gate(pooled)).unsqueeze(-1)
        chart_gate = torch.sigmoid(self.chart_vector_gate(pooled)).unsqueeze(-1)

        return {
            "mod_scalar": self.mod_scalar_head(pooled),
            "chart_scalar": self.chart_scalar_head(pooled),
            "mod_vector": vector_seed + mod_gate * mod_vec_tok,
            "chart_vector": vector_seed + chart_gate * chart_vec_tok,
            "token_summary": pooled,
        }


class FCLCLocalBlock(nn.Module):
    """Stage 3 local block whose outputs directly match Stage 4 input needs."""

    def __init__(
        self,
        cfg: Optional[Stage3LocalConfig] = None,
        structure_encoder: Optional["ExplicitStructureEncoder"] = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg or Stage3LocalConfig()

        scalar_es_dim = self.cfg.scalar_dim * 10
        vector_es_dim = self.cfg.vector_dim * 22

        # structure_encoder is shared across all BBlocks; created externally and passed in.
        if structure_encoder is not None:
            self.structure_encoder = structure_encoder
        else:
            # Fallback: create a local instance (used when FCLCLocalBlock is instantiated
            # standalone, e.g. in smoke tests or PseudoStage4Consumer contexts).
            self.structure_encoder = ExplicitStructureEncoder(self.cfg, scalar_es_dim, vector_es_dim)

        msg_in_dim = self.cfg.scalar_dim + self.cfg.vector_dim + self.cfg.edge_attr_dim
        self.scalar_msg = _MLP(msg_in_dim, self.cfg.token_dim, self.cfg.scalar_dim)
        self.vector_msg_gate = _MLP(msg_in_dim, self.cfg.token_dim, self.cfg.vector_dim)
        self.edge_to_vector = _MLP(self.cfg.edge_attr_dim, self.cfg.token_dim, self.cfg.vector_dim * 2)
        self.chart_scalar_gate = nn.Linear(self.cfg.scalar_dim, self.cfg.scalar_dim)
        self.chart_attn_bias = nn.Linear(self.cfg.scalar_dim, 1)
        self.edge_attn = _MLP(self.cfg.scalar_dim * 2 + self.cfg.edge_attr_dim, self.cfg.token_dim, 1)

        self.scalar_update = _MLP(
            self.cfg.scalar_dim * 3 + self.cfg.vector_dim,
            self.cfg.token_dim,
            self.cfg.scalar_dim,
        )
        self.scalar_update_norm = nn.LayerNorm(self.cfg.scalar_dim)
        self.vector_update_gate = _MLP(self.cfg.scalar_dim * 2, self.cfg.token_dim, self.cfg.vector_dim)
        self.scalar_to_vector = _MLP(self.cfg.scalar_dim, self.cfg.token_dim, self.cfg.vector_dim * 2)

        self.shared_scalar_merge = _MLP(
            self.cfg.scalar_dim * 2 + self.cfg.vector_dim * 2,
            self.cfg.token_dim,
            self.cfg.scalar_dim,
        )
        self.shared_scalar_merge_norm = nn.LayerNorm(self.cfg.scalar_dim)
        self.shared_vector_merge_gate = _MLP(self.cfg.scalar_dim * 2, self.cfg.token_dim, self.cfg.vector_dim)

    def init_local_state(self, batch: Stage3TensorBatch, shared_state: DualStreamState) -> DualStreamState:
        node_idx = batch.chart_membership[:, 1]
        chart_idx = batch.chart_membership[:, 0]
        local_scalar = shared_state.scalar[node_idx]
        chart_frame = batch.chart_frame_metadata["chart_frame"][chart_idx]
        local_vector = _rotate_vectors(
            shared_state.vector[node_idx],
            batch.node_tangent_basis[node_idx],
            chart_frame,
        )
        return DualStreamState(scalar=local_scalar, vector=local_vector)

    def merge_shared_state(
        self,
        batch: Stage3TensorBatch,
        shared_prev: DualStreamState,
        local_state: DualStreamState,
    ) -> DualStreamState:
        node_idx = batch.chart_membership[:, 1]
        chart_idx = batch.chart_membership[:, 0]
        num_nodes = shared_prev.scalar.shape[0]
        weight = batch.membership_weight.unsqueeze(-1)
        chart_frame = batch.chart_frame_metadata["chart_frame"][chart_idx]

        merged_scalar = _scatter_add(local_state.scalar * weight, node_idx, num_nodes)

        local_vec_in_node_basis = _rotate_vectors(
            local_state.vector,
            chart_frame,
            batch.node_tangent_basis[node_idx],
        )
        merged_vector = _scatter_add(local_vec_in_node_basis * weight.unsqueeze(-1), node_idx, num_nodes)

        scalar_in = torch.cat(
            [
                shared_prev.scalar,
                merged_scalar,
                _safe_norm(shared_prev.vector),
                _safe_norm(merged_vector),
            ],
            dim=-1,
        )
        scalar_next = self.shared_scalar_merge_norm(
            shared_prev.scalar + self.shared_scalar_merge(scalar_in)
        )
        vector_gate = torch.sigmoid(
            self.shared_vector_merge_gate(torch.cat([shared_prev.scalar, merged_scalar], dim=-1))
        ).unsqueeze(-1)
        vector_next = shared_prev.vector + vector_gate * merged_vector
        return DualStreamState(scalar=scalar_next, vector=vector_next)

    def compute_dynamic_es(
        self,
        batch: Stage3TensorBatch,
        local_state: DualStreamState,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        chart_idx = batch.chart_membership[:, 0]
        num_charts = batch.chart_es_geom_static.shape[0]

        scalar_means: List[torch.Tensor] = []
        scalar_vars: List[torch.Tensor] = []
        vector_means: List[torch.Tensor] = []
        vector_norms: List[torch.Tensor] = []
        vector_disp: List[torch.Tensor] = []

        for q in range(4):
            mask = batch.quadrant == q
            if mask.any():
                idx_q = chart_idx[mask]
                scalar_q = local_state.scalar[mask]
                mean_q = _scatter_mean(scalar_q, idx_q, num_charts)
                scalar_means.append(mean_q)

                centered = scalar_q - mean_q[idx_q]
                var_q = _scatter_mean(centered.pow(2), idx_q, num_charts)
                scalar_vars.append(var_q)

                vec_q = local_state.vector[mask]
                vec_mean_q = _scatter_mean(vec_q.flatten(start_dim=1), idx_q, num_charts).view(
                    num_charts, self.cfg.vector_dim, 2
                )
                vector_means.append(vec_mean_q)

                vec_norm_q = _scatter_mean(_safe_norm(vec_q), idx_q, num_charts)
                vector_norms.append(vec_norm_q)

                vec_disp_q = _scatter_mean(
                    (vec_q - vec_mean_q[idx_q]).pow(2).sum(dim=-1),
                    idx_q,
                    num_charts,
                )
                vector_disp.append(vec_disp_q)
            else:
                scalar_means.append(local_state.scalar.new_zeros((num_charts, self.cfg.scalar_dim)))
                scalar_vars.append(local_state.scalar.new_zeros((num_charts, self.cfg.scalar_dim)))
                vector_means.append(local_state.vector.new_zeros((num_charts, self.cfg.vector_dim, 2)))
                vector_norms.append(local_state.scalar.new_zeros((num_charts, self.cfg.vector_dim)))
                vector_disp.append(local_state.scalar.new_zeros((num_charts, self.cfg.vector_dim)))

        scalar_diff_02 = scalar_means[0] - scalar_means[2]
        scalar_diff_13 = scalar_means[1] - scalar_means[3]
        scalar_es = torch.cat(
            [
                *scalar_means,
                *scalar_vars,
                scalar_diff_02,
                scalar_diff_13,
            ],
            dim=-1,
        )

        vec_diff_02 = vector_means[0] - vector_means[2]
        vec_diff_13 = vector_means[1] - vector_means[3]
        vec_dot_02 = (vector_means[0] * vector_means[2]).sum(dim=-1)
        vec_dot_13 = (vector_means[1] * vector_means[3]).sum(dim=-1)
        vector_es = torch.cat(
            [
                *(v.flatten(start_dim=1) for v in vector_means),
                *vector_norms,
                *vector_disp,
                vec_diff_02.flatten(start_dim=1),
                vec_diff_13.flatten(start_dim=1),
                vec_dot_02,
                vec_dot_13,
            ],
            dim=-1,
        )
        return scalar_es, vector_es

    def encode_structure(
        self,
        batch: Stage3TensorBatch,
        local_state: DualStreamState,
    ) -> Tuple[DualStreamState, DualStreamState, Dict[str, torch.Tensor]]:
        scalar_es, vector_es = self.compute_dynamic_es(batch, local_state)
        # Normalise per-chart: extreme values in chart_es_geom_static (e.g. curvature-
        # derived features for sharp density regions) cause enc_g outputs to overflow
        # and can collapse LayerNorm variance → NaN inside ExplicitStructureEncoder.
        geom_scale = (
            batch.chart_es_geom_static.abs().amax(dim=-1, keepdim=True).clamp_min(1.0)
        )
        geom_norm = batch.chart_es_geom_static / geom_scale
        out = self.structure_encoder(geom_norm, scalar_es, vector_es)
        mod_state = DualStreamState(out["mod_scalar"], out["mod_vector"])
        chart_state = DualStreamState(out["chart_scalar"], out["chart_vector"])
        debug = {
            "scalar_es": scalar_es,
            "vector_es": vector_es,
            "token_summary": out["token_summary"],
        }
        return mod_state, chart_state, debug

    def local_message_step(
        self,
        batch: Stage3TensorBatch,
        local_state: DualStreamState,
        mod_state: DualStreamState,
    ) -> DualStreamState:
        if batch.local_knn_edge_index.numel() == 0:
            return local_state

        src = batch.local_knn_edge_index[0]
        dst = batch.local_knn_edge_index[1]
        dst_chart = batch.chart_membership[dst, 0]
        num_local = local_state.scalar.shape[0]

        src_scalar = local_state.scalar[src]
        dst_scalar = local_state.scalar[dst]
        src_vector = local_state.vector[src]
        edge_attr = batch.local_edge_attr
        edge_scale = edge_attr.abs().amax(dim=-1, keepdim=True).clamp_min(1.0)
        edge_attr = edge_attr / edge_scale    # (E_loc, 6), values in [-1, 1]

        src_vec_norm = _safe_norm(src_vector)
        msg_in = torch.cat([src_scalar, src_vec_norm, edge_attr], dim=-1)
        base_scalar = self.scalar_msg(msg_in)
        scalar_gate = torch.sigmoid(self.chart_scalar_gate(mod_state.scalar[dst_chart]))
        base_scalar = base_scalar * scalar_gate

        attn_in = torch.cat([dst_scalar, src_scalar, edge_attr], dim=-1)
        attn_logits = self.edge_attn(attn_in).squeeze(-1) + self.chart_attn_bias(mod_state.scalar[dst_chart]).squeeze(-1)
        alpha = _segment_softmax(attn_logits, dst, num_local)
        scalar_agg = _scatter_add(alpha.unsqueeze(-1) * base_scalar, dst, num_local)

        vector_gate = torch.sigmoid(self.vector_msg_gate(msg_in)).unsqueeze(-1)
        edge_vec = self.edge_to_vector(edge_attr).view(-1, self.cfg.vector_dim, 2)
        base_vector = src_vector * vector_gate + 0.1 * edge_vec + 0.1 * mod_state.vector[dst_chart]
        vector_agg = _scatter_add(alpha.view(-1, 1, 1) * base_vector, dst, num_local)

        local_chart = batch.chart_membership[:, 0]
        scalar_upd_in = torch.cat(
            [
                local_state.scalar,
                scalar_agg,
                _safe_norm(vector_agg),
                mod_state.scalar[local_chart],
            ],
            dim=-1,
        )
        scalar_next = self.scalar_update_norm(
            local_state.scalar + self.scalar_update(scalar_upd_in)
        )

        vec_upd_gate = torch.sigmoid(
            self.vector_update_gate(torch.cat([local_state.scalar, scalar_agg], dim=-1))
        ).unsqueeze(-1)
        vec_from_scalar = self.scalar_to_vector(scalar_next).view(-1, self.cfg.vector_dim, 2)
        vector_next = local_state.vector + vec_upd_gate * vector_agg + 0.1 * vec_from_scalar
        return DualStreamState(scalar=scalar_next, vector=vector_next)

    def forward(
        self,
        batch: Stage3TensorBatch,
        *,
        shared_state: DualStreamState,
        p_prev: Optional[DualStreamState] = None,
        return_debug: bool = False,
    ) -> Dict[str, Any]:
        local_state = self.init_local_state(batch, shared_state)

        mod_state, chart_state, debug0 = self.encode_structure(batch, local_state)

        # If a chart state from the previous BBlock is available, use it as the
        # starting point (residual addition) so chart information accumulates
        # across BBlock iterations.
        if p_prev is not None:
            chart_state = DualStreamState(
                scalar=p_prev.scalar + chart_state.scalar,
                vector=p_prev.vector + chart_state.vector,
            )
            mod_state = chart_state

        debug_steps: List[Dict[str, torch.Tensor]] = [debug0] if return_debug else []

        for _ in range(self.cfg.num_local_steps):
            local_state = self.local_message_step(batch, local_state, mod_state)
            mod_state, chart_delta, debug_step = self.encode_structure(batch, local_state)
            # Accumulate chart state across local steps (residual).
            chart_state = DualStreamState(
                scalar=chart_state.scalar + chart_delta.scalar,
                vector=chart_state.vector + chart_delta.vector,
            )
            if return_debug:
                debug_steps.append(debug_step)

        shared_next = self.merge_shared_state(batch, shared_state, local_state)
        out = {
            "node_state_shared_next": shared_next,
            "local_state_final": local_state,
            "p_next_local": chart_state,
            "intra_static_bundle": batch.intra_static_bundle(),
        }
        if return_debug:
            out["debug"] = {
                "steps": debug_steps,
                "geom_static": batch.chart_es_geom_static.detach(),
            }
        return out


class PseudoStage4Consumer(nn.Module):
    """Minimal Stage 4-shaped consumer used only for zero-repack smoke checks."""

    def __init__(self, cfg: Optional[Stage3LocalConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or Stage3LocalConfig()
        self.chart_msg = _MLP(self.cfg.scalar_dim * 2, self.cfg.token_dim, self.cfg.scalar_dim)
        self.chart_attn = _MLP(self.cfg.scalar_dim * 2, self.cfg.token_dim, 1)
        self.overlap_ctx = _MLP(
            self.cfg.scalar_dim + self.cfg.vector_dim * 2 + 2 + 1,
            self.cfg.token_dim,
            self.cfg.scalar_dim + self.cfg.vector_dim * 2,
        )
        self.chart_update = _MLP(
            self.cfg.scalar_dim * 3 + self.cfg.vector_dim,
            self.cfg.token_dim,
            self.cfg.scalar_dim,
        )
        self.chart_vector_gate = _MLP(self.cfg.scalar_dim * 2, self.cfg.token_dim, self.cfg.vector_dim)

    def forward(
        self,
        *,
        p_next_local: DualStreamState,
        local_state_final: DualStreamState,
        node_state_shared_next: DualStreamState,
        batch: Stage3TensorBatch,
        intra_static_bundle: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        bundle = intra_static_bundle or batch.intra_static_bundle()
        num_charts = p_next_local.scalar.shape[0]

        chart_scalar_agg = p_next_local.scalar.new_zeros((num_charts, self.cfg.scalar_dim))
        if bundle["chart_graph_edge_index"].numel() > 0:
            src = bundle["chart_graph_edge_index"][0]
            dst = bundle["chart_graph_edge_index"][1]
            logits = self.chart_attn(torch.cat([p_next_local.scalar[src], p_next_local.scalar[dst]], dim=-1)).squeeze(-1)
            alpha = _segment_softmax(logits, dst, num_charts)
            msg = self.chart_msg(torch.cat([p_next_local.scalar[src], p_next_local.scalar[dst]], dim=-1))
            chart_scalar_agg = _scatter_add(alpha.unsqueeze(-1) * msg, dst, num_charts)

        overlap_scalar_agg = p_next_local.scalar.new_zeros((num_charts, self.cfg.scalar_dim))
        overlap_vector_agg = p_next_local.vector.new_zeros((num_charts, self.cfg.vector_dim, 2))
        ptr = bundle["overlap_shared_ptr"]
        pairs = bundle["overlap_shared_membership_index"]
        edges = bundle["overlap_edge_index"]
        jaccard = bundle["overlap_jaccard"]

        for edge_idx in range(edges.shape[1]):
            start = int(ptr[edge_idx].item())
            end = int(ptr[edge_idx + 1].item())
            if end <= start:
                continue
            recv_m = pairs[start:end, 0]
            recv_chart = int(edges[1, edge_idx].item())
            shared_node = batch.chart_membership[recv_m, 1]
            shared_scalar = node_state_shared_next.scalar[shared_node].mean(dim=0)
            recv_vector = local_state_final.vector[recv_m].mean(dim=0).flatten()
            recv_coord = batch.local_coords[recv_m].mean(dim=0)
            token = torch.cat(
                [
                    shared_scalar,
                    recv_vector,
                    recv_coord,
                    jaccard[edge_idx:edge_idx + 1],
                ],
                dim=0,
            )
            ctx = self.overlap_ctx(token.unsqueeze(0)).squeeze(0)
            overlap_scalar_agg[recv_chart] = overlap_scalar_agg[recv_chart] + ctx[:self.cfg.scalar_dim]
            overlap_vector_agg[recv_chart] = overlap_vector_agg[recv_chart] + ctx[self.cfg.scalar_dim:].view(
                self.cfg.vector_dim, 2
            )

        scalar_upd = self.chart_update(
            torch.cat(
                [
                    p_next_local.scalar,
                    chart_scalar_agg,
                    overlap_scalar_agg,
                    _safe_norm(p_next_local.vector),
                ],
                dim=-1,
            )
        )
        chart_scalar_next = p_next_local.scalar + scalar_upd
        vector_gate = torch.sigmoid(
            self.chart_vector_gate(torch.cat([p_next_local.scalar, overlap_scalar_agg], dim=-1))
        ).unsqueeze(-1)
        chart_vector_next = p_next_local.vector + vector_gate * overlap_vector_agg

        return {
            "chart_state_intra_ready": DualStreamState(
                scalar=chart_scalar_next,
                vector=chart_vector_next,
            ),
            "bundle_identity_ok": bool(bundle["chart_graph_edge_index"] is batch.chart_graph_edge_index),
        }


__all__ = [
    "DualStreamState",
    "Stage3LocalConfig",
    "ExplicitStructureEncoder",
    "FCLCLocalBlock",
    "PseudoStage4Consumer",
    "_MLP",
    "_project_vectors",
]
