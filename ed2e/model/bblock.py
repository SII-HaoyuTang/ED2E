"""
B-block: one complete iteration of the ED2E message-passing pipeline.

B-block order:
    T_inter ∘ T_intra ∘ T_chart_encode ∘ (T_local_msg)³ ∘ T_init

``ED2EBBlockStack`` runs K B-blocks (independent weights) and returns the
chart-level ``DualStreamState`` after the final iteration.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as _ckpt

from ed2e.data.stage3_local import Stage3TensorBatch
from ed2e.model.stage3_local import (
    DualStreamState,
    ExplicitStructureEncoder,
    FCLCLocalBlock,
    Stage3LocalConfig,
    _MLP,
    _project_vectors,
)
from ed2e.model.stage4_intra import IntraLevelBlock, Stage4IntraConfig
from ed2e.model.stage5_inter import InterLevelBlock, Stage5InterConfig


@dataclass
class BBlockConfig:
    num_bblocks: int = 3
    local_cfg:   Stage3LocalConfig = field(default_factory=Stage3LocalConfig)
    intra_cfg:   Stage4IntraConfig = field(default_factory=Stage4IntraConfig)
    inter_cfg:   Stage5InterConfig = field(default_factory=Stage5InterConfig)
    use_gradient_checkpointing: bool = True   # recompute activations in backward to save ~35% memory


def _run_bblock(
    block: "BBlock",
    batch: Stage3TensorBatch,
    intra_static: dict,
    inter_static: dict,
    ss_scalar: torch.Tensor,
    ss_vector: torch.Tensor,
    pp_scalar,   # Tensor or None
    pp_vector,   # Tensor or None
):
    """Flat-tensor wrapper so torch.utils.checkpoint can handle DualStreamState."""
    shared = DualStreamState(scalar=ss_scalar, vector=ss_vector)
    p_prev = DualStreamState(scalar=pp_scalar, vector=pp_vector) if pp_scalar is not None else None
    out    = block(batch, shared, p_prev, intra_static, inter_static)
    p_new  = out["p_new"]
    nss    = out["node_state_shared"]
    return p_new.scalar, p_new.vector, nss.scalar, nss.vector


class BBlock(nn.Module):
    """
    One B-block iteration.

    forward() takes pre-computed static bundles (precomputed once per batch
    in BBlockStack) to avoid redundant work across iterations.

    Parameters
    ----------
    batch         : Stage3TensorBatch
    shared_state  : DualStreamState (N, 64) + (N, 8, 2), per-node — keyword-only
    p_prev        : DualStreamState (A, 64) + (A, 8, 2), per-chart from previous BBlock
    intra_static  : dict from batch.intra_static_bundle()  — precomputed once
    inter_static  : dict from batch.inter_static_bundle()  — precomputed once

    Returns
    -------
    dict with keys:
      "p_new"             : DualStreamState (A, 64) + (A, 8, 2)  — per-chart
      "node_state_shared" : DualStreamState (N, 64) + (N, 8, 2)  — per-node
    """

    def __init__(
        self,
        local_cfg: Stage3LocalConfig,
        intra_cfg: Stage4IntraConfig,
        inter_cfg:  Stage5InterConfig,
        structure_encoder: ExplicitStructureEncoder,
    ) -> None:
        super().__init__()
        self.local_block = FCLCLocalBlock(local_cfg, structure_encoder)
        self.intra_block = IntraLevelBlock(intra_cfg)
        self.inter_block = InterLevelBlock(inter_cfg)

    def forward(
        self,
        batch: Stage3TensorBatch,
        shared_state: DualStreamState,
        p_prev: Optional[DualStreamState],
        intra_static: Dict[str, Any],
        inter_static: Dict[str, Any],
    ) -> Dict[str, Any]:
        # Stage 3: T_init + (T_local_msg × num_local_steps) + T_chart_encode
        out3 = self.local_block(batch, shared_state=shared_state, p_prev=p_prev)
        p_mid       = out3["p_next_local"]        # (A, 64+8×2)  chart state
        loc_fin     = out3["local_state_final"]   # (N_M, ...)   node state for overlap ctx

        # Stage 4: T_intra
        p_bar = self.intra_block(p_mid, loc_fin, batch, intra_static)

        # Stage 5: T_inter
        p_new = self.inter_block(p_bar, batch, inter_static)

        return {
            "p_new":             p_new,
            "node_state_shared": out3["node_state_shared_next"],  # (N, ...) per-node
        }


class ED2EBBlockStack(nn.Module):
    """
    Stack of K B-blocks with independent weights, sharing one ExplicitStructureEncoder.

    Initialization (raw features → hidden dim) is performed once before the
    block loop.  Both the per-node shared state and the per-chart state (p_new)
    are propagated across BBlock iterations so chart information accumulates.

    Returns the chart-level DualStreamState after the last B-block.
    """

    def __init__(self, cfg: BBlockConfig) -> None:
        super().__init__()
        local_cfg = cfg.local_cfg

        # Raw-feature embedding — called once per forward pass (first BBlock only).
        self.shared_scalar_in = _MLP(
            local_cfg.raw_scalar_dim, local_cfg.scalar_dim, local_cfg.scalar_dim
        )
        self.shared_vector_in = _MLP(
            local_cfg.raw_vector_channels * 2,
            local_cfg.token_dim,
            local_cfg.vector_dim * 2,
        )

        # Shared ExplicitStructureEncoder — one instance reused by all K BBlocks.
        scalar_es_dim = local_cfg.scalar_dim * 10
        vector_es_dim = local_cfg.vector_dim * 22
        self.structure_encoder = ExplicitStructureEncoder(
            local_cfg, scalar_es_dim, vector_es_dim
        )

        # K BBlocks with independent message-passing weights, sharing structure_encoder.
        self.blocks = nn.ModuleList([
            BBlock(cfg.local_cfg, cfg.intra_cfg, cfg.inter_cfg, self.structure_encoder)
            for _ in range(cfg.num_bblocks)
        ])

    def _initialize_shared_state(self, batch: Stage3TensorBatch) -> DualStreamState:
        """Project raw node features to the hidden dimension (called once per forward)."""
        # Normalise scalar inputs per-node to prevent extreme physical values
        # (e.g. ‖∇ρ‖, Δρ can reach ~1e7 for steep density regions) from
        # collapsing LayerNorm variance and producing NaN.
        scalar_scale = batch.node_scalar_raw.abs().amax(dim=-1, keepdim=True).clamp_min(1.0)
        node_scalar_norm = batch.node_scalar_raw / scalar_scale   # (N, 5), values in [-1, 1]
        scalar = self.shared_scalar_in(node_scalar_norm)

        # Same normalisation for vector features (∇_M‖∇ρ‖, ∇_M H) — all-positive
        # norms after _safe_norm would otherwise cause near-zero variance in LayerNorm.
        vec_scale = (
            batch.node_vector_raw.abs()
            .amax(dim=(-2, -1), keepdim=True)   # (N, 1, 1)
            .clamp_min(1.0)
        )
        node_vector_norm = batch.node_vector_raw / vec_scale  # (N, 2, 3), values in [-1, 1]

        raw_vec_2d = _project_vectors(node_vector_norm, batch.node_tangent_basis)
        vector = self.shared_vector_in(raw_vec_2d.flatten(start_dim=1)).view(
            -1, self.blocks[0].local_block.cfg.vector_dim, 2
        )
        return DualStreamState(scalar=scalar, vector=vector)

    def forward(self, batch: Stage3TensorBatch) -> DualStreamState:
        intra_static = batch.intra_static_bundle()   # static, compute once
        inter_static = batch.inter_static_bundle()

        shared_state: DualStreamState = self._initialize_shared_state(batch)
        p_prev: Optional[DualStreamState] = None

        use_ckpt = self.cfg.use_gradient_checkpointing and self.training

        for block in self.blocks:
            ss_s, ss_v = shared_state.scalar, shared_state.vector
            pp_s = p_prev.scalar if p_prev is not None else None
            pp_v = p_prev.vector if p_prev is not None else None

            if use_ckpt:
                pns, pnv, nsss, nssv = _ckpt(
                    _run_bblock,
                    block, batch, intra_static, inter_static,
                    ss_s, ss_v, pp_s, pp_v,
                    use_reentrant=False,
                )
            else:
                out = block(batch, shared_state, p_prev, intra_static, inter_static)
                pns,  pnv  = out["p_new"].scalar,            out["p_new"].vector
                nsss, nssv = out["node_state_shared"].scalar, out["node_state_shared"].vector

            shared_state = DualStreamState(scalar=nsss, vector=nssv)
            p_prev       = DualStreamState(scalar=pns,  vector=pnv)

        return p_prev  # type: ignore[return-value]  # (A, 64) + (A, 8, 2)


__all__ = [
    "BBlockConfig",
    "BBlock",
    "ED2EBBlockStack",
]
