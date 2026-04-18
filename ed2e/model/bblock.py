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

from ed2e.data.stage3_local import Stage3TensorBatch
from ed2e.model.stage3_local import DualStreamState, FCLCLocalBlock, Stage3LocalConfig
from ed2e.model.stage4_intra import IntraLevelBlock, Stage4IntraConfig
from ed2e.model.stage5_inter import InterLevelBlock, Stage5InterConfig


@dataclass
class BBlockConfig:
    num_bblocks: int = 3
    local_cfg:   Stage3LocalConfig = field(default_factory=Stage3LocalConfig)
    intra_cfg:   Stage4IntraConfig = field(default_factory=Stage4IntraConfig)
    inter_cfg:   Stage5InterConfig = field(default_factory=Stage5InterConfig)


class BBlock(nn.Module):
    """
    One B-block iteration.

    forward() takes pre-computed static bundles (precomputed once per batch
    in BBlockStack) to avoid redundant work across iterations.

    Parameters
    ----------
    batch         : Stage3TensorBatch
    shared_state  : DualStreamState (N, 64) + (N, 8, 2), per-node — keyword-only
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
    ) -> None:
        super().__init__()
        self.local_block = FCLCLocalBlock(local_cfg)
        self.intra_block = IntraLevelBlock(intra_cfg)
        self.inter_block = InterLevelBlock(inter_cfg)

    def forward(
        self,
        batch: Stage3TensorBatch,
        shared_state: Optional[DualStreamState],
        intra_static: Dict[str, Any],
        inter_static: Dict[str, Any],
    ) -> Dict[str, Any]:
        # Stage 3: T_init + (T_local_msg)³ + T_chart_encode
        out3 = self.local_block(batch, shared_state=shared_state)
        p_mid       = out3["p_next_local"]        # (A, 64+8×2)  chart state
        loc_fin     = out3["local_state_final"]   # (N_M, ...)   node state for overlap ctx
        # out3["intra_static_bundle"] is identical to the pre-computed intra_static

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
    Stack of K B-blocks with independent weights.

    Each block receives the per-node ``shared_state`` produced by the
    previous block's T_init, allowing cross-iteration communication at
    node level.  Static graph bundles are computed once per forward pass.

    Returns the chart-level DualStreamState after the last B-block.
    """

    def __init__(self, cfg: BBlockConfig) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([
            BBlock(cfg.local_cfg, cfg.intra_cfg, cfg.inter_cfg)
            for _ in range(cfg.num_bblocks)
        ])

    def forward(self, batch: Stage3TensorBatch) -> DualStreamState:
        intra_static = batch.intra_static_bundle()   # static, compute once
        inter_static = batch.inter_static_bundle()

        shared_state: Optional[DualStreamState] = None
        for block in self.blocks:
            out          = block(batch, shared_state, intra_static, inter_static)
            p_new        = out["p_new"]
            shared_state = out["node_state_shared"]  # feeds next iteration's T_init

        return p_new  # (A, 64) + (A, 8, 2)


__all__ = [
    "BBlockConfig",
    "BBlock",
    "ED2EBBlockStack",
]
