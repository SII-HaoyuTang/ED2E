"""
ED2E full model: BBlockStack → MultiHeadChartReadout → EnergyHeads.

Predicts 6 molecular energy quantities from multi-level FCLC chart
representations of the electron density manifold.

Usage
-----
::

    from ed2e.model.ed2e import ED2EModel, ED2EConfig

    model = ED2EModel()                  # default config
    batch = collate_stage3_samples(...)  # Stage3TensorBatch on target device

    out = model(batch)
    # out["energy_pred"] : (B, 6)  z-score normalised energies

    out = model(batch, return_attn=True)
    # out["attn_weights"]      : (T, H, A)
    # out["chart_batch"]       : (A,)
    # out["chart_level_id"]    : (A,)
    # out["chart_center"]      : (A, 3)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from ed2e.data.stage3_local import Stage3TensorBatch
from ed2e.model.bblock import BBlockConfig, ED2EBBlockStack
from ed2e.model.readout import EnergyHeads, MultiHeadChartReadout, ReadoutConfig

TARGET_NAMES: Tuple[str, ...] = (
    "DF-RKS_Final",
    "Nuclear_Repulsion",
    "One_Electron",
    "Two_Electron",
    "DFT_XC",
    "Total",
)


@dataclass
class ED2EConfig:
    """Top-level configuration.  ``num_targets`` is the single source of truth."""
    bblock:             BBlockConfig  = field(default_factory=BBlockConfig)
    readout:            ReadoutConfig = field(default_factory=ReadoutConfig)
    num_targets:        int           = 6
    energy_head_hidden: int           = 128
    energy_dropout:     float         = 0.1
    target_names:       Tuple[str, ...] = TARGET_NAMES


class ED2EModel(nn.Module):
    """
    Full ED2E energy prediction model.

    forward()
    ---------
    batch       : Stage3TensorBatch
    return_attn : bool (default False)

    Returns dict:
      "energy_pred"      : (B, T)     — z-score normalised predictions
      when return_attn=True:
      "attn_weights"     : (T, H, A)  — per-chart attention weights
      "chart_batch"      : (A,)       — molecule index per chart
      "chart_level_id"   : (A,)       — manifold level per chart
      "chart_center"     : (A, 3)     — 3D chart centre coordinates
    """

    def __init__(self, cfg: ED2EConfig | None = None) -> None:
        super().__init__()
        cfg = cfg or ED2EConfig()
        self.cfg = cfg

        self.bblock_stack = ED2EBBlockStack(cfg.bblock)
        self.readout = MultiHeadChartReadout(cfg.readout, num_targets=cfg.num_targets)
        self.energy_heads = EnergyHeads(
            in_dim=cfg.readout.token_dim,
            hidden_dim=cfg.energy_head_hidden,
            num_targets=cfg.num_targets,
            dropout=cfg.energy_dropout,
        )

    def forward(
        self,
        batch: Stage3TensorBatch,
        return_attn: bool = False,
    ) -> Dict[str, Any]:
        # B-block stack: (A, 64) + (A, 8, 2)
        p_new = self.bblock_stack(batch)

        # Multi-head readout: (B, T, 96)
        ro = self.readout(p_new, batch, return_attn=return_attn)

        # Independent output heads: (B, T)
        pred = self.energy_heads(ro["global_features"])

        out: Dict[str, Any] = {"energy_pred": pred}
        if return_attn:
            out["attn_weights"]   = ro["attn_weights"]                           # (T, H, A)
            out["chart_batch"]    = batch.chart_batch                            # (A,)
            out["chart_level_id"] = batch.chart_frame_metadata["chart_level_id"] # (A,)
            out["chart_center"]   = batch.chart_frame_metadata["chart_center"]   # (A, 3)
        return out


__all__ = [
    "TARGET_NAMES",
    "ED2EConfig",
    "ED2EModel",
]
