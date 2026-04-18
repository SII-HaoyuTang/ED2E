"""ED2E model package."""

from .stage3_local import (
    DualStreamState,
    Stage3LocalConfig,
    FCLCLocalBlock,
    PseudoStage4Consumer,
)
from .stage4_intra import IntraLevelBlock, Stage4IntraConfig
from .stage5_inter import InterLevelBlock, Stage5InterConfig
from .bblock import BBlock, BBlockConfig, ED2EBBlockStack
from .readout import EnergyHeads, MultiHeadChartReadout, ReadoutConfig
from .ed2e import ED2EConfig, ED2EModel, TARGET_NAMES

__all__ = [
    # Stage 3
    "DualStreamState",
    "Stage3LocalConfig",
    "FCLCLocalBlock",
    "PseudoStage4Consumer",
    # Stage 4
    "IntraLevelBlock",
    "Stage4IntraConfig",
    # Stage 5
    "InterLevelBlock",
    "Stage5InterConfig",
    # B-block
    "BBlock",
    "BBlockConfig",
    "ED2EBBlockStack",
    # Readout
    "EnergyHeads",
    "MultiHeadChartReadout",
    "ReadoutConfig",
    # Full model
    "ED2EConfig",
    "ED2EModel",
    "TARGET_NAMES",
]
