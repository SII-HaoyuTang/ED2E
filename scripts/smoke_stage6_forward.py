"""
Smoke test for the full Stage 6 forward pass (BBlockStack + ReadOut + EnergyHeads).

Usage:
    python scripts/smoke_stage6_forward.py \
        --packed-dir data/ed_energy_5w/packed_stage3 --n-mols 4 --device cpu

Validates:
  1. ED2EModel instantiates without error
  2. forward() runs without error
  3. energy_pred shape (B, 6) and no NaN/Inf
  4. Attention weights (T, H, A), sum to 1 per molecule per target per head
  5. chart_batch / chart_level_id / chart_center shapes consistent with A
"""
from __future__ import annotations

import argparse
import os
import sys

import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 6 smoke test")
    p.add_argument("--packed-dir", required=True,
                   help="Path to packed stage3 directory (Stage3PackedDataset)")
    p.add_argument("--n-mols", type=int, default=4)
    p.add_argument("--device", default="cpu")
    p.add_argument("--num-bblocks", type=int, default=1,
                   help="Number of B-blocks (use 1 for fast smoke test)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    _HERE = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.dirname(_HERE))

    from ed2e.data.stage3_local import collate_stage3_samples
    from ed2e.data.stage3_packed import Stage3PackedDataset
    from ed2e.model.bblock import BBlockConfig
    from ed2e.model.ed2e import ED2EConfig, ED2EModel

    # ── Load samples ──────────────────────────────────────────────────────────
    print(f"\nLoading packed dataset from {args.packed_dir} …")
    dataset = Stage3PackedDataset(args.packed_dir)
    n = min(args.n_mols, len(dataset))
    if n == 0:
        print("ERROR: empty dataset", file=sys.stderr)
        sys.exit(1)
    samples = [dataset[i] for i in range(n)]
    batch = collate_stage3_samples(samples, device=device)

    A = batch.intra_geom_static.shape[0]
    B = int(batch.chart_batch.max().item()) + 1
    print(f"  B={B} molecules,  A={A} charts total")

    # ── Build model ───────────────────────────────────────────────────────────
    cfg = ED2EConfig(
        bblock=BBlockConfig(num_bblocks=args.num_bblocks),
    )
    model = ED2EModel(cfg).to(device)
    model.eval()

    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {num_params:,}")

    # ── Forward pass ─────────────────────────────────────────────────────────
    print(f"\nRunning forward pass (num_bblocks={args.num_bblocks}) …")
    with torch.no_grad():
        out = model(batch, return_attn=True)

    # ── Check 1: energy_pred shape ────────────────────────────────────────────
    pred = out["energy_pred"]
    assert pred.shape == (B, cfg.num_targets), (
        f"Expected energy_pred ({B}, {cfg.num_targets}), got {tuple(pred.shape)}"
    )
    print(f"  energy_pred  {tuple(pred.shape)}  ✓")

    # ── Check 2: no NaN/Inf ───────────────────────────────────────────────────
    assert not torch.isnan(pred).any(), "NaN in energy_pred"
    assert not torch.isinf(pred).any(), "Inf in energy_pred"
    print(f"  no NaN/Inf  ✓")

    # ── Check 3: attn_weights ─────────────────────────────────────────────────
    attn = out["attn_weights"]           # (T, H, A)
    T = cfg.num_targets
    H = cfg.readout.num_heads
    assert attn.shape == (T, H, A), (
        f"Expected attn_weights ({T}, {H}, {A}), got {tuple(attn.shape)}"
    )
    print(f"  attn_weights {tuple(attn.shape)}  ✓")

    # Per-molecule attention sums to 1 per target per head
    chart_batch = out["chart_batch"]  # (A,)
    for mol_idx in range(B):
        mask = chart_batch == mol_idx
        for t in range(T):
            for h in range(H):
                attn_sum = float(attn[t, h, mask].sum().item())
                assert abs(attn_sum - 1.0) < 1e-4, (
                    f"attn sum for mol={mol_idx} t={t} h={h}: {attn_sum:.6f} ≠ 1"
                )
    print(f"  attn sums to 1 per mol per target per head  ✓")

    # ── Check 4: metadata shapes ──────────────────────────────────────────────
    assert out["chart_level_id"].shape == (A,), (
        f"chart_level_id shape mismatch: {tuple(out['chart_level_id'].shape)}"
    )
    assert out["chart_center"].shape == (A, 3), (
        f"chart_center shape mismatch: {tuple(out['chart_center'].shape)}"
    )
    print(f"  chart_batch/level_id/center shapes  ✓")

    print(f"\nStage 6 forward OK")
    print(f"  energy_pred: {tuple(pred.shape)}")
    print(f"  attn_weights: {tuple(attn.shape)}")
    print(f"  Mean |pred|: {pred.abs().mean():.4f}")


if __name__ == "__main__":
    main()
