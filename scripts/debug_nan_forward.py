#!/usr/bin/env python3
"""
Single-molecule NaN diagnostic forward pass.

Loads one molecule from the packed dataset and runs the full model forward
pass with NaN/Inf checks inserted after each major module. Exits immediately
on the first NaN/Inf found, reporting which tensor and which flat index.

Usage:
    python scripts/debug_nan_forward.py \
        --packed-dir data/ed_energy_5w/packed_stage3 \
        --mol-idx 0 --device cpu --num-bblocks 1
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))


_EXTREME = 1e6   # threshold for "extreme but finite" values


def _check(name: str, t: object) -> None:
    """Print NaN/Inf/extreme status for a tensor; exit(1) if any found."""
    if not isinstance(t, torch.Tensor):
        return
    if not t.is_floating_point():
        return
    n_nan = int(t.isnan().sum())
    n_inf = int(t.isinf().sum())
    abs_max = float(t[~t.isnan() & ~t.isinf()].abs().max()) if t.numel() > 0 else 0.0
    extreme = abs_max > _EXTREME

    if n_nan == 0 and n_inf == 0 and not extreme:
        status = "ok"
    elif n_nan or n_inf:
        status = f"NaN={n_nan} Inf={n_inf}"
    else:
        status = f"extreme={abs_max:.2e}"

    print(f"  [{status:>16}]  {name}  {tuple(t.shape)}")
    if n_nan or n_inf:
        if n_nan:
            flat_nan = int(t.isnan().float().argmax())
            idx = list(np.unravel_index(flat_nan, t.shape))
            print(f"    !! First NaN at index {idx}")
        if n_inf:
            flat_inf = int(t.isinf().float().argmax())
            idx = list(np.unravel_index(flat_inf, t.shape))
            print(f"    !! First Inf at index {idx}")
        sys.exit(1)
    if extreme:
        print(f"    !! Extreme value {abs_max:.2e} (may cause overflow in LayerNorm/exp)")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Single-molecule NaN diagnostic forward pass.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--packed-dir", required=True,
                   help="Path to packed_stage3 directory.")
    p.add_argument("--mol-idx", type=int, default=0,
                   help="Index of the first molecule to load (0-based). Ignored if --mol-ids given.")
    p.add_argument("--mol-ids", nargs="+", default=None,
                   help="One or more mol_id strings to load (e.g. 1370746 2352995).")
    p.add_argument("--n-mols", type=int, default=1,
                   help="Number of molecules to collate into one batch (used with --mol-idx).")
    p.add_argument("--device", default="cpu")
    p.add_argument("--num-bblocks", type=int, default=1,
                   help="Number of BBlocks (use 1 for fast debugging).")
    args = p.parse_args()
    device = torch.device(args.device)

    from ed2e.data.stage3_local import collate_stage3_samples
    from ed2e.data.stage3_packed import Stage3PackedDataset
    from ed2e.model.bblock import BBlockConfig
    from ed2e.model.ed2e import ED2EConfig, ED2EModel

    # ── Load molecules ────────────────────────────────────────────────────────
    dataset = Stage3PackedDataset(args.packed_dir)

    if args.mol_ids is not None:
        id_to_idx = {mid: i for i, mid in enumerate(dataset.mol_ids)}
        missing = [m for m in args.mol_ids if m not in id_to_idx]
        if missing:
            print(f"ERROR: mol_ids not found in dataset: {missing}", file=sys.stderr)
            sys.exit(1)
        samples = [dataset[id_to_idx[m]] for m in args.mol_ids]
    else:
        if args.mol_idx >= len(dataset):
            print(f"ERROR: mol_idx={args.mol_idx} out of range (dataset size={len(dataset)})",
                  file=sys.stderr)
            sys.exit(1)
        end_idx = min(args.mol_idx + args.n_mols, len(dataset))
        samples = [dataset[i] for i in range(args.mol_idx, end_idx)]

    batch = collate_stage3_samples(samples, device=device)

    N = batch.node_xyz.shape[0]
    A = batch.chart_es_geom_static.shape[0]
    B = len(samples)
    print(f"Molecules: {batch.mol_ids}  (B={B})")
    print(f"  N={N} nodes,  A={A} charts,  "
          f"M={batch.chart_membership.shape[0]} memberships,  "
          f"E_loc={batch.local_knn_edge_index.shape[1]} local edges,  "
          f"E_inter={batch.inter_level_edge_index.shape[1]} inter edges")

    # ── Check raw batch tensors ───────────────────────────────────────────────
    print("\n=== Batch input tensors ===")
    for fname in [
        "node_scalar_raw", "node_vector_raw", "node_tangent_basis",
        "membership_weight", "membership_sr", "local_edge_attr",
        "chart_es_geom_static", "intra_geom_static",
        "inter_level_edge_attr", "inter_level_weights",
    ]:
        _check(fname, getattr(batch, fname, None))
    _check("chart_frame", batch.chart_frame_metadata.get("chart_frame"))
    _check("chart_center_normal", batch.chart_frame_metadata.get("chart_center_normal"))

    # ── Build model ───────────────────────────────────────────────────────────
    cfg = ED2EConfig(bblock=BBlockConfig(num_bblocks=args.num_bblocks))
    model = ED2EModel(cfg).to(device)
    model.eval()
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    stack = model.bblock_stack

    with torch.no_grad():
        # ── Shared state init ─────────────────────────────────────────────────
        print("\n=== _initialize_shared_state ===")
        shared_state = stack._initialize_shared_state(batch)
        _check("shared_state.scalar", shared_state.scalar)
        _check("shared_state.vector", shared_state.vector)

        intra_static = batch.intra_static_bundle()
        inter_static = batch.inter_static_bundle()

        p_prev = None
        for k, block in enumerate(stack.blocks):
            print(f"\n=== BBlock {k}: FCLCLocalBlock ===")
            out3 = block.local_block(batch, shared_state=shared_state, p_prev=p_prev)
            _check(f"B{k}.local.node_state_shared_next.scalar", out3["node_state_shared_next"].scalar)
            _check(f"B{k}.local.node_state_shared_next.vector", out3["node_state_shared_next"].vector)
            _check(f"B{k}.local.local_state_final.scalar",      out3["local_state_final"].scalar)
            _check(f"B{k}.local.local_state_final.vector",      out3["local_state_final"].vector)
            _check(f"B{k}.local.p_next_local.scalar",           out3["p_next_local"].scalar)
            _check(f"B{k}.local.p_next_local.vector",           out3["p_next_local"].vector)

            print(f"\n=== BBlock {k}: IntraLevelBlock ===")
            p_bar = block.intra_block(
                out3["p_next_local"], out3["local_state_final"], batch, intra_static
            )
            _check(f"B{k}.intra.p_bar.scalar", p_bar.scalar)
            _check(f"B{k}.intra.p_bar.vector", p_bar.vector)

            print(f"\n=== BBlock {k}: InterLevelBlock ===")
            p_new = block.inter_block(p_bar, batch, inter_static)
            _check(f"B{k}.inter.p_new.scalar", p_new.scalar)
            _check(f"B{k}.inter.p_new.vector", p_new.vector)

            shared_state = out3["node_state_shared_next"]
            p_prev = p_new

        # ── Readout ───────────────────────────────────────────────────────────
        print("\n=== MultiHeadChartReadout ===")
        readout_out = model.readout(p_prev, batch, return_attn=False)
        g = readout_out["global_features"]
        _check("global_features", g)

        # ── Energy heads ──────────────────────────────────────────────────────
        print("\n=== EnergyHeads ===")
        pred = model.energy_heads(g)
        _check("energy_pred", pred)

    print(f"\n✓  All checks passed.")
    print(f"   energy_pred shape : {tuple(pred.shape)}")
    print(f"   mean |pred|       : {pred.abs().mean():.4f}")
    for i, mol_id in enumerate(batch.mol_ids):
        print(f"   mol {mol_id}: {[f'{v:.4f}' for v in pred[i].tolist()]}")


if __name__ == "__main__":
    main()
