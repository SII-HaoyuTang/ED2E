"""
Smoke test for Stage 4 (T_intra) IntraLevelBlock forward pass.

Usage:
    python scripts/smoke_stage4_intra_forward.py \
        --cache-dir data/ed_energy_5w/cache_stage3 \
        --n-mols 4 --device cpu

Validates:
  1. New data fields present and correct shape
  2. ov2ce values in [0, E_chart)
  3. chart_to_ref values in [0, A)
  4. IntraLevelBlock.forward() runs without error
  5. Output shape and no NaN/Inf
"""
from __future__ import annotations

import argparse
import os
import sys

import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 4 intra-block smoke test")
    p.add_argument("--cache-dir", required=True, help="Path to stage3 cache directory")
    p.add_argument("--n-mols", type=int, default=4, help="Number of molecules to load")
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    # --- Load samples ---
    from ed2e.data.stage3_local import (
        Stage3Sample,
        collate_stage3_samples,
        load_stage3_entry,
        list_stage3_bundle_ids,
    )

    cache_dir = args.cache_dir
    # Support both merged zip bundle and individual pkl files
    bundle_files = sorted(f for f in os.listdir(cache_dir) if f.endswith(".zip") and "stage3" in f)
    pkl_files = sorted(f for f in os.listdir(cache_dir) if f.endswith(".pkl") and "stage3" in f)

    if bundle_files:
        bundle_path = os.path.join(cache_dir, bundle_files[0])
        mol_ids = list_stage3_bundle_ids(bundle_path)[: args.n_mols]
        if not mol_ids:
            print(f"ERROR: empty stage3 bundle {bundle_path}", file=sys.stderr)
            sys.exit(1)
        samples: list[Stage3Sample] = [load_stage3_entry(bundle_path, mid) for mid in mol_ids]
    elif pkl_files:
        if not pkl_files:
            print(f"ERROR: no stage3 files found in {cache_dir}", file=sys.stderr)
            sys.exit(1)
        samples = [load_stage3_entry(os.path.join(cache_dir, f), "") for f in pkl_files[: args.n_mols]]
    else:
        print(f"ERROR: no stage3 .zip or .pkl files found in {cache_dir}", file=sys.stderr)
        sys.exit(1)

    batch = collate_stage3_samples(samples, device=device)

    A   = batch.intra_geom_static.shape[0]
    E_c = batch.chart_graph_edge_index.shape[1]
    E_ov = batch.overlap_edge_index.shape[1]

    print(f"\nLoaded {len(samples)} molecules")
    print(f"  A={A}  E_chart={E_c}  E_overlap={E_ov}")

    # --- Check 1: intra_geom_static shape ---
    assert batch.intra_geom_static.shape == (A, 7), (
        f"Expected intra_geom_static (A,7) got {tuple(batch.intra_geom_static.shape)}"
    )
    print(f"  intra_geom_static  {tuple(batch.intra_geom_static.shape)}  ✓")

    # --- Check 2: ov2ce shape and range ---
    ov2ce = batch.overlap_edge_to_chart_edge_index
    assert ov2ce.shape == (E_ov,), (
        f"Expected ov2ce ({E_ov},) got {tuple(ov2ce.shape)}"
    )
    if E_ov > 0 and E_c > 0:
        assert int(ov2ce.min()) >= 0, "ov2ce has negative values"
        assert int(ov2ce.max()) < E_c, (
            f"ov2ce max={int(ov2ce.max())} >= E_chart={E_c}"
        )
    print(f"  ov2ce              ({E_ov},)  range [0, {E_c})  ✓")

    # --- Check 3: chart_to_ref range ---
    chart_to_ref = batch.chart_frame_metadata["chart_to_ref"]
    assert chart_to_ref.shape == (A,), (
        f"Expected chart_to_ref ({A},) got {tuple(chart_to_ref.shape)}"
    )
    assert int(chart_to_ref.min()) >= 0, "chart_to_ref has negative values"
    assert int(chart_to_ref.max()) < A, (
        f"chart_to_ref max={int(chart_to_ref.max())} >= A={A}"
    )
    print(f"  chart_to_ref       ({A},)  range [0, {A})  ✓")

    # --- Check 4: IntraLevelBlock forward ---
    from ed2e.model.stage3_local import DualStreamState
    from ed2e.model.stage4_intra import IntraLevelBlock

    block = IntraLevelBlock().to(device)
    block.eval()

    N_M = batch.chart_membership.shape[0]
    p_mid = DualStreamState(
        scalar=torch.randn(A, 64, device=device),
        vector=torch.randn(A, 8, 2, device=device),
    )
    local_state_final = DualStreamState(
        scalar=torch.randn(N_M, 64, device=device),
        vector=torch.randn(N_M, 8, 2, device=device),
    )

    intra_static = batch.intra_static_bundle()

    with torch.no_grad():
        out = block(p_mid, local_state_final, batch, intra_static)

    # --- Check 5: output shape and numerics ---
    assert out.scalar.shape == (A, 64), (
        f"Expected output scalar (A,64) got {tuple(out.scalar.shape)}"
    )
    assert out.vector.shape == (A, 8, 2), (
        f"Expected output vector (A,8,2) got {tuple(out.vector.shape)}"
    )
    assert not torch.isnan(out.scalar).any(), "NaN in output scalar"
    assert not torch.isnan(out.vector).any(), "NaN in output vector"
    assert not torch.isinf(out.scalar).any(), "Inf in output scalar"
    assert not torch.isinf(out.vector).any(), "Inf in output vector"

    print(f"\nStage4 forward OK")
    print(f"  scalar {tuple(out.scalar.shape)}  vector {tuple(out.vector.shape)}  no NaN ✓")


if __name__ == "__main__":
    main()
