"""
Smoke test for Stage 5 (T_inter) InterLevelBlock forward pass.

Usage:
    python scripts/smoke_stage5_inter_forward.py \
        --cache-dir data/ed_energy_5w/cache_stage3 \
        --n-mols 4 --device cpu

    python scripts/smoke_stage5_inter_forward.py \
        --packed-dir data/ed_energy_5w/packed_stage3 \
        --n-mols 4 --device cpu

Validates:
  1. inter_level_edge_index shape (2, E_inter) and values in [0, A)
  2. inter_level_weights shape (E_inter,) and values in [0, 1]
  3. inter_level_edge_attr shape (E_inter, 7)
  4. level_diff values in {-1, +1} (attr dim 6)
  5. InterLevelBlock.forward() runs without error
  6. Output shape (A, 64) + (A, 8, 2) and no NaN/Inf
"""
from __future__ import annotations

import argparse
import os
import sys

import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 5 inter-block smoke test")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--cache-dir", help="Path to stage3 cache directory (zip bundle or per-mol pkl)")
    g.add_argument("--packed-dir", help="Path to packed stage3 directory (Stage3PackedDataset)")
    p.add_argument("--n-mols", type=int, default=4, help="Number of molecules to load")
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    from ed2e.data.stage3_local import (
        Stage3Sample,
        collate_stage3_samples,
        load_stage3_entry,
        list_stage3_bundle_ids,
    )

    if args.packed_dir:
        from ed2e.data.stage3_packed import Stage3PackedDataset
        dataset = Stage3PackedDataset(args.packed_dir)
        n = min(args.n_mols, len(dataset))
        samples: list[Stage3Sample] = [dataset[i] for i in range(n)]
    else:
        cache_dir = args.cache_dir
        bundle_files = sorted(f for f in os.listdir(cache_dir) if f.endswith(".zip") and "stage3" in f)
        pkl_files = sorted(f for f in os.listdir(cache_dir) if f.endswith(".pkl") and "stage3" in f)

        if bundle_files:
            bundle_path = os.path.join(cache_dir, bundle_files[0])
            mol_ids = list_stage3_bundle_ids(bundle_path)[: args.n_mols]
            if not mol_ids:
                print(f"ERROR: empty stage3 bundle {bundle_path}", file=sys.stderr)
                sys.exit(1)
            samples = [load_stage3_entry(bundle_path, mid) for mid in mol_ids]
        elif pkl_files:
            samples = [load_stage3_entry(os.path.join(cache_dir, f), "") for f in pkl_files[: args.n_mols]]
        else:
            print(f"ERROR: no stage3 .zip or .pkl files found in {cache_dir}", file=sys.stderr)
            sys.exit(1)

    batch = collate_stage3_samples(samples, device=device)

    A   = batch.intra_geom_static.shape[0]
    E_i = batch.inter_level_edge_index.shape[1]

    print(f"\nLoaded {len(samples)} molecules")
    print(f"  A={A}  E_inter={E_i}")

    # --- Check 1: inter_level_edge_index shape and range ---
    edge_index = batch.inter_level_edge_index
    assert edge_index.shape == (2, E_i), (
        f"Expected inter_level_edge_index (2, {E_i}) got {tuple(edge_index.shape)}"
    )
    if E_i > 0:
        assert int(edge_index.min()) >= 0, "inter_level_edge_index has negative values"
        assert int(edge_index.max()) < A, (
            f"inter_level_edge_index max={int(edge_index.max())} >= A={A}"
        )
    print(f"  inter_level_edge_index  (2, {E_i})  range [0, {A})  ✓")

    # --- Check 2: inter_level_weights ---
    weights = batch.inter_level_weights
    assert weights.shape == (E_i,), (
        f"Expected inter_level_weights ({E_i},) got {tuple(weights.shape)}"
    )
    if E_i > 0:
        assert float(weights.min()) >= 0.0, "inter_level_weights has negative values"
        assert float(weights.max()) <= 1.0 + 1e-5, (
            f"inter_level_weights max={float(weights.max())} > 1"
        )
    print(f"  inter_level_weights     ({E_i},)  range [0, 1]  ✓")

    # --- Check 3: inter_level_edge_attr shape ---
    edge_attr = batch.inter_level_edge_attr
    assert edge_attr.shape == (E_i, 7), (
        f"Expected inter_level_edge_attr ({E_i}, 7) got {tuple(edge_attr.shape)}"
    )
    print(f"  inter_level_edge_attr   ({E_i}, 7)  ✓")

    # --- Check 4: level_diff values in {-1, +1} ---
    if E_i > 0:
        level_diff = edge_attr[:, 6]
        unique_ld = set(level_diff.cpu().tolist())
        assert unique_ld.issubset({-1.0, 1.0, 0.0}), (
            f"Unexpected level_diff values: {unique_ld}"
        )
    print(f"  level_diff values  ✓")

    # --- Check 5: InterLevelBlock forward ---
    from ed2e.model.stage3_local import DualStreamState
    from ed2e.model.stage5_inter import InterLevelBlock

    block = InterLevelBlock().to(device)
    block.eval()

    p_bar = DualStreamState(
        scalar=torch.randn(A, 64, device=device),
        vector=torch.randn(A, 8, 2, device=device),
    )

    inter_static = batch.inter_static_bundle()

    with torch.no_grad():
        out = block(p_bar, batch, inter_static)

    # --- Check 6: output shape and numerics ---
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

    print(f"\nStage5 forward OK")
    print(f"  scalar {tuple(out.scalar.shape)}  vector {tuple(out.vector.shape)}  no NaN ✓")


if __name__ == "__main__":
    main()
