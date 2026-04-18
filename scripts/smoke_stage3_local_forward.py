#!/usr/bin/env python3
"""
Stage 3 local block smoke forward.
"""
from __future__ import annotations

import argparse
import copy
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from ed2e.data.stage3_local import (  # noqa: E402
    build_stage3_sample,
    collate_stage3_samples,
    load_stage3_entry,
    load_stage3_sample,
    stage3_cache_path,
    summarize_stage3_sample,
    validate_stage3_sample,
)
from ed2e.data.stage3_packed import load_stage3_packed_sample  # noqa: E402
from ed2e.model.stage3_local import FCLCLocalBlock, PseudoStage4Consumer, Stage3LocalConfig  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Smoke-check Stage 3 local aggregation and Stage 4-ready interface.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mol-id", required=True)
    p.add_argument("--stage3-source", default=None, help="Stage 3 cache dir, single-file pkl, or merged zip bundle.")
    p.add_argument("--stage3-packed-dir", default=None, help="Packed Stage 3 mmap directory.")
    p.add_argument("--manifold-cache-dir", default=None, help="Build on the fly from Stage 1 + Stage 2.")
    p.add_argument("--manifold-pkl", default=None, help="Optional merged Stage 1 manifold pkl used only for smoke/debug.")
    p.add_argument("--fclc-source", default=None, help="Stage 2 cache dir or bundle path.")
    p.add_argument("--n-levels", type=int, default=4)
    p.add_argument("--smooth-sigma", type=float, default=0.5)
    p.add_argument("--tau-r", type=float, default=1.0)
    p.add_argument("--tau-2", type=float, default=1.5)
    p.add_argument("--local-knn-k", type=int, default=12)
    p.add_argument("--chart-knn-k", type=int, default=8)
    p.add_argument("--num-anchors", type=int, default=8)
    p.add_argument("--steps", type=int, default=2)
    p.add_argument("--device", default="cpu")
    p.add_argument("--scalar-dim", type=int, default=64)
    p.add_argument("--vector-dim", type=int, default=8)
    p.add_argument("--token-dim", type=int, default=96)
    p.add_argument("--verify-single-and-multi", action="store_true")
    return p.parse_args()


def _load_sample(args: argparse.Namespace):
    if args.stage3_packed_dir is not None:
        return load_stage3_packed_sample(args.stage3_packed_dir, args.mol_id)

    if args.stage3_source is not None:
        if os.path.isdir(args.stage3_source):
            path = stage3_cache_path(
                args.stage3_source,
                args.mol_id,
                args.local_knn_k,
                args.chart_knn_k,
                args.num_anchors,
            )
            return load_stage3_sample(path)
        return load_stage3_entry(args.stage3_source, args.mol_id)

    if args.fclc_source is None:
        raise ValueError("Building on the fly requires --fclc-source.")

    from ed2e.data.fclc import fclc_cache_path, load_fclc_entry, load_fclc_levels
    from ed2e.data.manifold import _patch_legacy_pickle_modules, load_manifold_levels, manifold_cache_path

    if args.manifold_pkl is not None:
        import pickle

        _patch_legacy_pickle_modules()
        with open(args.manifold_pkl, "rb") as f:
            manifold_levels = pickle.load(f)[args.mol_id]
    elif args.manifold_cache_dir is not None:
        manifold_levels = load_manifold_levels(
            manifold_cache_path(args.manifold_cache_dir, args.mol_id, args.n_levels, args.smooth_sigma)
        )
    else:
        raise ValueError("Building on the fly requires either --manifold-cache-dir or --manifold-pkl.")
    if os.path.isdir(args.fclc_source):
        fclc_levels = load_fclc_levels(fclc_cache_path(args.fclc_source, args.mol_id, args.tau_r, args.tau_2))
    else:
        fclc_levels = load_fclc_entry(args.fclc_source, args.mol_id)
    return build_stage3_sample(
        args.mol_id,
        manifold_levels,
        fclc_levels,
        local_knn_k=args.local_knn_k,
        chart_knn_k=args.chart_knn_k,
        num_anchors=args.num_anchors,
        inner_threads=1,
    )


def _run_once(sample, cfg: Stage3LocalConfig, device: str):
    batch = collate_stage3_samples([sample], device=device)
    local_block = FCLCLocalBlock(cfg).to(device)
    pseudo_stage4 = PseudoStage4Consumer(cfg).to(device)

    with torch.no_grad():
        local_out = local_block(batch, return_debug=True)
        intra_out = pseudo_stage4(
            p_next_local=local_out["p_next_local"],
            local_state_final=local_out["local_state_final"],
            node_state_shared_next=local_out["node_state_shared_next"],
            batch=batch,
            intra_static_bundle=local_out["intra_static_bundle"],
        )
    return batch, local_out, intra_out


def main() -> None:
    args = _parse_args()
    sample = _load_sample(args)
    validation = validate_stage3_sample(sample)
    summary = summarize_stage3_sample(sample)

    cfg = Stage3LocalConfig(
        scalar_dim=args.scalar_dim,
        vector_dim=args.vector_dim,
        token_dim=args.token_dim,
        num_local_steps=args.steps,
    )

    batch, local_out, intra_out = _run_once(sample, cfg, args.device)
    bundle = local_out["intra_static_bundle"]
    debug_steps = local_out["debug"]["steps"]

    geom_same = torch.allclose(local_out["debug"]["geom_static"], batch.chart_es_geom_static)
    scalar_changed = not torch.allclose(debug_steps[0]["scalar_es"], debug_steps[-1]["scalar_es"])
    vector_changed = not torch.allclose(debug_steps[0]["vector_es"], debug_steps[-1]["vector_es"])
    zero_repack = (
        bundle["chart_graph_edge_index"].data_ptr() == batch.chart_graph_edge_index.data_ptr()
        and bundle["overlap_edge_index"].data_ptr() == batch.overlap_edge_index.data_ptr()
        and bundle["overlap_shared_membership_index"].data_ptr() == batch.overlap_shared_membership_index.data_ptr()
    )

    print(f"mol_id={args.mol_id}")
    print(f"summary={summary}")
    print(f"validation={validation}")
    print(
        "local_output_shapes="
        f" shared_scalar={tuple(local_out['node_state_shared_next'].scalar.shape)}"
        f" shared_vector={tuple(local_out['node_state_shared_next'].vector.shape)}"
        f" local_scalar={tuple(local_out['local_state_final'].scalar.shape)}"
        f" local_vector={tuple(local_out['local_state_final'].vector.shape)}"
        f" chart_scalar={tuple(local_out['p_next_local'].scalar.shape)}"
        f" chart_vector={tuple(local_out['p_next_local'].vector.shape)}"
    )
    print(
        "checks="
        f" geom_static_stable={geom_same}"
        f" scalar_es_refreshed={scalar_changed}"
        f" vector_es_refreshed={vector_changed}"
        f" zero_repack={zero_repack}"
        f" pseudo_stage4_identity={intra_out['bundle_identity_ok']}"
    )
    print(
        "pseudo_stage4_shape="
        f" chart_scalar={tuple(intra_out['chart_state_intra_ready'].scalar.shape)}"
        f" chart_vector={tuple(intra_out['chart_state_intra_ready'].vector.shape)}"
    )

    if args.verify_single_and_multi:
        cfg_single = copy.deepcopy(cfg)
        cfg_single.num_local_steps = 1
        _, local_one, _ = _run_once(sample, cfg_single, args.device)
        print(
            "single_vs_multi="
            f" step1_chart_scalar={tuple(local_one['p_next_local'].scalar.shape)}"
            f" stepN_chart_scalar={tuple(local_out['p_next_local'].scalar.shape)}"
            f" step1_local_edges={int(batch.local_knn_edge_index.shape[1])}"
        )


if __name__ == "__main__":
    main()
