#!/usr/bin/env python3
"""
Unified Stage 2→3 preprocessing script.

Builds Stage 3 packed format directly from Stage 1 manifold cache,
running FCLC (Stage 2) and Stage 3/4/5 inter-edge computation in-memory
per molecule — no Stage 2 PKL intermediate required.

Output is a sharded Stage3PackedDataset written under --packed-dir.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import pickle
import sys
import time
from collections import defaultdict
from typing import List, Optional, Tuple

import numpy as np
from tqdm import tqdm

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from ed2e.data.stage3_packed import Stage3ShardedWriter  # noqa: E402

# ──────────────────────────────────────────────────────
# Global worker state (set once in main, inherited via fork)
# ──────────────────────────────────────────────────────
_g_manifold_cache_dir: str = ""
_g_manifold_merged: Optional[dict] = None
_g_n_levels: int = 4
_g_smooth_sigma: float = 0.5
_g_tau_r: float = 1.0
_g_tau_2: float = 1.5
_g_local_knn_k: int = 12
_g_chart_knn_k: int = 8
_g_num_anchors: int = 8
_g_threads_per_proc: int = 4
_g_mem_thresh: Optional[int] = None


def _worker(mol_id: str) -> Tuple[str, str, Optional[float], Optional[object]]:
    """Per-molecule worker: manifold → FCLC → Stage3Sample."""
    from ed2e.data.fclc import build_fclc_levels
    from ed2e.data.manifold import load_manifold_levels, manifold_cache_path
    from ed2e.data.stage3_local import build_stage3_sample

    t0 = time.perf_counter()
    try:
        if _g_manifold_merged is not None:
            manifold_levels = _g_manifold_merged[mol_id]
        else:
            path = manifold_cache_path(
                _g_manifold_cache_dir, mol_id, _g_n_levels, _g_smooth_sigma
            )
            if not os.path.exists(path):
                return mol_id, "error:missing_manifold", None, None
            manifold_levels = load_manifold_levels(path)

        fclc_levels = build_fclc_levels(manifold_levels, tau_r=_g_tau_r, tau_2=_g_tau_2,
                                         mem_thresh=_g_mem_thresh)
        sample = build_stage3_sample(
            mol_id,
            manifold_levels,
            fclc_levels,
            local_knn_k=_g_local_knn_k,
            chart_knn_k=_g_chart_knn_k,
            num_anchors=_g_num_anchors,
            inner_threads=_g_threads_per_proc,
        )
    except Exception as exc:
        return mol_id, f"error:{type(exc).__name__}: {exc}", None, None

    return mol_id, "ok", time.perf_counter() - t0, sample


def _discover_mol_ids(
    manifold_source: str,
    merged: Optional[dict],
    n_levels: int,
    smooth_sigma: float,
    max_samples: Optional[int],
) -> List[str]:
    if merged is not None:
        mol_ids = sorted(merged.keys())
    else:
        suffix = f"_nl{n_levels}_s{smooth_sigma:.2f}.pkl"
        mol_ids = sorted(
            name[: -len(suffix)]
            for name in os.listdir(manifold_source)
            if name.endswith(suffix)
        )
    if max_samples is not None:
        mol_ids = mol_ids[:max_samples]
    return mol_ids


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Unified Stage 2→3 preprocessing: manifold → FCLC inline → packed shard.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--manifold-cache-dir", default=None,
                   help="Stage 1 per-molecule cache directory.")
    p.add_argument("--manifold-pkl", default=None,
                   help="Optional merged Stage 1 manifold pkl.")
    p.add_argument("--packed-dir", required=True,
                   help="Output packed Stage 3 directory (sharded).")
    p.add_argument("--shard-size", type=int, default=2000,
                   help="Max molecules per shard.")
    p.add_argument("--workers", type=int,
                   default=max(1, (os.cpu_count() or 8) // 4))
    p.add_argument("--threads-per-proc", type=int, default=4,
                   help="Inner thread pool size per molecule.")
    p.add_argument("--parallel-mode", choices=("process", "thread"), default="process")
    p.add_argument("--n-levels", type=int, default=4)
    p.add_argument("--smooth-sigma", type=float, default=0.5)
    p.add_argument("--tau-r", type=float, default=1.0)
    p.add_argument("--tau-2", type=float, default=1.5)
    p.add_argument("--local-knn-k", type=int, default=12)
    p.add_argument("--chart-knn-k", type=int, default=8)
    p.add_argument("--num-anchors", type=int, default=8)
    p.add_argument("--mem-thresh", type=int, default=None,
                   help="Max mesh vertices per component for precomputing the full "
                        "dense geodesic distance matrix. Higher values trade RAM for "
                        "speed (eliminates per-Dijkstra fallback). Default: use "
                        "ED2E_FCLC_MEM_THRESH env var or module default (3000). "
                        "With 500 GB RAM, 15000–20000 is safe.")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--chunksize", type=int, default=4)
    p.add_argument("--resume", action="store_true",
                   help="Resume an interrupted run: skip mol_ids already written "
                        "in existing shard directories under --packed-dir.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if args.manifold_cache_dir is None and args.manifold_pkl is None:
        raise ValueError("Provide --manifold-cache-dir or --manifold-pkl.")

    global _g_manifold_cache_dir, _g_manifold_merged, _g_n_levels, _g_smooth_sigma
    global _g_tau_r, _g_tau_2, _g_local_knn_k, _g_chart_knn_k, _g_num_anchors
    global _g_threads_per_proc, _g_mem_thresh

    _g_manifold_cache_dir = args.manifold_cache_dir or ""
    _g_manifold_merged    = None
    _g_n_levels           = args.n_levels
    _g_smooth_sigma       = args.smooth_sigma
    _g_tau_r              = args.tau_r
    _g_tau_2              = args.tau_2
    _g_local_knn_k        = args.local_knn_k
    _g_chart_knn_k        = args.chart_knn_k
    _g_num_anchors        = args.num_anchors
    _g_threads_per_proc   = args.threads_per_proc
    _g_mem_thresh         = args.mem_thresh

    if args.manifold_pkl is not None:
        from ed2e.data.manifold import _patch_legacy_pickle_modules
        print(f"Loading merged Stage 1 manifold pkl from {args.manifold_pkl} ...")
        _patch_legacy_pickle_modules()
        with open(args.manifold_pkl, "rb") as f:
            _g_manifold_merged = pickle.load(f)

    mol_ids = _discover_mol_ids(
        _g_manifold_cache_dir,
        _g_manifold_merged,
        args.n_levels,
        args.smooth_sigma,
        args.max_samples,
    )
    print(f"Discovered {len(mol_ids)} molecules from Stage 1 source.")

    writer = Stage3ShardedWriter(args.packed_dir, shard_size=args.shard_size,
                                 resume=args.resume)
    if args.resume:
        done_set = writer.done_mol_ids()
        if done_set:
            mol_ids = [m for m in mol_ids if m not in done_set]
            print(f"Resume: {len(done_set)} already written "
                  f"({writer._shard_idx} shard(s)), {len(mol_ids)} remaining.")
        else:
            print("Resume: no existing shards found, starting fresh.")

    counts: defaultdict = defaultdict(int)
    proc_times: List[float] = []
    t0 = time.perf_counter()

    def _handle(mol_id: str, status: str, elapsed: Optional[float], sample: Optional[object]) -> None:
        key = status if status in ("ok",) else "error"
        counts[key] += 1
        if elapsed is not None:
            proc_times.append(elapsed)
        if sample is not None:
            writer.put(sample)  # type: ignore[arg-type]

    if args.parallel_mode == "thread":
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
            futures = [executor.submit(_worker, mid) for mid in mol_ids]
            with tqdm(total=len(mol_ids), desc="Stage2→3", unit="mol", dynamic_ncols=True) as pbar:
                for fut in futures:
                    mol_id, status, elapsed, sample = fut.result()
                    _handle(mol_id, status, elapsed, sample)
                    avg = float(np.mean(proc_times)) if proc_times else 0.0
                    pbar.set_postfix(ok=counts["ok"], err=counts["error"], avg=f"{avg:.2f}s", refresh=False)
                    pbar.update(1)
    else:
        ctx = mp.get_context("fork" if sys.platform != "win32" else "spawn")
        with ctx.Pool(processes=max(1, args.workers)) as pool:
            with tqdm(total=len(mol_ids), desc="Stage2→3", unit="mol", dynamic_ncols=True) as pbar:
                for mol_id, status, elapsed, sample in pool.imap_unordered(
                    _worker, mol_ids, chunksize=args.chunksize
                ):
                    _handle(mol_id, status, elapsed, sample)
                    avg = float(np.mean(proc_times)) if proc_times else 0.0
                    pbar.set_postfix(ok=counts["ok"], err=counts["error"], avg=f"{avg:.2f}s", refresh=False)
                    pbar.update(1)

    writer.finalize()

    elapsed = time.perf_counter() - t0
    print(
        f"\nStage2→3 finished in {elapsed:.1f}s | "
        f"ok={counts['ok']} error={counts['error']}"
    )
    print(f"Saved packed Stage 3 shards to {args.packed_dir}")


if __name__ == "__main__":
    main()
