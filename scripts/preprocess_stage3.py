#!/usr/bin/env python3
"""
Stage 3 preprocessing.

Builds Stage 3 local-aggregation samples from per-molecule Stage 1 manifold
cache + Stage 2 FCLC cache/bundle, then optionally merges them into a single
zip bundle.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import pickle
import sys
import time
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple

import numpy as np
from tqdm import tqdm

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from ed2e.data.stage3_local import (  # noqa: E402
    build_stage3_sample,
    save_stage3_bundle_entry,
    save_stage3_sample,
    stage3_cache_path,
)
from ed2e.data.stage3_packed import pack_stage3_cache  # noqa: E402

_g_manifold_cache_dir: str = ""
_g_manifold_merged: Optional[dict] = None
_g_n_levels: int = 4
_g_smooth_sigma: float = 0.5
_g_fclc_source: str = ""
_g_tau_r: float = 1.0
_g_tau_2: float = 1.5
_g_stage3_cache_dir: str = ""
_g_local_knn_k: int = 12
_g_chart_knn_k: int = 8
_g_num_anchors: int = 8
_g_threads_per_proc: int = 4


def _merged_path(cache_dir: str, local_knn_k: int, chart_knn_k: int, num_anchors: int) -> str:
    return os.path.join(cache_dir, f"all_stage3_lk{local_knn_k}_ck{chart_knn_k}_a{num_anchors}_ig7_il7.zip")


def _discover_mol_ids(
    fclc_source: str,
    tau_r: float,
    tau_2: float,
    max_samples: Optional[int],
) -> List[str]:
    from ed2e.data.fclc import list_fclc_bundle_ids

    if os.path.isdir(fclc_source):
        suffix = f"_fclc_tr{tau_r:.2f}_t2{tau_2:.2f}.pkl"
        mol_ids = sorted(
            name[:-len(suffix)]
            for name in os.listdir(fclc_source)
            if name.endswith(suffix)
        )
    else:
        mol_ids = list_fclc_bundle_ids(fclc_source)
    if max_samples is not None:
        mol_ids = mol_ids[:max_samples]
    return mol_ids


def _load_fclc_for_mol(mol_id: str) -> List:
    from ed2e.data.fclc import fclc_cache_path, load_fclc_entry, load_fclc_levels

    if os.path.isdir(_g_fclc_source):
        return load_fclc_levels(fclc_cache_path(_g_fclc_source, mol_id, _g_tau_r, _g_tau_2))
    return load_fclc_entry(_g_fclc_source, mol_id)


def _worker(mol_id: str) -> Tuple[str, str, Optional[float]]:
    from ed2e.data.manifold import load_manifold_levels, manifold_cache_path

    out_path = stage3_cache_path(
        _g_stage3_cache_dir,
        mol_id,
        _g_local_knn_k,
        _g_chart_knn_k,
        _g_num_anchors,
    )
    if os.path.exists(out_path):
        return mol_id, "cached", None

    t0 = time.perf_counter()
    try:
        if _g_manifold_merged is not None:
            manifold_levels = _g_manifold_merged[mol_id]
        else:
            manifold_path = manifold_cache_path(
                _g_manifold_cache_dir,
                mol_id,
                _g_n_levels,
                _g_smooth_sigma,
            )
            if not os.path.exists(manifold_path):
                return mol_id, "error:missing_manifold", None
            manifold_levels = load_manifold_levels(manifold_path)
        fclc_levels = _load_fclc_for_mol(mol_id)
        sample = build_stage3_sample(
            mol_id,
            manifold_levels,
            fclc_levels,
            local_knn_k=_g_local_knn_k,
            chart_knn_k=_g_chart_knn_k,
            num_anchors=_g_num_anchors,
            inner_threads=_g_threads_per_proc,
        )
        save_stage3_sample(out_path, sample)
    except Exception as exc:
        return mol_id, f"error:{type(exc).__name__}: {exc}", None
    return mol_id, "ok", time.perf_counter() - t0


def _merge_and_cleanup(
    mol_ids: List[str],
    cache_dir: str,
    local_knn_k: int,
    chart_knn_k: int,
    num_anchors: int,
) -> str:
    out_path = _merged_path(cache_dir, local_knn_k, chart_knn_k, num_anchors)
    kept_ids: List[str] = []
    print(f"\nMerging Stage 3 cache files -> {out_path}")
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_STORED) as zf:
        for mol_id in tqdm(mol_ids, desc="Merging", unit="mol", dynamic_ncols=True):
            path = stage3_cache_path(cache_dir, mol_id, local_knn_k, chart_knn_k, num_anchors)
            if not os.path.exists(path):
                continue
            with open(path, "rb") as f:
                sample = pickle.load(f)
            save_stage3_bundle_entry(zf, mol_id, sample)
            kept_ids.append(mol_id)
        zf.writestr(
            "meta/mol_ids.pkl",
            pickle.dumps(kept_ids, protocol=pickle.HIGHEST_PROTOCOL),
            compress_type=zipfile.ZIP_STORED,
        )

    for mol_id in tqdm(kept_ids, desc="Cleanup", unit="file", dynamic_ncols=True, leave=False):
        path = stage3_cache_path(cache_dir, mol_id, local_knn_k, chart_knn_k, num_anchors)
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
    return out_path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage 3 preprocessing: build local-aggregation samples + Stage 4 static bundle.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--manifold-cache-dir", default=None, help="Stage 1 per-molecule cache directory.")
    p.add_argument("--manifold-pkl", default=None, help="Optional merged Stage 1 manifold pkl fallback.")
    p.add_argument("--fclc-source", required=True, help="Stage 2 cache directory or merged bundle path.")
    p.add_argument("--cache-dir", required=True, help="Output Stage 3 cache directory.")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 8) // 4))
    p.add_argument("--threads-per-proc", type=int, default=4, help="Inner thread pool size per molecule.")
    p.add_argument("--parallel-mode", choices=("process", "thread"), default="process")
    p.add_argument("--n-levels", type=int, default=4)
    p.add_argument("--smooth-sigma", type=float, default=0.5)
    p.add_argument("--tau-r", type=float, default=1.0)
    p.add_argument("--tau-2", type=float, default=1.5)
    p.add_argument("--local-knn-k", type=int, default=12)
    p.add_argument("--chart-knn-k", type=int, default=8)
    p.add_argument("--num-anchors", type=int, default=8)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--chunksize", type=int, default=4)
    p.add_argument("--packed-dir", default=None, help="Optional packed Stage 3 output directory for training-time mmap reads.")
    p.add_argument("--packed-overwrite", action="store_true", help="Rewrite the packed Stage 3 directory if it already exists.")
    p.add_argument("--no-merge", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if args.manifold_cache_dir is None and args.manifold_pkl is None:
        raise ValueError("Provide --manifold-cache-dir or --manifold-pkl.")
    os.makedirs(args.cache_dir, exist_ok=True)

    mol_ids = _discover_mol_ids(args.fclc_source, args.tau_r, args.tau_2, args.max_samples)
    print(f"Discovered {len(mol_ids)} molecules from Stage 2 source.")

    global _g_manifold_cache_dir, _g_manifold_merged, _g_n_levels, _g_smooth_sigma
    global _g_fclc_source, _g_tau_r, _g_tau_2, _g_stage3_cache_dir
    global _g_local_knn_k, _g_chart_knn_k, _g_num_anchors, _g_threads_per_proc
    _g_manifold_cache_dir = args.manifold_cache_dir
    _g_manifold_merged = None
    _g_n_levels = args.n_levels
    _g_smooth_sigma = args.smooth_sigma
    _g_fclc_source = args.fclc_source
    _g_tau_r = args.tau_r
    _g_tau_2 = args.tau_2
    _g_stage3_cache_dir = args.cache_dir
    _g_local_knn_k = args.local_knn_k
    _g_chart_knn_k = args.chart_knn_k
    _g_num_anchors = args.num_anchors
    _g_threads_per_proc = args.threads_per_proc

    if args.manifold_pkl is not None:
        import pickle
        from ed2e.data.manifold import _patch_legacy_pickle_modules

        print(f"Loading merged Stage 1 manifold pkl from {args.manifold_pkl} ...")
        _patch_legacy_pickle_modules()
        with open(args.manifold_pkl, "rb") as f:
            _g_manifold_merged = pickle.load(f)

    counts = defaultdict(int)
    proc_times: List[float] = []
    t0 = time.perf_counter()

    if args.parallel_mode == "thread":
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
            futures = [executor.submit(_worker, mol_id) for mol_id in mol_ids]
            with tqdm(total=len(mol_ids), desc="Stage 3", unit="mol", dynamic_ncols=True) as pbar:
                for fut in futures:
                    mol_id, status, elapsed = fut.result()
                    key = status if status in ("ok", "cached") else "error"
                    counts[key] += 1
                    if elapsed is not None:
                        proc_times.append(elapsed)
                    avg = float(np.mean(proc_times)) if proc_times else 0.0
                    pbar.set_postfix(ok=counts["ok"], cached=counts["cached"], err=counts["error"], avg=f"{avg:.2f}s", refresh=False)
                    pbar.update(1)
    else:
        ctx = mp.get_context("fork" if sys.platform != "win32" else "spawn")
        with ctx.Pool(processes=max(1, args.workers)) as pool:
            with tqdm(total=len(mol_ids), desc="Stage 3", unit="mol", dynamic_ncols=True) as pbar:
                for mol_id, status, elapsed in pool.imap_unordered(_worker, mol_ids, chunksize=args.chunksize):
                    key = status if status in ("ok", "cached") else "error"
                    counts[key] += 1
                    if elapsed is not None:
                        proc_times.append(elapsed)
                    avg = float(np.mean(proc_times)) if proc_times else 0.0
                    pbar.set_postfix(ok=counts["ok"], cached=counts["cached"], err=counts["error"], avg=f"{avg:.2f}s", refresh=False)
                    pbar.update(1)

    elapsed = time.perf_counter() - t0
    print(
        f"\nStage 3 finished in {elapsed:.1f}s | "
        f"ok={counts['ok']} cached={counts['cached']} error={counts['error']}"
    )

    if args.packed_dir:
        packed_stats = pack_stage3_cache(
            args.cache_dir,
            args.packed_dir,
            mol_ids=mol_ids,
            overwrite=args.packed_overwrite,
            show_progress=True,
        )
        print(
            f"Saved packed Stage 3 cache to {args.packed_dir} | "
            f"samples={packed_stats['num_samples']} nodes={packed_stats['total_nodes']} "
            f"charts={packed_stats['total_charts']}"
        )

    if not args.no_merge:
        out_path = _merge_and_cleanup(
            mol_ids,
            args.cache_dir,
            args.local_knn_k,
            args.chart_knn_k,
            args.num_anchors,
        )
        print(f"Saved merged Stage 3 bundle to {out_path}")


if __name__ == "__main__":
    main()
