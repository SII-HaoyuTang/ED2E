#!/usr/bin/env python3
"""
Stage 2 preprocessing script (parallel version).

Builds the FCLC atlas for every molecule in the merged Stage 1 manifold PKL,
caches per-molecule results to disk, then merges everything into a single
bundle (default) or legacy consolidated .pkl file and removes the individual
files.

Parallelism strategy
--------------------
The merged manifold PKL is loaded once in the main process. By default the
script uses a thread pool so all workers share the same manifold data in a
single Python process. A fork-based process pool remains available as an
opt-in backend for cases where process-level parallelism is preferred.

Usage examples
--------------
# Test on 20 molecules, 4 workers (default thread mode)
python scripts/preprocess_stage2.py \\
    --manifold-pkl data/ed_energy_5w/cache_manifold/all_nl4_s0.50.pkl \\
    --cache-dir data/ed_energy_5w/cache_fclc \\
    --max-samples 20 --workers 4

# Full dataset (default thread mode)
python scripts/preprocess_stage2.py \\
    --manifold-pkl data/ed_energy_5w/cache_manifold/all_nl4_s0.50.pkl \\
    --cache-dir data/ed_energy_5w/cache_fclc \\
    --workers 8

# Process mode
python scripts/preprocess_stage2.py \\
    --manifold-pkl data/ed_energy_5w/cache_manifold/all_nl4_s0.50.pkl \\
    --cache-dir data/ed_energy_5w/cache_fclc \\
    --workers 8 --parallel-mode process

# Hybrid mode: 4 processes × 2 native threads/process
python scripts/preprocess_stage2.py \\
    --manifold-pkl data/ed_energy_5w/cache_manifold/all_nl4_s0.50.pkl \\
    --cache-dir data/ed_energy_5w/cache_fclc \\
    --workers 4 --parallel-mode process --native-threads 2

# Skip merging (keep individual per-molecule files)
python scripts/preprocess_stage2.py ... --no-merge

Merged output
-------------
  Default:
    {cache_dir}/all_fclc_tr{tau_r:.2f}_t2{tau_2:.2f}.zip
    Single-file bundle with one pickle member per molecule.

  Legacy:
    {cache_dir}/all_fclc_tr{tau_r:.2f}_t2{tau_2:.2f}.pkl
    Stores a dict: {mol_id: List[FCLCLevel]} and requires full in-memory merge.

  Individual per-molecule files are deleted after a successful merge.
"""
from __future__ import annotations

# Must be set before any numba import to avoid fork+JIT semaphore leaks.
# "workqueue" is numba's built-in thread pool and does not create semaphores.
import os
os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")

import argparse
import gc
import multiprocessing as mp
import pickle
import sys
import time
from contextlib import nullcontext
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import List, Optional, Tuple

import numpy as np
from tqdm import tqdm

try:
    from threadpoolctl import threadpool_limits
except Exception:  # pragma: no cover - optional runtime dependency
    threadpool_limits = None

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from ed2e.data.manifold import _patch_legacy_pickle_modules   # noqa: E402
from ed2e.data.fclc import (                                   # noqa: E402
    build_fclc_levels, fclc_cache_path, save_fclc_levels, load_fclc_levels,
    _warmup_numba_jit,
)

# ---------------------------------------------------------------------------
# Globals shared via fork
# ---------------------------------------------------------------------------
_g_manifold:     dict  = {}
_g_cache_dir:    str   = ""
_g_tau_r:        float = 1.0
_g_tau_2:        float = 1.5
_g_min_chart:    int   = 5
_g_compute_inter: bool = True
_g_mem_thresh:   int   = 3000
_g_native_threads: int = 1
_g_parallel_mode: str = "thread"


# ---------------------------------------------------------------------------
# Cache path helper (mirroring fclc.py but without importing it in global scope)
# ---------------------------------------------------------------------------

def _mol_cache_path(mol_id: str) -> str:
    return fclc_cache_path(_g_cache_dir, mol_id, _g_tau_r, _g_tau_2)


def _merged_path(cache_dir: str, tau_r: float, tau_2: float) -> str:
    return os.path.join(cache_dir, f"all_fclc_tr{tau_r:.2f}_t2{tau_2:.2f}.zip")


def _legacy_merged_path(cache_dir: str, tau_r: float, tau_2: float) -> str:
    return os.path.join(cache_dir, f"all_fclc_tr{tau_r:.2f}_t2{tau_2:.2f}.pkl")


def _configure_native_threads(native_threads: int) -> None:
    """Configure per-process native thread pools where supported."""
    if native_threads <= 0:
        return

    # Numba thread count can be adjusted dynamically per process.
    try:
        import numba
        numba.set_num_threads(native_threads)
    except Exception:
        pass


def _thread_limit_context(native_threads: int):
    """Context manager that limits BLAS/OpenMP thread pools for one task."""
    if native_threads <= 0 or threadpool_limits is None:
        return nullcontext()
    return threadpool_limits(limits=native_threads)


def _init_process_worker(native_threads: int) -> None:
    _configure_native_threads(native_threads)


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def _worker(mol_id: str) -> Tuple[str, str, Optional[float]]:
    """Process one molecule. Returns (mol_id, status, elapsed | None)."""
    path = _mol_cache_path(mol_id)
    if os.path.exists(path):
        return mol_id, "cached", None

    manifold_levels = _g_manifold.get(mol_id)
    if manifold_levels is None:
        return mol_id, "error:missing_manifold", None

    t0 = time.perf_counter()
    try:
        limit_ctx = (
            _thread_limit_context(_g_native_threads)
            if _g_parallel_mode == "process" else nullcontext()
        )
        with limit_ctx:
            fclc_levels = build_fclc_levels(
                manifold_levels,
                tau_r=_g_tau_r,
                tau_2=_g_tau_2,
                min_chart_size=_g_min_chart,
                compute_inter=_g_compute_inter,
                mem_thresh=_g_mem_thresh,
            )
    except Exception as exc:
        return mol_id, f"error:{type(exc).__name__}: {exc}", None

    elapsed = time.perf_counter() - t0

    total_charts = sum(len(lv.charts) for lv in fclc_levels)
    if total_charts == 0:
        del fclc_levels
        gc.collect()
        return mol_id, "empty", elapsed

    try:
        save_fclc_levels(path, fclc_levels)
        return mol_id, "ok", elapsed
    finally:
        del fclc_levels
        gc.collect()


# ---------------------------------------------------------------------------
# Merge step
# ---------------------------------------------------------------------------

def _merge_and_cleanup(
    mol_ids:   List[str],
    cache_dir: str,
    tau_r:     float,
    tau_2:     float,
    merge_format: str,
) -> str:
    """Merge per-molecule cache files and delete originals."""
    if merge_format == "dict":
        out_path = _legacy_merged_path(cache_dir, tau_r, tau_2)
    else:
        out_path = _merged_path(cache_dir, tau_r, tau_2)

    missing: List[str] = []
    merged_count = 0

    print(f"\nMerging individual cache files → {out_path}")
    if merge_format == "dict":
        merged: dict = {}
        for mol_id in tqdm(mol_ids, desc="Merging", unit="mol", dynamic_ncols=True):
            path = fclc_cache_path(cache_dir, mol_id, tau_r, tau_2)
            if not os.path.exists(path):
                missing.append(mol_id)
                continue
            merged[mol_id] = load_fclc_levels(path)
        merged_count = len(merged)
        print(f"  Loaded {merged_count} molecules  "
              f"({len(missing)} missing / failed, not included)")
        print(f"  Writing merged file …")
        with open(out_path, "wb") as f:
            pickle.dump(merged, f, protocol=pickle.HIGHEST_PROTOCOL)
        del merged
        gc.collect()
    else:
        import zipfile
        with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_STORED) as zf:
            kept_ids: List[str] = []
            for mol_id in tqdm(mol_ids, desc="Merging", unit="mol", dynamic_ncols=True):
                path = fclc_cache_path(cache_dir, mol_id, tau_r, tau_2)
                if not os.path.exists(path):
                    missing.append(mol_id)
                    continue
                zf.write(path, arcname=f"molecules/{mol_id}.pkl")
                kept_ids.append(mol_id)
                merged_count += 1
            zf.writestr(
                "meta/mol_ids.pkl",
                pickle.dumps(kept_ids, protocol=pickle.HIGHEST_PROTOCOL),
                compress_type=zipfile.ZIP_STORED,
            )
        print(f"  Packed {merged_count} molecules  "
              f"({len(missing)} missing / failed, not included)")

    size_gb = os.path.getsize(out_path) / 1e9
    print(f"  Saved {out_path}  ({size_gb:.2f} GB)")

    print(f"  Deleting {merged_count} individual files …")
    kept_for_cleanup = (mol_id for mol_id in mol_ids if mol_id not in set(missing))
    for mol_id in tqdm(kept_for_cleanup, desc="Cleanup", unit="file",
                       dynamic_ncols=True, leave=False):
        path = fclc_cache_path(cache_dir, mol_id, tau_r, tau_2)
        try:
            os.remove(path)
        except FileNotFoundError:
            pass

    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage 2 preprocessing: parallel FCLC atlas construction.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--manifold-pkl", required=True,
                   help="Path to merged Stage-1 manifold pkl "
                        "(all_nl*.pkl dict {mol_id: List[ManifoldLevel]})")
    p.add_argument("--cache-dir", required=True,
                   help="Directory for FCLC cache files.")
    p.add_argument("--workers", type=int, default=max(1, os.cpu_count() // 2),
                   help="Number of parallel workers (processes or threads).")
    p.add_argument(
        "--parallel-mode",
        choices=("thread", "process"),
        default="thread",
        help="Parallel execution backend. 'thread' keeps one Python process and "
             "shares manifold data across threads; 'process' uses fork workers.",
    )
    p.add_argument(
        "--native-threads",
        type=int,
        default=1,
        help="Native compute threads per worker process for Numba/BLAS/OpenMP "
             "controlled sections. Useful with --parallel-mode process for "
             "hybrid process+thread parallelism.",
    )
    p.add_argument("--tau-r", type=float, default=1.0,
                   help="Region compatibility threshold τ_r.")
    p.add_argument("--tau-2", type=float, default=1.5,
                   help="Two-point reference neighbourhood threshold τ_2.")
    p.add_argument("--min-chart-size", type=int, default=5,
                   help="Minimum vertex count per chart.")
    p.add_argument("--no-inter", action="store_true",
                   help="Skip inter-layer weight computation.")
    p.add_argument("--max-samples", type=int, default=None,
                   help="Cap on the number of molecules (for testing).")
    p.add_argument("--chunksize", type=int, default=2,
                   help="imap chunksize.")
    p.add_argument("--no-merge", action="store_true",
                   help="Skip the merge step; keep individual per-molecule files.")
    p.add_argument(
        "--mem-thresh",
        type=int,
        default=3000,
        help="Precompute dense geodesic distances only for components with at "
             "most this many vertices. Lower values reduce peak memory.",
    )
    p.add_argument(
        "--maxtasksperchild",
        type=int,
        default=32,
        help="Recycle worker processes after this many chunks to limit memory "
             "growth over long runs. Use 0 to disable recycling.",
    )
    p.add_argument(
        "--merge-format",
        choices=("zip", "dict"),
        default="zip",
        help="Merged output format. 'zip' streams per-molecule files into a "
             "single bundle without loading everything into RAM; 'dict' keeps "
             "the legacy single pickle dict format and is memory-heavy.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # ---- 1. Load Stage-1 manifold PKL ----------------------------------------
    print(f"Loading manifold PKL from {args.manifold_pkl} …")
    t0 = time.perf_counter()
    _patch_legacy_pickle_modules()
    with open(args.manifold_pkl, "rb") as f:
        raw_manifold: dict = pickle.load(f)

    mol_ids = list(raw_manifold.keys())
    if args.max_samples is not None:
        mol_ids = mol_ids[:args.max_samples]
    print(f"  {len(mol_ids)} molecules loaded in {time.perf_counter() - t0:.1f}s")

    os.makedirs(args.cache_dir, exist_ok=True)

    # ---- 2. Pre-warm Numba JIT kernels in main process ------------------------
    # In process mode, compiling here means forked workers inherit the compiled
    # code via copy-on-write. In thread mode it simply avoids first-task JIT
    # latency on a worker thread.
    print("  Pre-warming Numba JIT kernels …")
    _warmup_numba_jit()

    # ---- 3. Set globals (inherited by forked workers) -------------------------
    global _g_manifold, _g_cache_dir, _g_tau_r, _g_tau_2
    global _g_min_chart, _g_compute_inter
    global _g_mem_thresh, _g_native_threads, _g_parallel_mode
    if args.max_samples is None or len(mol_ids) == len(raw_manifold):
        _g_manifold = raw_manifold
    else:
        _g_manifold = {k: raw_manifold[k] for k in mol_ids}
    del raw_manifold
    gc.collect()
    _g_cache_dir     = args.cache_dir
    _g_tau_r         = args.tau_r
    _g_tau_2         = args.tau_2
    _g_min_chart     = args.min_chart_size
    _g_compute_inter = not args.no_inter
    _g_mem_thresh    = args.mem_thresh
    _g_parallel_mode = args.parallel_mode
    _g_native_threads = max(1, args.native_threads)
    if args.parallel_mode == "process":
        _configure_native_threads(_g_native_threads)
    else:
        _g_native_threads = 1

    n_workers = max(1, args.workers)
    if args.parallel_mode == "thread":
        if args.native_threads != 1:
            print("  Ignoring --native-threads in thread mode; using 1 to avoid "
                  "oversubscription.")
        print(f"  Starting {n_workers} worker thread(s) in one process "
              f"(native_threads={_g_native_threads}) …\n")
    else:
        print(f"  Starting {n_workers} worker process(es) via fork "
              f"(native_threads={_g_native_threads}) …\n")

    # ---- 3. Parallel processing -----------------------------------------------
    from collections import defaultdict
    counts: dict = defaultdict(int)
    proc_times: List[float] = []
    t_proc = time.perf_counter()

    with tqdm(total=len(mol_ids), desc="Stage 2", unit="mol",
              dynamic_ncols=True) as pbar:
        if args.parallel_mode == "thread":
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                mol_iter = iter(mol_ids)
                pending = {}
                while len(pending) < n_workers:
                    try:
                        mol_id = next(mol_iter)
                    except StopIteration:
                        break
                    fut = pool.submit(_worker, mol_id)
                    pending[fut] = mol_id

                while pending:
                    done, _ = wait(tuple(pending.keys()), return_when=FIRST_COMPLETED)
                    for fut in done:
                        pending.pop(fut, None)
                        mol_id, status, elapsed = fut.result()
                        _g_manifold.pop(mol_id, None)
                        if (counts["ok"] + counts["cached"] + counts["empty"] + counts["error"] + 1) % 32 == 0:
                            gc.collect()
                        try:
                            next_mol = next(mol_iter)
                            next_fut = pool.submit(_worker, next_mol)
                            pending[next_fut] = next_mol
                        except StopIteration:
                            pass
                        key = status if status in ("ok", "cached", "empty") else "error"
                        counts[key] += 1
                        if elapsed is not None:
                            proc_times.append(elapsed)

                        avg = float(np.mean(proc_times)) if proc_times else 0.0
                        pbar.set_postfix(
                            ok=counts["ok"], cached=counts["cached"],
                            empty=counts["empty"], err=counts["error"],
                            avg=f"{avg:.2f}s",
                            refresh=False,
                        )
                        pbar.update(1)

                        if key in ("error", "empty"):
                            tqdm.write(f"  [{key.upper()}] {mol_id}: {status}")
        else:
            ctx = mp.get_context("fork")
            pool_kwargs = {"processes": n_workers}
            if args.maxtasksperchild > 0:
                pool_kwargs["maxtasksperchild"] = args.maxtasksperchild
            pool_kwargs["initializer"] = _init_process_worker
            pool_kwargs["initargs"] = (_g_native_threads,)

            with ctx.Pool(**pool_kwargs) as pool:
                for mol_id, status, elapsed in pool.imap_unordered(
                    _worker, mol_ids, chunksize=args.chunksize
                ):
                    key = status if status in ("ok", "cached", "empty") else "error"
                    counts[key] += 1
                    _g_manifold.pop(mol_id, None)
                    if (counts["ok"] + counts["cached"] + counts["empty"] + counts["error"]) % 32 == 0:
                        gc.collect()
                    if elapsed is not None:
                        proc_times.append(elapsed)

                    avg = float(np.mean(proc_times)) if proc_times else 0.0
                    pbar.set_postfix(
                        ok=counts["ok"], cached=counts["cached"],
                        empty=counts["empty"], err=counts["error"],
                        avg=f"{avg:.2f}s",
                        refresh=False,
                    )
                    pbar.update(1)

                    if key in ("error", "empty"):
                        tqdm.write(f"  [{key.upper()}] {mol_id}: {status}")

    wall_proc = time.perf_counter() - t_proc
    _g_manifold.clear()
    gc.collect()

    # ---- 4. Summary -----------------------------------------------------------
    print("\n" + "=" * 55)
    print("FCLC atlas construction complete")
    print("=" * 55)
    for k in ("ok", "cached", "empty", "error"):
        print(f"  {k:>8}: {counts[k]}")
    if proc_times:
        print(f"  avg/mol  : {np.mean(proc_times):.3f}s")
        print(f"  median   : {np.median(proc_times):.3f}s")
        print(f"  max      : {np.max(proc_times):.3f}s")
    print(f"  wall     : {wall_proc:.1f}s  ({wall_proc / 60:.1f}min)")
    print("=" * 55)

    # ---- 5. Merge -------------------------------------------------------------
    if args.no_merge:
        print("\nSkipping merge (--no-merge).")
        return

    _merge_and_cleanup(
        mol_ids=mol_ids,
        cache_dir=args.cache_dir,
        tau_r=args.tau_r,
        tau_2=args.tau_2,
        merge_format=args.merge_format,
    )


if __name__ == "__main__":
    main()
