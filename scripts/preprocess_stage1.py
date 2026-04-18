#!/usr/bin/env python3
"""
Stage 1 preprocessing script (parallel version).

Runs multi-level isodensity manifold extraction on the EDBench PKL dataset,
caches per-molecule results to disk, then merges everything into a single
consolidated .pkl file and removes the individual files.

Parallelism strategy
--------------------
The PKL (~9 GB) is loaded once in the main process.  Worker processes are
created with "fork" so they inherit the parent's memory via copy-on-write
(no data copying), then each worker processes an independent set of molecules.

Usage examples
--------------
# Test on 20 molecules, 4 parallel workers
python scripts/preprocess_stage1.py \
    --pkl  data/ed_energy_5w/processed/mol_EDthresh0.05_data.pkl \
    --cache-dir data/ed_energy_5w/cache_manifold \
    --max-samples 20 --workers 4

# Full dataset (merge into single file at the end)
python scripts/preprocess_stage1.py \
    --pkl  data/ed_energy_5w/processed/mol_EDthresh0.05_data.pkl \
    --cache-dir data/ed_energy_5w/cache_manifold \
    --workers 8

# Skip merging (keep individual per-molecule files)
python scripts/preprocess_stage1.py ... --no-merge

Merged output
-------------
  {cache_dir}/all_nl{n_levels}_s{smooth_sigma:.2f}.pkl

  Stores a dict:  {mol_id: List[ManifoldLevel]}
  Individual per-molecule files are deleted after a successful merge.
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

from ed2e.data.dataset import EDBenchPKLDataset        # noqa: E402
from ed2e.data.manifold import extract_manifold_levels  # noqa: E402
from ed2e.data.manifold import _patch_legacy_pickle_modules  # noqa: E402

# ---------------------------------------------------------------------------
# Globals shared via fork
# ---------------------------------------------------------------------------
_g_raw:           dict                  = {}
_g_cache_dir:     str                   = ""
_g_n_levels:      int                   = 4
_g_smooth_sigma:  float                 = 0.5
_g_percentiles:   Optional[List[float]] = None
_g_min_comp_size: int                   = 10


# ---------------------------------------------------------------------------
# Cache path helpers
# ---------------------------------------------------------------------------

def _mol_cache_path(mol_id: str) -> str:
    tag = f"nl{_g_n_levels}_s{_g_smooth_sigma:.2f}"
    return os.path.join(_g_cache_dir, f"{mol_id}_{tag}.pkl")


def _merged_path(cache_dir: str, n_levels: int, smooth_sigma: float) -> str:
    return os.path.join(cache_dir, f"all_nl{n_levels}_s{smooth_sigma:.2f}.pkl")


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def _worker(mol_id: str) -> Tuple[str, str, Optional[float]]:
    """Process one molecule. Returns (mol_id, status, elapsed | None)."""
    path = _mol_cache_path(mol_id)
    if os.path.exists(path):
        _patch_legacy_pickle_modules()
        return mol_id, "cached", None

    entry     = _g_raw[mol_id]
    coords    = entry["electronic_density"]["coords"]
    densities = entry["electronic_density"]["density"]

    t0 = time.perf_counter()
    try:
        levels = extract_manifold_levels(
            coords, densities,
            n_levels=_g_n_levels,
            percentiles=_g_percentiles,
            smooth_sigma=_g_smooth_sigma,
            min_component_size=_g_min_comp_size,
        )
    except Exception as exc:
        return mol_id, f"error:{exc}", None

    elapsed = time.perf_counter() - t0

    if sum(len(lv.components) for lv in levels) == 0:
        return mol_id, "empty", elapsed

    with open(path, "wb") as f:
        pickle.dump(levels, f, protocol=pickle.HIGHEST_PROTOCOL)

    return mol_id, "ok", elapsed


# ---------------------------------------------------------------------------
# Merge step
# ---------------------------------------------------------------------------

def _merge_and_cleanup(
    mol_ids:      List[str],
    cache_dir:    str,
    n_levels:     int,
    smooth_sigma: float,
) -> str:
    """Load all per-molecule .pkl files, merge into one dict, delete originals.

    Returns the path of the merged file.
    """
    out_path = _merged_path(cache_dir, n_levels, smooth_sigma)
    tag      = f"nl{n_levels}_s{smooth_sigma:.2f}"

    merged:  dict       = {}
    missing: List[str]  = []

    print(f"\nMerging individual cache files → {out_path}")
    _patch_legacy_pickle_modules()
    for mol_id in tqdm(mol_ids, desc="Merging", unit="mol", dynamic_ncols=True):
        path = os.path.join(cache_dir, f"{mol_id}_{tag}.pkl")
        if not os.path.exists(path):
            missing.append(mol_id)
            continue
        with open(path, "rb") as f:
            merged[mol_id] = pickle.load(f)

    print(f"  Loaded {len(merged)} molecules  ({len(missing)} missing / failed, not included)")

    print(f"  Writing merged file …")
    with open(out_path, "wb") as f:
        pickle.dump(merged, f, protocol=pickle.HIGHEST_PROTOCOL)

    size_gb = os.path.getsize(out_path) / 1e9
    print(f"  Saved {out_path}  ({size_gb:.2f} GB)")

    print(f"  Deleting {len(merged)} individual files …")
    for mol_id in tqdm(merged, desc="Cleanup", unit="file",
                       dynamic_ncols=True, leave=False):
        path = os.path.join(cache_dir, f"{mol_id}_{tag}.pkl")
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
        description="Stage 1 preprocessing: parallel Marching Cubes manifold extraction.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--pkl", required=True,
                   help="Path to mol_EDthresh0.05_data.pkl")
    p.add_argument("--cache-dir", required=True,
                   help="Directory for cache files.")
    p.add_argument("--workers", type=int, default=max(1, os.cpu_count() // 2),
                   help="Number of parallel worker processes.")
    p.add_argument("--n-levels", type=int, default=4,
                   help="Number of isodensity levels K.")
    p.add_argument("--percentiles", type=float, nargs="+", default=None,
                   help="Percentile thresholds, e.g. --percentiles 20 40 60 80.")
    p.add_argument("--smooth-sigma", type=float, default=0.5,
                   help="Gaussian pre-smoothing sigma in Bohr (0 = disabled).")
    p.add_argument("--min-component-size", type=int, default=10,
                   help="Minimum vertex count to retain a mesh component.")
    p.add_argument("--max-samples", type=int, default=None,
                   help="Cap on the number of molecules (for testing).")
    p.add_argument("--chunksize", type=int, default=4,
                   help="imap chunksize — larger values reduce IPC overhead.")
    p.add_argument("--no-merge", action="store_true",
                   help="Skip the merge step; keep individual per-molecule files.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # ---- 1. Load PKL --------------------------------------------------------
    print(f"Loading PKL from {args.pkl} …")
    t0 = time.perf_counter()
    dataset = EDBenchPKLDataset(
        pkl_path=args.pkl,
        cache_dir=args.cache_dir,
        max_samples=args.max_samples,
    )
    mol_ids = dataset.mol_ids
    print(f"  {len(mol_ids)} molecules loaded in {time.perf_counter() - t0:.1f}s")

    os.makedirs(args.cache_dir, exist_ok=True)

    # ---- 2. Set globals (inherited by forked workers) -----------------------
    global _g_raw, _g_cache_dir, _g_n_levels, _g_smooth_sigma
    global _g_percentiles, _g_min_comp_size
    _g_raw           = dataset._raw
    _g_cache_dir     = args.cache_dir
    _g_n_levels      = args.n_levels
    _g_smooth_sigma  = args.smooth_sigma
    _g_percentiles   = args.percentiles
    _g_min_comp_size = args.min_component_size

    print(f"  Starting {max(1, args.workers)} worker process(es) via fork …\n")

    # ---- 3. Parallel extraction --------------------------------------------
    counts:     dict        = defaultdict(int)
    proc_times: List[float] = []
    t_proc = time.perf_counter()

    ctx = mp.get_context("fork")
    with ctx.Pool(processes=max(1, args.workers)) as pool:
        with tqdm(total=len(mol_ids), desc="Stage 1", unit="mol",
                  dynamic_ncols=True) as pbar:
            for mol_id, status, elapsed in pool.imap_unordered(
                _worker, mol_ids, chunksize=args.chunksize
            ):
                key = status if status in ("ok", "cached", "empty") else "error"
                counts[key] += 1
                if elapsed is not None:
                    proc_times.append(elapsed)

                avg = float(np.mean(proc_times)) if proc_times else 0.0
                pbar.set_postfix(
                    ok=counts["ok"],
                    cached=counts["cached"],
                    empty=counts["empty"],
                    err=counts["error"],
                    avg=f"{avg:.2f}s",
                    refresh=False,
                )
                pbar.update(1)

                if key in ("error", "empty"):
                    tqdm.write(f"  [{key.upper()}] {mol_id}: {status}")

    wall_proc = time.perf_counter() - t_proc

    # ---- 4. Summary ---------------------------------------------------------
    print("\n" + "=" * 55)
    print("Extraction complete")
    print("=" * 55)
    for k in ("ok", "cached", "empty", "error"):
        print(f"  {k:>8}: {counts[k]}")
    if proc_times:
        print(f"  avg/mol  : {np.mean(proc_times):.3f}s")
        print(f"  median   : {np.median(proc_times):.3f}s")
        print(f"  max      : {np.max(proc_times):.3f}s")
    print(f"  wall     : {wall_proc:.1f}s  ({wall_proc / 60:.1f}min)")
    print("=" * 55)

    # ---- 5. Merge -----------------------------------------------------------
    if args.no_merge:
        print("\nSkipping merge (--no-merge).")
        return

    _merge_and_cleanup(
        mol_ids=mol_ids,
        cache_dir=args.cache_dir,
        n_levels=args.n_levels,
        smooth_sigma=args.smooth_sigma,
    )


if __name__ == "__main__":
    main()
