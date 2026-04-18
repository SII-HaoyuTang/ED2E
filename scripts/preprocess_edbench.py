#!/usr/bin/env python3
"""
End-to-end EDBench preprocessing pipeline.

Converts the raw EDBench PKL + CSV into a training-ready data directory:

    mol_EDthresh0.05_data.pkl  +  ed_energy_5w.csv
      [Stage 1]  → cache_manifold/all_nl4_s0.50.pkl
      [Stage 2+3]→ packed_stage3/  (sharded Stage3PackedDataset)
      [Split]    → split.json      (train/val/test mol_id lists)
      [Stats]    → energy_stats.json  (per-target z-score statistics)

Each step is skipped automatically if its output already exists.
Use --skip-stage1 / --skip-stage23 to force-skip individual steps.

Usage
-----
::

    # Full pipeline (all 47 k molecules)
    python scripts/preprocess_edbench.py \\
        --pkl      data/ed_energy_5w/processed/mol_EDthresh0.05_data.pkl \\
        --csv      data/ed_energy_5w/raw/ed_energy_5w.csv \\
        --data-dir data/ed_energy_5w \\
        --workers  8

    # Quick test on first 200 molecules
    python scripts/preprocess_edbench.py ... --max-samples 200

    # Resume: skip already-done Stage 1
    python scripts/preprocess_edbench.py ... --skip-stage1
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import pickle
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from tqdm import tqdm

# ── project root on sys.path ──────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from ed2e.data.energy_stats import (   # noqa: E402
    compute_energy_stats,
    load_energy_labels,
    load_split_ids,
    save_energy_stats,
)
from ed2e.data.stage3_packed import Stage3PackedDataset   # noqa: E402

# ══════════════════════════════════════════════════════════════════════════════
# Stage 1 helpers (adapted from scripts/preprocess_stage1.py)
# ══════════════════════════════════════════════════════════════════════════════

# Worker-process globals (shared via fork, set in main before Pool creation)
_s1_raw:           dict  = {}
_s1_cache_dir:     str   = ""
_s1_n_levels:      int   = 4
_s1_smooth_sigma:  float = 0.5
_s1_min_comp_size: int   = 10
_s1_percentiles:   Optional[List[float]] = None


def _s1_mol_cache_path(mol_id: str) -> str:
    tag = f"nl{_s1_n_levels}_s{_s1_smooth_sigma:.2f}"
    return os.path.join(_s1_cache_dir, f"{mol_id}_{tag}.pkl")


def _s1_merged_path(cache_dir: str, n_levels: int, smooth_sigma: float) -> str:
    return os.path.join(cache_dir, f"all_nl{n_levels}_s{smooth_sigma:.2f}.pkl")


def _s1_worker(mol_id: str) -> Tuple[str, str, Optional[float]]:
    """Stage 1 worker: extract manifold levels for one molecule."""
    from ed2e.data.manifold import extract_manifold_levels, _patch_legacy_pickle_modules

    path = _s1_mol_cache_path(mol_id)
    if os.path.exists(path):
        _patch_legacy_pickle_modules()
        return mol_id, "cached", None

    entry     = _s1_raw[mol_id]
    coords    = entry["electronic_density"]["coords"]
    densities = entry["electronic_density"]["density"]

    t0 = time.perf_counter()
    try:
        levels = extract_manifold_levels(
            coords, densities,
            n_levels=_s1_n_levels,
            percentiles=_s1_percentiles,
            smooth_sigma=_s1_smooth_sigma,
            min_component_size=_s1_min_comp_size,
        )
    except Exception as exc:
        return mol_id, f"error:{exc}", None

    elapsed = time.perf_counter() - t0

    if sum(len(lv.components) for lv in levels) == 0:
        return mol_id, "empty", elapsed

    with open(path, "wb") as f:
        pickle.dump(levels, f, protocol=pickle.HIGHEST_PROTOCOL)

    return mol_id, "ok", elapsed


def _run_stage1(
    mol_ids: List[str],
    raw:     dict,
    cache_dir: str,
    n_levels: int,
    smooth_sigma: float,
    min_comp_size: int,
    percentiles: Optional[List[float]],
    workers: int,
    chunksize: int,
) -> str:
    """Run Stage 1 extraction, merge into single PKL, return merged path."""
    global _s1_raw, _s1_cache_dir, _s1_n_levels, _s1_smooth_sigma
    global _s1_min_comp_size, _s1_percentiles

    _s1_raw           = raw
    _s1_cache_dir     = cache_dir
    _s1_n_levels      = n_levels
    _s1_smooth_sigma  = smooth_sigma
    _s1_min_comp_size = min_comp_size
    _s1_percentiles   = percentiles

    os.makedirs(cache_dir, exist_ok=True)

    counts:     Dict[str, int] = defaultdict(int)
    proc_times: List[float]    = []

    ctx = mp.get_context("fork" if sys.platform != "win32" else "spawn")
    with ctx.Pool(processes=workers) as pool:
        with tqdm(total=len(mol_ids), desc="Stage 1", unit="mol",
                  dynamic_ncols=True) as pbar:
            for mol_id, status, elapsed in pool.imap_unordered(
                _s1_worker, mol_ids, chunksize=chunksize
            ):
                key = status if status in ("ok", "cached", "empty") else "error"
                counts[key] += 1
                if elapsed is not None:
                    proc_times.append(elapsed)
                avg = float(np.mean(proc_times)) if proc_times else 0.0
                pbar.set_postfix(ok=counts["ok"], cached=counts["cached"],
                                 err=counts["error"], avg=f"{avg:.2f}s",
                                 refresh=False)
                pbar.update(1)
                if key in ("error", "empty"):
                    tqdm.write(f"  [{key.upper()}] {mol_id}: {status}")

    # ── Merge ──
    merged_path = _s1_merged_path(cache_dir, n_levels, smooth_sigma)
    print(f"\nMerging individual files → {merged_path}")

    from ed2e.data.manifold import _patch_legacy_pickle_modules
    _patch_legacy_pickle_modules()

    merged: dict = {}
    tag = f"nl{n_levels}_s{smooth_sigma:.2f}"
    for mol_id in tqdm(mol_ids, desc="Merging", unit="mol", dynamic_ncols=True):
        p = os.path.join(cache_dir, f"{mol_id}_{tag}.pkl")
        if os.path.exists(p):
            with open(p, "rb") as f:
                merged[mol_id] = pickle.load(f)

    with open(merged_path, "wb") as f:
        pickle.dump(merged, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  Saved {merged_path}  ({os.path.getsize(merged_path)/1e9:.2f} GB)")

    for mol_id in tqdm(merged, desc="Cleanup", leave=False):
        p = os.path.join(cache_dir, f"{mol_id}_{tag}.pkl")
        try:
            os.remove(p)
        except FileNotFoundError:
            pass

    return merged_path


# ══════════════════════════════════════════════════════════════════════════════
# Stage 2+3 helpers (adapted from scripts/preprocess_stage2_to_packed.py)
# ══════════════════════════════════════════════════════════════════════════════

_s23_manifold_merged: Optional[dict] = None
_s23_tau_r:           float = 1.0
_s23_tau_2:           float = 1.5
_s23_local_knn_k:     int   = 12
_s23_chart_knn_k:     int   = 8
_s23_num_anchors:     int   = 8
_s23_threads_per_proc: int  = 4
_s23_mem_thresh:      Optional[int] = None


def _s23_worker(mol_id: str) -> Tuple[str, str, Optional[float], Optional[object]]:
    """Stage 2+3 worker: manifold → FCLC → Stage3Sample."""
    from ed2e.data.fclc import build_fclc_levels
    from ed2e.data.stage3_local import build_stage3_sample

    t0 = time.perf_counter()
    try:
        manifold_levels = _s23_manifold_merged[mol_id]  # type: ignore[index]
        fclc_levels = build_fclc_levels(
            manifold_levels, tau_r=_s23_tau_r, tau_2=_s23_tau_2,
            mem_thresh=_s23_mem_thresh,
        )
        sample = build_stage3_sample(
            mol_id, manifold_levels, fclc_levels,
            local_knn_k=_s23_local_knn_k,
            chart_knn_k=_s23_chart_knn_k,
            num_anchors=_s23_num_anchors,
            inner_threads=_s23_threads_per_proc,
        )
    except Exception as exc:
        return mol_id, f"error:{type(exc).__name__}: {exc}", None, None
    return mol_id, "ok", time.perf_counter() - t0, sample


def _run_stage23(
    mol_ids: List[str],
    manifold_merged: dict,
    packed_dir: str,
    tau_r: float,
    tau_2: float,
    local_knn_k: int,
    chart_knn_k: int,
    num_anchors: int,
    workers: int,
    threads_per_proc: int,
    shard_size: int,
    chunksize: int,
    mem_thresh: Optional[int] = None,
) -> None:
    """Run Stage 2+3 pipeline, write to packed_dir."""
    from ed2e.data.stage3_packed import Stage3ShardedWriter

    global _s23_manifold_merged, _s23_tau_r, _s23_tau_2
    global _s23_local_knn_k, _s23_chart_knn_k, _s23_num_anchors
    global _s23_threads_per_proc, _s23_mem_thresh

    _s23_manifold_merged  = manifold_merged
    _s23_tau_r            = tau_r
    _s23_tau_2            = tau_2
    _s23_local_knn_k      = local_knn_k
    _s23_chart_knn_k      = chart_knn_k
    _s23_num_anchors      = num_anchors
    _s23_threads_per_proc = threads_per_proc
    _s23_mem_thresh       = mem_thresh

    writer = Stage3ShardedWriter(packed_dir, shard_size=shard_size)
    counts:     Dict[str, int] = defaultdict(int)
    proc_times: List[float]    = []

    ctx = mp.get_context("fork" if sys.platform != "win32" else "spawn")
    with ctx.Pool(processes=workers) as pool:
        with tqdm(total=len(mol_ids), desc="Stage 2+3", unit="mol",
                  dynamic_ncols=True) as pbar:
            for mol_id, status, elapsed, sample in pool.imap_unordered(
                _s23_worker, mol_ids, chunksize=chunksize
            ):
                key = "ok" if status == "ok" else "error"
                counts[key] += 1
                if elapsed is not None:
                    proc_times.append(elapsed)
                if sample is not None:
                    writer.put(sample)  # type: ignore[arg-type]
                avg = float(np.mean(proc_times)) if proc_times else 0.0
                pbar.set_postfix(ok=counts["ok"], err=counts["error"],
                                 avg=f"{avg:.2f}s", refresh=False)
                pbar.update(1)
    writer.finalize()
    print(f"  Stage 2+3 done: ok={counts['ok']} error={counts['error']}")


# ══════════════════════════════════════════════════════════════════════════════
# Split + stats helpers
# ══════════════════════════════════════════════════════════════════════════════

def _build_split_json(
    csv_path:   str,
    packed_dir: str,
    split_col:  str,
    out_path:   str,
) -> Dict[str, List[str]]:
    """
    Read split column from CSV, intersect with actually-processed mol_ids,
    write split.json, return the dict.
    """
    splits    = load_split_ids(csv_path, split_col=split_col)
    processed = set(Stage3PackedDataset(packed_dir).mol_ids)

    splits_clean = {
        k: [m for m in v if m in processed]
        for k, v in splits.items()
    }

    n_all = sum(len(v) for v in splits.values())
    n_ok  = sum(len(v) for v in splits_clean.values())
    print(f"  Split intersection: {n_ok}/{n_all} molecules retained")
    for tag, ids in splits_clean.items():
        print(f"    {tag:>5}: {len(ids)}")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(splits_clean, f)
    print(f"  Saved {out_path}")
    return splits_clean


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="End-to-end EDBench preprocessing: PKL+CSV → packed Stage3.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--pkl",       required=True,
                   help="Path to mol_EDthresh0.05_data.pkl")
    p.add_argument("--csv",       required=True,
                   help="Path to ed_energy_5w.csv")
    p.add_argument("--data-dir",  required=True,
                   help="Output root directory.  Sub-directories are created automatically.")

    # skip flags
    p.add_argument("--skip-stage1",  action="store_true",
                   help="Skip Stage 1 if cache_manifold/ already exists.")
    p.add_argument("--skip-stage23", action="store_true",
                   help="Skip Stage 2+3 if packed_stage3/manifest.json already exists.")

    # parallelism
    p.add_argument("--workers",         type=int, default=max(1, (os.cpu_count() or 8) // 4))
    p.add_argument("--threads-per-proc",type=int, default=4)
    p.add_argument("--chunksize",       type=int, default=4)
    p.add_argument("--shard-size",      type=int, default=2000)

    # molecule limits
    p.add_argument("--max-samples", type=int, default=None,
                   help="Cap on molecules (for testing).")

    # Stage 1 params
    p.add_argument("--n-levels",          type=int,   default=4)
    p.add_argument("--smooth-sigma",      type=float, default=0.5)
    p.add_argument("--min-component-size",type=int,   default=10)

    # Stage 2+3 params
    p.add_argument("--tau-r",       type=float, default=1.0)
    p.add_argument("--tau-2",       type=float, default=1.5)
    p.add_argument("--local-knn-k", type=int,   default=12)
    p.add_argument("--chart-knn-k", type=int,   default=8)
    p.add_argument("--num-anchors", type=int,   default=8)
    p.add_argument("--mem-thresh",  type=int,   default=None,
                   help="Max vertices per manifold component for precomputing the "
                        "full dense geodesic distance matrix. Eliminates per-Dijkstra "
                        "fallback and greatly speeds up Stage 2. With 500 GB RAM, "
                        "15000–20000 is recommended.")

    # split
    p.add_argument("--split-col", default="scaffold_split",
                   choices=["scaffold_split", "random_split"],
                   help="CSV column to use for train/val/test split.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    data_dir    = args.data_dir
    cache_dir   = os.path.join(data_dir, "cache_manifold")
    packed_dir  = os.path.join(data_dir, "packed_stage3")
    split_path  = os.path.join(data_dir, "split.json")
    stats_path  = os.path.join(data_dir, "energy_stats.json")
    merged_path = os.path.join(
        cache_dir,
        f"all_nl{args.n_levels}_s{args.smooth_sigma:.2f}.pkl",
    )

    t_total = time.perf_counter()

    # ── Stage 1 ───────────────────────────────────────────────────────────────
    if args.skip_stage1 or os.path.exists(merged_path):
        print(f"[Stage 1] Skipping — {merged_path} already exists.")
    else:
        print(f"\n{'='*60}")
        print(f"[Stage 1] Loading PKL from {args.pkl} …")
        t0 = time.perf_counter()
        from ed2e.data.dataset import EDBenchPKLDataset
        dataset = EDBenchPKLDataset(
            pkl_path=args.pkl,
            cache_dir=cache_dir,
            max_samples=args.max_samples,
        )
        mol_ids = dataset.mol_ids
        print(f"  {len(mol_ids)} molecules loaded in {time.perf_counter()-t0:.1f}s")

        _run_stage1(
            mol_ids=mol_ids,
            raw=dataset._raw,
            cache_dir=cache_dir,
            n_levels=args.n_levels,
            smooth_sigma=args.smooth_sigma,
            min_comp_size=args.min_component_size,
            percentiles=None,
            workers=args.workers,
            chunksize=args.chunksize,
        )
        print(f"[Stage 1] Done in {time.perf_counter()-t0:.1f}s")

    # ── Stage 2+3 ─────────────────────────────────────────────────────────────
    manifest_path = os.path.join(packed_dir, "manifest.json")
    if args.skip_stage23 or os.path.exists(manifest_path):
        print(f"\n[Stage 2+3] Skipping — {manifest_path} already exists.")
    else:
        print(f"\n{'='*60}")
        print(f"[Stage 2+3] Loading manifold cache from {merged_path} …")
        t0 = time.perf_counter()
        from ed2e.data.manifold import _patch_legacy_pickle_modules
        _patch_legacy_pickle_modules()
        with open(merged_path, "rb") as f:
            manifold_merged: dict = pickle.load(f)

        mol_ids_s23 = sorted(manifold_merged.keys())
        if args.max_samples is not None:
            mol_ids_s23 = mol_ids_s23[: args.max_samples]
        print(f"  {len(mol_ids_s23)} molecules to process …")

        _run_stage23(
            mol_ids=mol_ids_s23,
            manifold_merged=manifold_merged,
            packed_dir=packed_dir,
            tau_r=args.tau_r,
            tau_2=args.tau_2,
            local_knn_k=args.local_knn_k,
            chart_knn_k=args.chart_knn_k,
            num_anchors=args.num_anchors,
            workers=args.workers,
            threads_per_proc=args.threads_per_proc,
            shard_size=args.shard_size,
            chunksize=args.chunksize,
            mem_thresh=args.mem_thresh,
        )
        print(f"[Stage 2+3] Done in {time.perf_counter()-t0:.1f}s")

    # ── Split JSON ────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"[Split] Building split.json from CSV column '{args.split_col}' …")
    splits = _build_split_json(
        csv_path=args.csv,
        packed_dir=packed_dir,
        split_col=args.split_col,
        out_path=split_path,
    )

    # ── Energy stats ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"[Stats] Computing per-target energy statistics from training split …")
    labels = load_energy_labels(args.csv)
    stats  = compute_energy_stats(labels, train_mol_ids=splits["train"])
    save_energy_stats(stats, stats_path)

    from ed2e.data.energy_stats import TARGET_NAMES
    print("  Per-target mean ± std (Hartree):")
    for name, m, s in zip(TARGET_NAMES, stats["mean"], stats["std"]):
        print(f"    {name:>22}: {m:+.4f} ± {s:.4f}")

    # ── Summary ───────────────────────────────────────────────────────────────
    total = time.perf_counter() - t_total
    print(f"\n{'='*60}")
    print(f"EDBench preprocessing complete in {total:.1f}s ({total/60:.1f}min)")
    print(f"  cache_manifold  → {cache_dir}")
    print(f"  packed_stage3   → {packed_dir}")
    print(f"  split.json      → {split_path}")
    print(f"  energy_stats    → {stats_path}")


if __name__ == "__main__":
    main()
