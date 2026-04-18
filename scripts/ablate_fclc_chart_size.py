#!/usr/bin/env python3
"""
FCLC chart-size ablation experiment.

Sweeps a grid of (tau_r, tau_2) hyper-parameter combinations on a fixed subset
of molecules and reports chart-size statistics, helping the user choose
parameters that yield charts of a moderate, well-defined size.

Usage
-----
python scripts/ablate_fclc_chart_size.py \\
    --manifold-pkl data/ed_energy_5w/cache_manifold/all_nl4_s0.50.pkl \\
    --n-mols 50 --output-csv data/fclc_ablation.csv

# Custom grid
python scripts/ablate_fclc_chart_size.py \\
    --manifold-pkl ... \\
    --tau-r 0.75 1.0 1.25 --tau-2 1.0 1.5 2.0 \\
    --n-mols 50 --output-csv data/fclc_ablation.csv

Output columns
--------------
  tau_r, tau_2,
  n_charts_mean, n_charts_std,
  chart_size_mean, chart_size_median, chart_size_p10, chart_size_p90,
  coverage_rate,   (fraction of components 100 % covered)
  wall_time_s
"""
from __future__ import annotations

import argparse
import csv
import os
import pickle
import sys
import time
from typing import List, Tuple

import numpy as np
from tqdm import tqdm

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from ed2e.data.manifold import _patch_legacy_pickle_modules   # noqa: E402
from ed2e.data.fclc import build_fclc_levels                  # noqa: E402


# ---------------------------------------------------------------------------
# Statistics helper
# ---------------------------------------------------------------------------

def _run_one(
    mol_ids:       List[str],
    manifold_dict: dict,
    tau_r:         float,
    tau_2:         float,
    min_chart:     int = 5,
) -> dict:
    """Run build_fclc_levels on a batch and collect statistics."""
    all_chart_sizes: List[int] = []
    n_charts_per_mol: List[int] = []
    covered_total = 0
    covered_fully = 0
    t0 = time.perf_counter()

    for mol_id in tqdm(mol_ids, desc=f"τ_r={tau_r:.2f} τ_2={tau_2:.2f}",
                       unit="mol", leave=False, dynamic_ncols=True):
        manifold_levels = manifold_dict.get(mol_id)
        if manifold_levels is None:
            continue
        try:
            fclc_levels = build_fclc_levels(
                manifold_levels,
                tau_r=tau_r, tau_2=tau_2,
                min_chart_size=min_chart,
                compute_inter=False,
            )
        except Exception:
            continue

        mol_charts = sum(len(lv.charts) for lv in fclc_levels)
        n_charts_per_mol.append(mol_charts)

        for lv_idx, (ml, fl) in enumerate(
                zip(manifold_levels, fclc_levels)):
            for comp in ml.components:
                V = len(comp.verts)
                if V == 0:
                    continue
                # Collect coverage for this component
                covered_verts: set = set()
                for ch in fl.charts:
                    if ch.component_id == comp.component_id:
                        covered_verts.update(ch.vert_indices.tolist())
                covered_total += 1
                if len(covered_verts) >= V:
                    covered_fully += 1

        for lv in fclc_levels:
            for ch in lv.charts:
                all_chart_sizes.append(len(ch.vert_indices))

    wall = time.perf_counter() - t0
    sizes = np.array(all_chart_sizes, dtype=np.float64)
    n_per_mol = np.array(n_charts_per_mol, dtype=np.float64)

    return {
        "tau_r": tau_r,
        "tau_2": tau_2,
        "n_charts_mean": float(n_per_mol.mean()) if len(n_per_mol) else 0.0,
        "n_charts_std":  float(n_per_mol.std())  if len(n_per_mol) else 0.0,
        "chart_size_mean":   float(sizes.mean())               if len(sizes) else 0.0,
        "chart_size_median": float(np.median(sizes))           if len(sizes) else 0.0,
        "chart_size_p10":    float(np.percentile(sizes, 10))   if len(sizes) else 0.0,
        "chart_size_p90":    float(np.percentile(sizes, 90))   if len(sizes) else 0.0,
        "coverage_rate": covered_fully / max(covered_total, 1),
        "wall_time_s": wall,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sweep tau_r × tau_2 to characterise FCLC chart sizes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--manifold-pkl", required=True,
                   help="Path to merged Stage-1 manifold pkl.")
    p.add_argument("--n-mols", type=int, default=50,
                   help="Number of molecules to use (deterministic prefix).")
    p.add_argument("--tau-r", type=float, nargs="+",
                   default=[0.5, 0.75, 1.0, 1.25, 1.5],
                   help="tau_r values to sweep.")
    p.add_argument("--tau-2", type=float, nargs="+",
                   default=[1.0, 1.5, 2.0],
                   help="tau_2 values to sweep.")
    p.add_argument("--min-chart-size", type=int, default=5)
    p.add_argument("--output-csv", default="fclc_ablation.csv",
                   help="Output CSV file path.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    print(f"Loading manifold PKL from {args.manifold_pkl} …")
    _patch_legacy_pickle_modules()
    with open(args.manifold_pkl, "rb") as f:
        manifold_dict: dict = pickle.load(f)

    mol_ids = list(manifold_dict.keys())[:args.n_mols]
    print(f"  Using {len(mol_ids)} molecules.")

    # Build parameter grid (deterministic order)
    grid: List[Tuple[float, float]] = [
        (tr, t2) for tr in sorted(args.tau_r) for t2 in sorted(args.tau_2)
    ]
    print(f"  Grid size: {len(grid)} combinations "
          f"(tau_r × tau_2 = {len(args.tau_r)} × {len(args.tau_2)})")

    rows = []
    for tau_r, tau_2 in grid:
        stats = _run_one(mol_ids, manifold_dict, tau_r, tau_2,
                         min_chart=args.min_chart_size)
        rows.append(stats)
        print(
            f"  τ_r={tau_r:.2f}  τ_2={tau_2:.2f} → "
            f"charts/mol={stats['n_charts_mean']:.1f}  "
            f"size_median={stats['chart_size_median']:.0f}  "
            f"[p10={stats['chart_size_p10']:.0f}, p90={stats['chart_size_p90']:.0f}]  "
            f"cov={stats['coverage_rate']*100:.1f}%  "
            f"t={stats['wall_time_s']:.1f}s"
        )

    # Write CSV
    fieldnames = list(rows[0].keys())
    with open(args.output_csv, "w", newline="") as csvf:
        writer = csv.DictWriter(csvf, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved ablation results → {args.output_csv}")


if __name__ == "__main__":
    main()
