#!/usr/bin/env python3
"""
Preprocessing quality analysis for Stage 1–5 packed Stage3 data.

Computes per-molecule quality metrics, runs Pass/Fail checks, writes summary
statistics, and optionally generates distribution plots and geometric
visualizations for selected molecules.

Usage
-----
::

    # Statistical analysis on 1000 molecules
    python scripts/analyze_preprocessing.py \
        --packed-dir data/ed_energy_5w/packed_stage3 \
        --out-dir    data/ed_energy_5w/preprocess_analysis \
        --n-mols     1000 \
        --plot-mols  308 42 100

    # Quick Pass/Fail check only (no plots)
    python scripts/analyze_preprocessing.py \
        --packed-dir data/ed_energy_5w/packed_stage3 \
        --quick-check --n-mols 200
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import warnings
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))


# ══════════════════════════════════════════════════════════════════════════════
# Per-molecule metrics
# ══════════════════════════════════════════════════════════════════════════════

def _compute_mol_metrics(sample) -> Dict[str, Any]:
    """Extract quality metrics from a single Stage3Sample."""
    # Aliases
    membership      = np.asarray(sample.chart_membership)       # (M, 2): chart_idx, node_idx
    mem_weight      = np.asarray(sample.membership_weight)      # (M,)
    chart_es        = np.asarray(sample.chart_es_geom_static)   # (A, ?)
    anchor_mask     = np.asarray(sample.chart_anchor_mask)      # (A, num_anchors)
    jaccard         = np.asarray(sample.overlap_jaccard)        # (P,) may be empty
    inter_ei        = np.asarray(sample.inter_level_edge_index) # (2, E_inter)
    inter_ea        = np.asarray(sample.inter_level_edge_attr)  # (E_inter, 7)
    level_id        = np.asarray(sample.chart_frame_metadata["chart_level_id"])  # (A,)
    node_xyz        = np.asarray(sample.node_xyz)               # (N, 3)

    N = node_xyz.shape[0]
    A = chart_es.shape[0]
    E_inter = inter_ei.shape[1] if inter_ei.ndim == 2 else 0
    M = membership.shape[0]

    # 1. Node coverage: unique nodes present in any membership entry
    covered_nodes = np.unique(membership[:, 1]) if M > 0 else np.array([], dtype=np.int64)
    node_coverage = float(len(covered_nodes)) / max(N, 1)

    # 2. Average membership count per node
    avg_membership_per_node = float(M) / max(N, 1)

    # 3. Membership weight normalisation error: sum of weights per node should == 1
    if M > 0:
        node_ids = membership[:, 1]
        weight_sum = np.zeros(N, dtype=np.float64)
        np.add.at(weight_sum, node_ids, mem_weight.astype(np.float64))
        # Only check covered nodes
        covered_sums = weight_sum[covered_nodes]
        max_weight_error = float(np.abs(covered_sums - 1.0).max()) if len(covered_sums) else 0.0
    else:
        max_weight_error = 0.0

    # 4. Charts per level
    num_levels = int(level_id.max()) + 1 if A > 0 else 0
    charts_per_level = {}
    for lv in range(num_levels):
        charts_per_level[lv] = int(np.sum(level_id == lv))

    # 5. Anchor utilisation
    anchor_util = float(anchor_mask.mean()) if anchor_mask.size > 0 else float("nan")

    # 6. Overlap Jaccard stats
    if len(jaccard) > 0:
        overlap_jaccard_mean = float(jaccard.mean())
        overlap_jaccard_min  = float(jaccard.min())
        overlap_jaccard_max  = float(jaccard.max())
    else:
        overlap_jaccard_mean = float("nan")
        overlap_jaccard_min  = float("nan")
        overlap_jaccard_max  = float("nan")

    # 7. Inter-level edge statistics
    if E_inter > 0:
        inter_nn_dist_mean    = float(inter_ea[:, 0].mean())
        inter_normal_dev_mean = float(inter_ea[:, 1].mean())
        level_diff_vals       = set(inter_ea[:, 6].tolist())
        edge_index_max        = int(inter_ei.max())
        edge_index_min        = int(inter_ei.min())
    else:
        inter_nn_dist_mean    = float("nan")
        inter_normal_dev_mean = float("nan")
        level_diff_vals       = set()
        edge_index_max        = -1
        edge_index_min        = 0

    # 8. NaN checks
    has_nan_inter_ea = bool(np.isnan(inter_ea).any()) if E_inter > 0 else False
    has_nan_anchor   = bool(np.isnan(anchor_mask.astype(float)).any())

    # ── Pass/Fail ──
    passes: Dict[str, bool] = {
        "node_coverage":     node_coverage > 0.99,
        "weight_norm":       max_weight_error < 1e-4,
        "level_diff_valid":  level_diff_vals.issubset({-1.0, 1.0}) if E_inter > 0 else True,
        "edge_index_range":  (edge_index_max < A and edge_index_min >= 0) if E_inter > 0 else True,
        "no_nan_edge_attr":  not has_nan_inter_ea,
        "inter_edges_exist": E_inter > 0 if num_levels >= 2 else True,
    }

    return {
        "mol_id":                  str(getattr(sample, "mol_id", "")),
        "N":                       N,
        "A":                       A,
        "E_inter":                 E_inter,
        "M":                       M,
        "node_coverage":           node_coverage,
        "avg_membership_per_node": avg_membership_per_node,
        "max_weight_error":        max_weight_error,
        "num_levels":              num_levels,
        "charts_per_level":        charts_per_level,
        "anchor_util":             anchor_util,
        "overlap_jaccard_mean":    overlap_jaccard_mean,
        "overlap_jaccard_min":     overlap_jaccard_min,
        "overlap_jaccard_max":     overlap_jaccard_max,
        "inter_nn_dist_mean":      inter_nn_dist_mean,
        "inter_normal_dev_mean":   inter_normal_dev_mean,
        "passes":                  passes,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Aggregation & reporting
# ══════════════════════════════════════════════════════════════════════════════

_SCALAR_KEYS = [
    "N", "A", "E_inter", "M",
    "node_coverage", "avg_membership_per_node", "max_weight_error",
    "anchor_util", "overlap_jaccard_mean",
    "inter_nn_dist_mean", "inter_normal_dev_mean",
]

_CHECK_KEYS = [
    "node_coverage", "weight_norm", "level_diff_valid",
    "edge_index_range", "no_nan_edge_attr", "inter_edges_exist",
]


def _aggregate_stats(metrics_list: List[Dict]) -> Dict[str, Any]:
    """Compute summary statistics over all molecules."""
    result: Dict[str, Any] = {}
    for key in _SCALAR_KEYS:
        vals = [m[key] for m in metrics_list if not isinstance(m[key], float) or not np.isnan(m[key])]
        if not vals:
            result[key] = {}
            continue
        arr = np.array(vals, dtype=float)
        result[key] = {
            "mean": float(arr.mean()),
            "std":  float(arr.std()),
            "min":  float(arr.min()),
            "max":  float(arr.max()),
            "p5":   float(np.percentile(arr, 5)),
            "p25":  float(np.percentile(arr, 25)),
            "p50":  float(np.percentile(arr, 50)),
            "p75":  float(np.percentile(arr, 75)),
            "p95":  float(np.percentile(arr, 95)),
        }
    # Check pass rates
    result["check_pass_rates"] = {}
    for ck in _CHECK_KEYS:
        rate = float(np.mean([m["passes"][ck] for m in metrics_list]))
        result["check_pass_rates"][ck] = rate
    return result


def _write_per_mol_csv(metrics_list: List[Dict], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    check_cols = [f"pass_{k}" for k in _CHECK_KEYS]
    all_pass   = "all_pass"
    fieldnames = (
        ["mol_id"] + _SCALAR_KEYS + check_cols + [all_pass]
    )
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for m in metrics_list:
            row = {"mol_id": m["mol_id"]}
            for k in _SCALAR_KEYS:
                v = m[k]
                row[k] = "" if (isinstance(v, float) and np.isnan(v)) else v
            for ck in _CHECK_KEYS:
                row[f"pass_{ck}"] = int(m["passes"][ck])
            row[all_pass] = int(all(m["passes"].values()))
            writer.writerow(row)


def _write_flagged_json(metrics_list: List[Dict], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    flagged = []
    for m in metrics_list:
        failed = [k for k, v in m["passes"].items() if not v]
        if failed:
            flagged.append({"mol_id": m["mol_id"], "failed_checks": failed})
    with open(path, "w") as f:
        json.dump(flagged, f, indent=2)
    print(f"  Flagged molecules: {len(flagged)} / {len(metrics_list)}")


# ══════════════════════════════════════════════════════════════════════════════
# Distribution plots (matplotlib, optional)
# ══════════════════════════════════════════════════════════════════════════════

def _try_import_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        return None


def _make_plots(metrics_list: List[Dict], plots_dir: str) -> None:
    plt = _try_import_matplotlib()
    if plt is None:
        print("  matplotlib not available — skipping plots.")
        return

    os.makedirs(plots_dir, exist_ok=True)

    def _hist(values, title, xlabel, fname, bins=40, log_y=False):
        vals = [v for v in values if not (isinstance(v, float) and np.isnan(v))]
        if not vals:
            return
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(vals, bins=bins, color="steelblue", edgecolor="white", linewidth=0.3)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        if log_y:
            ax.set_yscale("log")
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, fname), dpi=120)
        plt.close(fig)

    def _boxplot_by_level(values_per_level: Dict[int, List[float]], title, ylabel, fname):
        levels = sorted(values_per_level.keys())
        data   = [values_per_level[lv] for lv in levels]
        if not any(data):
            return
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.boxplot(data, labels=[f"L{lv}" for lv in levels], patch_artist=True)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, fname), dpi=120)
        plt.close(fig)

    # 1. chart count histogram
    _hist([m["A"] for m in metrics_list], "Chart count per molecule", "Charts (A)", "chart_count.png")

    # 2. chart count by level (box)
    by_level: Dict[int, List[float]] = defaultdict(list)
    for m in metrics_list:
        for lv, cnt in m["charts_per_level"].items():
            by_level[lv].append(float(cnt))
    _boxplot_by_level(by_level, "Charts per level (box)", "Chart count", "chart_count_by_level.png")

    # 3. inter_nn_dist histogram
    _hist([m["inter_nn_dist_mean"] for m in metrics_list],
          "Inter-level mean NN distance", "Distance (Å)", "inter_nn_dist.png")

    # 4. inter normal deviation
    _hist([m["inter_normal_dev_mean"] for m in metrics_list],
          "Inter-level mean normal deviation", "1 − cos(n_src, n_tgt)", "inter_normal_dev.png")

    # 5. membership per node
    _hist([m["avg_membership_per_node"] for m in metrics_list],
          "Avg membership count / node", "Memberships per node", "membership_per_node.png")

    # 6. overlap Jaccard
    _hist([m["overlap_jaccard_mean"] for m in metrics_list],
          "Overlap Jaccard (mean per mol)", "Jaccard", "overlap_jaccard.png")

    # 7. anchor utilization
    _hist([m["anchor_util"] for m in metrics_list],
          "Anchor utilisation", "Fraction of anchors used", "anchor_utilization.png")

    # 8. node coverage histogram
    _hist([m["node_coverage"] for m in metrics_list],
          "Node coverage fraction", "Coverage", "node_coverage.png", bins=20)

    # 9. max weight error (log scale)
    _hist([m["max_weight_error"] for m in metrics_list],
          "Max membership weight error", "|Σw − 1| (max over nodes)", "weight_error.png",
          bins=40, log_y=True)

    # 10. chart size distribution by level
    chart_sizes_by_level: Dict[int, List[float]] = defaultdict(list)
    for m in metrics_list:
        if m["A"] == 0:
            continue
        for lv, cnt in m["charts_per_level"].items():
            if cnt > 0:
                # Approximate: A total / level charts
                chart_sizes_by_level[lv].append(float(m["N"]) / max(cnt, 1))
    _boxplot_by_level(chart_sizes_by_level, "Approx chart size by level",
                      "Nodes / Chart", "chart_size_by_level.png")

    print(f"  Plots saved to {plots_dir}")


# ══════════════════════════════════════════════════════════════════════════════
# Geometric visualisation (plotly, optional, per mol)
# ══════════════════════════════════════════════════════════════════════════════

def _visualize_mol(sample, out_path: str) -> None:
    """
    Interactive plotly scatter of:
     - Chart centres coloured by level
     - Intra-chart graph edges (chart_graph_edge_index if present)
     - Inter-level edges coloured by level_diff
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("  plotly not available — skipping geometric visualisation.")
        return

    level_id = np.asarray(sample.chart_frame_metadata["chart_level_id"])  # (A,)
    centers  = np.asarray(sample.chart_frame_metadata["chart_center"])    # (A, 3)
    A        = centers.shape[0]

    inter_ei = np.asarray(sample.inter_level_edge_index)  # (2, E)
    inter_ea = np.asarray(sample.inter_level_edge_attr)   # (E, 7)
    E_inter  = inter_ei.shape[1] if inter_ei.ndim == 2 else 0

    # ── Chart centre scatter ──
    palette = ["#4477AA", "#EE6677", "#228833", "#CCBB44"]
    traces = []
    num_levels = int(level_id.max()) + 1 if A > 0 else 0
    for lv in range(num_levels):
        mask = level_id == lv
        if not mask.any():
            continue
        c = centers[mask]
        traces.append(go.Scatter3d(
            x=c[:, 0], y=c[:, 1], z=c[:, 2],
            mode="markers",
            marker=dict(size=4, color=palette[lv % len(palette)], opacity=0.8),
            name=f"Level {lv}",
        ))

    # ── Intra-chart edges (chart_graph_edge_index) ──
    if hasattr(sample, "chart_graph_edge_index"):
        cg_ei = np.asarray(sample.chart_graph_edge_index)  # (2, E_intra)
        if cg_ei.shape[1] > 0:
            edge_x, edge_y, edge_z = [], [], []
            for e in range(cg_ei.shape[1]):
                s, d = int(cg_ei[0, e]), int(cg_ei[1, e])
                for coord_list, coord in zip([edge_x, edge_y, edge_z],
                                             [centers[:, 0], centers[:, 1], centers[:, 2]]):
                    coord_list += [float(coord[s]), float(coord[d]), None]
            traces.append(go.Scatter3d(
                x=edge_x, y=edge_y, z=edge_z,
                mode="lines",
                line=dict(color="rgba(100,100,100,0.3)", width=1),
                name="Intra edges",
            ))

    # ── Inter-level edges ──
    if E_inter > 0:
        level_diff = inter_ea[:, 6]  # ±1
        for ld, color in [(-1.0, "rgba(255,100,50,0.6)"), (1.0, "rgba(50,180,255,0.6)")]:
            mask_e = np.abs(level_diff - ld) < 0.5
            if not mask_e.any():
                continue
            ei_sub = inter_ei[:, mask_e]
            edge_x, edge_y, edge_z = [], [], []
            for e in range(ei_sub.shape[1]):
                s, d = int(ei_sub[0, e]), int(ei_sub[1, e])
                for coord_list, coord in zip([edge_x, edge_y, edge_z],
                                             [centers[:, 0], centers[:, 1], centers[:, 2]]):
                    coord_list += [float(coord[s]), float(coord[d]), None]
            label = "k←k+1" if ld < 0 else "k+1←k"
            traces.append(go.Scatter3d(
                x=edge_x, y=edge_y, z=edge_z,
                mode="lines",
                line=dict(color=color, width=2),
                name=f"Inter {label}",
            ))

    mol_id = getattr(sample, "mol_id", "?")
    fig = go.Figure(data=traces)
    fig.update_layout(
        title=f"Molecule {mol_id}: chart graph (A={A}, E_inter={E_inter})",
        scene=dict(xaxis_title="x (Å)", yaxis_title="y (Å)", zaxis_title="z (Å)"),
        margin=dict(l=0, r=0, b=0, t=40),
    )
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.write_html(out_path)
    print(f"    Chart graph: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="EDBench preprocessing quality analysis.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--packed-dir", required=True,
                   help="Path to packed_stage3 directory.")
    p.add_argument("--out-dir",    default=None,
                   help="Output directory for stats/CSV/plots. Defaults to "
                        "<packed-dir>/../preprocess_analysis.")
    p.add_argument("--n-mols",     type=int, default=500,
                   help="Number of molecules to analyse (random sample if "
                        "larger than available).")
    p.add_argument("--plot-mols",  nargs="*", default=None,
                   help="Specific mol_ids to generate 3D chart-graph HTML for.")
    p.add_argument("--quick-check", action="store_true",
                   help="Run Pass/Fail checks only, skip all plots.")
    p.add_argument("--seed",       type=int, default=42,
                   help="Random seed for molecule sampling.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    out_dir = args.out_dir or os.path.join(
        os.path.dirname(args.packed_dir.rstrip("/\\")), "preprocess_analysis"
    )
    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)

    # ── Load dataset ──────────────────────────────────────────────────────────
    print(f"\nLoading packed dataset from {args.packed_dir} …")
    from ed2e.data.stage3_packed import Stage3PackedDataset
    dataset = Stage3PackedDataset(args.packed_dir)
    all_mol_ids = dataset.mol_ids
    print(f"  {len(all_mol_ids)} molecules available.")

    rng = np.random.default_rng(args.seed)
    n   = min(args.n_mols, len(all_mol_ids))
    if n < len(all_mol_ids):
        chosen_mol_ids = rng.choice(all_mol_ids, size=n, replace=False).tolist()
        # Build a mol_id→index map for quick access
        mol_to_idx = {m: i for i, m in enumerate(all_mol_ids)}
        indices = [mol_to_idx[m] for m in chosen_mol_ids]
    else:
        indices = list(range(len(all_mol_ids)))
        chosen_mol_ids = all_mol_ids

    print(f"  Analysing {n} molecules …")

    # ── Compute metrics ───────────────────────────────────────────────────────
    metrics_list: List[Dict] = []
    for idx in tqdm(indices, desc="Metrics", unit="mol", dynamic_ncols=True):
        sample = dataset[idx]
        try:
            m = _compute_mol_metrics(sample)
        except Exception as exc:
            tqdm.write(f"  [ERROR] mol={getattr(sample, 'mol_id', idx)}: {exc}")
            continue
        metrics_list.append(m)

    if not metrics_list:
        print("ERROR: no metrics computed. Check dataset.")
        sys.exit(1)

    # ── Summary stats ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Summary over {len(metrics_list)} molecules:")
    stats = _aggregate_stats(metrics_list)

    for key in _SCALAR_KEYS:
        if key not in stats or not stats[key]:
            continue
        s = stats[key]
        print(f"  {key:>30}:  mean={s['mean']:.4g}  std={s['std']:.4g}"
              f"  [{s['min']:.4g}, {s['max']:.4g}]")

    print(f"\nPass/Fail rates:")
    all_good = True
    for ck, rate in stats["check_pass_rates"].items():
        icon = "✓" if rate == 1.0 else ("⚠" if rate > 0.95 else "✗")
        print(f"  {icon} {ck:>22}: {rate*100:.1f}%")
        if rate < 1.0:
            all_good = False
    if all_good:
        print("  All checks passed for all molecules.")

    # ── Write outputs ─────────────────────────────────────────────────────────
    summary_path = os.path.join(out_dir, "summary_stats.json")
    with open(summary_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\n  summary_stats.json → {summary_path}")

    csv_path = os.path.join(out_dir, "per_mol_stats.csv")
    _write_per_mol_csv(metrics_list, csv_path)
    print(f"  per_mol_stats.csv  → {csv_path}")

    flagged_path = os.path.join(out_dir, "flagged_mols.json")
    _write_flagged_json(metrics_list, flagged_path)
    print(f"  flagged_mols.json  → {flagged_path}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    if not args.quick_check:
        print(f"\nGenerating distribution plots …")
        _make_plots(metrics_list, plots_dir)

    # ── Per-mol geometric visualisation ──────────────────────────────────────
    if args.plot_mols and not args.quick_check:
        print(f"\nGenerating geometric visualisations for {args.plot_mols} …")
        mol_to_idx = {m: i for i, m in enumerate(all_mol_ids)}
        vis_dir    = os.path.join(out_dir, "plots")
        for mol_id in args.plot_mols:
            if mol_id not in mol_to_idx:
                print(f"  mol_id {mol_id!r} not in dataset, skipping.")
                continue
            sample = dataset[mol_to_idx[mol_id]]
            html_path = os.path.join(vis_dir, f"mol_{mol_id}_chart_graph.html")
            try:
                _visualize_mol(sample, html_path)
            except Exception as exc:
                print(f"  [ERROR] visualise mol {mol_id}: {exc}")

    print(f"\nDone. Output: {out_dir}")


if __name__ == "__main__":
    main()
