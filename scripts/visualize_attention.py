#!/usr/bin/env python3
"""
BERT-style attention visualisation for ED2E.

For each specified molecule, generates:
  1. Interactive plotly 3D scatter of chart centres coloured by attention weight
     (one HTML per target × molecule combination)
  2. CSV export: mol_id, chart_idx, level_id, x, y, z, attn_t0…t5
  3. Aggregate statistics plots:
       - attention vs level (box plot per target)
       - attention vs chart size (scatter per target)

Usage
-----
::

    python scripts/visualize_attention.py \\
        --checkpoint runs/stage6_k3/best.pt \\
        --packed-dir data/ed_energy_5w/packed_stage3 \\
        --mol-ids 308 42 100 \\
        --out-dir  figs/attention

    # Use first N molecules from the packed dataset:
    python scripts/visualize_attention.py \\
        --checkpoint runs/stage6_k3/best.pt \\
        --packed-dir data/ed_energy_5w/packed_stage3 \\
        --n-mols 8 --out-dir figs/attention
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _try_plotly():
    try:
        import plotly.graph_objects as go
        import plotly.subplots as sp
        return go, sp
    except ImportError:
        return None, None


def _try_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Per-molecule visualisation
# ══════════════════════════════════════════════════════════════════════════════

def _visualize_one_mol(
    mol_idx: int,
    mol_id: str,
    attn_mol:  torch.Tensor,      # (T, H, A_mol)  attention weights
    centers:   np.ndarray,         # (A_mol, 3)
    level_ids: np.ndarray,         # (A_mol,) int
    target_names: List[str],
    out_dir: str,
) -> None:
    go, _ = _try_plotly()
    if go is None:
        print("  plotly not available — skipping 3D scatter.")
        return

    T = len(target_names)
    # Average across heads for visualisation
    attn_mean = attn_mol.mean(dim=1).cpu().numpy()   # (T, A_mol)

    palette = ["#4477AA", "#EE6677", "#228833", "#CCBB44", "#AA3377", "#BBBBBB"]
    num_levels = int(level_ids.max()) + 1 if len(level_ids) > 0 else 1

    for t, name in enumerate(target_names):
        weights = attn_mean[t]   # (A_mol,)
        traces = []
        for lv in range(num_levels):
            mask = level_ids == lv
            if not mask.any():
                continue
            c = centers[mask]
            w = weights[mask]
            traces.append(go.Scatter3d(
                x=c[:, 0], y=c[:, 1], z=c[:, 2],
                mode="markers",
                marker=dict(
                    size=6,
                    color=w,
                    colorscale="Viridis",
                    colorbar=dict(title="Attn") if lv == 0 else None,
                    opacity=0.85,
                    cmin=float(attn_mean[t].min()),
                    cmax=float(attn_mean[t].max()),
                ),
                name=f"Level {lv}",
                text=[f"L{lv} attn={wi:.4f}" for wi in w.tolist()],
            ))

        fig = go.Figure(data=traces)
        fig.update_layout(
            title=f"Mol {mol_id} | Target: {name} | attn per chart",
            scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z"),
            margin=dict(l=0, r=0, b=0, t=40),
        )
        html_path = os.path.join(out_dir, f"mol_{mol_id}_{name.replace(' ', '_')}.html")
        fig.write_html(html_path)
        print(f"    {html_path}")


# ══════════════════════════════════════════════════════════════════════════════
# CSV export
# ══════════════════════════════════════════════════════════════════════════════

def _export_csv(
    rows: List[Dict],
    target_names: List[str],
    out_path: str,
) -> None:
    attn_cols = [f"attn_{n.replace(' ', '_')}" for n in target_names]
    fieldnames = ["mol_id", "chart_idx", "level_id", "x", "y", "z"] + attn_cols
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  CSV: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Aggregate statistics plots
# ══════════════════════════════════════════════════════════════════════════════

def _aggregate_plots(
    all_rows: List[Dict],
    target_names: List[str],
    out_dir: str,
) -> None:
    plt = _try_matplotlib()
    if plt is None:
        print("  matplotlib not available — skipping aggregate plots.")
        return

    attn_cols = [f"attn_{n.replace(' ', '_')}" for n in target_names]
    T = len(target_names)

    # attention vs level (box plot)
    from collections import defaultdict
    by_level: Dict[int, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))
    for row in all_rows:
        lv = int(row["level_id"])
        for t, col in enumerate(attn_cols):
            by_level[lv][t].append(float(row[col]))

    levels = sorted(by_level.keys())
    fig, axes = plt.subplots(1, T, figsize=(4 * T, 4), sharey=False)
    if T == 1:
        axes = [axes]
    for t, (ax, name) in enumerate(zip(axes, target_names)):
        data   = [by_level[lv][t] for lv in levels]
        labels = [f"L{lv}" for lv in levels]
        ax.boxplot(data, labels=labels, patch_artist=True)
        ax.set_title(name, fontsize=8)
        ax.set_xlabel("Level")
        ax.set_ylabel("Attention weight" if t == 0 else "")
    fig.suptitle("Attention weight vs manifold level", fontsize=10)
    fig.tight_layout()
    path = os.path.join(out_dir, "attn_by_level.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  {path}")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ED2E attention visualisation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint",  required=True, help="Path to .pt checkpoint")
    p.add_argument("--packed-dir",  required=True, help="packed_stage3 directory")
    p.add_argument("--out-dir",     required=True, help="Output directory")

    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--mol-ids",  nargs="+", help="Specific mol_ids to visualise")
    g.add_argument("--n-mols",   type=int,  help="Take first N molecules from dataset")

    p.add_argument("--device",   default="cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    from ed2e.data.stage3_local import collate_stage3_samples
    from ed2e.data.stage3_packed import Stage3PackedDataset
    from ed2e.model.ed2e import ED2EModel

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"Loading checkpoint {args.checkpoint} …")
    ckpt  = torch.load(args.checkpoint, map_location="cpu")
    model = ED2EModel(ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device)
    model.eval()
    target_names = list(model.cfg.target_names)
    T = model.cfg.num_targets
    print(f"  {T} targets: {target_names}")

    # ── Load samples ──────────────────────────────────────────────────────────
    print(f"Loading packed dataset from {args.packed_dir} …")
    dataset   = Stage3PackedDataset(args.packed_dir)
    all_mids  = dataset.mol_ids
    mid2idx   = {m: i for i, m in enumerate(all_mids)}

    if args.mol_ids:
        sel_mids = [str(m) for m in args.mol_ids]
        missing  = [m for m in sel_mids if m not in mid2idx]
        if missing:
            print(f"WARNING: mol_ids not found: {missing}")
            sel_mids = [m for m in sel_mids if m in mid2idx]
    else:
        sel_mids = all_mids[: args.n_mols]

    if not sel_mids:
        print("No valid molecules selected. Exiting.")
        sys.exit(1)
    print(f"  Visualising {len(sel_mids)} molecules …")

    all_rows: List[Dict] = []

    for mol_id in sel_mids:
        print(f"\n  mol_id={mol_id}")
        sample  = dataset[mid2idx[mol_id]]
        batch   = collate_stage3_samples([sample], device=device)

        with torch.no_grad():
            out = model(batch, return_attn=True)

        attn        = out["attn_weights"]       # (T, H, A_batch)
        chart_batch = out["chart_batch"]        # (A_batch,)
        level_id    = out["chart_level_id"]     # (A_batch,)
        centers     = out["chart_center"]       # (A_batch, 3)

        # Since B=1, all charts belong to mol_idx=0
        mask       = chart_batch == 0
        attn_mol   = attn[:, :, mask]           # (T, H, A_mol)
        centers_np = centers[mask].cpu().numpy() # (A_mol, 3)
        level_np   = level_id[mask].cpu().numpy().astype(int)  # (A_mol,)
        A_mol      = int(mask.sum())

        # attn averaged over heads: (T, A_mol)
        attn_mean_np = attn_mol.mean(dim=1).cpu().numpy()

        # ── 3D visualisation ──────────────────────────────────────────────────
        _visualize_one_mol(
            mol_idx=0,
            mol_id=mol_id,
            attn_mol=attn_mol,
            centers=centers_np,
            level_ids=level_np,
            target_names=target_names,
            out_dir=args.out_dir,
        )

        # ── Accumulate CSV rows ────────────────────────────────────────────────
        for a in range(A_mol):
            row: Dict = {
                "mol_id":    mol_id,
                "chart_idx": a,
                "level_id":  int(level_np[a]),
                "x":         float(centers_np[a, 0]),
                "y":         float(centers_np[a, 1]),
                "z":         float(centers_np[a, 2]),
            }
            for t, name in enumerate(target_names):
                row[f"attn_{name.replace(' ', '_')}"] = float(attn_mean_np[t, a])
            all_rows.append(row)

    # ── Export CSV ────────────────────────────────────────────────────────────
    csv_path = os.path.join(args.out_dir, "attention_data.csv")
    _export_csv(all_rows, target_names, csv_path)

    # ── Aggregate plots ───────────────────────────────────────────────────────
    print(f"\nGenerating aggregate plots …")
    _aggregate_plots(all_rows, target_names, args.out_dir)

    print(f"\nDone. Outputs in {args.out_dir}")


if __name__ == "__main__":
    main()
