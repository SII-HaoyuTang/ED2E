#!/usr/bin/env python3
"""
Point-cloud benchmark analysis script (PointMetaBase-S-X3D).

Phases:
  A) Training curves from metrics.json  (skipped gracefully if not present)
     - train_loss (log scale) + val_mean_mae vs epoch + learning rate
  B) Full-dataset inference on all splits, collecting:
     - preds / targets (Hartree)
     - per-molecule point-cloud statistics (mean_rho, extent, …)
  C) Pred vs True scatter plots  (combined + per-split)
     – adds Spearman ρ to annotation box (voxel script only shows Pearson r)
  D) Residual deep-dives  (point-cloud-specific, not in voxel script)
     D1 – residual histograms with KDE per energy target
     D2 – |error| vs mean electron density per target
     D3 – |error| vs spatial extent (bounding-box diagonal) per target
     D4 – 6×6 target-error correlation heat map
  E) Cumulative error distribution (CDF) per target — test split
  F) analysis_summary.json  (superset of voxel format; adds Spearman ρ)

Checkpoint compatibility:
  Old format (repro-x3d-v1): keys "model", "args", "target_mean", "target_std"
  New format (≥ unified arch): keys "model_state", "config"

Usage:
  python -m benchmark.point_cloude.analyze_results \\
      --run-dir  benchmark/outputs/checkpoints/benchmark/repro-x3d-v1 \\
      --pkl-path data/ed_energy_5w/processed/mol_EDthresh0.05_data.pkl \\
      --csv-path data/ed_energy_5w/raw/ed_energy_5w.csv \\
      --cache-dir data/ed_energy_5w/cache_fps \\
      --output-dir benchmark/outputs/analysis/pointcloud \\
      --device cpu --batch-size 32
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Optional

import numpy as np
import torch
from scipy.stats import spearmanr
from torch.utils.data import DataLoader
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from benchmark.point_cloude.data.energy_dataset import (
    EDBenchEnergyDataset,
    energy_collate_fn,
)
from benchmark.point_cloude.models.backbone.pointmetabase_x3d import PointMetaBaseX3D

LABEL_NAMES = [
    "E1_Final", "E2_NucRepul", "E3_OneElec",
    "E4_TwoElec", "E5_XC", "E6_Total",
]

_TARGET_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c",
    "#d62728", "#9467bd", "#8c564b",
]


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _save_json(path: str, obj: object) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


# ---------------------------------------------------------------------------
# Checkpoint compatibility
# ---------------------------------------------------------------------------

def _load_compat(
    run_dir: str,
) -> tuple[dict, dict, np.ndarray, np.ndarray, int, list | None]:
    """
    Load best.pt and return:
        (state_dict, model_cfg, target_mean, target_std, epoch, history_or_None)

    Handles both old format ("model"/"args") and new format ("model_state"/"config").
    """
    ckpt_path = os.path.join(run_dir, "best.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if "model_state" in ckpt:            # new unified arch format
        state_dict  = ckpt["model_state"]
        cfg         = ckpt["config"]
        target_mean = np.array(cfg["target_mean"], dtype=np.float32)
        target_std  = np.array(cfg["target_std"],  dtype=np.float32)
    else:                                 # legacy format (repro-x3d-v1)
        state_dict  = ckpt["model"]
        cfg         = ckpt["args"]
        target_mean = np.asarray(ckpt["target_mean"], dtype=np.float32)
        target_std  = np.asarray(ckpt["target_std"],  dtype=np.float32)

    epoch = int(ckpt["epoch"])

    met_path = os.path.join(run_dir, "metrics.json")
    history: list | None = None
    if os.path.exists(met_path):
        with open(met_path) as f:
            history = json.load(f)["history"]

    return state_dict, cfg, target_mean, target_std, epoch, history


def _build_model(cfg: dict) -> PointMetaBaseX3D:
    return PointMetaBaseX3D(
        in_channels=4,
        width=cfg.get("width", 32),
        num_targets=6,
        npoint_start=cfg.get("npoint", cfg.get("npoint_start", 2048)),
        radius=cfg.get("radius", 0.15),
        radius_mult=cfg.get("radius_mult", 1.5),
        K=cfg.get("K", 32),
        mlp_layers=[512, 256],
        dropout=cfg.get("dropout", 0.5),
    )


# ---------------------------------------------------------------------------
# Phase A — Training curves
# ---------------------------------------------------------------------------

def plot_training_curves(
    history: list[dict],
    best_epoch: int,
    output_path: str,
) -> None:
    epochs     = [h["epoch"]        for h in history]
    train_loss = [h["train_loss"]   for h in history]
    val_mae    = [h["val_mean_mae"] for h in history]
    lr         = [h["lr"]           for h in history]

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle("PointMetaBase-S-X3D — Training Curves", fontsize=14, fontweight="bold")

    color_loss = "#2196F3"
    color_mae  = "#F44336"

    ax1 = axes[0]
    l1, = ax1.semilogy(epochs, train_loss, color=color_loss, lw=1.5, label="Train Loss (MSE, log)")
    ax1.set_ylabel("Train Loss (log scale)", color=color_loss)
    ax1.tick_params(axis="y", labelcolor=color_loss)

    ax1r = ax1.twinx()
    l2, = ax1r.plot(epochs, val_mae, color=color_mae, lw=1.5, label="Val Mean MAE")
    ax1r.set_ylabel("Val Mean MAE (Hartree)", color=color_mae)
    ax1r.tick_params(axis="y", labelcolor=color_mae)

    best_mae = min(val_mae)
    ax1.axvline(best_epoch, color="gray", ls="--", lw=1, alpha=0.7)
    ax1r.scatter([best_epoch], [best_mae], color=color_mae, zorder=5, s=50)
    ax1r.annotate(
        f"Best ep={best_epoch}\nMAE={best_mae:.2f}",
        xy=(best_epoch, best_mae),
        xytext=(best_epoch + max(len(epochs) // 20, 5), best_mae * 1.05),
        fontsize=8,
        arrowprops=dict(arrowstyle="->", lw=0.8),
        color="gray",
    )
    ax1.set_xlabel("Epoch")
    ax1.set_xlim(1, max(epochs))
    ax1.legend([l1, l2], [l1.get_label(), l2.get_label()], loc="upper right", fontsize=9)

    ax2 = axes[1]
    ax2.plot(epochs, lr, color="#4CAF50", lw=1.5)
    ax2.axvline(best_epoch, color="gray", ls="--", lw=1, alpha=0.7)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Learning Rate")
    ax2.set_xlim(1, max(epochs))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Phase A] Saved training curves → {output_path}")


# ---------------------------------------------------------------------------
# Phase B — Full-dataset inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    loader: DataLoader,
    target_mean: np.ndarray,
    target_std: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """
    Returns:
        preds      (N, 6)  Hartree
        targets    (N, 6)  Hartree
        mol_stats  dict of (N,) arrays: mean_rho, std_rho, max_rho, extent
    """
    model.eval()
    all_preds, all_targets = [], []
    all_mean_rho, all_std_rho, all_max_rho, all_extent = [], [], [], []

    t_mean = torch.from_numpy(target_mean).to(device)
    t_std  = torch.from_numpy(target_std).to(device)

    for batch in loader:
        pc     = batch["point_cloud"].to(device)   # (B, N, 4)
        target = batch["energies"].to(device)       # (B, 6)

        pred_norm = model(pc)
        pred = pred_norm * t_std + t_mean

        # Per-molecule point-cloud statistics (computed on CPU)
        pc_cpu    = pc.cpu()
        xyz       = pc_cpu[:, :, :3]                              # (B, N, 3)
        rho       = pc_cpu[:, :, 3]                               # (B, N)
        xyz_range = xyz.max(dim=1).values - xyz.min(dim=1).values # (B, 3)

        all_mean_rho.append(rho.mean(dim=1).numpy())
        all_std_rho.append(rho.std(dim=1).numpy())
        all_max_rho.append(rho.max(dim=1).values.numpy())
        all_extent.append(xyz_range.norm(dim=1).numpy())

        all_preds.append(pred.cpu().numpy())
        all_targets.append(target.cpu().numpy())

    preds   = np.concatenate(all_preds,   axis=0)
    targets = np.concatenate(all_targets, axis=0)
    mol_stats = {
        "mean_rho": np.concatenate(all_mean_rho),
        "std_rho":  np.concatenate(all_std_rho),
        "max_rho":  np.concatenate(all_max_rho),
        "extent":   np.concatenate(all_extent),
    }
    return preds, targets, mol_stats


# ---------------------------------------------------------------------------
# Phase C — Scatter plots (pred vs true)
# ---------------------------------------------------------------------------

_SPLIT_STYLES: dict[str, dict] = {
    "train": dict(color="#999999", alpha=0.12, s=2, marker=".",  zorder=1, label="train"),
    "valid": dict(color="#1565C0", alpha=0.45, s=6, marker="^",  zorder=2, label="valid"),
    "val":   dict(color="#1565C0", alpha=0.45, s=6, marker="^",  zorder=2, label="valid"),
    "test":  dict(color="#E64A19", alpha=0.55, s=6, marker="o",  zorder=3, label="test"),
}

_DRAW_ORDER = ("train", "valid", "val", "test")


def _draw_one_scatter_panel(
    ax: plt.Axes,
    results: dict[str, tuple[np.ndarray, np.ndarray]],
    col_idx: int,
    name: str,
    splits_to_draw: list[str],
    annotation_split: Optional[str] = "test",
) -> None:
    all_true_vals = []
    for split in _DRAW_ORDER:
        if split not in splits_to_draw or split not in results:
            continue
        preds, targets = results[split][:2]
        t = targets[:, col_idx]
        p = preds[:, col_idx]
        sty = _SPLIT_STYLES[split]
        ax.scatter(t, p, color=sty["color"], alpha=sty["alpha"], s=sty["s"],
                   marker=sty["marker"], zorder=sty["zorder"],
                   linewidths=0, rasterized=True)
        all_true_vals.append(t)

    if not all_true_vals:
        return

    all_t = np.concatenate(all_true_vals)
    lo, hi = all_t.min(), all_t.max()
    margin = (hi - lo) * 0.05
    diag = np.array([lo - margin, hi + margin])
    ax.plot(diag, diag, color="#D32F2F", ls="--", lw=1.2, zorder=10)
    ax.set_xlim(diag[0], diag[1])
    ax.set_ylim(diag[0], diag[1])

    if annotation_split and annotation_split in results:
        t_ann = results[annotation_split][1][:, col_idx]
        p_ann = results[annotation_split][0][:, col_idx]
        mae  = float(np.abs(p_ann - t_ann).mean())
        rmse = float(np.sqrt(((p_ann - t_ann) ** 2).mean()))
        r    = float(np.corrcoef(t_ann, p_ann)[0, 1])
        sr   = float(spearmanr(t_ann, p_ann)[0])
        lbl  = _SPLIT_STYLES[annotation_split]["label"]
        ax.text(
            0.04, 0.96,
            f"{lbl} MAE={mae:.2f}\n{lbl} RMSE={rmse:.2f}\n"
            f"{lbl} r={r:.3f}\n{lbl} ρ={sr:.3f}",
            transform=ax.transAxes,
            va="top", ha="left", fontsize=7,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
        )

    ax.set_xlabel("True (Hartree)", fontsize=8)
    ax.set_ylabel("Predicted (Hartree)", fontsize=8)
    ax.set_title(name, fontsize=9, fontweight="bold")
    ax.set_aspect("equal", adjustable="datalim")


def _make_legend_elements(splits_shown: list[str]) -> list:
    seen_labels: set[str] = set()
    elements = []
    for s in _DRAW_ORDER:
        if s not in splits_shown:
            continue
        sty = _SPLIT_STYLES[s]
        if sty["label"] in seen_labels:
            continue
        seen_labels.add(sty["label"])
        elements.append(
            Line2D([0], [0], marker=sty["marker"], color="w",
                   markerfacecolor=sty["color"], markeredgewidth=0,
                   markersize=7, label=sty["label"], alpha=0.9)
        )
    elements.append(Line2D([0], [0], color="#D32F2F", ls="--", lw=1.2, label="y = x"))
    return elements


def plot_scatter(
    results: dict[str, tuple],
    output_dir: str,
    base_name: str = "pc_scatter",
) -> None:
    """
    Outputs:
        {base_name}.png          — all splits overlaid
        {base_name}_{split}.png  — one plot per split
    """
    present_splits = [s for s in _DRAW_ORDER if s in results]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(
        "PointMetaBase-S-X3D — Predicted vs True Energy (all splits, best checkpoint)",
        fontsize=13, fontweight="bold",
    )
    for col_idx, name in enumerate(LABEL_NAMES):
        ax = axes[col_idx // 3][col_idx % 3]
        _draw_one_scatter_panel(ax, results, col_idx, name,
                                splits_to_draw=present_splits,
                                annotation_split="test" if "test" in results else None)
    legend_el = _make_legend_elements(present_splits)
    fig.legend(handles=legend_el, loc="lower center", ncol=len(legend_el),
               fontsize=9, frameon=True, bbox_to_anchor=(0.5, -0.01))
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    combined_path = os.path.join(output_dir, f"{base_name}.png")
    plt.savefig(combined_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"[Phase C] Saved combined scatter → {combined_path}")

    for split in present_splits:
        sty = _SPLIT_STYLES[split]
        split_label = sty["label"]
        fig2, axes2 = plt.subplots(2, 3, figsize=(15, 10))
        fig2.suptitle(
            f"PointMetaBase-S-X3D — Predicted vs True Energy ({split_label} split, best checkpoint)",
            fontsize=13, fontweight="bold",
        )
        for col_idx, name in enumerate(LABEL_NAMES):
            ax = axes2[col_idx // 3][col_idx % 3]
            _draw_one_scatter_panel(ax, results, col_idx, name,
                                    splits_to_draw=[split],
                                    annotation_split=split)
        legend_el2 = _make_legend_elements([split])
        fig2.legend(handles=legend_el2, loc="lower center", ncol=len(legend_el2),
                    fontsize=9, frameon=True, bbox_to_anchor=(0.5, -0.01))
        plt.tight_layout(rect=[0, 0.04, 1, 1])
        split_path = os.path.join(output_dir, f"{base_name}_{split_label}.png")
        plt.savefig(split_path, dpi=250, bbox_inches="tight")
        plt.close(fig2)
        print(f"[Phase C] Saved {split_label} scatter → {split_path}")


# ---------------------------------------------------------------------------
# Phase D — Residual deep-dives
# ---------------------------------------------------------------------------

def plot_residual_hist(
    preds: np.ndarray,
    targets: np.ndarray,
    output_path: str,
) -> None:
    """D1: per-target residual (pred − true) histogram with KDE, test split."""
    from scipy.stats import gaussian_kde

    residuals = preds - targets   # (N, 6)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(
        "PointMetaBase-S-X3D — Residuals (pred − true), test split",
        fontsize=13, fontweight="bold",
    )
    for col_idx, (name, color) in enumerate(zip(LABEL_NAMES, _TARGET_COLORS)):
        ax = axes[col_idx // 3][col_idx % 3]
        res = residuals[:, col_idx]
        ax.hist(res, bins=60, color=color, alpha=0.4, density=True, label="histogram")

        # KDE overlay
        try:
            kde = gaussian_kde(res)
            x_range = np.linspace(res.min(), res.max(), 300)
            ax.plot(x_range, kde(x_range), color=color, lw=2, label="KDE")
        except Exception:
            pass

        ax.axvline(0, color="black", ls="--", lw=1, alpha=0.7)
        bias = float(res.mean())
        std  = float(res.std())
        ax.text(
            0.97, 0.95,
            f"bias={bias:.2f}\nstd={std:.2f}",
            transform=ax.transAxes, va="top", ha="right", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
        )
        ax.set_title(name, fontsize=9, fontweight="bold")
        ax.set_xlabel("pred − true (Hartree)", fontsize=8)
        ax.set_ylabel("Density", fontsize=8)
        ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Phase D1] Saved residual hist → {output_path}")


def _plot_error_vs_feature(
    preds: np.ndarray,
    targets: np.ndarray,
    feature: np.ndarray,
    xlabel: str,
    title: str,
    output_path: str,
) -> None:
    """Shared helper for D2 and D3."""
    abs_errors = np.abs(preds - targets)   # (N, 6)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    for col_idx, (name, color) in enumerate(zip(LABEL_NAMES, _TARGET_COLORS)):
        ax = axes[col_idx // 3][col_idx % 3]
        err = abs_errors[:, col_idx]
        ax.scatter(feature, err, color=color, alpha=0.25, s=3, linewidths=0, rasterized=True)

        sr = float(spearmanr(feature, err)[0])
        ax.text(
            0.97, 0.95,
            f"Spearman ρ={sr:.3f}",
            transform=ax.transAxes, va="top", ha="right", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
        )
        ax.set_title(name, fontsize=9, fontweight="bold")
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel("|error| (Hartree)", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {output_path}")


def plot_error_vs_density(
    preds: np.ndarray,
    targets: np.ndarray,
    mean_rho: np.ndarray,
    output_path: str,
) -> None:
    """D2: |error| vs mean electron density per target, test split."""
    _plot_error_vs_feature(
        preds, targets, mean_rho,
        xlabel="Mean electron density (a.u.)",
        title="PointMetaBase-S-X3D — |Error| vs Mean Density, test split",
        output_path=output_path,
    )
    print(f"[Phase D2] Saved error vs density → {output_path}")


def plot_error_vs_size(
    preds: np.ndarray,
    targets: np.ndarray,
    extent: np.ndarray,
    output_path: str,
) -> None:
    """D3: |error| vs spatial extent (bounding-box diagonal) per target, test split."""
    _plot_error_vs_feature(
        preds, targets, extent,
        xlabel="Spatial extent — bbox diagonal (Bohr)",
        title="PointMetaBase-S-X3D — |Error| vs Molecular Size, test split",
        output_path=output_path,
    )
    print(f"[Phase D3] Saved error vs size → {output_path}")


def plot_error_correlation(
    preds: np.ndarray,
    targets: np.ndarray,
    output_path: str,
) -> None:
    """D4: 6×6 Pearson correlation matrix of per-target |errors|, test split."""
    abs_errors = np.abs(preds - targets)   # (N, 6)
    corr = np.corrcoef(abs_errors.T)       # (6, 6)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
    plt.colorbar(im, ax=ax, label="Pearson r")

    ax.set_xticks(range(6))
    ax.set_yticks(range(6))
    ax.set_xticklabels(LABEL_NAMES, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(LABEL_NAMES, fontsize=8)

    for i in range(6):
        for j in range(6):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center",
                    fontsize=7, color="black" if abs(corr[i, j]) < 0.7 else "white")

    ax.set_title("PointMetaBase-S-X3D — Per-target |Error| Correlation\n(test split)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Phase D4] Saved error correlation → {output_path}")


# ---------------------------------------------------------------------------
# Phase E — Cumulative error distribution (CDF)
# ---------------------------------------------------------------------------

def plot_cdf(
    preds: np.ndarray,
    targets: np.ndarray,
    output_path: str,
) -> None:
    """CDF of |error| per target for the test split."""
    abs_errors = np.abs(preds - targets)   # (N, 6)

    # x-axis upper limit = 95th percentile across all targets
    x_max = float(np.percentile(abs_errors, 95))

    fig, ax = plt.subplots(figsize=(8, 5))
    for col_idx, (name, color) in enumerate(zip(LABEL_NAMES, _TARGET_COLORS)):
        err = np.sort(abs_errors[:, col_idx])
        cdf = np.arange(1, len(err) + 1) / len(err)
        ax.plot(err, cdf, color=color, lw=1.8, label=name)

    ax.set_xlim(0, x_max)
    ax.set_ylim(0, 1)
    ax.set_xlabel("|error| (Hartree)", fontsize=10)
    ax.set_ylabel("Cumulative fraction", fontsize=10)
    ax.set_title(
        "PointMetaBase-S-X3D — Cumulative Error Distribution, test split",
        fontsize=11, fontweight="bold",
    )
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Phase E] Saved CDF → {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Analyze point-cloud benchmark training results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--run-dir",   required=True,
                   help="Path to the run checkpoint dir (must contain best.pt).")
    p.add_argument("--pkl-path",  required=True,
                   help="Path to mol_EDthresh0.05_data.pkl.")
    p.add_argument("--csv-path",  required=True,
                   help="Path to ed_energy_5w.csv.")
    p.add_argument("--cache-dir", required=True,
                   help="Path to FPS cache dir.")
    p.add_argument("--output-dir", default="benchmark/outputs/analysis/pointcloud",
                   help="Directory to write analysis outputs.")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--splits", nargs="+", default=["train", "valid", "test"],
                   help="Splits to run inference on.")
    p.add_argument("--skip-inference", action="store_true",
                   help="Skip Phases B-F (only produce Phase A training curves).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    # ── Load checkpoint (compat) ──────────────────────────────────────────
    print(f"Run dir: {args.run_dir}")
    state_dict, cfg, target_mean, target_std, best_epoch, history = _load_compat(args.run_dir)
    print(f"Best epoch: {best_epoch}")

    # ── Phase A ───────────────────────────────────────────────────────────
    if history is not None:
        best_ep_idx = int(np.argmin([h["val_mean_mae"] for h in history]))
        best_epoch  = history[best_ep_idx]["epoch"]
        curves_path = os.path.join(args.output_dir, "pc_training_curves.png")
        plot_training_curves(history, best_epoch, curves_path)
    else:
        print("[Phase A] WARNING: metrics.json not found — training curves skipped.")

    if args.skip_inference:
        print("Phases B-F skipped (--skip-inference).")
        return

    # ── Load model ────────────────────────────────────────────────────────
    print("[Phase B] Building model and loading weights ...")
    model = _build_model(cfg).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    # ── Phase B — Inference ───────────────────────────────────────────────
    # results[split] = (preds (N,6), targets (N,6), mol_stats dict)
    results: dict[str, tuple[np.ndarray, np.ndarray, dict]] = {}

    for split in args.splits:
        print(f"[Phase B] Inference on split={split} ...")
        dataset = EDBenchEnergyDataset(
            pkl_path=args.pkl_path,
            csv_path=args.csv_path,
            cache_dir=args.cache_dir,
            split=split,
        )
        effective_workers = 0 if dataset.has_missing_cache() else args.num_workers
        if dataset.has_missing_cache():
            print(f"  Cache incomplete for {split}; forcing num_workers=0.")
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=energy_collate_fn,
            num_workers=effective_workers,
            pin_memory=(device.type == "cuda"),
        )
        loader_bar = tqdm(loader, desc=f"  {split}", unit="batch", leave=False)
        preds, targets, mol_stats = run_inference(
            model, loader_bar, target_mean, target_std, device
        )
        results[split] = (preds, targets, mol_stats)
        print(f"  {split}: {len(targets)} samples")

    # ── Phase C — Scatter plots ───────────────────────────────────────────
    plot_scatter(results, args.output_dir)

    # ── Phase D — Residual deep-dives (test split) ────────────────────────
    test_split = "test" if "test" in results else (list(results.keys())[-1] if results else None)
    if test_split:
        preds_t, targets_t, stats_t = results[test_split]

        plot_residual_hist(
            preds_t, targets_t,
            os.path.join(args.output_dir, "pc_residual_hist.png"),
        )
        plot_error_vs_density(
            preds_t, targets_t, stats_t["mean_rho"],
            os.path.join(args.output_dir, "pc_error_vs_density.png"),
        )
        plot_error_vs_size(
            preds_t, targets_t, stats_t["extent"],
            os.path.join(args.output_dir, "pc_error_vs_size.png"),
        )
        plot_error_correlation(
            preds_t, targets_t,
            os.path.join(args.output_dir, "pc_error_correlation.png"),
        )
    else:
        print("[Phase D] Skipped — no test split available.")

    # ── Phase E — CDF ─────────────────────────────────────────────────────
    if test_split:
        preds_t, targets_t, _ = results[test_split]
        plot_cdf(preds_t, targets_t, os.path.join(args.output_dir, "pc_cdf.png"))
    else:
        print("[Phase E] Skipped — no test split available.")

    # ── Phase F — Summary JSON ────────────────────────────────────────────
    summary: dict = {
        "run_dir":          args.run_dir,
        "best_epoch":       best_epoch,
        "best_val_mean_mae": float(np.min([h["val_mean_mae"] for h in history]))
                             if history else None,
        "splits": {},
    }
    for split, (preds, targets, _) in results.items():
        split_metrics: dict = {}
        maes = []
        for i, name in enumerate(LABEL_NAMES):
            t, p = targets[:, i], preds[:, i]
            mae  = float(np.abs(p - t).mean())
            rmse = float(np.sqrt(((p - t) ** 2).mean()))
            r    = float(np.corrcoef(t, p)[0, 1])
            sr   = float(spearmanr(t, p)[0])
            split_metrics[name] = {
                "mae": mae, "rmse": rmse,
                "pearson_r": r, "spearman_r": sr,
            }
            maes.append(mae)
        split_metrics["mean_mae"] = float(np.mean(maes))
        summary["splits"][split] = split_metrics

    # Console test summary
    if "test" in summary["splits"]:
        print("\n=== Test Set ===")
        for name in LABEL_NAMES:
            m = summary["splits"]["test"][name]
            print(f"  {name:<14}  MAE={m['mae']:8.4f}  RMSE={m['rmse']:9.4f}"
                  f"  r={m['pearson_r']:.4f}  ρ={m['spearman_r']:.4f}")
        print(f"  {'Mean MAE':<14}  {summary['splits']['test']['mean_mae']:.4f}")

    summary_path = os.path.join(args.output_dir, "analysis_summary.json")
    _save_json(summary_path, summary)
    print(f"\n[Phase F] Saved analysis summary → {summary_path}")
    print(f"\nDone. Outputs in {args.output_dir}")


if __name__ == "__main__":
    main()
