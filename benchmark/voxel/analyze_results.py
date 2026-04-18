#!/usr/bin/env python3
"""
Voxel benchmark analysis script.

Phases:
  A) Training curves from metrics.json
     - train_loss (log scale) + val_mean_mae vs epoch
     - learning rate vs epoch
  B) Full-dataset inference on all splits, collecting:
     - preds / targets (Hartree)
     - per-molecule voxel statistics (mean_density, max_density, nonzero_frac)
  C) Pred vs True scatter plots (combined + per-split)
     – adds Spearman ρ to annotation box
  D) Residual deep-dives
     D1 – residual (pred − true) histograms with KDE per energy target
     D2 – |error| vs mean voxel density per target
     D3 – |error| vs nonzero voxel fraction (molecular volume proxy) per target
     D4 – 6×6 target-error Pearson correlation heat map
  E) Cumulative error distribution (CDF) per target — test split
  F) analysis_summary.json  (adds Spearman ρ vs old format)

Usage:
  python -m benchmark.voxel.analyze_results \\
      --run-dir benchmark/outputs/checkpoints/train/voxel_full_density \\
      --csv-path data/ed_energy_5w/raw/ed_energy_5w.csv \\
      --cache-dir data/ed_energy_5w/cache_voxel \\
      --output-dir benchmark/outputs/analysis/voxel \\
      --device cpu
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

from benchmark.voxel.data.energy_dataset import (
    LABEL_NAMES,
    EDBenchVoxelDataset,
    voxel_collate_fn,
)
from benchmark.voxel.models.voxel_densenet import VoxelDenseNetRegressor


_TARGET_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c",
    "#d62728", "#9467bd", "#8c564b",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _save_json(path: str, obj: object) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def _build_model(cfg: dict) -> VoxelDenseNetRegressor:
    return VoxelDenseNetRegressor(
        in_channels=len(cfg["channels"]),
        out_dim=6,
        growth_rate=cfg.get("growth_rate", 12),
        block_config=(cfg.get("dense1", 16), cfg.get("dense2", 16)),
        num_init_features=cfg.get("num_init_features", 64),
        drop_rate=cfg.get("drop_rate", 0.0),
        small_inputs=True,
    )


# ---------------------------------------------------------------------------
# Phase A — Training curves
# ---------------------------------------------------------------------------

def plot_training_curves(
    history: list[dict],
    best_epoch: int,
    output_path: str,
) -> None:
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_mae = [h["val_mean_mae"] for h in history]
    lr = [h["lr"] for h in history]

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle("Voxel DenseNet — Training Curves", fontsize=14, fontweight="bold")

    # --- Upper: loss + val MAE ---
    ax1 = axes[0]
    color_loss = "#2196F3"
    color_mae = "#F44336"

    l1, = ax1.semilogy(epochs, train_loss, color=color_loss, lw=1.5, label="Train Loss (MSE, log)")
    ax1.set_ylabel("Train Loss (log scale)", color=color_loss)
    ax1.tick_params(axis="y", labelcolor=color_loss)

    ax1r = ax1.twinx()
    l2, = ax1r.plot(epochs, val_mae, color=color_mae, lw=1.5, label="Val Mean MAE")
    ax1r.set_ylabel("Val Mean MAE (Hartree)", color=color_mae)
    ax1r.tick_params(axis="y", labelcolor=color_mae)

    # Mark best epoch
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
    lines = [l1, l2]
    ax1.legend(lines, [l.get_label() for l in lines], loc="upper right", fontsize=9)

    # --- Lower: learning rate ---
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
# Phase B — Inference + per-molecule voxel statistics
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
        mol_stats  dict of (N,) arrays: mean_density, max_density, nonzero_frac
    """
    model.eval()
    all_preds, all_targets = [], []
    all_mean_density, all_max_density, all_nonzero_frac = [], [], []

    t_mean = torch.from_numpy(target_mean).to(device)
    t_std = torch.from_numpy(target_std).to(device)

    for batch in loader:
        voxel = batch["voxel"].to(device)        # (B, C, G, G, G)
        target = batch["target"].to(device)       # (B, 6) — raw Hartree

        pred_norm = model(voxel)                  # (B, 6) — normalized output
        pred = pred_norm * t_std + t_mean         # denormalize

        # Per-molecule voxel statistics (computed on CPU)
        voxel_cpu = voxel.cpu()
        density = voxel_cpu[:, 0]                 # (B, G, G, G) — first channel is density
        density_flat = density.flatten(1)         # (B, G*G*G)

        all_mean_density.append(density_flat.mean(dim=1).numpy())
        all_max_density.append(density_flat.max(dim=1).values.numpy())
        all_nonzero_frac.append((density_flat > 0.01).float().mean(dim=1).numpy())

        all_preds.append(pred.cpu().numpy())
        all_targets.append(target.cpu().numpy())

    preds   = np.concatenate(all_preds,   axis=0)
    targets = np.concatenate(all_targets, axis=0)
    mol_stats = {
        "mean_density":  np.concatenate(all_mean_density),
        "max_density":   np.concatenate(all_max_density),
        "nonzero_frac":  np.concatenate(all_nonzero_frac),
    }
    return preds, targets, mol_stats


# ---------------------------------------------------------------------------
# Phase C — Scatter plots (pred vs true)
# ---------------------------------------------------------------------------

_SPLIT_STYLES: dict[str, dict] = {
    "train": dict(color="#999999", alpha=0.12, s=2,  marker=".",  zorder=1, label="train"),
    "valid": dict(color="#1565C0", alpha=0.45, s=6,  marker="^",  zorder=2, label="valid"),
    "val":   dict(color="#1565C0", alpha=0.45, s=6,  marker="^",  zorder=2, label="valid"),
    "test":  dict(color="#E64A19", alpha=0.55, s=6,  marker="o",  zorder=3, label="test"),
}

_DRAW_ORDER = ("train", "valid", "val", "test")


def _draw_one_scatter_panel(
    ax: plt.Axes,
    results: dict[str, tuple],
    col_idx: int,
    name: str,
    splits_to_draw: list[str],
    annotation_split: Optional[str] = "test",
) -> None:
    """Draw a single (true, pred) panel for energy target `col_idx`."""
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

    # y=x diagonal
    all_t = np.concatenate(all_true_vals)
    lo, hi = all_t.min(), all_t.max()
    margin = (hi - lo) * 0.05
    diag = np.array([lo - margin, hi + margin])
    ax.plot(diag, diag, color="#D32F2F", ls="--", lw=1.2, zorder=10)
    ax.set_xlim(diag[0], diag[1])
    ax.set_ylim(diag[0], diag[1])

    # Metrics annotation
    if annotation_split and annotation_split in results:
        t_ann = results[annotation_split][1][:, col_idx]
        p_ann = results[annotation_split][0][:, col_idx]
        mae  = float(np.abs(p_ann - t_ann).mean())
        rmse = float(np.sqrt(((p_ann - t_ann) ** 2).mean()))
        r    = float(np.corrcoef(t_ann, p_ann)[0, 1])
        sr   = float(spearmanr(t_ann, p_ann)[0])
        label = _SPLIT_STYLES[annotation_split]["label"]
        ax.text(
            0.04, 0.96,
            f"{label} MAE={mae:.2f}\n{label} RMSE={rmse:.2f}\n"
            f"{label} r={r:.3f}\n{label} ρ={sr:.3f}",
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
    base_name: str = "voxel_scatter",
) -> None:
    """
    Generate scatter plots (pred vs true).

    Outputs:
      {base_name}.png          — all splits overlaid
      {base_name}_{split}.png  — one plot per split
    """
    present_splits = [s for s in _DRAW_ORDER if s in results]

    # ------------------------------------------------------------------ #
    # 1. Combined plot (all splits overlaid)
    # ------------------------------------------------------------------ #
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(
        "Voxel DenseNet — Predicted vs True Energy (all splits, best checkpoint)",
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

    # ------------------------------------------------------------------ #
    # 2. Per-split plots
    # ------------------------------------------------------------------ #
    for split in present_splits:
        sty = _SPLIT_STYLES[split]
        split_label = sty["label"]
        fig2, axes2 = plt.subplots(2, 3, figsize=(15, 10))
        fig2.suptitle(
            f"Voxel DenseNet — Predicted vs True Energy ({split_label} split, best checkpoint)",
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
        "Voxel DenseNet — Residuals (pred − true), test split",
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
    mean_density: np.ndarray,
    output_path: str,
) -> None:
    """D2: |error| vs mean voxel density per target, test split."""
    _plot_error_vs_feature(
        preds, targets, mean_density,
        xlabel="Mean voxel density (a.u.)",
        title="Voxel DenseNet — |Error| vs Mean Voxel Density, test split",
        output_path=output_path,
    )
    print(f"[Phase D2] Saved error vs density → {output_path}")


def plot_error_vs_size(
    preds: np.ndarray,
    targets: np.ndarray,
    nonzero_frac: np.ndarray,
    output_path: str,
) -> None:
    """D3: |error| vs nonzero voxel fraction (molecular volume proxy) per target, test split."""
    _plot_error_vs_feature(
        preds, targets, nonzero_frac,
        xlabel="Nonzero voxel fraction (volume proxy)",
        title="Voxel DenseNet — |Error| vs Molecular Volume, test split",
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

    ax.set_title("Voxel DenseNet — Per-target |Error| Correlation\n(test split)",
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
        "Voxel DenseNet — Cumulative Error Distribution, test split",
        fontsize=11, fontweight="bold",
    )
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Phase E] Saved CDF → {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze voxel benchmark training results.")
    p.add_argument("--run-dir", required=True,
                   help="Path to the run checkpoint dir (contains config.json, metrics.json, best.pt).")
    p.add_argument("--csv-path", required=True,
                   help="Path to ed_energy_5w.csv.")
    p.add_argument("--cache-dir", required=True,
                   help="Path to voxel cache dir.")
    p.add_argument("--output-dir", default="benchmark/outputs/analysis/voxel",
                   help="Directory to write analysis outputs.")
    p.add_argument("--batch-size", type=int, default=128,
                   help="Batch size for inference (default 128).")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--splits", nargs="+", default=["train", "valid", "test"],
                   help="Splits to run inference on (default: train valid test).")
    p.add_argument("--skip-inference", action="store_true",
                   help="Skip Phases B-F (useful if you only want training curves).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    # --- Load config & metrics ---
    cfg_path = os.path.join(args.run_dir, "config.json")
    met_path = os.path.join(args.run_dir, "metrics.json")
    ckpt_path = os.path.join(args.run_dir, "best.pt")

    cfg = _load_json(cfg_path)
    met = _load_json(met_path)

    history = met["history"]
    best_val_mae = met["best_val_mean_mae"]
    best_epoch = int(np.argmin([h["val_mean_mae"] for h in history])) + 1  # 1-indexed

    target_mean = np.array(cfg["target_mean"], dtype=np.float32)
    target_std  = np.array(cfg["target_std"],  dtype=np.float32)

    channels        = cfg["channels"]
    grid_length     = cfg["grid_length"]
    cube_size_bohr  = cfg["cube_size_bohr"]
    gaussian_sigma  = cfg.get("gaussian_sigma", 0.0)
    split_col       = cfg.get("split_col", "scaffold_split")

    print(f"Run dir   : {args.run_dir}")
    print(f"Best epoch: {best_epoch}  (val mean MAE = {best_val_mae:.4f})")

    # -----------------------------------------------------------------------
    # Phase A — Training curves
    # -----------------------------------------------------------------------
    curves_path = os.path.join(args.output_dir, "voxel_training_curves.png")
    plot_training_curves(history, best_epoch, curves_path)

    if args.skip_inference:
        print("[Phases B-F] Skipped (--skip-inference).")
        return

    # -----------------------------------------------------------------------
    # Phase B — Full-dataset inference
    # -----------------------------------------------------------------------
    print("[Phase B] Loading model from best.pt ...")
    model = _build_model(cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_key = "model_state" if "model_state" in ckpt else "model"
    model.load_state_dict(ckpt[state_key])
    model.eval()

    # results[split] = (preds (N,6), targets (N,6), mol_stats dict)
    inference_results: dict[str, tuple] = {}

    for split in args.splits:
        print(f"[Phase B] Inference on split={split} ...")
        dataset = EDBenchVoxelDataset(
            csv_path=args.csv_path,
            cache_dir=args.cache_dir,
            grid_length=grid_length,
            cube_size_bohr=cube_size_bohr,
            channels=channels,
            split=split,
            split_col=split_col,
            gaussian_sigma=gaussian_sigma,
            require_all_cached=True,
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=voxel_collate_fn,
            num_workers=args.num_workers,
            pin_memory=(args.device == "cuda"),
        )
        loader_with_progress = tqdm(loader, desc=f"  {split}", unit="batch", leave=False)
        preds, targets, mol_stats = run_inference(
            model, loader_with_progress, target_mean, target_std, device
        )
        inference_results[split] = (preds, targets, mol_stats)
        print(f"  {split}: {len(targets)} samples")

    # -----------------------------------------------------------------------
    # Phase C — Scatter plots
    # -----------------------------------------------------------------------
    plot_scatter(inference_results, args.output_dir)

    # -----------------------------------------------------------------------
    # Phase D — Residual deep-dives (test split)
    # -----------------------------------------------------------------------
    test_split = "test" if "test" in inference_results else (
        list(inference_results.keys())[-1] if inference_results else None
    )
    if test_split:
        preds_t, targets_t, stats_t = inference_results[test_split]

        plot_residual_hist(
            preds_t, targets_t,
            os.path.join(args.output_dir, "voxel_residual_hist.png"),
        )
        plot_error_vs_density(
            preds_t, targets_t, stats_t["mean_density"],
            os.path.join(args.output_dir, "voxel_error_vs_density.png"),
        )
        plot_error_vs_size(
            preds_t, targets_t, stats_t["nonzero_frac"],
            os.path.join(args.output_dir, "voxel_error_vs_size.png"),
        )
        plot_error_correlation(
            preds_t, targets_t,
            os.path.join(args.output_dir, "voxel_error_correlation.png"),
        )
    else:
        print("[Phase D] Skipped — no test split available.")

    # -----------------------------------------------------------------------
    # Phase E — CDF
    # -----------------------------------------------------------------------
    if test_split:
        preds_t, targets_t, _ = inference_results[test_split]
        plot_cdf(preds_t, targets_t, os.path.join(args.output_dir, "voxel_cdf.png"))
    else:
        print("[Phase E] Skipped — no test split available.")

    # -----------------------------------------------------------------------
    # Phase F — Summary JSON
    # -----------------------------------------------------------------------
    summary: dict = {
        "run_dir": args.run_dir,
        "best_epoch": best_epoch,
        "best_val_mean_mae": float(best_val_mae),
        "splits": {},
    }
    for split, (preds, targets, _) in inference_results.items():
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

    # Print test summary
    if "test" in summary["splits"]:
        print("\n=== Test Set ===")
        for name in LABEL_NAMES:
            m = summary["splits"]["test"][name]
            print(f"  {name:<18}  MAE={m['mae']:8.4f}  RMSE={m['rmse']:9.4f}"
                  f"  r={m['pearson_r']:.4f}  ρ={m['spearman_r']:.4f}")
        print(f"  {'Mean MAE':<18}  {summary['splits']['test']['mean_mae']:.4f}")

    summary_path = os.path.join(args.output_dir, "analysis_summary.json")
    _save_json(summary_path, summary)
    print(f"\n[Phase F] Saved analysis summary → {summary_path}")
    print(f"\nDone. Outputs in {args.output_dir}")


if __name__ == "__main__":
    main()
