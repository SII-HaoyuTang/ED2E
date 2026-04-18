#!/usr/bin/env python3
"""
Training script for EDBench ED5-EC energy prediction (PointMetaBase-S-X3D).

Usage:
    # Smoke test
    python -m benchmark.point_cloude.train_energy \\
        --pkl-path data/ed_energy_5w/processed/mol_EDthresh0.05_data.pkl \\
        --csv-path data/ed_energy_5w/raw/ed_energy_5w.csv \\
        --cache-dir data/ed_energy_5w/cache_fps \\
        --max-samples 128 --epochs 2 --npoint 512 --device cpu

    # Full training (matches paper config)
    python -m benchmark.point_cloude.train_energy \\
        --pkl-path data/ed_energy_5w/processed/mol_EDthresh0.05_data.pkl \\
        --csv-path data/ed_energy_5w/raw/ed_energy_5w.csv \\
        --cache-dir data/ed_energy_5w/cache_fps \\
        --npoint 2048 --batch-size 32 --lr 1e-3 --epochs 100 --device cuda
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import wandb as _wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

from benchmark.point_cloude.data.energy_dataset import EDBenchEnergyDataset, energy_collate_fn
from benchmark.point_cloude.models.backbone.pointmetabase_x3d import PointMetaBaseX3D

LABEL_NAMES = [
    "E1_Final", "E2_NucRepul", "E3_OneElec",
    "E4_TwoElec", "E5_XC", "E6_Total",
]


@dataclass
class EvalResult:
    mean_mae: float
    mae: list[float]
    rmse: list[float]
    pearson: list[float]
    spearman: list[float]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_target_stats(energies: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = energies.mean(axis=0).astype(np.float32, copy=False)
    std = energies.std(axis=0).astype(np.float32, copy=False)
    std = np.clip(std, a_min=1e-6, a_max=None)
    return mean, std


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EDBench ED5-EC energy prediction (point cloud)")
    # Data
    p.add_argument("--pkl-path",  required=True, help="Path to .pkl density file")
    p.add_argument("--csv-path",  required=True, help="Path to ed_energy_5w.csv")
    p.add_argument("--cache-dir", required=True, help="Cache dir for FPS .pt files")
    p.add_argument("--npoint",    type=int,   default=2048,
                   help="Points per molecule after FPS (default 2048)")
    p.add_argument("--max-samples",  type=int, default=None, help="Cap dataset size (debug)")
    p.add_argument("--max-train-samples", type=int, default=None)
    p.add_argument("--max-val-samples",   type=int, default=None)
    p.add_argument("--max-test-samples",  type=int, default=None)
    # Model
    p.add_argument("--width",       type=int,   default=32,
                   help="Base channel width (default 32)")
    p.add_argument("--radius",      type=float, default=0.15,
                   help="Base ball-query radius in Bohr (default 0.15)")
    p.add_argument("--radius-mult", type=float, default=1.5,
                   help="Radius multiplier per stage (default 1.5)")
    p.add_argument("--K",           type=int,   default=32,
                   help="Neighbours per query point (default 32)")
    p.add_argument("--dropout",     type=float, default=0.5)
    # Training
    p.add_argument("--batch-size",    type=int,   default=32)
    p.add_argument("--lr",            type=float, default=1e-3)
    p.add_argument("--weight-decay",  type=float, default=0.05)
    p.add_argument("--epochs",        type=int,   default=100)
    p.add_argument("--grad-clip",     type=float, default=1.0,
                   help="Gradient clip max norm")
    p.add_argument("--num-workers",   type=int,   default=4)
    p.add_argument("--seed",          type=int,   default=22)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # Output
    p.add_argument("--output-dir",      default=None,
                   help="Optional explicit run directory override.")
    p.add_argument("--checkpoint-root", default="benchmark/outputs/checkpoints")
    p.add_argument("--wandb-root",      default="benchmark/outputs/wandb")
    p.add_argument("--run-kind", choices=("benchmark", "train"), default="benchmark")
    p.add_argument("--run-name",  default=None)
    p.add_argument("--save-every", type=int, default=10)
    # W&B
    p.add_argument("--wandb",        action="store_true", help="Enable W&B logging")
    p.add_argument("--wandb-project", default="edbench-pointcloud")
    p.add_argument("--wandb-entity",  default=None)
    p.add_argument("--wandb-group",   default=None)
    p.add_argument("--wandb-tags",    default="")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_loader(
    dataset: EDBenchEnergyDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=energy_collate_fn,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def save_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def resolve_run_name(args: argparse.Namespace) -> str:
    if args.run_name:
        return args.run_name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"pointcloud_{args.run_kind}_{timestamp}"


def resolve_run_dir(args: argparse.Namespace, run_name: str) -> str:
    if args.output_dir:
        return args.output_dir
    return os.path.join(args.checkpoint_root, args.run_kind, run_name)


def make_checkpoint_payload(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    best_val: float,
    config_payload: dict,
) -> dict:
    return {
        "model_state":     model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "config":          config_payload,
        "epoch":           epoch,
        "best_val_mean_mae": best_val,
    }


# ---------------------------------------------------------------------------
# Train / Evaluate
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    device: torch.device,
    grad_clip: float,
) -> float:
    model.train()
    total_loss  = 0.0
    total_count = 0
    criterion   = nn.MSELoss()

    progress = tqdm(loader, desc="Train", leave=False, unit="batch")
    for batch in progress:
        pc = batch["point_cloud"].to(device)   # (B, N, 4)
        y  = batch["energies"].to(device)      # (B, 6)
        y_norm = (y - target_mean) / target_std

        optimizer.zero_grad(set_to_none=True)
        pred_norm = model(pc)
        loss = criterion(pred_norm, y_norm)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        bs = pc.shape[0]
        total_loss  += loss.item() * bs
        total_count += bs
        progress.set_postfix(loss=f"{loss.item():.5f}")

    progress.close()
    return total_loss / max(total_count, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    device: torch.device,
) -> EvalResult:
    """
    Returns per-target MAE, RMSE, Pearson r, Spearman ρ.
    Matches EDBench's metric_reg_multitask evaluation.
    """
    model.eval()
    all_preds   = []
    all_targets = []

    for batch in loader:
        pc = batch["point_cloud"].to(device)
        y  = batch["energies"].to(device)
        pred_norm = model(pc)
        pred = pred_norm * target_std + target_mean
        all_preds.append(pred.cpu().numpy())
        all_targets.append(y.cpu().numpy())

    preds   = np.concatenate(all_preds,   axis=0)   # (N_total, 6)
    targets = np.concatenate(all_targets, axis=0)

    mae_list, rmse_list, pearson_list, spearman_list = [], [], [], []
    for i in range(len(LABEL_NAMES)):
        p, t = preds[:, i], targets[:, i]
        mae_list.append(float(np.abs(p - t).mean()))
        rmse_list.append(float(np.sqrt(((p - t) ** 2).mean())))
        pearson_list.append(float(pearsonr(p, t)[0]))
        spearman_list.append(float(spearmanr(p, t)[0]))

    return EvalResult(
        mean_mae=float(np.mean(mae_list)),
        mae=mae_list,
        rmse=rmse_list,
        pearson=pearson_list,
        spearman=spearman_list,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    run_name  = resolve_run_name(args)
    run_dir   = resolve_run_dir(args, run_name)
    wandb_dir = os.path.join(args.wandb_root, args.run_kind)
    os.makedirs(run_dir,   exist_ok=True)
    os.makedirs(wandb_dir, exist_ok=True)

    # --- Effective per-split max_samples ---
    max_train = args.max_train_samples if args.max_train_samples is not None else args.max_samples
    max_val   = args.max_val_samples   if args.max_val_samples   is not None else args.max_samples
    max_test  = args.max_test_samples  if args.max_test_samples  is not None else args.max_samples

    # --- Datasets ---
    train_set = EDBenchEnergyDataset(
        pkl_path=args.pkl_path, csv_path=args.csv_path,
        cache_dir=args.cache_dir, split="train",
        npoint=args.npoint, max_samples=max_train,
    )
    val_set = EDBenchEnergyDataset(
        pkl_path=args.pkl_path, csv_path=args.csv_path,
        cache_dir=args.cache_dir, split="valid",
        npoint=args.npoint, max_samples=max_val,
    )
    test_set = EDBenchEnergyDataset(
        pkl_path=args.pkl_path, csv_path=args.csv_path,
        cache_dir=args.cache_dir, split="test",
        npoint=args.npoint, max_samples=max_test,
    )

    mean_np, std_np = compute_target_stats(train_set.energies)
    target_mean = torch.from_numpy(mean_np).to(device)
    target_std  = torch.from_numpy(std_np).to(device)

    effective_workers = args.num_workers
    if args.num_workers > 0 and (
        train_set.has_missing_cache()
        or val_set.has_missing_cache()
        or test_set.has_missing_cache()
    ):
        print(
            "Cache is incomplete; forcing num_workers=0 so the 9 GB PKL is not "
            "replicated across worker processes during on-demand cache building."
        )
        effective_workers = 0

    train_loader = make_loader(train_set, args.batch_size, shuffle=True,  num_workers=effective_workers)
    val_loader   = make_loader(val_set,   args.batch_size, shuffle=False, num_workers=effective_workers)
    test_loader  = make_loader(test_set,  args.batch_size, shuffle=False, num_workers=effective_workers)

    # --- Model ---
    model = PointMetaBaseX3D(
        in_channels=4,
        width=args.width,
        num_targets=6,
        npoint_start=args.npoint,
        radius=args.radius,
        radius_mult=args.radius_mult,
        K=args.K,
        mlp_layers=[512, 256],
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"PointMetaBase-S-X3D  params: {n_params:,}", flush=True)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    # --- Config snapshot ---
    config_payload = vars(args).copy()
    config_payload["run_name"]            = run_name
    config_payload["resolved_run_dir"]    = run_dir
    config_payload["resolved_wandb_dir"]  = wandb_dir
    config_payload["num_train_samples"]   = len(train_set)
    config_payload["num_val_samples"]     = len(val_set)
    config_payload["num_test_samples"]    = len(test_set)
    config_payload["num_parameters"]      = n_params
    config_payload["label_names"]         = list(LABEL_NAMES)
    config_payload["target_mean"]         = mean_np.tolist()
    config_payload["target_std"]          = std_np.tolist()
    save_json(os.path.join(run_dir, "config.json"), config_payload)

    # --- W&B ---
    wandb_run = None
    if args.wandb:
        if not _WANDB_AVAILABLE:
            print("WARNING: wandb not installed, skipping W&B logging.", flush=True)
        else:
            tags = [t for t in (x.strip() for x in args.wandb_tags.split(",")) if t]
            wandb_run = _wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                group=args.wandb_group,
                name=run_name,
                dir=wandb_dir,
                config=config_payload,
                tags=tags,
            )
            wandb_run.summary["num_parameters"]    = n_params
            wandb_run.summary["num_train_samples"] = len(train_set)
            wandb_run.summary["num_val_samples"]   = len(val_set)
            wandb_run.summary["num_test_samples"]  = len(test_set)

    best_val  = math.inf
    best_path = os.path.join(run_dir, "best.pt")
    last_path = os.path.join(run_dir, "last.pt")
    history: list[dict] = []

    # --- Training loop ---
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            target_mean=target_mean,
            target_std=target_std,
            device=device,
            grad_clip=args.grad_clip,
        )
        val_res = evaluate(model, val_loader, target_mean, target_std, device)
        scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        history.append({
            "epoch": epoch, "train_loss": train_loss,
            "val_mean_mae": val_res.mean_mae, "lr": lr,
        })

        print(
            f"Epoch {epoch:4d}/{args.epochs}  "
            f"train_loss={train_loss:.6f}  val_mean_MAE={val_res.mean_mae:.6f}  lr={lr:.2e}",
            flush=True,
        )
        for name, mae, rmse, pr, sr in zip(
            LABEL_NAMES, val_res.mae, val_res.rmse, val_res.pearson, val_res.spearman
        ):
            print(
                f"  {name:<14} MAE={mae:.6f}  RMSE={rmse:.6f}  "
                f"r={pr:.4f}  ρ={sr:.4f}",
                flush=True,
            )

        epoch_payload = make_checkpoint_payload(
            model=model, optimizer=optimizer, scheduler=scheduler,
            epoch=epoch, best_val=best_val, config_payload=config_payload,
        )
        torch.save(epoch_payload, last_path)
        if args.save_every > 0 and (epoch % args.save_every == 0 or epoch == args.epochs):
            torch.save(epoch_payload, os.path.join(run_dir, f"epoch_{epoch:04d}.pt"))

        if wandb_run is not None:
            log_payload = {
                "epoch": epoch,
                "train/loss": train_loss,
                "val/mean_mae": val_res.mean_mae,
                "train/lr": lr,
            }
            for name, mae, rmse, pr, sr in zip(
                LABEL_NAMES, val_res.mae, val_res.rmse, val_res.pearson, val_res.spearman
            ):
                log_payload[f"val/{name}_mae"]     = mae
                log_payload[f"val/{name}_rmse"]    = rmse
                log_payload[f"val/{name}_pearson"] = pr
                log_payload[f"val/{name}_spearman"] = sr
            wandb_run.log(log_payload, step=epoch)

        if val_res.mean_mae < best_val:
            best_val = val_res.mean_mae
            best_payload = make_checkpoint_payload(
                model=model, optimizer=optimizer, scheduler=scheduler,
                epoch=epoch, best_val=best_val, config_payload=config_payload,
            )
            torch.save(best_payload, best_path)
            if wandb_run is not None:
                wandb_run.summary["best_val_mean_mae"] = best_val
                wandb_run.summary["best_epoch"]        = epoch

    # --- Final test evaluation (load best checkpoint) ---
    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])

    test_res = evaluate(model, test_loader, target_mean, target_std, device)
    print("=== Test Set Evaluation ===", flush=True)
    print(f"  Mean MAE: {test_res.mean_mae:.6f}", flush=True)
    for name, mae, rmse, pr, sr in zip(
        LABEL_NAMES, test_res.mae, test_res.rmse, test_res.pearson, test_res.spearman
    ):
        print(
            f"  {name:<14} MAE={mae:.6f}  RMSE={rmse:.6f}  "
            f"r={pr:.4f}  ρ={sr:.4f}",
            flush=True,
        )

    if wandb_run is not None:
        test_payload: dict = {"test/mean_mae": test_res.mean_mae}
        for name, mae, rmse, pr, sr in zip(
            LABEL_NAMES, test_res.mae, test_res.rmse, test_res.pearson, test_res.spearman
        ):
            test_payload[f"test/{name}_mae"]     = mae
            test_payload[f"test/{name}_rmse"]    = rmse
            test_payload[f"test/{name}_pearson"] = pr
            test_payload[f"test/{name}_spearman"] = sr
        wandb_run.log(test_payload)
        wandb_run.summary["best_checkpoint"] = best_path
        wandb_run.summary["last_checkpoint"] = last_path
        wandb_run.finish()

    save_json(
        os.path.join(run_dir, "metrics.json"),
        {
            "run_name":         run_name,
            "run_kind":         args.run_kind,
            "run_dir":          run_dir,
            "wandb_dir":        wandb_dir,
            "best_val_mean_mae": best_val,
            "test_mean_mae":    test_res.mean_mae,
            "test_mae":         test_res.mae,
            "test_rmse":        test_res.rmse,
            "test_pearson":     test_res.pearson,
            "test_spearman":    test_res.spearman,
            "history":          history,
        },
    )

    print(f"\nDone. Best val mean MAE: {best_val:.6f}", flush=True)
    print(f"Checkpoints in: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
