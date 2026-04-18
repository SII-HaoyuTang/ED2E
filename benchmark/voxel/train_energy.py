#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
from datetime import datetime
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import wandb as _wandb

    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

from benchmark.voxel.data.energy_dataset import (
    LABEL_NAMES,
    EDBenchVoxelDataset,
    build_voxel_cache,
    compute_target_stats,
    filter_rows,
    load_energy_rows,
    metadata_path,
    voxel_cache_tag,
    voxel_collate_fn,
)
from benchmark.voxel.models.voxel_densenet import VoxelDenseNetRegressor


@dataclass
class EvalResult:
    mean_mae: float
    mae: list[float]
    rmse: list[float]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a voxelized 3D DenseNet on EDBench.")
    p.add_argument("--pkl-path", default="data/ed_energy_5w/processed/mol_EDthresh0.05_data.pkl")
    p.add_argument("--csv-path", default="data/ed_energy_5w/raw/ed_energy_5w.csv")
    p.add_argument("--cache-dir", default="data/ed_energy_5w/cache_voxel")
    p.add_argument("--output-dir", default=None, help="Optional explicit run directory override.")
    p.add_argument("--checkpoint-root", default="benchmark/outputs/checkpoints")
    p.add_argument("--wandb-root", default="benchmark/outputs/wandb")
    p.add_argument("--run-kind", choices=("benchmark", "train"), default="benchmark")
    p.add_argument("--run-name", default=None)
    p.add_argument("--split-col", choices=("scaffold_split", "random_split"), default="scaffold_split")
    p.add_argument("--grid-length", type=int, default=14)
    p.add_argument("--cube-size-bohr", type=float, default=32.0)
    p.add_argument("--channels", default="density", help="Comma-separated voxel channels.")
    p.add_argument("--gaussian-sigma", type=float, default=0.0)
    p.add_argument("--save-dtype", choices=("float16", "float32"), default="float16")
    p.add_argument("--prepare-cache", action="store_true", help="Build voxel cache before training.")
    p.add_argument("--prepare-only", action="store_true", help="Only build voxel cache and exit.")
    p.add_argument("--overwrite-cache", action="store_true")
    p.add_argument("--prepare-cache-workers", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=5e-2)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=22)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--dense1", type=int, default=16)
    p.add_argument("--dense2", type=int, default=16)
    p.add_argument("--growth-rate", type=int, default=12)
    p.add_argument("--num-init-features", type=int, default=64)
    p.add_argument("--drop-rate", type=float, default=0.0)
    p.add_argument("--max-train-samples", type=int, default=None)
    p.add_argument("--max-val-samples", type=int, default=None)
    p.add_argument("--max-test-samples", type=int, default=None)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    p.add_argument("--wandb-project", default="edbench-voxel")
    p.add_argument("--wandb-entity", default=None)
    p.add_argument("--wandb-group", default=None)
    p.add_argument("--wandb-tags", default="")
    return p.parse_args()


def make_loader(
    dataset: EDBenchVoxelDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=voxel_collate_fn,
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    device: torch.device,
    grad_clip: float,
) -> float:
    model.train()
    total_loss = 0.0
    total_count = 0
    use_amp = scaler is not None
    autocast_device = "cuda" if device.type == "cuda" else "cpu"

    progress = tqdm(loader, desc="Train", leave=False, unit="batch")
    for batch in progress:
        x = batch["voxel"].to(device, non_blocking=True)
        y = batch["target"].to(device, non_blocking=True)
        y_norm = (y - target_mean) / target_std

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=autocast_device, enabled=use_amp):
            pred = model(x)
            loss = torch.nn.functional.mse_loss(pred, y_norm)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        bs = x.shape[0]
        total_loss += loss.item() * bs
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
    model.eval()
    preds = []
    targets = []
    for batch in loader:
        x = batch["voxel"].to(device, non_blocking=True)
        y = batch["target"].to(device, non_blocking=True)
        pred_norm = model(x)
        pred = pred_norm * target_std + target_mean
        preds.append(pred.cpu())
        targets.append(y.cpu())

    pred_all = torch.cat(preds, dim=0)
    target_all = torch.cat(targets, dim=0)
    diff = pred_all - target_all
    mae = diff.abs().mean(dim=0)
    rmse = torch.sqrt((diff * diff).mean(dim=0))
    return EvalResult(
        mean_mae=float(mae.mean().item()),
        mae=[float(x) for x in mae.tolist()],
        rmse=[float(x) for x in rmse.tolist()],
    )


def ensure_cache(args: argparse.Namespace, channels: list[str]) -> str:
    tag = voxel_cache_tag(args.grid_length, args.cube_size_bohr, channels, args.gaussian_sigma)
    meta_path = metadata_path(args.cache_dir, tag)
    max_samples_by_split = {
        "train": args.max_train_samples,
        "valid": args.max_val_samples,
        "test": args.max_test_samples,
    }
    if all(value is None for value in max_samples_by_split.values()):
        max_samples_by_split = None
    if args.prepare_cache or not os.path.exists(meta_path):
        stats = build_voxel_cache(
            pkl_path=args.pkl_path,
            csv_path=args.csv_path,
            cache_dir=args.cache_dir,
            grid_length=args.grid_length,
            cube_size_bohr=args.cube_size_bohr,
            channels=channels,
            gaussian_sigma=args.gaussian_sigma,
            split_col=args.split_col,
            splits=("train", "valid", "test"),
            max_samples_per_split=None,
            max_samples_by_split=max_samples_by_split,
            overwrite=args.overwrite_cache,
            save_dtype=args.save_dtype,
            workers=max(args.prepare_cache_workers, 1),
        )
        print(f"Cache build stats: {stats}", flush=True)
    return tag


def save_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def resolve_run_name(args: argparse.Namespace) -> str:
    if args.run_name:
        return args.run_name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"voxel_{args.run_kind}_{timestamp}"


def resolve_run_dir(args: argparse.Namespace, run_name: str) -> str:
    if args.output_dir:
        return args.output_dir
    return os.path.join(args.checkpoint_root, args.run_kind, run_name)


def make_checkpoint_payload(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler | None,
    epoch: int,
    best_val: float,
    config_payload: dict,
) -> dict:
    return {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": None if scaler is None else scaler.state_dict(),
        "config": config_payload,
        "epoch": epoch,
        "best_val_mean_mae": best_val,
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    channels = [x.strip() for x in args.channels.split(",") if x.strip()]
    if not channels:
        raise ValueError("At least one channel is required.")

    run_name = resolve_run_name(args)
    run_dir = resolve_run_dir(args, run_name)
    wandb_dir = os.path.join(args.wandb_root, args.run_kind)

    tag = ensure_cache(args, channels)
    if args.prepare_only:
        return

    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(wandb_dir, exist_ok=True)

    rows = load_energy_rows(args.csv_path)
    train_rows = filter_rows(rows, "train", split_col=args.split_col, max_samples=args.max_train_samples)
    mean_np, std_np = compute_target_stats(train_rows)
    device = torch.device(args.device)
    target_mean = torch.from_numpy(mean_np).to(device)
    target_std = torch.from_numpy(std_np).to(device)

    train_ds = EDBenchVoxelDataset(
        csv_path=args.csv_path,
        cache_dir=args.cache_dir,
        grid_length=args.grid_length,
        cube_size_bohr=args.cube_size_bohr,
        channels=channels,
        split="train",
        split_col=args.split_col,
        gaussian_sigma=args.gaussian_sigma,
        max_samples=args.max_train_samples,
    )
    val_ds = EDBenchVoxelDataset(
        csv_path=args.csv_path,
        cache_dir=args.cache_dir,
        grid_length=args.grid_length,
        cube_size_bohr=args.cube_size_bohr,
        channels=channels,
        split="valid",
        split_col=args.split_col,
        gaussian_sigma=args.gaussian_sigma,
        max_samples=args.max_val_samples,
    )
    test_ds = EDBenchVoxelDataset(
        csv_path=args.csv_path,
        cache_dir=args.cache_dir,
        grid_length=args.grid_length,
        cube_size_bohr=args.cube_size_bohr,
        channels=channels,
        split="test",
        split_col=args.split_col,
        gaussian_sigma=args.gaussian_sigma,
        max_samples=args.max_test_samples,
    )

    train_loader = make_loader(train_ds, args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = make_loader(val_ds, args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = make_loader(test_ds, args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = VoxelDenseNetRegressor(
        in_channels=len(channels),
        out_dim=len(LABEL_NAMES),
        growth_rate=args.growth_rate,
        block_config=(args.dense1, args.dense2),
        num_init_features=args.num_init_features,
        drop_rate=args.drop_rate,
        small_inputs=True,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"VoxelDenseNet params: {n_params:,}", flush=True)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    scaler = torch.amp.GradScaler("cuda") if (args.amp and device.type == "cuda") else None

    best_val = math.inf
    best_path = os.path.join(run_dir, "best.pt")
    last_path = os.path.join(run_dir, "last.pt")
    history: list[dict] = []

    config_payload = vars(args).copy()
    config_payload["channels"] = channels
    config_payload["cache_tag"] = tag
    config_payload["run_name"] = run_name
    config_payload["resolved_run_dir"] = run_dir
    config_payload["resolved_wandb_dir"] = wandb_dir
    config_payload["num_train_samples"] = len(train_ds)
    config_payload["num_val_samples"] = len(val_ds)
    config_payload["num_test_samples"] = len(test_ds)
    config_payload["num_parameters"] = n_params
    config_payload["label_names"] = list(LABEL_NAMES)
    config_payload["target_mean"] = mean_np.tolist()
    config_payload["target_std"] = std_np.tolist()
    save_json(os.path.join(run_dir, "config.json"), config_payload)

    wandb_run = None
    if args.wandb:
        if not _WANDB_AVAILABLE:
            print("WARNING: wandb not installed, skipping W&B logging.", flush=True)
        else:
            tags = [tag for tag in (x.strip() for x in args.wandb_tags.split(",")) if tag]
            wandb_run = _wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                group=args.wandb_group,
                name=run_name,
                dir=wandb_dir,
                config=config_payload,
                tags=tags,
            )
            wandb_run.summary["num_parameters"] = n_params
            wandb_run.summary["cache_tag"] = tag
            wandb_run.summary["num_train_samples"] = len(train_ds)
            wandb_run.summary["num_val_samples"] = len(val_ds)
            wandb_run.summary["num_test_samples"] = len(test_ds)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            target_mean=target_mean,
            target_std=target_std,
            device=device,
            grad_clip=args.grad_clip,
        )
        val_res = evaluate(model, val_loader, target_mean, target_std, device)
        scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        history.append(
            {"epoch": epoch, "train_loss": train_loss, "val_mean_mae": val_res.mean_mae, "lr": lr}
        )
        print(
            f"Epoch {epoch:4d}/{args.epochs}  "
            f"train_loss={train_loss:.6f}  val_mean_MAE={val_res.mean_mae:.6f}  lr={lr:.2e}",
            flush=True,
        )
        for name, mae, rmse in zip(LABEL_NAMES, val_res.mae, val_res.rmse):
            print(f"  {name:<14} MAE={mae:.6f}  RMSE={rmse:.6f}", flush=True)

        epoch_payload = make_checkpoint_payload(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            best_val=best_val,
            config_payload=config_payload,
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
            for name, mae, rmse in zip(LABEL_NAMES, val_res.mae, val_res.rmse):
                log_payload[f"val/{name}_mae"] = mae
                log_payload[f"val/{name}_rmse"] = rmse
            wandb_run.log(log_payload, step=epoch)

        if val_res.mean_mae < best_val:
            best_val = val_res.mean_mae
            best_payload = make_checkpoint_payload(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                best_val=best_val,
                config_payload=config_payload,
            )
            torch.save(
                best_payload,
                best_path,
            )
            if wandb_run is not None:
                wandb_run.summary["best_val_mean_mae"] = best_val
                wandb_run.summary["best_epoch"] = epoch

    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])

    test_res = evaluate(model, test_loader, target_mean, target_std, device)
    print("=== Test Set Evaluation ===", flush=True)
    print(f"  Mean MAE: {test_res.mean_mae:.6f}", flush=True)
    for name, mae, rmse in zip(LABEL_NAMES, test_res.mae, test_res.rmse):
        print(f"  {name:<14} MAE={mae:.6f}  RMSE={rmse:.6f}", flush=True)

    if wandb_run is not None:
        test_payload = {"test/mean_mae": test_res.mean_mae}
        for name, mae, rmse in zip(LABEL_NAMES, test_res.mae, test_res.rmse):
            test_payload[f"test/{name}_mae"] = mae
            test_payload[f"test/{name}_rmse"] = rmse
        wandb_run.log(test_payload)
        wandb_run.summary["best_checkpoint"] = best_path
        wandb_run.summary["last_checkpoint"] = last_path
        wandb_run.finish()

    save_json(
        os.path.join(run_dir, "metrics.json"),
        {
            "run_name": run_name,
            "run_kind": args.run_kind,
            "run_dir": run_dir,
            "wandb_dir": wandb_dir,
            "best_val_mean_mae": best_val,
            "test_mean_mae": test_res.mean_mae,
            "test_mae": test_res.mae,
            "test_rmse": test_res.rmse,
            "history": history,
        },
    )


if __name__ == "__main__":
    main()
