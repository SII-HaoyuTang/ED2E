#!/usr/bin/env python3
from __future__ import annotations

import argparse

from benchmark.voxel.data.energy_dataset import build_voxel_cache


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Precompute voxel caches for EDBench.")
    p.add_argument("--pkl-path", default="data/ed_energy_5w/processed/mol_EDthresh0.05_data.pkl")
    p.add_argument("--csv-path", default="data/ed_energy_5w/raw/ed_energy_5w.csv")
    p.add_argument("--cache-dir", default="data/ed_energy_5w/cache_voxel")
    p.add_argument("--grid-length", type=int, default=14)
    p.add_argument("--cube-size-bohr", type=float, default=32.0)
    p.add_argument("--channels", default="density", help="Comma-separated voxel channels.")
    p.add_argument("--gaussian-sigma", type=float, default=0.0)
    p.add_argument("--split-col", choices=("scaffold_split", "random_split"), default="scaffold_split")
    p.add_argument("--max-samples-per-split", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--save-dtype", choices=("float16", "float32"), default="float16")
    p.add_argument("--workers", type=int, default=4, help="Thread workers used within one process.")
    p.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bar.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    channels = [x.strip() for x in args.channels.split(",") if x.strip()]
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
        max_samples_per_split=args.max_samples_per_split,
        overwrite=args.overwrite,
        save_dtype=args.save_dtype,
        workers=max(args.workers, 1),
        show_progress=not args.no_progress,
    )
    print(stats, flush=True)


if __name__ == "__main__":
    main()
