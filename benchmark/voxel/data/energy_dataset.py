from __future__ import annotations

import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import pickle
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

try:
    from scipy.ndimage import gaussian_filter
except Exception:  # pragma: no cover
    gaussian_filter = None


LABEL_NAMES = (
    "E1_Final",
    "E2_NucRepul",
    "E3_OneElec",
    "E4_TwoElec",
    "E5_XC",
    "E6_Total",
)

CHANNEL_OPTIONS = ("density", "atom_occupancy", "atom_z")


@dataclass(frozen=True)
class EnergyRow:
    mol_id: str
    labels: np.ndarray
    scaffold_split: str
    random_split: str


def voxel_cache_tag(
    grid_length: int,
    cube_size_bohr: float,
    channels: Sequence[str],
    gaussian_sigma: float,
) -> str:
    ch_tag = "-".join(channels)
    sigma_tag = f"{gaussian_sigma:.2f}".replace(".", "p")
    cube_tag = f"{cube_size_bohr:.2f}".replace(".", "p")
    return f"g{grid_length}_c{cube_tag}_s{sigma_tag}_{ch_tag}"


def voxel_cache_path(cache_dir: str, mol_id: str, tag: str) -> str:
    return os.path.join(cache_dir, f"{mol_id}_{tag}.pt")


def metadata_path(cache_dir: str, tag: str) -> str:
    return os.path.join(cache_dir, f"meta_{tag}.json")


def load_energy_rows(csv_path: str) -> list[EnergyRow]:
    rows: list[EnergyRow] = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels = np.fromstring(row["label"], sep=" ", dtype=np.float32)
            if labels.shape != (6,):
                raise ValueError(
                    f"Expected 6 labels for mol_id={row['index']}, got {labels.shape}"
                )
            rows.append(
                EnergyRow(
                    mol_id=row["index"],
                    labels=labels,
                    scaffold_split=row["scaffold_split"],
                    random_split=row["random_split"],
                )
            )
    return rows


def filter_rows(
    rows: Sequence[EnergyRow],
    split: str,
    split_col: str = "scaffold_split",
    max_samples: Optional[int] = None,
) -> list[EnergyRow]:
    if split_col not in {"scaffold_split", "random_split"}:
        raise ValueError(f"Unsupported split_col={split_col}")
    selected = [r for r in rows if getattr(r, split_col) == split]
    if max_samples is not None:
        selected = selected[:max_samples]
    return selected


def _weighted_center(
    atom_coords: np.ndarray,
    ed_coords: np.ndarray,
    densities: np.ndarray,
) -> np.ndarray:
    total = float(densities.sum())
    if total > 1e-12:
        return (ed_coords * densities[:, None]).sum(axis=0) / total
    return atom_coords.mean(axis=0)


def voxelize_edbench_entry(
    atom_coords: np.ndarray,
    atom_types: np.ndarray,
    ed_coords: np.ndarray,
    densities: np.ndarray,
    grid_length: int,
    cube_size_bohr: float,
    channels: Sequence[str],
    gaussian_sigma: float = 0.0,
) -> np.ndarray:
    channels = tuple(channels)
    for ch in channels:
        if ch not in CHANNEL_OPTIONS:
            raise ValueError(f"Unsupported channel '{ch}'")

    grid = np.zeros((len(channels), grid_length, grid_length, grid_length), dtype=np.float32)
    center = _weighted_center(atom_coords, ed_coords, densities).astype(np.float32)
    half = 0.5 * float(cube_size_bohr)
    voxel_size = float(cube_size_bohr) / float(grid_length)

    def _coords_to_idx(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        rel = coords - center[None, :]
        idx = np.floor((rel + half) / voxel_size).astype(np.int64)
        valid = ((idx >= 0) & (idx < grid_length)).all(axis=1)
        return idx, valid

    ed_idx, ed_valid = _coords_to_idx(ed_coords)
    atom_idx, atom_valid = _coords_to_idx(atom_coords)

    for ci, ch in enumerate(channels):
        if ch == "density":
            valid_idx = ed_idx[ed_valid]
            valid_den = densities[ed_valid].astype(np.float32, copy=False)
            np.add.at(grid[ci], (valid_idx[:, 0], valid_idx[:, 1], valid_idx[:, 2]), valid_den)
        elif ch == "atom_occupancy":
            valid_idx = atom_idx[atom_valid]
            np.add.at(grid[ci], (valid_idx[:, 0], valid_idx[:, 1], valid_idx[:, 2]), 1.0)
        elif ch == "atom_z":
            valid_idx = atom_idx[atom_valid]
            valid_z = atom_types[atom_valid].astype(np.float32, copy=False)
            np.add.at(grid[ci], (valid_idx[:, 0], valid_idx[:, 1], valid_idx[:, 2]), valid_z)

    if gaussian_sigma > 0.0:
        if gaussian_filter is None:
            raise RuntimeError("scipy.ndimage.gaussian_filter is required when gaussian_sigma > 0")
        for ci in range(grid.shape[0]):
            grid[ci] = gaussian_filter(grid[ci], sigma=gaussian_sigma)

    return grid


def build_voxel_cache(
    pkl_path: str,
    csv_path: str,
    cache_dir: str,
    grid_length: int,
    cube_size_bohr: float,
    channels: Sequence[str],
    gaussian_sigma: float = 0.0,
    split_col: str = "scaffold_split",
    splits: Sequence[str] = ("train", "valid", "test"),
    max_samples_per_split: Optional[int] = None,
    max_samples_by_split: Optional[dict[str, Optional[int]]] = None,
    overwrite: bool = False,
    save_dtype: str = "float16",
    workers: int = 1,
    show_progress: bool = True,
) -> dict[str, int]:
    os.makedirs(cache_dir, exist_ok=True)
    rows = load_energy_rows(csv_path)

    with open(pkl_path, "rb") as f:
        raw: dict = pickle.load(f)

    tag = voxel_cache_tag(grid_length, cube_size_bohr, channels, gaussian_sigma)
    built = 0
    skipped = 0
    missing = 0
    split_counts: dict[str, int] = {}
    work_items: list[tuple[EnergyRow, str]] = []

    for split in splits:
        split_max_samples = max_samples_per_split
        if max_samples_by_split is not None and split in max_samples_by_split:
            split_max_samples = max_samples_by_split[split]
        split_rows = filter_rows(rows, split=split, split_col=split_col, max_samples=split_max_samples)
        split_counts[split] = len(split_rows)
        for row in split_rows:
            out_path = voxel_cache_path(cache_dir, row.mol_id, tag)
            work_items.append((row, out_path))

    if save_dtype not in {"float16", "float32"}:
        raise ValueError("save_dtype must be 'float16' or 'float32'")

    def _build_one(row: EnergyRow, out_path: str) -> str:
        if os.path.exists(out_path) and not overwrite:
            return "skipped"

        entry = raw.get(row.mol_id)
        if entry is None:
            return "missing"

        voxel = voxelize_edbench_entry(
            atom_coords=entry["mol"]["coords"].astype(np.float32),
            atom_types=entry["mol"]["x"].astype(np.int64),
            ed_coords=entry["electronic_density"]["coords"].astype(np.float32),
            densities=entry["electronic_density"]["density"].astype(np.float32),
            grid_length=grid_length,
            cube_size_bohr=cube_size_bohr,
            channels=channels,
            gaussian_sigma=gaussian_sigma,
        )
        if save_dtype == "float16":
            voxel = voxel.astype(np.float16)
        else:
            voxel = voxel.astype(np.float32)

        torch.save(
            {
                "voxel": torch.from_numpy(voxel),
                "mol_id": row.mol_id,
                "labels": torch.from_numpy(row.labels.copy()),
                "tag": tag,
            },
            out_path,
        )
        return "built"

    progress = tqdm(
        total=len(work_items),
        desc=f"Voxel cache [{tag}]",
        unit="mol",
        disable=(not show_progress),
    )
    try:
        if workers <= 1:
            for row, out_path in work_items:
                status = _build_one(row, out_path)
                if status == "built":
                    built += 1
                elif status == "skipped":
                    skipped += 1
                else:
                    missing += 1
                progress.update(1)
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(_build_one, row, out_path) for row, out_path in work_items]
                for future in as_completed(futures):
                    status = future.result()
                    if status == "built":
                        built += 1
                    elif status == "skipped":
                        skipped += 1
                    else:
                        missing += 1
                    progress.update(1)
    finally:
        progress.close()

    with open(metadata_path(cache_dir, tag), "w") as f:
        json.dump(
            {
                "pkl_path": pkl_path,
                "csv_path": csv_path,
                "grid_length": grid_length,
                "cube_size_bohr": cube_size_bohr,
                "channels": list(channels),
                "gaussian_sigma": gaussian_sigma,
                "split_col": split_col,
                "splits": list(splits),
                "split_counts": split_counts,
                "max_samples_per_split": max_samples_per_split,
                "max_samples_by_split": max_samples_by_split,
                "save_dtype": save_dtype,
                "workers": workers,
                "tag": tag,
            },
            f,
            indent=2,
            sort_keys=True,
        )

    return {"built": built, "skipped": skipped, "missing": missing, **split_counts}


class EDBenchVoxelDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        cache_dir: str,
        grid_length: int,
        cube_size_bohr: float,
        channels: Sequence[str],
        split: str,
        split_col: str = "scaffold_split",
        gaussian_sigma: float = 0.0,
        max_samples: Optional[int] = None,
        require_all_cached: bool = True,
    ) -> None:
        self.rows = filter_rows(
            load_energy_rows(csv_path),
            split=split,
            split_col=split_col,
            max_samples=max_samples,
        )
        self.tag = voxel_cache_tag(grid_length, cube_size_bohr, channels, gaussian_sigma)
        self.cache_dir = cache_dir
        self.items: list[EnergyRow] = []
        missing: list[str] = []
        for row in self.rows:
            path = voxel_cache_path(cache_dir, row.mol_id, self.tag)
            if os.path.exists(path):
                self.items.append(row)
            else:
                missing.append(row.mol_id)
        if missing and require_all_cached:
            raise FileNotFoundError(
                f"{len(missing)} cached voxel files are missing for tag={self.tag}. "
                f"First few: {missing[:5]}"
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        row = self.items[idx]
        path = voxel_cache_path(self.cache_dir, row.mol_id, self.tag)
        payload = torch.load(path, map_location="cpu")
        return {
            "voxel": payload["voxel"].float(),
            "target": torch.from_numpy(row.labels.copy()).float(),
            "mol_id": row.mol_id,
        }


def compute_target_stats(rows: Iterable[EnergyRow]) -> tuple[np.ndarray, np.ndarray]:
    arr = np.stack([row.labels for row in rows], axis=0).astype(np.float32)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    std = np.where(std > 1e-8, std, 1.0).astype(np.float32)
    return mean.astype(np.float32), std


def voxel_collate_fn(batch: list[dict[str, torch.Tensor | str]]) -> dict[str, torch.Tensor | list[str]]:
    voxels = torch.stack([item["voxel"] for item in batch], dim=0)
    targets = torch.stack([item["target"] for item in batch], dim=0)
    mol_ids = [str(item["mol_id"]) for item in batch]
    return {"voxel": voxels, "target": targets, "mol_id": mol_ids}
