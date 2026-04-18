"""
Energy label utilities for the EDBench dataset.

The CSV stores all 6 energy values in a single space-separated ``label``
column.  The order is fixed:

    0  DF-RKS Final
    1  Nuclear Repulsion
    2  One-Electron
    3  Two-Electron
    4  DFT XC
    5  Total

Usage
-----
::

    from ed2e.data.energy_stats import (
        TARGET_NAMES, NUM_TARGETS,
        load_energy_labels, load_split_ids, compute_energy_stats,
    )

    labels   = load_energy_labels("data/.../ed_energy_5w.csv")
    splits   = load_split_ids("data/.../ed_energy_5w.csv", split_col="scaffold_split")
    stats    = compute_energy_stats(labels, train_mol_ids=splits["train"])
    # stats["mean"]  (6,) float32
    # stats["std"]   (6,) float32
"""
from __future__ import annotations

import csv
import json
import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── Public constants ──────────────────────────────────────────────────────────

TARGET_NAMES: Tuple[str, ...] = (
    "DF-RKS Final",
    "Nuclear Repulsion",
    "One-Electron",
    "Two-Electron",
    "DFT XC",
    "Total",
)

NUM_TARGETS: int = len(TARGET_NAMES)


# ── CSV helpers ───────────────────────────────────────────────────────────────

def _parse_label(label_str: str) -> np.ndarray:
    """Parse a single space-separated label string → (6,) float32."""
    parts = label_str.strip().split()
    if len(parts) != NUM_TARGETS:
        raise ValueError(
            f"Expected {NUM_TARGETS} energy values, got {len(parts)}: {label_str!r}"
        )
    return np.array([float(v) for v in parts], dtype=np.float32)


def load_energy_labels(csv_path: str) -> Dict[str, np.ndarray]:
    """
    Load all energy labels from the EDBench CSV.

    Returns
    -------
    Dict[mol_id_str, np.ndarray of shape (6,)]
        Energies in order: DF-RKS Final, Nuclear Repulsion, One-Electron,
        Two-Electron, DFT XC, Total.
    """
    labels: Dict[str, np.ndarray] = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mol_id = str(row["index"])
            try:
                labels[mol_id] = _parse_label(row["label"])
            except (KeyError, ValueError):
                continue  # skip malformed rows
    return labels


def load_split_ids(
    csv_path: str,
    split_col: str = "scaffold_split",
) -> Dict[str, List[str]]:
    """
    Read train/val/test mol_id lists from the CSV split column.

    Parameters
    ----------
    csv_path  : Path to ed_energy_5w.csv
    split_col : Column name; one of ``"scaffold_split"`` (default) or
                ``"random_split"``.

    Returns
    -------
    {"train": [...], "val": [...], "test": [...]}
    """
    splits: Dict[str, List[str]] = {"train": [], "val": [], "test": []}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tag = row.get(split_col, "").strip()
            if tag in splits:
                splits[tag].append(str(row["index"]))
    return splits


# ── Statistics ────────────────────────────────────────────────────────────────

def compute_energy_stats(
    labels: Dict[str, np.ndarray],
    train_mol_ids: List[str],
) -> Dict[str, np.ndarray]:
    """
    Compute per-target mean and std from the training set.

    Only molecules present in ``labels`` **and** ``train_mol_ids`` are used.
    ``std`` is clamped to ≥ 1e-6 to avoid division by zero.

    Returns
    -------
    {"mean": (6,) float32, "std": (6,) float32}
    """
    rows = []
    for mol_id in train_mol_ids:
        if mol_id in labels:
            rows.append(labels[mol_id])
    if not rows:
        raise ValueError("No training molecules found in label dict.")

    arr  = np.stack(rows, axis=0)           # (N_train, 6)
    mean = arr.mean(axis=0).astype(np.float32)
    std  = arr.std(axis=0).astype(np.float32)
    std  = np.where(std < 1e-6, np.float32(1.0), std)
    return {"mean": mean, "std": std}


def save_energy_stats(stats: Dict[str, np.ndarray], path: str) -> None:
    """Save mean/std to a JSON file (human-readable)."""
    payload = {
        "target_names": list(TARGET_NAMES),
        "mean": stats["mean"].tolist(),
        "std":  stats["std"].tolist(),
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def load_energy_stats(path: str) -> Dict[str, np.ndarray]:
    """Load mean/std from a JSON file produced by :func:`save_energy_stats`."""
    with open(path) as f:
        payload = json.load(f)
    return {
        "mean": np.array(payload["mean"], dtype=np.float32),
        "std":  np.array(payload["std"],  dtype=np.float32),
    }


# ── Normalisation helpers ─────────────────────────────────────────────────────

def normalise(energies: np.ndarray, stats: Dict[str, np.ndarray]) -> np.ndarray:
    """Apply z-score normalisation:  (x - mean) / std."""
    return (energies - stats["mean"]) / stats["std"]


def denormalise(normed: np.ndarray, stats: Dict[str, np.ndarray]) -> np.ndarray:
    """Invert z-score normalisation:  x * std + mean."""
    return normed * stats["std"] + stats["mean"]


__all__ = [
    "TARGET_NAMES",
    "NUM_TARGETS",
    "load_energy_labels",
    "load_split_ids",
    "compute_energy_stats",
    "save_energy_stats",
    "load_energy_stats",
    "normalise",
    "denormalise",
]
