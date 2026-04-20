"""
Packed Stage 3 cache for high-throughput training IO.

The default Stage 3 cache format is one pickle per molecule (or a zip bundle of
those pickles). That is convenient for debugging but suboptimal for training,
because every sample load pays Python pickle deserialization overhead.

This module provides a packed, mmap-friendly representation:

- each tensor field is stored as one flat `.npy` array
- per-sample boundaries are stored as pointer arrays in `index.npz`
- training workers lazily open read-only memmaps once and then slice views

The runtime goal is to keep training-time sample loading on the "slice + batch"
path rather than the "rebuild Python object graph from pickle" path.
"""
from __future__ import annotations

import json
import os
import pickle
import re
import zipfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from numpy.lib.format import open_memmap
from torch.utils.data import Dataset

from .stage3_local import (
    Stage3Sample,
    collate_stage3_samples,
    is_stage3_bundle_path,
    list_stage3_bundle_ids,
    load_stage3_sample,
    stage3_bundle_member,
)

PACKED_FORMAT = "stage3_packed_v1"
PACKED_META_NAME = "meta.json"
PACKED_INDEX_NAME = "index.npz"

_DIR_FILE_RE = re.compile(r"^(?P<mol_id>.+)_stage3_lk(?P<lk>\d+)_ck(?P<ck>\d+)_a(?P<a>\d+)(?P<extra>[^.]*)\.pkl$")

_COUNT_KEYS = (
    "node",
    "chart",
    "membership",
    "local_edge",
    "group",
    "chart_edge",
    "overlap_edge",
    "overlap_shared_pair",
    "overlap_shared_ptr",
    "inter_edge",   # Stage 5 新增
)

_FIELD_SPECS = (
    ("node_xyz", "node", np.float32),
    ("node_normal", "node", np.float32),
    ("node_scalar_raw", "node", np.float32),
    ("node_vector_raw", "node", np.float32),
    ("node_tangent_basis", "node", np.float32),
    ("chart_membership", "membership", np.int32),
    ("membership_sr", "membership", np.float32),
    ("membership_weight", "membership", np.float32),
    ("local_coords", "membership", np.float32),
    ("quadrant", "membership", np.int8),
    ("local_knn_edge_index", "local_edge", np.int32),
    ("local_edge_attr", "local_edge", np.float32),
    ("chart_es_geom_static", "chart", np.float32),
    ("chart_anchor_pos", "chart", np.float32),
    ("chart_anchor_mask", "chart", np.bool_),
    ("reference_chart_id", "group", np.int32),
    ("chart_graph_edge_index", "chart_edge", np.int32),
    ("overlap_edge_index", "overlap_edge", np.int32),
    ("overlap_shared_membership_index", "overlap_shared_pair", np.int32),
    ("overlap_shared_ptr", "overlap_shared_ptr", np.int32),
    ("overlap_jaccard", "overlap_edge", np.float32),
    ("chart_center", "chart", np.float32),
    ("chart_center_normal", "chart", np.float32),
    ("chart_frame", "chart", np.float32),
    ("chart_level_id", "chart", np.int32),
    ("chart_component_id", "chart", np.int32),
    ("chart_group_id", "chart", np.int32),
    ("chart_seed_node_index", "chart", np.int32),
    ("chart_stage2_id", "chart", np.int32),
    ("group_level_id", "group", np.int32),
    ("group_component_id", "group", np.int32),
    # Stage 4 新增
    ("intra_geom_static", "chart", np.float32),
    ("chart_to_ref", "chart", np.int32),
    ("overlap_edge_to_chart_edge_index", "overlap_edge", np.int32),
    # Stage 5 新增
    ("inter_level_edge_index", "inter_edge", np.int32),
    ("inter_level_weights", "inter_edge", np.float32),
    ("inter_level_edge_attr", "inter_edge", np.float32),
)

_EDGE_FIELDS = {
    "local_knn_edge_index",
    "chart_graph_edge_index",
    "overlap_edge_index",
    "inter_level_edge_index",   # Stage 5 新增
}

_META_CHART_FIELDS = (
    "chart_center",
    "chart_center_normal",
    "chart_frame",
    "chart_level_id",
    "chart_component_id",
    "chart_group_id",
    "chart_seed_node_index",
    "chart_stage2_id",
    "chart_to_ref",   # Stage 4 新增
)

_META_GROUP_FIELDS = (
    "group_level_id",
    "group_component_id",
)


def stage3_packed_meta_path(packed_dir: str) -> str:
    return os.path.join(packed_dir, PACKED_META_NAME)


def stage3_packed_index_path(packed_dir: str) -> str:
    return os.path.join(packed_dir, PACKED_INDEX_NAME)


def is_stage3_packed_dir(path: str) -> bool:
    return os.path.isdir(path) and os.path.exists(stage3_packed_meta_path(path)) and os.path.exists(stage3_packed_index_path(path))


def list_stage3_packed_ids(packed_dir: str) -> List[str]:
    with np.load(stage3_packed_index_path(packed_dir), allow_pickle=False) as index:
        return [str(x) for x in index["mol_ids"].tolist()]


def load_stage3_packed_meta(packed_dir: str) -> Dict[str, Any]:
    with open(stage3_packed_meta_path(packed_dir), "r", encoding="utf-8") as f:
        return json.load(f)


def _storage_view(field: str, arr: np.ndarray) -> np.ndarray:
    if field in _EDGE_FIELDS:
        return np.asarray(arr).T
    return np.asarray(arr)


def _restore_view(field: str, arr: np.ndarray) -> np.ndarray:
    if field in _EDGE_FIELDS:
        return np.asarray(arr).T
    return np.asarray(arr)


def _extract_field(sample: Stage3Sample, field: str) -> np.ndarray:
    if field in _META_CHART_FIELDS or field in _META_GROUP_FIELDS:
        return np.asarray(sample.chart_frame_metadata[field])
    return np.asarray(getattr(sample, field))


def _sample_entity_counts(sample: Stage3Sample) -> Dict[str, int]:
    return {
        "node": int(len(sample.node_xyz)),
        "chart": int(len(sample.chart_es_geom_static)),
        "membership": int(len(sample.chart_membership)),
        "local_edge": int(sample.local_knn_edge_index.shape[1]),
        "group": int(len(sample.reference_chart_id)),
        "chart_edge": int(sample.chart_graph_edge_index.shape[1]),
        "overlap_edge": int(sample.overlap_edge_index.shape[1]),
        "overlap_shared_pair": int(len(sample.overlap_shared_membership_index)),
        "overlap_shared_ptr": int(len(sample.overlap_shared_ptr)),
        "inter_edge": int(sample.inter_level_edge_index.shape[1]),   # Stage 5 新增
    }


def _assert_sample_schema(sample: Stage3Sample, field_shapes: Dict[str, List[int]]) -> None:
    counts = _sample_entity_counts(sample)
    for field, count_key, _ in _FIELD_SPECS:
        storage = _storage_view(field, _extract_field(sample, field))
        if storage.ndim == 0:
            raise ValueError(f"Stage3 field '{field}' must be at least 1-D")
        if int(storage.shape[0]) != counts[count_key]:
            raise ValueError(
                f"Stage3 field '{field}' count mismatch: "
                f"shape[0]={storage.shape[0]} vs expected {counts[count_key]}"
            )
        trailing = list(storage.shape[1:])
        prev = field_shapes.get(field)
        if prev is None:
            field_shapes[field] = trailing
        elif prev != trailing:
            raise ValueError(
                f"Inconsistent trailing shape for field '{field}': "
                f"{prev} vs {trailing}"
            )


@dataclass
class _SourceSpec:
    kind: str
    source: str
    mol_ids: List[str]
    dir_map: Optional[Dict[str, str]] = None


class _Stage3SourceReader:
    def __init__(self, source: str, mol_ids: Optional[Sequence[str]] = None) -> None:
        self.spec = self._build_spec(source, mol_ids)
        self._zf: Optional[zipfile.ZipFile] = None

    @staticmethod
    def _build_spec(source: str, mol_ids: Optional[Sequence[str]]) -> _SourceSpec:
        if os.path.isdir(source):
            dir_map: Dict[str, str] = {}
            tag_set = set()
            for name in os.listdir(source):
                match = _DIR_FILE_RE.match(name)
                if not match:
                    continue
                mol_id = match.group("mol_id")
                tag_set.add((match.group("lk"), match.group("ck"), match.group("a")))
                dir_map[mol_id] = os.path.join(source, name)
            if not dir_map:
                raise FileNotFoundError(f"No Stage 3 cache files found under {source}")
            if len(tag_set) > 1:
                raise ValueError(
                    "Packed Stage 3 source directory mixes multiple Stage 3 tags; "
                    "use a directory with a single cache configuration."
                )
            ids = sorted(dir_map.keys())
            if mol_ids is not None:
                want = list(mol_ids)
                missing = [mol_id for mol_id in want if mol_id not in dir_map]
                if missing:
                    raise KeyError(f"{len(missing)} requested mol_id values are missing. First few: {missing[:5]}")
                ids = want
            return _SourceSpec(kind="dir", source=source, mol_ids=ids, dir_map=dir_map)

        if is_stage3_bundle_path(source):
            ids = list_stage3_bundle_ids(source)
            if mol_ids is not None:
                want = list(mol_ids)
                have = set(ids)
                missing = [mol_id for mol_id in want if mol_id not in have]
                if missing:
                    raise KeyError(f"{len(missing)} requested mol_id values are missing. First few: {missing[:5]}")
                ids = want
            return _SourceSpec(kind="bundle", source=source, mol_ids=ids)

        sample = load_stage3_sample(source)
        ids = [sample.mol_id]
        if mol_ids is not None and list(mol_ids) != ids:
            raise KeyError(f"Requested mol_ids {list(mol_ids)} do not match the single-sample source {ids}")
        return _SourceSpec(kind="single", source=source, mol_ids=ids)

    def __enter__(self) -> "_Stage3SourceReader":
        if self.spec.kind == "bundle":
            self._zf = zipfile.ZipFile(self.spec.source, "r")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._zf is not None:
            self._zf.close()
            self._zf = None

    @property
    def mol_ids(self) -> List[str]:
        return self.spec.mol_ids

    def load(self, mol_id: str) -> Stage3Sample:
        if self.spec.kind == "dir":
            assert self.spec.dir_map is not None
            return load_stage3_sample(self.spec.dir_map[mol_id])
        if self.spec.kind == "bundle":
            if self._zf is None:
                raise RuntimeError("Bundle source reader must be used as a context manager")
            with self._zf.open(stage3_bundle_member(mol_id), "r") as f:
                return pickle.load(f)
        return load_stage3_sample(self.spec.source)


def pack_stage3_cache(
    source: str,
    packed_dir: str,
    *,
    mol_ids: Optional[Sequence[str]] = None,
    max_samples: Optional[int] = None,
    overwrite: bool = False,
    show_progress: bool = True,
) -> Dict[str, int]:
    from tqdm import tqdm

    if os.path.exists(packed_dir):
        if not os.path.isdir(packed_dir):
            raise ValueError(f"Packed output path exists and is not a directory: {packed_dir}")
        if not overwrite and os.listdir(packed_dir):
            raise FileExistsError(
                f"Packed output directory already exists and is not empty: {packed_dir}. "
                "Use overwrite=True to rewrite the packed arrays."
            )
    os.makedirs(packed_dir, exist_ok=True)

    with _Stage3SourceReader(source, mol_ids=mol_ids) as reader:
        pack_ids = reader.mol_ids[:max_samples] if max_samples is not None else reader.mol_ids
        if not pack_ids:
            raise ValueError("No Stage 3 samples selected for packing.")

        ptrs: Dict[str, List[int]] = {key: [0] for key in _COUNT_KEYS}
        field_shapes: Dict[str, List[int]] = {}

        scan_bar = tqdm(pack_ids, desc="Scan Stage3", unit="mol", disable=(not show_progress))
        for mol_id in scan_bar:
            sample = reader.load(mol_id)
            _assert_sample_schema(sample, field_shapes)
            counts = _sample_entity_counts(sample)
            for key in _COUNT_KEYS:
                ptrs[key].append(ptrs[key][-1] + counts[key])
        scan_bar.close()

    field_meta: Dict[str, Dict[str, Any]] = {}
    arrays: Dict[str, np.memmap] = {}
    for field, count_key, dtype in _FIELD_SPECS:
        trailing = tuple(field_shapes[field])
        shape = (ptrs[count_key][-1],) + trailing
        arrays[field] = open_memmap(
            os.path.join(packed_dir, f"{field}.npy"),
            mode="w+",
            dtype=np.dtype(dtype),
            shape=shape,
        )
        field_meta[field] = {
            "file": f"{field}.npy",
            "dtype": np.dtype(dtype).name,
            "shape_tail": list(trailing),
            "count_key": count_key,
        }

    with _Stage3SourceReader(source, mol_ids=pack_ids) as reader:
        write_bar = tqdm(pack_ids, desc="Pack Stage3", unit="mol", disable=(not show_progress))
        for sample_idx, mol_id in enumerate(write_bar):
            sample = reader.load(mol_id)
            for field, count_key, dtype in _FIELD_SPECS:
                start = ptrs[count_key][sample_idx]
                end = ptrs[count_key][sample_idx + 1]
                storage = _storage_view(field, _extract_field(sample, field)).astype(dtype, copy=False)
                arrays[field][start:end] = storage
        write_bar.close()

    for arr in arrays.values():
        arr.flush()

    mol_arr = np.asarray(pack_ids, dtype=f"<U{max(1, max(len(mol_id) for mol_id in pack_ids))}")
    np.savez(
        stage3_packed_index_path(packed_dir),
        mol_ids=mol_arr,
        **{f"{key}_ptr": np.asarray(value, dtype=np.int64) for key, value in ptrs.items()},
    )

    with open(stage3_packed_meta_path(packed_dir), "w", encoding="utf-8") as f:
        json.dump(
            {
                "format": PACKED_FORMAT,
                "source": source,
                "num_samples": len(pack_ids),
                "fields": field_meta,
            },
            f,
            indent=2,
            sort_keys=True,
        )

    return {
        "num_samples": len(pack_ids),
        "total_nodes": ptrs["node"][-1],
        "total_charts": ptrs["chart"][-1],
        "total_membership": ptrs["membership"][-1],
        "total_local_edges": ptrs["local_edge"][-1],
        "total_chart_edges": ptrs["chart_edge"][-1],
        "total_overlap_edges": ptrs["overlap_edge"][-1],
    }


class Stage3PackedDataset(Dataset):
    """Lazy mmap-backed Stage 3 dataset for training-time sample reads.

    Supports both a single packed directory and a sharded layout (a root
    directory containing a ``manifest.json`` produced by ``Stage3ShardedWriter``).
    In the sharded case the dataset presents a merged view across all shards.
    """

    def __init__(
        self,
        packed_dir: str,
        *,
        mol_ids: Optional[Sequence[str]] = None,
    ) -> None:
        manifest_path = os.path.join(packed_dir, MANIFEST_NAME)
        if os.path.isfile(manifest_path):
            # Sharded layout: load each shard and merge
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            shard_dirs = [os.path.join(packed_dir, s) for s in manifest["shards"]]
            if not shard_dirs:
                raise ValueError(f"manifest.json in {packed_dir} lists no shards")

            # Load first shard to get field metadata
            first_meta = load_stage3_packed_meta(shard_dirs[0])
            if first_meta.get("format") != PACKED_FORMAT:
                raise ValueError(
                    f"Unsupported packed Stage 3 format {first_meta.get('format')}; "
                    f"expected {PACKED_FORMAT}"
                )
            self.packed_dir = packed_dir
            self.meta = first_meta  # field shapes are consistent across shards
            self._shards: List[str] = shard_dirs

            # Build merged mol_ids list and per-shard sample-index mapping
            all_mol_ids: List[str] = []
            # (shard_idx, local_sample_idx) for each global index
            self._shard_sample_map: List[tuple] = []
            shard_ptrs_list: List[Dict[str, np.ndarray]] = []
            for shard_dir in shard_dirs:
                with np.load(stage3_packed_index_path(shard_dir), allow_pickle=False) as idx:
                    shard_mids = [str(x) for x in idx["mol_ids"].tolist()]
                    shard_ptrs = {
                        key: np.asarray(idx[f"{key}_ptr"], dtype=np.int64)
                        for key in _COUNT_KEYS
                    }
                all_mol_ids.extend(shard_mids)
                shard_ptrs_list.append(shard_ptrs)
                shard_idx = len(shard_ptrs_list) - 1
                for local_idx in range(len(shard_mids)):
                    self._shard_sample_map.append((shard_idx, local_idx))

            self._all_mol_ids = all_mol_ids
            self._shard_ptrs_list = shard_ptrs_list
            self._ptrs: Optional[Dict[str, np.ndarray]] = None  # not used in sharded mode
            self._is_sharded = True
        else:
            # Single packed directory
            if not is_stage3_packed_dir(packed_dir):
                raise FileNotFoundError(f"Not a packed Stage 3 directory: {packed_dir}")
            self.packed_dir = packed_dir
            self.meta = load_stage3_packed_meta(packed_dir)
            if self.meta.get("format") != PACKED_FORMAT:
                raise ValueError(
                    f"Unsupported packed Stage 3 format {self.meta.get('format')}; "
                    f"expected {PACKED_FORMAT}"
                )
            with np.load(stage3_packed_index_path(packed_dir), allow_pickle=False) as index:
                self._all_mol_ids = [str(x) for x in index["mol_ids"].tolist()]
                self._ptrs = {
                    key: np.asarray(index[f"{key}_ptr"], dtype=np.int64)
                    for key in _COUNT_KEYS
                }
            self._shards = [packed_dir]
            self._shard_ptrs_list = [self._ptrs]
            self._shard_sample_map = [(0, i) for i in range(len(self._all_mol_ids))]
            self._is_sharded = False

        self._mol_to_index = {mol_id: i for i, mol_id in enumerate(self._all_mol_ids)}
        if mol_ids is None:
            self._sample_indices = np.arange(len(self._all_mol_ids), dtype=np.int64)
        else:
            missing = [mol_id for mol_id in mol_ids if mol_id not in self._mol_to_index]
            if missing:
                raise KeyError(f"{len(missing)} requested mol_id values are missing. First few: {missing[:5]}")
            self._sample_indices = np.asarray(
                [self._mol_to_index[mol_id] for mol_id in mol_ids],
                dtype=np.int64,
            )

        # Per-shard lazy array dicts
        self._shard_arrays: List[Optional[Dict[str, np.ndarray]]] = [None] * len(self._shards)

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        state["_shard_arrays"] = [None] * len(self._shards)
        return state

    def __len__(self) -> int:
        return int(len(self._sample_indices))

    @property
    def mol_ids(self) -> List[str]:
        return [self._all_mol_ids[int(i)] for i in self._sample_indices.tolist()]

    def _ensure_shard_arrays(self, shard_idx: int) -> None:
        if self._shard_arrays[shard_idx] is not None:
            return
        shard_dir = self._shards[shard_idx]
        arrays = {}
        for field, info in self.meta["fields"].items():
            arrays[field] = np.load(
                os.path.join(shard_dir, info["file"]),
                mmap_mode="r",
                allow_pickle=False,
            )
        self._shard_arrays[shard_idx] = arrays

    def _slice_field(self, shard_idx: int, local_sample_index: int, field: str, count_key: str) -> np.ndarray:
        self._ensure_shard_arrays(shard_idx)
        arrays = self._shard_arrays[shard_idx]
        assert arrays is not None
        ptrs = self._shard_ptrs_list[shard_idx]
        start = int(ptrs[count_key][local_sample_index])
        end = int(ptrs[count_key][local_sample_index + 1])
        return _restore_view(field, arrays[field][start:end])

    def __getitem__(self, idx: int) -> Stage3Sample:
        global_index = int(self._sample_indices[idx])
        shard_idx, local_sample_index = self._shard_sample_map[global_index]
        mol_id = self._all_mol_ids[global_index]

        def _sf(field: str, count_key: str) -> np.ndarray:
            return self._slice_field(shard_idx, local_sample_index, field, count_key)

        chart_frame_metadata = {field: _sf(field, "chart") for field in _META_CHART_FIELDS}
        chart_frame_metadata.update({field: _sf(field, "group") for field in _META_GROUP_FIELDS})

        return Stage3Sample(
            mol_id=mol_id,
            node_xyz=_sf("node_xyz", "node"),
            node_normal=_sf("node_normal", "node"),
            node_scalar_raw=_sf("node_scalar_raw", "node"),
            node_vector_raw=_sf("node_vector_raw", "node"),
            node_tangent_basis=_sf("node_tangent_basis", "node"),
            chart_membership=_sf("chart_membership", "membership"),
            membership_sr=_sf("membership_sr", "membership"),
            membership_weight=_sf("membership_weight", "membership"),
            local_coords=_sf("local_coords", "membership"),
            quadrant=_sf("quadrant", "membership"),
            local_knn_edge_index=_sf("local_knn_edge_index", "local_edge"),
            local_edge_attr=_sf("local_edge_attr", "local_edge"),
            chart_es_geom_static=_sf("chart_es_geom_static", "chart"),
            chart_anchor_pos=_sf("chart_anchor_pos", "chart"),
            chart_anchor_mask=_sf("chart_anchor_mask", "chart"),
            reference_chart_id=_sf("reference_chart_id", "group"),
            chart_graph_edge_index=_sf("chart_graph_edge_index", "chart_edge"),
            overlap_edge_index=_sf("overlap_edge_index", "overlap_edge"),
            overlap_shared_membership_index=_sf("overlap_shared_membership_index", "overlap_shared_pair"),
            overlap_shared_ptr=_sf("overlap_shared_ptr", "overlap_shared_ptr"),
            overlap_jaccard=_sf("overlap_jaccard", "overlap_edge"),
            chart_frame_metadata=chart_frame_metadata,
            intra_geom_static=_sf("intra_geom_static", "chart"),
            overlap_edge_to_chart_edge_index=_sf("overlap_edge_to_chart_edge_index", "overlap_edge"),
            inter_level_edge_index=_sf("inter_level_edge_index", "inter_edge"),
            inter_level_weights=_sf("inter_level_weights", "inter_edge"),
            inter_level_edge_attr=_sf("inter_level_edge_attr", "inter_edge"),
        )


def pack_stage3_samples(
    samples: List[Stage3Sample],
    packed_dir: str,
    *,
    overwrite: bool = False,
    show_progress: bool = False,
) -> Dict[str, int]:
    """Pack an in-memory list of Stage3Sample into a packed directory.

    Equivalent to pack_stage3_cache() but accepts a pre-loaded list instead of
    a source path.  Used by Stage3ShardedWriter for streaming packed output.
    """
    from tqdm import tqdm

    if not samples:
        raise ValueError("pack_stage3_samples received an empty list")

    if os.path.exists(packed_dir):
        if not os.path.isdir(packed_dir):
            raise ValueError(f"Packed output path exists and is not a directory: {packed_dir}")
        if not overwrite and os.listdir(packed_dir):
            raise FileExistsError(
                f"Packed output directory already exists and is not empty: {packed_dir}. "
                "Use overwrite=True to rewrite the packed arrays."
            )
    os.makedirs(packed_dir, exist_ok=True)

    ptrs: Dict[str, List[int]] = {key: [0] for key in _COUNT_KEYS}
    field_shapes: Dict[str, List[int]] = {}

    for sample in samples:
        _assert_sample_schema(sample, field_shapes)
        counts = _sample_entity_counts(sample)
        for key in _COUNT_KEYS:
            ptrs[key].append(ptrs[key][-1] + counts[key])

    field_meta: Dict[str, Dict[str, Any]] = {}
    arrays: Dict[str, np.memmap] = {}
    for field, count_key, dtype in _FIELD_SPECS:
        trailing = tuple(field_shapes[field])
        shape = (ptrs[count_key][-1],) + trailing
        arrays[field] = open_memmap(
            os.path.join(packed_dir, f"{field}.npy"),
            mode="w+",
            dtype=np.dtype(dtype),
            shape=shape,
        )
        field_meta[field] = {
            "file": f"{field}.npy",
            "dtype": np.dtype(dtype).name,
            "shape_tail": list(trailing),
            "count_key": count_key,
        }

    pack_bar = tqdm(samples, desc="Pack Stage3", unit="mol", disable=(not show_progress))
    for sample_idx, sample in enumerate(pack_bar):
        for field, count_key, dtype in _FIELD_SPECS:
            start = ptrs[count_key][sample_idx]
            end = ptrs[count_key][sample_idx + 1]
            storage = _storage_view(field, _extract_field(sample, field)).astype(dtype, copy=False)
            arrays[field][start:end] = storage
    pack_bar.close()

    for arr in arrays.values():
        arr.flush()

    pack_ids = [s.mol_id for s in samples]
    mol_arr = np.asarray(pack_ids, dtype=f"<U{max(1, max(len(m) for m in pack_ids))}")
    np.savez(
        stage3_packed_index_path(packed_dir),
        mol_ids=mol_arr,
        **{f"{key}_ptr": np.asarray(value, dtype=np.int64) for key, value in ptrs.items()},
    )

    with open(stage3_packed_meta_path(packed_dir), "w", encoding="utf-8") as f:
        json.dump(
            {
                "format": PACKED_FORMAT,
                "source": "in-memory",
                "num_samples": len(pack_ids),
                "fields": field_meta,
            },
            f,
            indent=2,
            sort_keys=True,
        )

    return {
        "num_samples": len(pack_ids),
        "total_nodes": ptrs["node"][-1],
        "total_charts": ptrs["chart"][-1],
        "total_membership": ptrs["membership"][-1],
        "total_local_edges": ptrs["local_edge"][-1],
        "total_chart_edges": ptrs["chart_edge"][-1],
        "total_overlap_edges": ptrs["overlap_edge"][-1],
        "total_inter_edges": ptrs["inter_edge"][-1],
    }


MANIFEST_NAME = "manifest.json"


class Stage3ShardedWriter:
    """Streaming Stage3 packed writer that automatically shards output.

    Usage::

        writer = Stage3ShardedWriter(root="/path/to/packed", shard_size=2000)
        for mol_id in mol_ids:
            sample = build_stage3_sample(...)
            writer.put(sample)
        writer.finalize()

    After ``finalize()``, the root directory contains a ``manifest.json`` listing
    all shard subdirectories.  ``Stage3PackedDataset`` will detect this manifest
    and present a merged view across all shards.

    Resume support
    --------------
    Pass ``resume=True`` to re-open an interrupted run.  The writer scans
    ``root`` for existing complete shards (``shard_0000/``, ``shard_0001/``,
    …) and skips over them.  Call ``done_mol_ids()`` before starting the
    worker loop to filter out already-processed molecules::

        writer = Stage3ShardedWriter(root, shard_size=2000, resume=True)
        done = writer.done_mol_ids()
        remaining = [m for m in all_mol_ids if m not in done]
        for mol_id in remaining:
            writer.put(build_stage3_sample(...))
        writer.finalize()   # manifest covers old + new shards
    """

    def __init__(self, root: str, shard_size: int = 2000, resume: bool = False) -> None:
        self.root = root
        self.shard_size = shard_size
        self._buf: List[Stage3Sample] = []
        os.makedirs(root, exist_ok=True)

        if resume:
            # Scan for consecutive complete shards already on disk.
            # Stop at the first gap (missing or incomplete shard).
            existing: List[str] = []
            idx = 0
            while True:
                d = os.path.join(root, f"shard_{idx:04d}")
                if os.path.isdir(d) and is_stage3_packed_dir(d):
                    existing.append(d)
                    idx += 1
                else:
                    break
            self._shard_dirs = existing
            self._shard_idx = idx
        else:
            self._shard_dirs = []
            self._shard_idx = 0

    def done_mol_ids(self) -> "set[str]":
        """Return the set of mol_ids already stored in completed shards.

        Call this after constructing with ``resume=True`` and before
        starting the processing loop to determine which molecules can be
        skipped.
        """
        done: set = set()
        for shard_dir in self._shard_dirs:
            with np.load(stage3_packed_index_path(shard_dir), allow_pickle=False) as idx:
                for mol_id in idx["mol_ids"].tolist():
                    done.add(str(mol_id))
        return done

    def put(self, sample: Stage3Sample) -> None:
        self._buf.append(sample)
        if len(self._buf) >= self.shard_size:
            self._flush()

    def finalize(self) -> None:
        if self._buf:
            self._flush()
        manifest = {"shards": [os.path.basename(d) for d in self._shard_dirs]}
        with open(os.path.join(self.root, MANIFEST_NAME), "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    def _flush(self) -> None:
        shard_dir = os.path.join(self.root, f"shard_{self._shard_idx:04d}")
        pack_stage3_samples(self._buf, shard_dir, overwrite=True)
        self._shard_dirs.append(shard_dir)
        self._buf.clear()
        self._shard_idx += 1


def load_stage3_packed_sample(packed_dir: str, mol_id: str) -> Stage3Sample:
    dataset = Stage3PackedDataset(packed_dir, mol_ids=[mol_id])
    return dataset[0]


def collate_stage3_packed_samples(
    samples: List[Stage3Sample],
    *,
    device: Optional[str] = None,
):
    return collate_stage3_samples(samples, device=device)


__all__ = [
    "PACKED_FORMAT",
    "PACKED_META_NAME",
    "PACKED_INDEX_NAME",
    "MANIFEST_NAME",
    "stage3_packed_meta_path",
    "stage3_packed_index_path",
    "is_stage3_packed_dir",
    "list_stage3_packed_ids",
    "load_stage3_packed_meta",
    "pack_stage3_cache",
    "pack_stage3_samples",
    "Stage3PackedDataset",
    "Stage3ShardedWriter",
    "load_stage3_packed_sample",
    "collate_stage3_packed_samples",
]
