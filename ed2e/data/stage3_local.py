"""
Stage 3 / Stage 4 linked data pipeline.

This module upgrades Stage 1 manifold + Stage 2 FCLC outputs into a flat,
tensor-friendly per-molecule sample that can be consumed directly by the
Stage 3 local block and then handed to Stage 4 intra-chart aggregation
without Python-side object rebuilding.
"""
from __future__ import annotations

import os
import pickle
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra as sp_dijkstra
from scipy.spatial import cKDTree

if TYPE_CHECKING:
    from .fclc import FCLCChart, FCLCLevel
    from .manifold import ManifoldComponent, ManifoldLevel

_EPS = 1e-8


@dataclass
class Stage3Sample:
    """Flat per-molecule cache entry for Stage 3 local aggregation."""

    mol_id: str

    node_xyz: np.ndarray
    node_normal: np.ndarray
    node_scalar_raw: np.ndarray
    node_vector_raw: np.ndarray
    node_tangent_basis: np.ndarray

    chart_membership: np.ndarray
    membership_sr: np.ndarray
    membership_weight: np.ndarray
    local_coords: np.ndarray
    quadrant: np.ndarray
    local_knn_edge_index: np.ndarray
    local_edge_attr: np.ndarray

    chart_es_geom_static: np.ndarray
    chart_anchor_pos: np.ndarray
    chart_anchor_mask: np.ndarray
    reference_chart_id: np.ndarray
    chart_graph_edge_index: np.ndarray
    overlap_edge_index: np.ndarray
    overlap_shared_membership_index: np.ndarray
    overlap_shared_ptr: np.ndarray
    overlap_jaccard: np.ndarray
    chart_frame_metadata: Dict[str, np.ndarray]

    # Stage 4 新增：intra-level chart 图显式结构描述 + overlap→chart_edge 映射
    intra_geom_static: np.ndarray                  # (A, 7)
    overlap_edge_to_chart_edge_index: np.ndarray   # (E_ov,)

    # Stage 5 新增：inter-level chart 图边
    inter_level_edge_index: np.ndarray             # (2, E_inter)  int64
    inter_level_weights: np.ndarray                # (E_inter,)    float32
    inter_level_edge_attr: np.ndarray              # (E_inter, 7)  float32


@dataclass
class Stage3TensorBatch:
    """Collated batch that Stage 3 / Stage 4 code can consume directly."""

    mol_ids: List[str]

    node_xyz: torch.Tensor
    node_normal: torch.Tensor
    node_scalar_raw: torch.Tensor
    node_vector_raw: torch.Tensor
    node_tangent_basis: torch.Tensor

    chart_membership: torch.Tensor
    membership_sr: torch.Tensor
    membership_weight: torch.Tensor
    local_coords: torch.Tensor
    quadrant: torch.Tensor
    local_knn_edge_index: torch.Tensor
    local_edge_attr: torch.Tensor

    chart_es_geom_static: torch.Tensor
    chart_anchor_pos: torch.Tensor
    chart_anchor_mask: torch.Tensor
    reference_chart_id: torch.Tensor
    chart_graph_edge_index: torch.Tensor
    overlap_edge_index: torch.Tensor
    overlap_shared_membership_index: torch.Tensor
    overlap_shared_ptr: torch.Tensor
    overlap_jaccard: torch.Tensor
    chart_frame_metadata: Dict[str, torch.Tensor]

    # Stage 4 新增
    intra_geom_static: torch.Tensor                  # (A, 7)
    overlap_edge_to_chart_edge_index: torch.Tensor   # (E_ov,)

    # Stage 5 新增
    inter_level_edge_index: torch.Tensor             # (2, E_inter)
    inter_level_weights: torch.Tensor                # (E_inter,)
    inter_level_edge_attr: torch.Tensor              # (E_inter, 7)

    node_batch: torch.Tensor
    chart_batch: torch.Tensor
    membership_batch: torch.Tensor

    def intra_static_bundle(self) -> Dict[str, Any]:
        return {
            "chart_graph_edge_index": self.chart_graph_edge_index,
            "overlap_edge_index": self.overlap_edge_index,
            "overlap_shared_membership_index": self.overlap_shared_membership_index,
            "overlap_shared_ptr": self.overlap_shared_ptr,
            "overlap_jaccard": self.overlap_jaccard,
            "reference_chart_id": self.reference_chart_id,
            "chart_anchor_pos": self.chart_anchor_pos,
            "chart_anchor_mask": self.chart_anchor_mask,
            "chart_frame_metadata": self.chart_frame_metadata,
            # Stage 4 新增
            "intra_geom_static": self.intra_geom_static,
            "chart_to_ref": self.chart_frame_metadata["chart_to_ref"],
            "overlap_edge_to_chart_edge_index": self.overlap_edge_to_chart_edge_index,
        }

    def inter_static_bundle(self) -> Dict[str, Any]:
        return {
            "inter_level_edge_index": self.inter_level_edge_index,
            "inter_level_weights": self.inter_level_weights,
            "inter_level_edge_attr": self.inter_level_edge_attr,
        }

    def pin_memory(self) -> "Stage3TensorBatch":
        tensor_fields = (
            "node_xyz",
            "node_normal",
            "node_scalar_raw",
            "node_vector_raw",
            "node_tangent_basis",
            "chart_membership",
            "membership_sr",
            "membership_weight",
            "local_coords",
            "quadrant",
            "local_knn_edge_index",
            "local_edge_attr",
            "chart_es_geom_static",
            "chart_anchor_pos",
            "chart_anchor_mask",
            "reference_chart_id",
            "chart_graph_edge_index",
            "overlap_edge_index",
            "overlap_shared_membership_index",
            "overlap_shared_ptr",
            "overlap_jaccard",
            "node_batch",
            "chart_batch",
            "membership_batch",
            "intra_geom_static",
            "overlap_edge_to_chart_edge_index",
            "inter_level_edge_index",
            "inter_level_weights",
            "inter_level_edge_attr",
        )
        for name in tensor_fields:
            setattr(self, name, getattr(self, name).pin_memory())
        self.chart_frame_metadata = {
            key: value.pin_memory()
            for key, value in self.chart_frame_metadata.items()
        }
        return self

    def to(
        self,
        device: str | torch.device,
        *,
        non_blocking: bool = False,
    ) -> "Stage3TensorBatch":
        tensor_fields = (
            "node_xyz",
            "node_normal",
            "node_scalar_raw",
            "node_vector_raw",
            "node_tangent_basis",
            "chart_membership",
            "membership_sr",
            "membership_weight",
            "local_coords",
            "quadrant",
            "local_knn_edge_index",
            "local_edge_attr",
            "chart_es_geom_static",
            "chart_anchor_pos",
            "chart_anchor_mask",
            "reference_chart_id",
            "chart_graph_edge_index",
            "overlap_edge_index",
            "overlap_shared_membership_index",
            "overlap_shared_ptr",
            "overlap_jaccard",
            "node_batch",
            "chart_batch",
            "membership_batch",
            "intra_geom_static",
            "overlap_edge_to_chart_edge_index",
            "inter_level_edge_index",
            "inter_level_weights",
            "inter_level_edge_attr",
        )
        for name in tensor_fields:
            setattr(
                self,
                name,
                getattr(self, name).to(device=device, non_blocking=non_blocking),
            )
        self.chart_frame_metadata = {
            key: value.to(device=device, non_blocking=non_blocking)
            for key, value in self.chart_frame_metadata.items()
        }
        return self


@dataclass
class _ComponentContext:
    level_id: int
    component_id: int
    component: ManifoldComponent
    node_offset: int
    group_id: int
    adjacency: csr_matrix
    scalar_f_norm: np.ndarray
    dist_row_cache: Dict[int, np.ndarray]


@dataclass
class _ChartStaticResult:
    local_edge_index: np.ndarray
    local_edge_attr: np.ndarray
    geom_static: np.ndarray
    anchor_pos: np.ndarray
    anchor_mask: np.ndarray


def _zscore_channels(arr: np.ndarray) -> np.ndarray:
    mu = arr.mean(axis=0, keepdims=True)
    std = arr.std(axis=0, keepdims=True)
    return (arr - mu) / np.where(std > 1e-8, std, 1.0)


def build_tangent_basis(normals: np.ndarray) -> np.ndarray:
    """Deterministic tangent basis from per-node normals."""
    normals = np.asarray(normals, dtype=np.float32)
    basis = np.zeros((len(normals), 2, 3), dtype=np.float32)
    ref_x = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    ref_y = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    for i, n in enumerate(normals):
        nn = n.astype(np.float64)
        nn /= np.linalg.norm(nn) + _EPS
        ref = ref_x if abs(nn @ ref_x) < 0.9 else ref_y
        e1 = ref - (ref @ nn) * nn
        e1 /= np.linalg.norm(e1) + _EPS
        e2 = np.cross(nn, e1)
        e2 /= np.linalg.norm(e2) + _EPS
        basis[i, 0] = e1.astype(np.float32)
        basis[i, 1] = e2.astype(np.float32)
    return basis


def project_vectors_to_basis(vectors_3d: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Project (..., C, 3) vectors onto (..., 2, 3) tangent bases -> (..., C, 2)."""
    return np.einsum("...cj,...bj->...cb", vectors_3d, basis, optimize=True).astype(np.float32)


def reconstruct_vectors_from_basis(vectors_2d: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Lift (..., C, 2) tangent coordinates back to 3-D."""
    return np.einsum("...cb,...bj->...cj", vectors_2d, basis, optimize=True).astype(np.float32)


def rotate_vectors_between_bases(
    vectors_2d: np.ndarray,
    src_basis: np.ndarray,
    dst_basis: np.ndarray,
) -> np.ndarray:
    vec3 = reconstruct_vectors_from_basis(vectors_2d, src_basis)
    return project_vectors_to_basis(vec3, dst_basis)


def stage3_cache_path(
    cache_dir: str,
    mol_id: str,
    local_knn_k: int = 12,
    chart_knn_k: int = 8,
    num_anchors: int = 8,
) -> str:
    tag = f"stage3_lk{local_knn_k}_ck{chart_knn_k}_a{num_anchors}_ig7_il7"
    return os.path.join(cache_dir, f"{mol_id}_{tag}.pkl")


def is_stage3_bundle_path(path: str) -> bool:
    return path.endswith(".zip")


def stage3_bundle_member(mol_id: str) -> str:
    return f"molecules/{mol_id}.pkl"


def save_stage3_sample(path: str, sample: Stage3Sample) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(sample, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_stage3_sample(path: str) -> Stage3Sample:
    with open(path, "rb") as f:
        return pickle.load(f)


def save_stage3_bundle_entry(zf: zipfile.ZipFile, mol_id: str, sample: Stage3Sample) -> None:
    zf.writestr(
        stage3_bundle_member(mol_id),
        pickle.dumps(sample, protocol=pickle.HIGHEST_PROTOCOL),
        compress_type=zipfile.ZIP_STORED,
    )


def load_stage3_entry(path: str, mol_id: str) -> Stage3Sample:
    if is_stage3_bundle_path(path):
        with zipfile.ZipFile(path, "r") as zf:
            with zf.open(stage3_bundle_member(mol_id), "r") as f:
                return pickle.load(f)
    return load_stage3_sample(path)


def list_stage3_bundle_ids(path: str) -> List[str]:
    if is_stage3_bundle_path(path):
        with zipfile.ZipFile(path, "r") as zf:
            if "meta/mol_ids.pkl" in zf.namelist():
                with zf.open("meta/mol_ids.pkl", "r") as f:
                    return pickle.load(f)
            return sorted(
                name[len("molecules/"):-4]
                for name in zf.namelist()
                if name.startswith("molecules/") and name.endswith(".pkl")
            )
    sample = load_stage3_sample(path)
    return [sample.mol_id]


def _get_component_dist_row(ctx: _ComponentContext, seed_local: int) -> np.ndarray:
    if seed_local not in ctx.dist_row_cache:
        ctx.dist_row_cache[seed_local] = sp_dijkstra(
            ctx.adjacency,
            indices=[seed_local],
            directed=False,
            return_predecessors=False,
        )[0].astype(np.float32)
    return ctx.dist_row_cache[seed_local]


def _resolve_chart_seed_local(chart: FCLCChart, component: ManifoldComponent) -> int:
    seed = int(getattr(chart, "seed_vertex_idx", -1))
    if 0 <= seed < len(component.verts):
        return seed
    dist = np.linalg.norm(component.verts - chart.center[None, :], axis=1)
    return int(np.argmin(dist))


def _resolve_chart_membership_sr(
    chart: FCLCChart,
    ctx: _ComponentContext,
    seed_local: int,
    lam: Tuple[float, float, float],
) -> np.ndarray:
    from .fclc import compute_region_compatibility_scores

    sr = getattr(chart, "membership_sr", None)
    if sr is not None and len(sr) == len(chart.vert_indices):
        return np.asarray(sr, dtype=np.float32).copy()

    dist_row = _get_component_dist_row(ctx, seed_local)
    scores = compute_region_compatibility_scores(
        seed=seed_local,
        dist_row=dist_row,
        normals=ctx.component.normals,
        scalar_f_norm=ctx.scalar_f_norm,
        lam=lam,
    )
    return scores[np.asarray(chart.vert_indices, dtype=np.int64)].astype(np.float32)


def _build_chart_geom_static(
    local_coords: np.ndarray,
    quadrant: np.ndarray,
    scalar_feats: np.ndarray,
    member_normals: np.ndarray,
    center_normal: np.ndarray,
) -> np.ndarray:
    block: List[float] = []
    total = max(1, len(local_coords))
    center_normal = center_normal.astype(np.float64)
    center_normal /= np.linalg.norm(center_normal) + _EPS

    for q in range(4):
        mask = quadrant == q
        count = int(mask.sum())
        occ = count / total
        block.append(float(occ))

    for q in range(4):
        mask = quadrant == q
        if mask.any():
            mu = local_coords[mask].mean(axis=0)
        else:
            mu = np.zeros(2, dtype=np.float32)
        block.extend([float(mu[0]), float(mu[1])])

    for q in range(4):
        mask = quadrant == q
        if mask.any():
            coords = local_coords[mask]
            centered = coords - coords.mean(axis=0, keepdims=True)
            cov = centered.T @ centered / max(1, len(coords))
            trace = float(np.trace(cov))
            det = float(np.linalg.det(cov))
        else:
            trace = 0.0
            det = 0.0
        block.extend([trace, det])

    H = scalar_feats[:, 2]
    K = scalar_feats[:, 3]
    for q in range(4):
        mask = quadrant == q
        block.append(float(H[mask].mean()) if mask.any() else 0.0)
    for q in range(4):
        mask = quadrant == q
        block.append(float(K[mask].mean()) if mask.any() else 0.0)

    for q in range(4):
        mask = quadrant == q
        if mask.any():
            dots = np.clip((member_normals[mask] * center_normal[None, :]).sum(axis=1), -1.0, 1.0)
            disp = float(np.mean(1.0 - dots))
        else:
            disp = 0.0
        block.append(disp)

    return np.asarray(block, dtype=np.float32)


def _farthest_point_anchors(local_coords: np.ndarray, num_anchors: int) -> Tuple[np.ndarray, np.ndarray]:
    pos = np.zeros((num_anchors, 2), dtype=np.float32)
    mask = np.zeros((num_anchors,), dtype=bool)
    if len(local_coords) == 0:
        return pos, mask

    if len(local_coords) <= num_anchors:
        pos[:len(local_coords)] = local_coords.astype(np.float32)
        mask[:len(local_coords)] = True
        return pos, mask

    radii = np.linalg.norm(local_coords, axis=1)
    selected = [int(np.argmax(radii))]
    min_dist = np.linalg.norm(local_coords - local_coords[selected[0]][None, :], axis=1)
    while len(selected) < num_anchors:
        next_idx = int(np.argmax(min_dist))
        selected.append(next_idx)
        dist = np.linalg.norm(local_coords - local_coords[next_idx][None, :], axis=1)
        min_dist = np.minimum(min_dist, dist)

    selected = np.asarray(selected, dtype=np.int64)
    pos[:] = local_coords[selected].astype(np.float32)
    mask[:] = True
    return pos, mask


def _build_symmetric_local_knn(
    local_coords: np.ndarray,
    quadrant: np.ndarray,
    k: int,
) -> Tuple[np.ndarray, np.ndarray]:
    n = len(local_coords)
    if n <= 1:
        return (
            np.zeros((2, 0), dtype=np.int64),
            np.zeros((0, 6), dtype=np.float32),
        )

    nn = min(k + 1, n)
    tree = cKDTree(local_coords)
    _, idx = tree.query(local_coords, k=nn)
    idx = np.asarray(idx)
    if idx.ndim == 1:
        idx = idx[:, None]

    edges = set()
    for dst in range(n):
        for src in idx[dst, 1:]:
            src_i = int(src)
            if src_i == dst:
                continue
            edges.add((src_i, dst))
            edges.add((dst, src_i))

    if not edges:
        return (
            np.zeros((2, 0), dtype=np.int64),
            np.zeros((0, 6), dtype=np.float32),
        )

    ordered = sorted(edges)
    src = np.asarray([p[0] for p in ordered], dtype=np.int64)
    dst = np.asarray([p[1] for p in ordered], dtype=np.int64)
    delta = local_coords[src] - local_coords[dst]
    dist = np.linalg.norm(delta, axis=1, keepdims=True)
    same_quad = (quadrant[src] == quadrant[dst]).astype(np.float32)[:, None]
    q_src = (quadrant[src].astype(np.float32) / 3.0)[:, None]
    q_dst = (quadrant[dst].astype(np.float32) / 3.0)[:, None]
    edge_attr = np.concatenate([delta.astype(np.float32), dist.astype(np.float32), same_quad, q_src, q_dst], axis=1)
    edge_index = np.stack([src, dst], axis=0)
    return edge_index, edge_attr


def _build_chart_static_job(
    local_coords: np.ndarray,
    quadrant: np.ndarray,
    scalar_feats: np.ndarray,
    member_normals: np.ndarray,
    center_normal: np.ndarray,
    local_knn_k: int,
    num_anchors: int,
) -> _ChartStaticResult:
    local_edge_index, local_edge_attr = _build_symmetric_local_knn(local_coords, quadrant, local_knn_k)
    geom_static = _build_chart_geom_static(local_coords, quadrant, scalar_feats, member_normals, center_normal)
    anchor_pos, anchor_mask = _farthest_point_anchors(local_coords, num_anchors)
    return _ChartStaticResult(
        local_edge_index=local_edge_index,
        local_edge_attr=local_edge_attr,
        geom_static=geom_static,
        anchor_pos=anchor_pos,
        anchor_mask=anchor_mask,
    )


def build_stage3_sample(
    mol_id: str,
    manifold_levels: List[ManifoldLevel],
    fclc_levels: List[FCLCLevel],
    *,
    local_knn_k: int = 12,
    chart_knn_k: int = 8,
    num_anchors: int = 8,
    inner_threads: int = 4,
    lam: Tuple[float, float, float] = (0.4, 0.3, 0.3),
) -> Stage3Sample:
    """Build one flat Stage 3 sample from Stage 1 + Stage 2 outputs."""
    from .fclc import build_mesh_adjacency

    if len(manifold_levels) != len(fclc_levels):
        raise ValueError(
            f"Stage1/Stage2 level count mismatch for mol_id={mol_id}: "
            f"{len(manifold_levels)} vs {len(fclc_levels)}"
        )

    node_xyz_list: List[np.ndarray] = []
    node_normal_list: List[np.ndarray] = []
    node_scalar_list: List[np.ndarray] = []
    node_vector_list: List[np.ndarray] = []
    component_ctx: Dict[Tuple[int, int], _ComponentContext] = {}

    node_offset = 0
    group_counter = 0
    for level in manifold_levels:
        for comp in level.components:
            n = len(comp.verts)
            node_xyz_list.append(comp.verts.astype(np.float32))
            node_normal_list.append(comp.normals.astype(np.float32))
            node_scalar_list.append(comp.scalar_features.astype(np.float32))
            node_vector_list.append(comp.vector_features.astype(np.float32))

            component_ctx[(level.level_id, comp.component_id)] = _ComponentContext(
                level_id=level.level_id,
                component_id=comp.component_id,
                component=comp,
                node_offset=node_offset,
                group_id=group_counter,
                adjacency=build_mesh_adjacency(comp.verts, comp.faces),
                scalar_f_norm=_zscore_channels(comp.scalar_features.astype(np.float32)),
                dist_row_cache={},
            )
            node_offset += n
            group_counter += 1

    if not node_xyz_list:
        raise ValueError(f"No manifold components available for mol_id={mol_id}")

    node_xyz = np.concatenate(node_xyz_list, axis=0)
    node_normal = np.concatenate(node_normal_list, axis=0)
    node_scalar_raw = np.concatenate(node_scalar_list, axis=0)
    node_vector_raw = np.concatenate(node_vector_list, axis=0)
    node_tangent_basis = build_tangent_basis(node_normal)

    membership_pairs: List[np.ndarray] = []
    membership_sr_list: List[np.ndarray] = []
    local_coords_list: List[np.ndarray] = []
    quadrant_list: List[np.ndarray] = []

    chart_level_id: List[int] = []
    chart_component_id: List[int] = []
    chart_group_id: List[int] = []
    chart_seed_global: List[int] = []
    chart_center: List[np.ndarray] = []
    chart_center_normal: List[np.ndarray] = []
    chart_frame: List[np.ndarray] = []
    chart_original_id: List[int] = []
    chart_membership_range: List[Tuple[int, int]] = []

    chart_jobs: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []

    membership_offset = 0
    local_chart_id = 0
    for level_idx, fclc_level in enumerate(fclc_levels):
        manifold_level = manifold_levels[level_idx]
        if fclc_level.level_id != manifold_level.level_id:
            raise ValueError(
                f"Level id mismatch for mol_id={mol_id}: "
                f"stage1={manifold_level.level_id}, stage2={fclc_level.level_id}"
            )

        for chart in fclc_level.charts:
            ctx = component_ctx[(chart.level_id, chart.component_id)]
            seed_local = _resolve_chart_seed_local(chart, ctx.component)
            seed_global = ctx.node_offset + seed_local

            vert_local = np.asarray(chart.vert_indices, dtype=np.int64)
            node_idx = ctx.node_offset + vert_local
            sr = _resolve_chart_membership_sr(chart, ctx, seed_local, lam)
            lc = np.asarray(chart.local_coords, dtype=np.float32)
            quad = np.asarray(chart.quadrant, dtype=np.int8)

            if len(node_idx) != len(sr) or len(node_idx) != len(lc) or len(node_idx) != len(quad):
                raise ValueError(
                    f"Chart field length mismatch for mol_id={mol_id}, chart={chart.chart_id}"
                )

            chart_col = np.full((len(node_idx), 1), local_chart_id, dtype=np.int64)
            membership_pairs.append(np.concatenate([chart_col, node_idx[:, None]], axis=1))
            membership_sr_list.append(sr.astype(np.float32))
            local_coords_list.append(lc.astype(np.float32))
            quadrant_list.append(quad.astype(np.int64))

            chart_level_id.append(chart.level_id)
            chart_component_id.append(chart.component_id)
            chart_group_id.append(ctx.group_id)
            chart_seed_global.append(seed_global)
            chart_center.append(np.asarray(chart.center, dtype=np.float32))
            chart_center_normal.append(np.asarray(chart.center_normal, dtype=np.float32))
            chart_frame.append(np.asarray(chart.frame, dtype=np.float32))
            chart_original_id.append(int(chart.chart_id))
            chart_membership_range.append((membership_offset, membership_offset + len(node_idx)))

            chart_jobs.append((
                lc.astype(np.float32),
                quad.astype(np.int64),
                node_scalar_raw[node_idx],
                node_normal[node_idx],
                np.asarray(chart.center_normal, dtype=np.float32),
            ))

            membership_offset += len(node_idx)
            local_chart_id += 1

    if local_chart_id == 0:
        raise ValueError(f"No FCLC charts available for mol_id={mol_id}")

    chart_membership = np.concatenate(membership_pairs, axis=0).astype(np.int64)
    membership_sr = np.concatenate(membership_sr_list, axis=0).astype(np.float32)
    local_coords = np.concatenate(local_coords_list, axis=0).astype(np.float32)
    quadrant = np.concatenate(quadrant_list, axis=0).astype(np.int64)

    node_ids = chart_membership[:, 1]
    # Compute membership weights: softmax(-sr) per node, in float64 to avoid
    # float32 underflow when sr is large (which caused weights to silently collapse
    # to 0 for ~45% of molecules in the original float32 implementation).
    sr_f64 = membership_sr.astype(np.float64)
    sr_min = np.full(len(node_xyz), np.inf, dtype=np.float64)
    np.minimum.at(sr_min, node_ids, sr_f64)           # per-node minimum sr
    shifted = np.exp(-(sr_f64 - sr_min[node_ids]))    # in [0, 1]; most-compatible = 1
    denom = np.bincount(node_ids, weights=shifted, minlength=len(node_xyz))
    membership_weight = (shifted / np.maximum(denom[node_ids], 1e-30)).astype(np.float32)

    chart_results: List[_ChartStaticResult] = []
    if inner_threads > 1 and len(chart_jobs) > 1:
        with ThreadPoolExecutor(max_workers=inner_threads) as executor:
            futures = [
                executor.submit(
                    _build_chart_static_job,
                    coords,
                    quad,
                    scalar_feats,
                    normals,
                    center_normal,
                    local_knn_k,
                    num_anchors,
                )
                for coords, quad, scalar_feats, normals, center_normal in chart_jobs
            ]
            for fut in futures:
                chart_results.append(fut.result())
    else:
        for coords, quad, scalar_feats, normals, center_normal in chart_jobs:
            chart_results.append(
                _build_chart_static_job(
                    coords,
                    quad,
                    scalar_feats,
                    normals,
                    center_normal,
                    local_knn_k,
                    num_anchors,
                )
            )

    local_edge_index_parts: List[np.ndarray] = []
    local_edge_attr_parts: List[np.ndarray] = []
    chart_es_geom_static = np.zeros((local_chart_id, len(chart_results[0].geom_static)), dtype=np.float32)
    chart_anchor_pos = np.zeros((local_chart_id, num_anchors, 2), dtype=np.float32)
    chart_anchor_mask = np.zeros((local_chart_id, num_anchors), dtype=bool)

    for chart_idx, result in enumerate(chart_results):
        start, _ = chart_membership_range[chart_idx]
        if result.local_edge_index.shape[1] > 0:
            local_edge_index_parts.append(result.local_edge_index + start)
            local_edge_attr_parts.append(result.local_edge_attr)
        chart_es_geom_static[chart_idx] = result.geom_static
        chart_anchor_pos[chart_idx] = result.anchor_pos
        chart_anchor_mask[chart_idx] = result.anchor_mask

    local_knn_edge_index = (
        np.concatenate(local_edge_index_parts, axis=1).astype(np.int64)
        if local_edge_index_parts else np.zeros((2, 0), dtype=np.int64)
    )
    local_edge_attr = (
        np.concatenate(local_edge_attr_parts, axis=0).astype(np.float32)
        if local_edge_attr_parts else np.zeros((0, 6), dtype=np.float32)
    )

    group_to_charts: Dict[int, List[int]] = {}
    for chart_idx, gid in enumerate(chart_group_id):
        group_to_charts.setdefault(int(gid), []).append(chart_idx)

    group_ids = sorted(group_to_charts.keys())
    reference_chart_id = np.zeros((len(group_ids),), dtype=np.int64)
    group_level_id = np.zeros((len(group_ids),), dtype=np.int64)
    group_component_id = np.zeros((len(group_ids),), dtype=np.int64)

    chart_membership_nodes = {
        chart_idx: chart_membership[start:end, 1]
        for chart_idx, (start, end) in enumerate(chart_membership_range)
    }
    chart_membership_idx = {
        chart_idx: np.arange(start, end, dtype=np.int64)
        for chart_idx, (start, end) in enumerate(chart_membership_range)
    }

    chart_graph_edges: set[Tuple[int, int]] = set()
    overlap_edges: List[Tuple[int, int]] = []
    overlap_shared_pairs: List[np.ndarray] = []
    overlap_shared_ptr: List[int] = [0]
    overlap_jaccard: List[float] = []

    # Stage 4 intra-geom 辅助数组（在 group 循环内填充）
    chart_centers_arr  = np.stack(chart_center, axis=0).astype(np.float32)   # (A, 3)
    chart_normals_arr  = np.stack(chart_center_normal, axis=0).astype(np.float32)  # (A, 3)
    chart_frames_arr   = np.stack(chart_frame, axis=0).astype(np.float32)    # (A, 2, 3)
    chart_sizes        = np.array(
        [r - l for l, r in chart_membership_range], dtype=np.float32
    )  # (A,)
    chart_to_ref_arr   = np.zeros(local_chart_id, dtype=np.int64)            # (A,)
    intra_geom         = np.zeros((local_chart_id, 7), dtype=np.float32)     # (A, 7)

    for group_order, gid in enumerate(group_ids):
        charts_in_group = group_to_charts[gid]
        first_chart = charts_in_group[0]
        group_level_id[group_order] = chart_level_id[first_chart]
        group_component_id[group_order] = chart_component_id[first_chart]

        ctx = component_ctx[(chart_level_id[first_chart], chart_component_id[first_chart])]

        pairwise: Dict[Tuple[int, int], float] = {}
        for dst_chart in charts_in_group:
            dst_seed_local = chart_seed_global[dst_chart] - ctx.node_offset
            dist_row = _get_component_dist_row(ctx, int(dst_seed_local))
            for src_chart in charts_in_group:
                if src_chart == dst_chart:
                    continue
                src_seed_local = chart_seed_global[src_chart] - ctx.node_offset
                pairwise[(src_chart, dst_chart)] = float(dist_row[int(src_seed_local)])

        if len(charts_in_group) == 1:
            reference_chart_id[group_order] = charts_in_group[0]
        else:
            total_dist = []
            for c in charts_in_group:
                s = 0.0
                for other in charts_in_group:
                    if other == c:
                        continue
                    d = pairwise.get((other, c), np.inf)
                    s += float(d if np.isfinite(d) else 1e6)
                total_dist.append((s, c))
            total_dist.sort()
            reference_chart_id[group_order] = total_dist[0][1]

        # --- Stage 4: intra_geom_static 预计算（在 pairwise 可用时） ---
        ref = int(reference_chart_id[group_order])
        c0  = chart_centers_arr[ref]
        n0  = chart_normals_arr[ref]
        f0  = chart_frames_arr[ref]   # (2, 3)
        for a in charts_in_group:
            chart_to_ref_arr[a] = ref
            d_geod  = float(pairwise.get((a, ref), pairwise.get((ref, a), 0.0)))
            proj2   = f0 @ (chart_centers_arr[a] - c0)          # (2,)
            nn_dev  = 1.0 - float(np.clip(
                float(np.dot(chart_normals_arr[a], n0)), -1.0, 1.0
            ))
            e_proj  = f0 @ chart_frames_arr[a, 0]               # (2,)  cosθ, sinθ
            log_ar  = float(np.log(
                (chart_sizes[a] + 1.0) / (chart_sizes[ref] + 1.0)
            ))
            intra_geom[a] = [
                d_geod, proj2[0], proj2[1], nn_dev,
                e_proj[0], e_proj[1], log_ar,
            ]

        for dst_chart in charts_in_group:
            candidates: List[Tuple[float, int]] = []
            for src_chart in charts_in_group:
                if src_chart == dst_chart:
                    continue
                d = pairwise.get((src_chart, dst_chart), np.inf)
                if np.isfinite(d):
                    candidates.append((float(d), src_chart))
            candidates.sort(key=lambda x: (x[0], x[1]))
            for _, src_chart in candidates[:min(chart_knn_k, len(candidates))]:
                chart_graph_edges.add((int(src_chart), int(dst_chart)))

        for i, chart_a in enumerate(charts_in_group):
            nodes_a = chart_membership_nodes[chart_a]
            idx_a = chart_membership_idx[chart_a]
            for chart_b in charts_in_group[i + 1:]:
                nodes_b = chart_membership_nodes[chart_b]
                idx_b = chart_membership_idx[chart_b]
                shared, ia, ib = np.intersect1d(nodes_a, nodes_b, return_indices=True)
                if len(shared) == 0:
                    continue
                union = np.union1d(nodes_a, nodes_b)
                jac = float(len(shared) / max(1, len(union)))

                recv_a = np.stack([idx_a[ia], idx_b[ib]], axis=1).astype(np.int64)
                recv_b = np.stack([idx_b[ib], idx_a[ia]], axis=1).astype(np.int64)

                overlap_edges.append((chart_b, chart_a))
                overlap_shared_pairs.append(recv_a)
                overlap_shared_ptr.append(overlap_shared_ptr[-1] + len(recv_a))
                overlap_jaccard.append(jac)
                chart_graph_edges.add((chart_b, chart_a))

                overlap_edges.append((chart_a, chart_b))
                overlap_shared_pairs.append(recv_b)
                overlap_shared_ptr.append(overlap_shared_ptr[-1] + len(recv_b))
                overlap_jaccard.append(jac)
                chart_graph_edges.add((chart_a, chart_b))

    chart_graph_edge_index = (
        np.asarray(sorted(chart_graph_edges), dtype=np.int64).T
        if chart_graph_edges else np.zeros((2, 0), dtype=np.int64)
    )
    overlap_edge_index = (
        np.asarray(overlap_edges, dtype=np.int64).T
        if overlap_edges else np.zeros((2, 0), dtype=np.int64)
    )
    overlap_shared_membership_index = (
        np.concatenate(overlap_shared_pairs, axis=0).astype(np.int64)
        if overlap_shared_pairs else np.zeros((0, 2), dtype=np.int64)
    )

    # Stage 4: overlap_edge_to_chart_edge_index (ov2ce)
    # 每条 overlap 边在 chart_graph_edge_index 中的位置
    if overlap_edge_index.shape[1] > 0 and chart_graph_edge_index.shape[1] > 0:
        ce_dict = {
            (int(chart_graph_edge_index[0, i]), int(chart_graph_edge_index[1, i])): i
            for i in range(chart_graph_edge_index.shape[1])
        }
        ov2ce = np.array(
            [
                ce_dict[(int(overlap_edge_index[0, j]), int(overlap_edge_index[1, j]))]
                for j in range(overlap_edge_index.shape[1])
            ],
            dtype=np.int64,
        )
    else:
        ov2ce = np.zeros((0,), dtype=np.int64)

    # Stage 5: inter-level edges
    # Build mapping from stage-2 chart_id → global chart index
    stage2id_to_global: Dict[int, int] = {
        int(cid): gi for gi, cid in enumerate(chart_original_id)
    }
    chart_level_ids_arr = np.asarray(chart_level_id, dtype=np.int64)  # (A,)

    inter_edges:     List[Tuple[int, int]] = []
    inter_w_list:    List[float]           = []
    inter_attr_list: List[List[float]]     = []

    def _collect_inter_edges_inner(
        iw_dict: Optional[Dict],
    ) -> None:
        for recv_s2id, weight_list in (iw_dict or {}).items():
            if recv_s2id not in stage2id_to_global:
                continue
            g_recv = stage2id_to_global[recv_s2id]
            for tup in (weight_list or []):
                if len(tup) < 4:
                    continue
                send_s2id, w, mean_d, mean_nd = tup
                if send_s2id not in stage2id_to_global:
                    continue
                g_send = stage2id_to_global[send_s2id]
                c_recv = chart_centers_arr[g_recv]
                c_send = chart_centers_arr[g_send]
                f_recv = chart_frames_arr[g_recv]   # (2, 3)
                f_send = chart_frames_arr[g_send]
                inter_edges.append((g_send, g_recv))
                inter_w_list.append(float(w))
                inter_attr_list.append([
                    float(mean_d),
                    float(mean_nd),
                    float(np.linalg.norm(c_recv - c_send)),
                    float(np.dot(f_recv[0], f_send[0])),   # cos_theta
                    float(np.dot(f_recv[1], f_send[0])),   # sin_theta
                    float(np.log((chart_sizes[g_recv] + 1.0) / (chart_sizes[g_send] + 1.0))),
                    float(chart_level_ids_arr[g_recv] - chart_level_ids_arr[g_send]),  # ±1
                ])

    for ki in range(len(fclc_levels) - 1):
        _collect_inter_edges_inner(fclc_levels[ki].inter_weights)
        _collect_inter_edges_inner(fclc_levels[ki + 1].inter_weights_up)

    if inter_edges:
        inter_level_edge_index = np.array(inter_edges, dtype=np.int64).T    # (2, E_inter)
        inter_level_weights    = np.array(inter_w_list, dtype=np.float32)   # (E_inter,)
        inter_level_edge_attr  = np.array(inter_attr_list, dtype=np.float32)  # (E_inter, 7)
    else:
        inter_level_edge_index = np.empty((2, 0), dtype=np.int64)
        inter_level_weights    = np.empty((0,), dtype=np.float32)
        inter_level_edge_attr  = np.empty((0, 7), dtype=np.float32)

    chart_frame_metadata = {
        "chart_center": np.stack(chart_center, axis=0).astype(np.float32),
        "chart_center_normal": np.stack(chart_center_normal, axis=0).astype(np.float32),
        "chart_frame": np.stack(chart_frame, axis=0).astype(np.float32),
        "chart_level_id": np.asarray(chart_level_id, dtype=np.int64),
        "chart_component_id": np.asarray(chart_component_id, dtype=np.int64),
        "chart_group_id": np.asarray(chart_group_id, dtype=np.int64),
        "chart_seed_node_index": np.asarray(chart_seed_global, dtype=np.int64),
        "chart_stage2_id": np.asarray(chart_original_id, dtype=np.int64),
        "group_level_id": group_level_id.astype(np.int64),
        "group_component_id": group_component_id.astype(np.int64),
        "chart_to_ref": chart_to_ref_arr.astype(np.int64),   # Stage 4 新增
    }

    return Stage3Sample(
        mol_id=mol_id,
        node_xyz=node_xyz.astype(np.float32),
        node_normal=node_normal.astype(np.float32),
        node_scalar_raw=node_scalar_raw.astype(np.float32),
        node_vector_raw=node_vector_raw.astype(np.float32),
        node_tangent_basis=node_tangent_basis.astype(np.float32),
        chart_membership=chart_membership.astype(np.int64),
        membership_sr=membership_sr.astype(np.float32),
        membership_weight=membership_weight.astype(np.float32),
        local_coords=local_coords.astype(np.float32),
        quadrant=quadrant.astype(np.int64),
        local_knn_edge_index=local_knn_edge_index.astype(np.int64),
        local_edge_attr=local_edge_attr.astype(np.float32),
        chart_es_geom_static=chart_es_geom_static.astype(np.float32),
        chart_anchor_pos=chart_anchor_pos.astype(np.float32),
        chart_anchor_mask=chart_anchor_mask.astype(bool),
        reference_chart_id=reference_chart_id.astype(np.int64),
        chart_graph_edge_index=chart_graph_edge_index.astype(np.int64),
        overlap_edge_index=overlap_edge_index.astype(np.int64),
        overlap_shared_membership_index=overlap_shared_membership_index.astype(np.int64),
        overlap_shared_ptr=np.asarray(overlap_shared_ptr, dtype=np.int64),
        overlap_jaccard=np.asarray(overlap_jaccard, dtype=np.float32),
        chart_frame_metadata=chart_frame_metadata,
        intra_geom_static=intra_geom.astype(np.float32),
        overlap_edge_to_chart_edge_index=ov2ce,
        inter_level_edge_index=inter_level_edge_index,
        inter_level_weights=inter_level_weights,
        inter_level_edge_attr=inter_level_edge_attr,
    )


def collate_stage3_samples(
    samples: List[Stage3Sample],
    *,
    device: Optional[str | torch.device] = None,
) -> Stage3TensorBatch:
    if not samples:
        raise ValueError("collate_stage3_samples received an empty list")

    node_offset = 0
    chart_offset = 0
    membership_offset = 0
    group_offset = 0
    shared_pair_offset = 0
    chart_edge_offset = 0  # Stage 4: tracks offset into chart_graph_edge_index per sample

    mol_ids: List[str] = []

    node_xyz = []
    node_normal = []
    node_scalar_raw = []
    node_vector_raw = []
    node_tangent_basis = []
    chart_membership = []
    membership_sr = []
    membership_weight = []
    local_coords = []
    quadrant = []
    local_knn_edge_index = []
    local_edge_attr = []
    chart_es_geom_static = []
    chart_anchor_pos = []
    chart_anchor_mask = []
    reference_chart_id = []
    chart_graph_edge_index = []
    overlap_edge_index = []
    overlap_shared_membership_index = []
    overlap_shared_ptr = [0]
    overlap_jaccard = []
    intra_geom_static = []        # Stage 4 新增
    ov2ce_list: List[np.ndarray] = []  # Stage 4 新增
    inter_edge_index_list: List[np.ndarray] = []  # Stage 5 新增
    inter_weights_list: List[np.ndarray] = []     # Stage 5 新增
    inter_edge_attr_list: List[np.ndarray] = []   # Stage 5 新增

    meta_lists: Dict[str, List[np.ndarray]] = {}

    node_batch = []
    chart_batch = []
    membership_batch = []

    for batch_idx, sample in enumerate(samples):
        mol_ids.append(sample.mol_id)

        n_nodes = len(sample.node_xyz)
        n_charts = len(sample.chart_es_geom_static)
        n_membership = len(sample.chart_membership)
        n_groups = len(sample.reference_chart_id)

        node_xyz.append(sample.node_xyz)
        node_normal.append(sample.node_normal)
        node_scalar_raw.append(sample.node_scalar_raw)
        node_vector_raw.append(sample.node_vector_raw)
        node_tangent_basis.append(sample.node_tangent_basis)

        cm = sample.chart_membership.copy()
        cm[:, 0] += chart_offset
        cm[:, 1] += node_offset
        chart_membership.append(cm)
        membership_sr.append(sample.membership_sr)
        membership_weight.append(sample.membership_weight)
        local_coords.append(sample.local_coords)
        quadrant.append(sample.quadrant)

        if sample.local_knn_edge_index.shape[1] > 0:
            local_knn_edge_index.append(sample.local_knn_edge_index + membership_offset)
            local_edge_attr.append(sample.local_edge_attr)

        chart_es_geom_static.append(sample.chart_es_geom_static)
        chart_anchor_pos.append(sample.chart_anchor_pos)
        chart_anchor_mask.append(sample.chart_anchor_mask)
        reference_chart_id.append(sample.reference_chart_id + chart_offset)

        if sample.chart_graph_edge_index.shape[1] > 0:
            chart_graph_edge_index.append(sample.chart_graph_edge_index + chart_offset)
        if sample.overlap_edge_index.shape[1] > 0:
            overlap_edge_index.append(sample.overlap_edge_index + chart_offset)
            overlap_shared_membership_index.append(
                sample.overlap_shared_membership_index + membership_offset
            )
            for ptr in sample.overlap_shared_ptr[1:]:
                overlap_shared_ptr.append(int(ptr) + shared_pair_offset)
            shared_pair_offset += len(sample.overlap_shared_membership_index)
            overlap_jaccard.append(sample.overlap_jaccard)

        # Stage 4: intra_geom_static（无 offset）
        intra_geom_static.append(sample.intra_geom_static)

        # Stage 4: ov2ce（+chart_edge_offset），先 append 后累加
        if sample.overlap_edge_to_chart_edge_index.shape[0] > 0:
            ov2ce_list.append(sample.overlap_edge_to_chart_edge_index + chart_edge_offset)
        chart_edge_offset += sample.chart_graph_edge_index.shape[1]

        # Stage 5: inter-level edges（both rows += chart_offset）
        if sample.inter_level_edge_index.shape[1] > 0:
            inter_edge_index_list.append(sample.inter_level_edge_index + chart_offset)
            inter_weights_list.append(sample.inter_level_weights)
            inter_edge_attr_list.append(sample.inter_level_edge_attr)

        for key, value in sample.chart_frame_metadata.items():
            arr = value.copy()
            if key == "chart_group_id":
                arr = arr + group_offset
            elif key == "chart_to_ref":          # Stage 4 新增：chart 索引需加 offset
                arr = arr + chart_offset
            meta_lists.setdefault(key, []).append(arr)

        node_batch.append(np.full((n_nodes,), batch_idx, dtype=np.int64))
        chart_batch.append(np.full((n_charts,), batch_idx, dtype=np.int64))
        membership_batch.append(np.full((n_membership,), batch_idx, dtype=np.int64))

        node_offset += n_nodes
        chart_offset += n_charts
        membership_offset += n_membership
        group_offset += n_groups

    def _torch(x: np.ndarray, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        tensor = torch.from_numpy(x)
        if dtype is not None:
            tensor = tensor.to(dtype=dtype)
        if device is not None:
            tensor = tensor.to(device)
        return tensor

    meta_tensors: Dict[str, torch.Tensor] = {}
    for key, arrays in meta_lists.items():
        if key in {"chart_center", "chart_center_normal", "chart_frame"}:
            merged = np.concatenate(arrays, axis=0).astype(np.float32, copy=False)
            meta_tensors[key] = _torch(merged, torch.float32)
        else:
            merged = np.concatenate(arrays, axis=0).astype(np.int64, copy=False)
            meta_tensors[key] = _torch(merged, torch.long)

    return Stage3TensorBatch(
        mol_ids=mol_ids,
        node_xyz=_torch(np.concatenate(node_xyz, axis=0).astype(np.float32, copy=False), torch.float32),
        node_normal=_torch(np.concatenate(node_normal, axis=0).astype(np.float32, copy=False), torch.float32),
        node_scalar_raw=_torch(np.concatenate(node_scalar_raw, axis=0).astype(np.float32, copy=False), torch.float32),
        node_vector_raw=_torch(np.concatenate(node_vector_raw, axis=0).astype(np.float32, copy=False), torch.float32),
        node_tangent_basis=_torch(np.concatenate(node_tangent_basis, axis=0).astype(np.float32, copy=False), torch.float32),
        chart_membership=_torch(np.concatenate(chart_membership, axis=0).astype(np.int64, copy=False), torch.long),
        membership_sr=_torch(np.concatenate(membership_sr, axis=0).astype(np.float32, copy=False), torch.float32),
        membership_weight=_torch(np.concatenate(membership_weight, axis=0).astype(np.float32, copy=False), torch.float32),
        local_coords=_torch(np.concatenate(local_coords, axis=0).astype(np.float32, copy=False), torch.float32),
        quadrant=_torch(np.concatenate(quadrant, axis=0).astype(np.int64, copy=False), torch.long),
        local_knn_edge_index=_torch(
            np.concatenate(local_knn_edge_index, axis=1).astype(np.int64, copy=False)
            if local_knn_edge_index else np.zeros((2, 0), dtype=np.int64),
            torch.long,
        ),
        local_edge_attr=_torch(
            np.concatenate(local_edge_attr, axis=0).astype(np.float32, copy=False)
            if local_edge_attr else np.zeros((0, 6), dtype=np.float32),
            torch.float32,
        ),
        chart_es_geom_static=_torch(np.concatenate(chart_es_geom_static, axis=0).astype(np.float32, copy=False), torch.float32),
        chart_anchor_pos=_torch(np.concatenate(chart_anchor_pos, axis=0).astype(np.float32, copy=False), torch.float32),
        chart_anchor_mask=_torch(np.concatenate(chart_anchor_mask, axis=0).astype(bool, copy=False), torch.bool),
        reference_chart_id=_torch(np.concatenate(reference_chart_id, axis=0).astype(np.int64, copy=False), torch.long),
        chart_graph_edge_index=_torch(
            np.concatenate(chart_graph_edge_index, axis=1).astype(np.int64, copy=False)
            if chart_graph_edge_index else np.zeros((2, 0), dtype=np.int64),
            torch.long,
        ),
        overlap_edge_index=_torch(
            np.concatenate(overlap_edge_index, axis=1).astype(np.int64, copy=False)
            if overlap_edge_index else np.zeros((2, 0), dtype=np.int64),
            torch.long,
        ),
        overlap_shared_membership_index=_torch(
            np.concatenate(overlap_shared_membership_index, axis=0).astype(np.int64, copy=False)
            if overlap_shared_membership_index else np.zeros((0, 2), dtype=np.int64),
            torch.long,
        ),
        overlap_shared_ptr=_torch(np.asarray(overlap_shared_ptr, dtype=np.int64), torch.long),
        overlap_jaccard=_torch(
            np.concatenate(overlap_jaccard, axis=0).astype(np.float32, copy=False)
            if overlap_jaccard else np.zeros((0,), dtype=np.float32),
            torch.float32,
        ),
        chart_frame_metadata=meta_tensors,
        node_batch=_torch(np.concatenate(node_batch, axis=0).astype(np.int64), torch.long),
        chart_batch=_torch(np.concatenate(chart_batch, axis=0).astype(np.int64), torch.long),
        membership_batch=_torch(np.concatenate(membership_batch, axis=0).astype(np.int64), torch.long),
        intra_geom_static=_torch(
            np.concatenate(intra_geom_static, axis=0).astype(np.float32, copy=False),
            torch.float32,
        ),
        overlap_edge_to_chart_edge_index=_torch(
            np.concatenate(ov2ce_list, axis=0).astype(np.int64, copy=False)
            if ov2ce_list else np.zeros((0,), dtype=np.int64),
            torch.long,
        ),
        inter_level_edge_index=_torch(
            np.concatenate(inter_edge_index_list, axis=1).astype(np.int64, copy=False)
            if inter_edge_index_list else np.zeros((2, 0), dtype=np.int64),
            torch.long,
        ),
        inter_level_weights=_torch(
            np.concatenate(inter_weights_list, axis=0).astype(np.float32, copy=False)
            if inter_weights_list else np.zeros((0,), dtype=np.float32),
            torch.float32,
        ),
        inter_level_edge_attr=_torch(
            np.concatenate(inter_edge_attr_list, axis=0).astype(np.float32, copy=False)
            if inter_edge_attr_list else np.zeros((0, 7), dtype=np.float32),
            torch.float32,
        ),
    )


def summarize_stage3_sample(sample: Stage3Sample) -> Dict[str, int]:
    return {
        "num_nodes": int(len(sample.node_xyz)),
        "num_charts": int(len(sample.chart_es_geom_static)),
        "num_memberships": int(len(sample.chart_membership)),
        "num_local_edges": int(sample.local_knn_edge_index.shape[1]),
        "num_chart_edges": int(sample.chart_graph_edge_index.shape[1]),
        "num_overlap_edges": int(sample.overlap_edge_index.shape[1]),
        "num_overlap_pairs": int(len(sample.overlap_shared_membership_index)),
        "num_reference_charts": int(len(sample.reference_chart_id)),
    }


def check_membership_weights(sample: Stage3Sample, atol: float = 1e-5) -> Dict[str, float | bool]:
    node_idx = sample.chart_membership[:, 1]
    weight_sum = np.bincount(node_idx, weights=sample.membership_weight, minlength=len(sample.node_xyz))
    touched = np.unique(node_idx)
    err = float(np.max(np.abs(weight_sum[touched] - 1.0))) if len(touched) else 0.0
    return {
        "ok": bool(err <= atol),
        "max_abs_error": err,
        "num_touched_nodes": int(len(touched)),
    }


def check_overlap_jaccard(sample: Stage3Sample, atol: float = 1e-6) -> Dict[str, float | bool]:
    chart_nodes: Dict[int, np.ndarray] = {}
    for chart_idx in range(len(sample.chart_es_geom_static)):
        mask = sample.chart_membership[:, 0] == chart_idx
        chart_nodes[chart_idx] = sample.chart_membership[mask, 1]

    max_err = 0.0
    for edge_idx in range(sample.overlap_edge_index.shape[1]):
        src = int(sample.overlap_edge_index[0, edge_idx])
        dst = int(sample.overlap_edge_index[1, edge_idx])
        a = chart_nodes[dst]
        b = chart_nodes[src]
        inter = len(np.intersect1d(a, b))
        union = len(np.union1d(a, b))
        jac = float(inter / max(1, union))
        max_err = max(max_err, abs(jac - float(sample.overlap_jaccard[edge_idx])))
    return {
        "ok": bool(max_err <= atol),
        "max_abs_error": float(max_err),
        "num_overlap_edges": int(sample.overlap_edge_index.shape[1]),
    }


def check_chart_plane_residual(
    sample: Stage3Sample,
    mean_atol: float = 0.2,
    p95_atol: float = 0.35,
) -> Dict[str, float | bool]:
    if len(sample.chart_membership) == 0:
        return {"ok": True, "mean_abs_residual": 0.0, "p95_abs_residual": 0.0, "max_abs_residual": 0.0}

    membership = sample.chart_membership
    chart_idx = membership[:, 0]
    node_idx = membership[:, 1]
    xyz = sample.node_xyz[node_idx]
    centers = sample.chart_frame_metadata["chart_center"][chart_idx]
    normals = sample.chart_frame_metadata["chart_center_normal"][chart_idx]
    residual = np.abs(np.sum((xyz - centers) * normals, axis=1))
    mean_val = float(residual.mean())
    p95_val = float(np.percentile(residual, 95))
    max_val = float(residual.max())
    return {
        "ok": bool(mean_val <= mean_atol and p95_val <= p95_atol),
        "mean_abs_residual": mean_val,
        "p95_abs_residual": p95_val,
        "max_abs_residual": max_val,
    }


def check_chart_normal_alignment(
    sample: Stage3Sample,
    mean_atol: float = 0.15,
    p95_atol: float = 0.35,
) -> Dict[str, float | bool]:
    if len(sample.chart_membership) == 0:
        return {"ok": True, "mean_deviation": 0.0, "p95_deviation": 0.0, "max_deviation": 0.0}

    membership = sample.chart_membership
    chart_idx = membership[:, 0]
    node_idx = membership[:, 1]
    node_normals = sample.node_normal[node_idx]
    chart_normals = sample.chart_frame_metadata["chart_center_normal"][chart_idx]
    deviation = 1.0 - np.clip(np.sum(node_normals * chart_normals, axis=1), -1.0, 1.0)
    mean_val = float(deviation.mean())
    p95_val = float(np.percentile(deviation, 95))
    max_val = float(deviation.max())
    return {
        "ok": bool(mean_val <= mean_atol and p95_val <= p95_atol),
        "mean_deviation": mean_val,
        "p95_deviation": p95_val,
        "max_deviation": max_val,
    }


def check_chart_vector_projection(
    sample: Stage3Sample,
    mean_rel_atol: float = 0.25,
    p95_rel_atol: float = 0.6,
) -> Dict[str, float | bool]:
    if len(sample.chart_membership) == 0:
        return {
            "ok": True,
            "mean_relative_loss": 0.0,
            "p95_relative_loss": 0.0,
            "max_relative_loss": 0.0,
            "mean_absolute_loss": 0.0,
        }

    membership = sample.chart_membership
    chart_idx = membership[:, 0]
    node_idx = membership[:, 1]
    chart_basis = sample.chart_frame_metadata["chart_frame"][chart_idx]
    vec3 = sample.node_vector_raw[node_idx]
    vec_chart_2d = project_vectors_to_basis(vec3, chart_basis)
    vec_chart_3d = reconstruct_vectors_from_basis(vec_chart_2d, chart_basis)
    abs_loss = np.linalg.norm(vec3 - vec_chart_3d, axis=-1)
    vec_norm = np.linalg.norm(vec3, axis=-1)
    rel_loss = abs_loss / np.maximum(vec_norm, _EPS)
    mean_rel = float(rel_loss.mean())
    p95_rel = float(np.percentile(rel_loss, 95))
    max_rel = float(rel_loss.max())
    mean_abs = float(abs_loss.mean())
    return {
        "ok": bool(mean_rel <= mean_rel_atol and p95_rel <= p95_rel_atol),
        "mean_relative_loss": mean_rel,
        "p95_relative_loss": p95_rel,
        "max_relative_loss": max_rel,
        "mean_absolute_loss": mean_abs,
    }


def validate_stage3_sample(sample: Stage3Sample) -> Dict[str, Dict[str, float | bool | int]]:
    summary = summarize_stage3_sample(sample)
    return {
        "summary": summary,
        "membership_weight": check_membership_weights(sample),
        "overlap_jaccard": check_overlap_jaccard(sample),
        "chart_plane_residual": check_chart_plane_residual(sample),
        "chart_normal_alignment": check_chart_normal_alignment(sample),
        "chart_vector_projection": check_chart_vector_projection(sample),
    }


__all__ = [
    "Stage3Sample",
    "Stage3TensorBatch",
    "build_stage3_sample",
    "build_tangent_basis",
    "project_vectors_to_basis",
    "reconstruct_vectors_from_basis",
    "rotate_vectors_between_bases",
    "stage3_cache_path",
    "save_stage3_sample",
    "load_stage3_sample",
    "load_stage3_entry",
    "save_stage3_bundle_entry",
    "list_stage3_bundle_ids",
    "collate_stage3_samples",
    "summarize_stage3_sample",
    "check_membership_weights",
    "check_overlap_jaccard",
    "check_chart_plane_residual",
    "check_chart_normal_alignment",
    "check_chart_vector_projection",
    "validate_stage3_sample",
]
