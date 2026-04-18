"""
Stage 2: Feature-Compatible Local Chart (FCLC) Atlas Construction.

For each ManifoldComponent (a connected triangular mesh with per-vertex
physical features from Stage 1), this module builds a set of overlapping
local charts (FCLC atlas) via a deterministic frontier-driven growth
algorithm, then computes:

  - Local PCA frame (e_{a,1}, e_{a,2}) and 4-quadrant partition
  - Explicit structure descriptor ES_a^local (geometry + scalar + vector blocks)
  - Inter-layer directional correspondence weights (counting-based)

Design principles
-----------------
  - No randomness: all algorithms are fully deterministic.
    Geodesic medoid uses stride-based candidate sub-sampling.
    PCA frame uses np.linalg.eigh (deterministic ascending eigenvalue order).
  - Parallelism: exposed through build_fclc_levels() which iterates over
    ManifoldComponents; the caller (preprocess_stage2.py) handles
    molecule-level parallelism via fork-based multiprocessing.
  - Caching: List[FCLCLevel] is serialised to pickle by save_fclc_levels().

Performance notes
-----------------
  - Full geodesic distance matrix is precomputed once per component via
    scipy.sparse.csgraph.shortest_path only for moderately sized components
    (default V ≤ 3000, configurable). All subsequent chart-growth calls use
    O(1) row slices instead of repeated Dijkstra.
  - Frontier computation and scoring are Numba-JIT-compiled (@njit, cache=True)
    and operate directly on raw CSR int32 arrays — no Python loops.
  - covered_mask is a numpy bool array; no Python set rebuilding per iteration.

Spatial units: Bohr throughout (inherited from Stage 1).
"""
from __future__ import annotations

import math
import os
import pickle
import zipfile
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra as sp_dijkstra
from scipy.sparse.csgraph import shortest_path as sp_shortest_path
from scipy.spatial import cKDTree

try:
    import numba
except Exception:  # pragma: no cover - runtime fallback when numba is unavailable
    class _NumbaTypedList:
        @staticmethod
        def empty_list(_dtype):
            return []

    class _NumbaStub:
        int32 = np.int32
        typed = type("typed", (), {"List": _NumbaTypedList})

        @staticmethod
        def njit(*args, **kwargs):
            def _decorator(fn):
                return fn
            return _decorator

    numba = _NumbaStub()

if TYPE_CHECKING:
    from ed2e.data.manifold import ManifoldComponent, ManifoldLevel

# ---------------------------------------------------------------------------
# Memory threshold: precompute full (V,V) distance matrix only when V ≤ this.
# A dense float64 matrix costs ~8 * V^2 bytes, so V=5000 already implies
# roughly 200 MB for the distance matrix alone. Keeping the default lower
# avoids late-run worker deaths when several large components are processed in
# parallel, while still allowing callers to override the threshold explicitly.
# ---------------------------------------------------------------------------
_DEFAULT_MEM_THRESH = 3000


def _resolve_mem_thresh(mem_thresh: Optional[int]) -> int:
    """Resolve the dense distance-matrix threshold.

    Priority:
      1. explicit function argument
      2. ED2E_FCLC_MEM_THRESH environment variable
      3. module default
    """
    if mem_thresh is not None:
        return max(0, int(mem_thresh))

    env_val = os.environ.get("ED2E_FCLC_MEM_THRESH")
    if env_val is None:
        return _DEFAULT_MEM_THRESH

    try:
        return max(0, int(env_val))
    except ValueError:
        return _DEFAULT_MEM_THRESH


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FCLCChart:
    """One Feature-Compatible Local Chart P_a.

    All vertex references are local indices into the parent ManifoldComponent.
    """
    chart_id:      int
    level_id:      int
    component_id:  int

    # Vertices (indices into ManifoldComponent.verts / .scalar_features / etc.)
    vert_indices:  np.ndarray   # (V_a,) int32

    # Geometry
    center:        np.ndarray   # (3,)     float32 — chart centre in Bohr
    center_normal: np.ndarray   # (3,)     float32 — unit normal at centre
    frame:         np.ndarray   # (2, 3)   float32 — PCA tangent frame (e_{a,1}, e_{a,2})
    local_coords:  np.ndarray   # (V_a, 2) float32 — 2-D tangent-plane projections
    quadrant:      np.ndarray   # (V_a,)   int8    — quadrant labels {0,1,2,3}

    # Features (rows extracted from ManifoldComponent)
    scalar_feats:  np.ndarray   # (V_a, 5)    float32
    vector_feats:  np.ndarray   # (V_a, 2, 3) float32

    # Precomputed explicit structure descriptor (supplied after construction)
    es_local:      np.ndarray   # (D_local,) float32

    # Stage 3 needs the original seed vertex and per-membership region score.
    # Defaults keep older Stage 2 pickles loadable.
    seed_vertex_idx: int = -1
    membership_sr: Optional[np.ndarray] = None


@dataclass
class FCLCLevel:
    """All FCLC charts for one density level M_k."""
    level_id:   int
    threshold:  float
    charts:     List[FCLCChart] = field(default_factory=list)

    # Inter-layer directional weights: k (receiver) ← k+1 (sender)
    # inter_weights[chart_id_a] = [(chart_id_b, w̃_{a←b}, mean_nn_dist, mean_normal_dev), ...]
    # mean_nn_dist:   mean distance from projected vertices to their NN in the target level
    # mean_normal_dev: mean (1 - cos(n_src, n_target_nn)) for those projections
    inter_weights: Optional[Dict[int, List[Tuple[int, float, float, float]]]] = None

    # Reverse direction: k+1 (receiver) ← k (sender), stored on level k+1
    # inter_weights_up[chart_id_b] = [(chart_id_a, w̃_{b←a}, mean_nn_dist, mean_normal_dev), ...]
    inter_weights_up: Optional[Dict[int, List[Tuple[int, float, float, float]]]] = None


# ---------------------------------------------------------------------------
# Step 1 — Mesh adjacency graph
# ---------------------------------------------------------------------------

def build_mesh_adjacency(
    verts: np.ndarray,   # (V, 3)
    faces: np.ndarray,   # (F, 3)
) -> csr_matrix:
    """Build a sparse adjacency matrix weighted by 3-D Euclidean edge length."""
    V = len(verts)
    i0 = faces[:, 0]; i1 = faces[:, 1]; i2 = faces[:, 2]

    # Three directed edges per triangle
    rows = np.concatenate([i0, i1, i1, i2, i2, i0])
    cols = np.concatenate([i1, i0, i2, i1, i0, i2])

    # Edge lengths
    d01 = np.linalg.norm(verts[i0] - verts[i1], axis=1)
    d12 = np.linalg.norm(verts[i1] - verts[i2], axis=1)
    d20 = np.linalg.norm(verts[i2] - verts[i0], axis=1)
    data = np.concatenate([d01, d01, d12, d12, d20, d20])

    adj = csr_matrix((data, (rows, cols)), shape=(V, V))
    return adj


# ---------------------------------------------------------------------------
# Step 2 — Geodesic medoid (deterministic)
# ---------------------------------------------------------------------------

def geodesic_medoid(
    adj:            csr_matrix,
    n_verts:        int,
    dist_mat:       Optional[np.ndarray] = None,
    max_candidates: int = 64,
) -> int:
    """Return the vertex index that minimises the sum of geodesic distances
    to all other candidate vertices.

    Candidate set: stride-based sub-sampling (no randomness).
    If dist_mat (V, V) is provided, uses precomputed distances.
    """
    stride = max(1, n_verts // max_candidates)
    candidates = list(range(0, n_verts, stride))

    if len(candidates) == 1:
        return candidates[0]

    cand_arr = np.array(candidates, dtype=np.int32)

    if dist_mat is not None:
        sub = dist_mat[np.ix_(cand_arr, cand_arr)]
    else:
        # Fallback: run Dijkstra only for candidate rows
        sub = sp_dijkstra(adj, indices=cand_arr, directed=False,
                          return_predecessors=False)[:, cand_arr]

    np.fill_diagonal(sub, 0.0)
    sub = np.where(np.isinf(sub), 0.0, sub)
    total = sub.sum(axis=1)
    return int(cand_arr[np.argmin(total)])


# ---------------------------------------------------------------------------
# Step 3 — Region compatibility scoring and chart growth
# ---------------------------------------------------------------------------

def _normalise_scalar_features(scalar_f: np.ndarray) -> np.ndarray:
    """Z-score normalise each of the 5 scalar channels independently."""
    mu  = scalar_f.mean(axis=0, keepdims=True)
    std = scalar_f.std(axis=0, keepdims=True)
    return (scalar_f - mu) / np.where(std > 1e-8, std, 1.0)


def compute_region_compatibility_scores(
    seed:          int,
    dist_row:      np.ndarray,        # (V,) precomputed geodesic distances from seed
    normals:       np.ndarray,        # (V, 3)
    scalar_f_norm: np.ndarray,        # (V, 5) — pre-normalised
    lam:           Tuple[float, float, float] = (0.4, 0.3, 0.3),
    sigma_d:       float = 1.0,
) -> np.ndarray:
    """Return region-compatibility scores S_R(seed, q) for all vertices."""
    lam_g, lam_s, lam_r = lam

    seed_normal = normals[seed]
    seed_scalar = scalar_f_norm[seed]

    # S_R components
    r_g = dist_row / (sigma_d + 1e-8)                                    # (V,)
    r_s = np.linalg.norm(scalar_f_norm - seed_scalar, axis=1)             # (V,)
    r_r = 1.0 - np.clip((normals * seed_normal).sum(axis=1), -1, 1)       # (V,)

    return lam_g * r_g + lam_s * r_s + lam_r * r_r


def grow_chart(
    seed:          int,
    dist_row:      np.ndarray,        # (V,) precomputed geodesic distances from seed
    normals:       np.ndarray,        # (V, 3)
    scalar_f_norm: np.ndarray,        # (V, 5) — pre-normalised
    tau_r:         float = 1.0,
    lam:           Tuple[float, float, float] = (0.4, 0.3, 0.3),
    sigma_d:       float = 1.0,
    return_scores: bool = False,
) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
    """Grow a chart from *seed* by accepting neighbours whose region
    compatibility score S_R < tau_r.

    Returns:
        sorted array of accepted vertex indices, or
        (accepted_vertex_indices, accepted_membership_sr) if return_scores=True.

    dist_row must be the precomputed geodesic distance from seed to all V verts.
    """
    S_R = compute_region_compatibility_scores(
        seed=seed,
        dist_row=dist_row,
        normals=normals,
        scalar_f_norm=scalar_f_norm,
        lam=lam,
        sigma_d=sigma_d,
    )

    mask = (S_R < tau_r) & ~np.isinf(dist_row)
    accepted = np.where(mask)[0].astype(np.int32)
    if return_scores:
        return accepted, S_R[accepted].astype(np.float32)
    return accepted


# ---------------------------------------------------------------------------
# Step 4 — Frontier-driven atlas growth  (JIT-compiled helpers)
# ---------------------------------------------------------------------------

@numba.njit(cache=True)
def _compute_frontier_jit(
    covered_mask: np.ndarray,   # (V,) bool
    indptr:       np.ndarray,   # (V+1,) int32 — CSR row pointers
    indices:      np.ndarray,   # (nnz,) int32 — CSR column indices
) -> np.ndarray:                # (F,) int32 frontier vertex indices
    """Return covered vertices that have at least one uncovered neighbour."""
    frontier = numba.typed.List.empty_list(numba.int32)
    n = len(covered_mask)
    for v in range(n):
        if not covered_mask[v]:
            continue
        for j in range(indptr[v], indptr[v + 1]):
            nbr = indices[j]
            if not covered_mask[nbr]:
                frontier.append(numba.int32(v))
                break
    result = np.empty(len(frontier), dtype=np.int32)
    for i in range(len(frontier)):
        result[i] = frontier[i]
    return result


@numba.njit(cache=True)
def _frontier_scores_jit(
    frontier:     np.ndarray,   # (F,) int32
    covered_mask: np.ndarray,   # (V,) bool
    indptr:       np.ndarray,   # (V+1,) int32
    indices:      np.ndarray,   # (nnz,) int32
    normals:      np.ndarray,   # (V, 3) float64
    lam_U:        float,
    lam_D:        float,
    lam_C:        float,
) -> np.ndarray:                # (F,) float64 scores
    """Compute S_frontier(b) = lam_U·U + lam_D·D + lam_C·C for each frontier point."""
    n_f = len(frontier)
    scores = np.zeros(n_f, dtype=np.float64)
    for idx in range(n_f):
        b = frontier[idx]
        start = indptr[b]
        end   = indptr[b + 1]
        deg   = end - start
        if deg == 0:
            continue

        U = 0.0
        D = 0.0
        n_mean = np.zeros(3, dtype=np.float64)

        for j in range(start, end):
            nbr = indices[j]
            if not covered_mask[nbr]:
                U += 1.0
                D += 1.0
            n_mean[0] += normals[nbr, 0]
            n_mean[1] += normals[nbr, 1]
            n_mean[2] += normals[nbr, 2]

        U /= deg
        D /= deg

        # Normalise mean normal
        norm_sq = n_mean[0]*n_mean[0] + n_mean[1]*n_mean[1] + n_mean[2]*n_mean[2]
        if norm_sq > 1e-16:
            inv = 1.0 / math.sqrt(norm_sq)
            n_mean[0] *= inv
            n_mean[1] *= inv
            n_mean[2] *= inv

        # Mean angular deviation from mean normal → C = exp(-pac)
        pac = 0.0
        for j in range(start, end):
            nbr = indices[j]
            dot = (normals[nbr, 0] * n_mean[0] +
                   normals[nbr, 1] * n_mean[1] +
                   normals[nbr, 2] * n_mean[2])
            if dot > 1.0:
                dot = 1.0
            elif dot < -1.0:
                dot = -1.0
            pac += 1.0 - dot
        pac /= deg
        C = math.exp(-pac)

        scores[idx] = lam_U * U + lam_D * D + lam_C * C
    return scores


def _s2_score(
    b:             int,
    candidates:    np.ndarray,    # (N,) vertex indices
    dist_row_b:    np.ndarray,    # (V,) geodesic distances from b
    normals:       np.ndarray,    # (V, 3)
    scalar_f_norm: np.ndarray,    # (V, 5)
    sigma_d:       float = 1.0,
    w:             Tuple[float, float, float] = (0.4, 0.3, 0.3),
) -> np.ndarray:
    """Two-point symmetric score S_2(b, q) for each candidate q."""
    w_d, w_n, w_f = w
    s_d = dist_row_b[candidates] / (sigma_d + 1e-8)
    s_n = 1.0 - np.clip(
        (normals[candidates] * normals[b]).sum(axis=1), -1, 1)
    s_f = np.linalg.norm(
        scalar_f_norm[candidates] - scalar_f_norm[b], axis=1)
    return w_d * s_d + w_n * s_n + w_f * s_f


def _candidate_quality(
    P_verts:       np.ndarray,    # chart candidate vertices
    covered_mask:  np.ndarray,    # (V,) bool
    dist_row_e:    np.ndarray,    # (V,) geodesic distances from seed e (reuse from grow_chart)
    normals:       np.ndarray,
    scalar_f_norm: np.ndarray,
    seed:          int,
    lam:           Tuple,
    alpha:         Tuple[float, float, float, float] = (0.4, 0.3, 0.3, 0.2),
) -> float:
    """Compute Q(e) for a candidate chart rooted at seed.

    Reuses dist_row_e already computed by grow_chart — no extra Dijkstra.
    """
    alpha_U, alpha_D, alpha_C, alpha_F = alpha

    omega_mask_P = covered_mask[P_verts]
    n_P = len(P_verts)
    n_new = int((~omega_mask_P).sum())

    U_P = n_new / (n_P + 1e-8)
    D_P = U_P  # same by definition in our simplified form

    # C_P: internal compatibility = exp(-mean S_R)
    seed_normal = normals[seed]
    seed_scalar = scalar_f_norm[seed]
    lam_g, lam_s, lam_r = lam
    r_g = dist_row_e[P_verts]
    r_s = np.linalg.norm(scalar_f_norm[P_verts] - seed_scalar, axis=1)
    r_r = 1.0 - np.clip((normals[P_verts] * seed_normal).sum(axis=1), -1, 1)
    mean_SR = (lam_g * r_g + lam_s * r_s + lam_r * r_r).mean()
    C_P = float(np.exp(-mean_SR))

    n_overlap = int(omega_mask_P.sum())
    L_front = n_overlap / (n_P + 1e-8)

    return alpha_U * U_P + alpha_D * D_P + alpha_C * C_P - alpha_F * L_front


def _get_dist_row(
    v:            int,
    dist_mat:     Optional[np.ndarray],
    adj:          csr_matrix,
) -> np.ndarray:
    """Return geodesic distance row from vertex v. Uses precomputed mat if available."""
    if dist_mat is not None:
        return dist_mat[v]
    return sp_dijkstra(adj, indices=[v], directed=False,
                       return_predecessors=False)[0]


def build_fclc_atlas(
    verts:          np.ndarray,   # (V, 3)
    faces:          np.ndarray,   # (F, 3)
    normals:        np.ndarray,   # (V, 3)
    scalar_f:       np.ndarray,   # (V, 5)
    vector_f:       np.ndarray,   # (V, 2, 3)
    component_id:   int,
    level_id:       int,
    tau_r:          float = 1.0,
    tau_2:          float = 1.5,
    min_chart_size: int   = 5,
    lam:            Tuple[float, float, float] = (0.4, 0.3, 0.3),
    alpha:          Tuple[float, float, float, float] = (0.4, 0.3, 0.3, 0.2),
    mem_thresh:     Optional[int] = None,
) -> List[FCLCChart]:
    """Build FCLC atlas for one ManifoldComponent.

    Returns list of FCLCChart objects (frames and es_local populated separately
    by build_fclc_levels after this function returns).
    """
    V = len(verts)
    adj = build_mesh_adjacency(verts, faces)
    scalar_f_norm = _normalise_scalar_features(scalar_f)

    # --- precompute full distance matrix (O(V²) space, eliminates all Dijkstra in loop) ---
    resolved_mem_thresh = _resolve_mem_thresh(mem_thresh)
    if V <= resolved_mem_thresh:
        dist_mat: Optional[np.ndarray] = sp_shortest_path(
            adj, directed=False, return_predecessors=False)
    else:
        dist_mat = None

    # Raw CSR arrays for Numba kernels
    indptr  = adj.indptr.astype(np.int32)
    csr_indices = adj.indices.astype(np.int32)
    normals_f64 = normals.astype(np.float64)

    lam_U, lam_D, lam_C = 0.4, 0.3, 0.3  # frontier score weights

    # --- seed the first chart ---
    seed0 = geodesic_medoid(adj, V, dist_mat=dist_mat)
    dist0 = _get_dist_row(seed0, dist_mat, adj)
    P0, P0_sr = grow_chart(
        seed0,
        dist0,
        normals,
        scalar_f_norm,
        tau_r=tau_r,
        lam=lam,
        return_scores=True,
    )
    if len(P0) < min_chart_size:
        P0 = np.arange(V, dtype=np.int32)
        P0_sr = compute_region_compatibility_scores(
            seed=seed0,
            dist_row=dist0,
            normals=normals,
            scalar_f_norm=scalar_f_norm,
            lam=lam,
        )[P0].astype(np.float32)

    charts: List[FCLCChart] = []
    covered_mask = np.zeros(V, dtype=np.bool_)

    def _accept_chart(
        seed: int,
        P_verts: np.ndarray,
        membership_sr: np.ndarray,
        chart_id: int,
    ) -> FCLCChart:
        frame, lc, quad = compute_pca_frame_and_coords(
            verts[P_verts], verts[seed], normals[seed])
        return FCLCChart(
            chart_id=chart_id,
            level_id=level_id,
            component_id=component_id,
            vert_indices=P_verts.astype(np.int32),
            center=verts[seed].copy(),
            center_normal=normals[seed].copy(),
            frame=frame,
            local_coords=lc,
            quadrant=quad,
            scalar_feats=scalar_f[P_verts].copy(),
            vector_feats=vector_f[P_verts].copy(),
            es_local=np.zeros(1, dtype=np.float32),  # filled later
            seed_vertex_idx=int(seed),
            membership_sr=np.asarray(membership_sr, dtype=np.float32).copy(),
        )

    ch = _accept_chart(seed0, P0, P0_sr, 0)
    charts.append(ch)
    covered_mask[P0] = True

    max_iters = V * 4   # safety cap
    itr = 0

    while covered_mask.sum() / V < 0.99 and itr < max_iters:
        itr += 1

        # --- collect frontier (JIT) ---
        frontier = _compute_frontier_jit(covered_mask, indptr, csr_indices)
        if len(frontier) == 0:
            break

        # --- score frontier (JIT) ---
        fscores = _frontier_scores_jit(
            frontier, covered_mask, indptr, csr_indices,
            normals_f64, lam_U, lam_D, lam_C)
        b_star = int(frontier[np.argmax(fscores)])

        # --- two-point reference neighbourhood ---
        uncov_arr = np.where(~covered_mask)[0].astype(np.int32)
        if len(uncov_arr) == 0:
            break

        dist_b = _get_dist_row(b_star, dist_mat, adj)
        s2 = _s2_score(b_star, uncov_arr, dist_b, normals, scalar_f_norm)
        E_b = uncov_arr[s2 <= tau_2]

        if len(E_b) == 0:
            # No compatible candidates; use uncovered adjacents of b_star
            E_b = np.array([n for n in csr_indices[indptr[b_star]:indptr[b_star+1]]
                            if not covered_mask[n]], dtype=np.int32)

        if len(E_b) == 0:
            covered_mask[uncov_arr[0]] = True
            continue

        # --- evaluate candidate charts ---
        best_q = -1e9
        best_seed = -1
        best_P: Optional[np.ndarray] = None
        for e in E_b:
            dist_e = _get_dist_row(int(e), dist_mat, adj)
            P_e = grow_chart(int(e), dist_e, normals, scalar_f_norm,
                             tau_r=tau_r, lam=lam)
            if len(P_e) < min_chart_size:
                continue
            q = _candidate_quality(P_e, covered_mask, dist_e,
                                   normals, scalar_f_norm, int(e), lam, alpha)
            if q > best_q:
                best_q = q
                best_seed = int(e)
                best_P = P_e

        if best_P is None:
            best_seed = int(uncov_arr[0])
            best_P = np.array([best_seed], dtype=np.int32)
            best_sr = compute_region_compatibility_scores(
                seed=best_seed,
                dist_row=_get_dist_row(best_seed, dist_mat, adj),
                normals=normals,
                scalar_f_norm=scalar_f_norm,
                lam=lam,
            )[best_P].astype(np.float32)
        else:
            best_sr = compute_region_compatibility_scores(
                seed=best_seed,
                dist_row=_get_dist_row(best_seed, dist_mat, adj),
                normals=normals,
                scalar_f_norm=scalar_f_norm,
                lam=lam,
            )[best_P].astype(np.float32)

        ch = _accept_chart(best_seed, best_P, best_sr, len(charts))
        charts.append(ch)
        covered_mask[best_P] = True

    # --- ensure full coverage: assign remaining uncovered vertices to
    #     the chart whose centre is closest ---
    remaining = np.where(~covered_mask)[0].astype(np.int32)
    if len(remaining) > 0:
        centers = np.array([c.center for c in charts], dtype=np.float32)
        tree = cKDTree(centers)
        _, chart_for_vert = tree.query(verts[remaining])
        for vi, ci in zip(remaining, chart_for_vert):
            c = charts[ci]
            c.vert_indices = np.append(c.vert_indices, vi).astype(np.int32)
            c.scalar_feats = np.vstack([c.scalar_feats, scalar_f[vi]])
            c.vector_feats = np.vstack([c.vector_feats, vector_f[vi:vi+1]])
            dist_seed = _get_dist_row(int(c.seed_vertex_idx), dist_mat, adj)
            sr_val = compute_region_compatibility_scores(
                seed=int(c.seed_vertex_idx),
                dist_row=dist_seed,
                normals=normals,
                scalar_f_norm=scalar_f_norm,
                lam=lam,
            )[vi]
            if c.membership_sr is None:
                c.membership_sr = np.array([sr_val], dtype=np.float32)
            else:
                c.membership_sr = np.append(
                    c.membership_sr,
                    np.float32(sr_val),
                ).astype(np.float32)
            c.local_coords = np.vstack([
                c.local_coords,
                _project_to_local(verts[vi], c.center, c.center_normal, c.frame)
            ])
            s = np.sign(c.local_coords[-1])
            c.quadrant = np.append(c.quadrant, _sign_to_quadrant(s[0], s[1]))

    return charts


def _project_to_local(
    p: np.ndarray,      # (3,)
    center: np.ndarray,
    normal: np.ndarray,
    frame:  np.ndarray, # (2, 3)
) -> np.ndarray:
    """Project point p to local 2-D chart coordinates."""
    xi = p - center
    xi -= (xi @ normal) * normal
    return np.array([xi @ frame[0], xi @ frame[1]], dtype=np.float32)


def _sign_to_quadrant(s1: float, s2: float) -> int:
    """Map (sign(u1), sign(u2)) to quadrant {0,1,2,3}."""
    if s1 >= 0 and s2 >= 0:
        return 0
    if s1 < 0 and s2 >= 0:
        return 1
    if s1 < 0 and s2 < 0:
        return 2
    return 3


# ---------------------------------------------------------------------------
# Step 5 — Local PCA frame and 4-quadrant partition
# ---------------------------------------------------------------------------

def compute_pca_frame_and_coords(
    chart_verts:   np.ndarray,   # (V_a, 3)
    center:        np.ndarray,   # (3,)
    center_normal: np.ndarray,   # (3,)
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute PCA-based local tangent frame and 2-D coordinates.

    Returns:
        frame       : (2, 3) float32 — (e_{a,1}, e_{a,2}) orthonormal tangent basis
        local_coords: (V_a, 2) float32 — 2-D projections
        quadrant    : (V_a,) int8
    """
    n = center_normal.astype(np.float64)
    n /= (np.linalg.norm(n) + 1e-12)

    # Tangent-plane projections ξ_i = (I - n n^T)(x_i - c)
    delta = (chart_verts - center).astype(np.float64)        # (V_a, 3)
    xi = delta - (delta @ n)[:, None] * n                     # (V_a, 3)

    # Use Gram-Schmidt to get an initial tangent basis
    ref = np.array([1.0, 0.0, 0.0])
    if abs(n @ ref) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    t1 = ref - (ref @ n) * n
    t1 /= (np.linalg.norm(t1) + 1e-12)
    t2 = np.cross(n, t1)
    t2 /= (np.linalg.norm(t2) + 1e-12)

    # 2-D coordinates in this preliminary basis
    u = xi @ np.stack([t1, t2], axis=1)    # (V_a, 2)

    # PCA on 2-D coords for deterministic, data-driven principal direction
    if len(u) >= 2:
        cov = (u.T @ u) / max(len(u) - 1, 1)    # (2, 2)
        eigvals, eigvecs = np.linalg.eigh(cov)    # ascending order
        e1_2d = eigvecs[:, 1]                     # largest eigenvalue direction
        e1 = e1_2d[0] * t1 + e1_2d[1] * t2
    else:
        e1 = t1.copy()

    e1 /= (np.linalg.norm(e1) + 1e-12)
    e2 = np.cross(n, e1)
    e2 /= (np.linalg.norm(e2) + 1e-12)

    frame = np.stack([e1, e2], axis=0).astype(np.float32)   # (2, 3)

    # Recompute 2-D coords in the final frame
    lc = np.column_stack([xi @ e1, xi @ e2]).astype(np.float32)  # (V_a, 2)

    # Quadrant labels
    s1 = np.sign(lc[:, 0])
    s2 = np.sign(lc[:, 1])
    quad = np.zeros(len(lc), dtype=np.int8)
    quad[(s1 >= 0) & (s2 >= 0)] = 0
    quad[(s1 < 0)  & (s2 >= 0)] = 1
    quad[(s1 < 0)  & (s2 < 0)]  = 2
    quad[(s1 >= 0) & (s2 < 0)]  = 3

    return frame, lc, quad


# ---------------------------------------------------------------------------
# Step 6 — Explicit structure descriptor ES_a^local
# ---------------------------------------------------------------------------

def compute_es_local(chart: FCLCChart) -> np.ndarray:
    """Build the explicit structure descriptor ES_a^local.

    Three blocks, each broken down by quadrant:

    Geometry block (~12 dims):
        per-quadrant vertex fraction (4), chart total area approx (1),
        H mean/std (2), K mean/std (2), max geodesic radius approx (1)

    Scalar block (~25 dims):
        5 physical quantities × 4 quadrants × mean   (20)
        5 physical quantities × global std            (5)

    Vector block (~16 dims):
        2 tangential gradient fields × 4 quadrants × 2 frame projections  (16)

    Total: 53 dimensions.
    """
    quad = chart.quadrant          # (V_a,)
    sc   = chart.scalar_feats      # (V_a, 5)
    vf   = chart.vector_feats      # (V_a, 2, 3)
    lc   = chart.local_coords      # (V_a, 2)
    e1   = chart.frame[0]          # (3,)
    e2   = chart.frame[1]          # (3,)
    V_a  = len(quad)

    # ---- geometry block ----
    geom = []

    for q in range(4):
        geom.append(np.sum(quad == q) / (V_a + 1e-8))

    extent = float(np.linalg.norm(lc, axis=1).max()) if V_a > 0 else 0.0
    geom.append(extent)

    H_vals = sc[:, 2];  K_vals = sc[:, 3]
    geom += [float(H_vals.mean()), float(H_vals.std()),
             float(K_vals.mean()), float(K_vals.std())]

    geom.append(extent)   # max-geodesic-radius proxy

    while len(geom) < 12:
        geom.append(0.0)
    geom = geom[:12]

    # ---- scalar block ----
    scalar_blk = []
    for feat_idx in range(5):
        vals = sc[:, feat_idx]
        for q in range(4):
            mask = quad == q
            scalar_blk.append(float(vals[mask].mean()) if mask.any() else 0.0)
    for feat_idx in range(5):
        scalar_blk.append(float(sc[:, feat_idx].std()))

    # ---- vector block ----
    vector_blk = []
    for vec_idx in range(2):
        vec3 = vf[:, vec_idx, :]   # (V_a, 3)
        u1 = (vec3 * e1).sum(axis=1)   # (V_a,)
        u2 = (vec3 * e2).sum(axis=1)   # (V_a,)
        for q in range(4):
            mask = quad == q
            v1 = float(u1[mask].mean()) if mask.any() else 0.0
            v2 = float(u2[mask].mean()) if mask.any() else 0.0
            vector_blk += [v1, v2]

    es = np.array(geom + scalar_blk + vector_blk, dtype=np.float32)
    assert len(es) == 53, f"ES_local expected 53 dims, got {len(es)}"
    return es


# ---------------------------------------------------------------------------
# Step 7 — Inter-layer directional correspondence weights
# ---------------------------------------------------------------------------

def compute_inter_layer_weights(
    charts_k:   List[FCLCChart],   # receiver (inner, level k)
    charts_k1:  List[FCLCChart],   # sender   (outer, level k+1)
    verts_k:    np.ndarray,        # (V_k, 3)
    normals_k:  np.ndarray,        # (V_k, 3)
    verts_k1:   np.ndarray,        # (V_{k+1}, 3)
) -> Dict[int, List[Tuple[int, float]]]:
    """Compute normalised counting-based correspondence weights w̃_{a←b}.

    For each vertex x in chart P_a^(k), project along its normal n(x) and
    find the nearest vertex in M_{k+1} (KD-tree nearest-neighbour).
    Count how many projections land in each chart P_b^(k+1); normalise.

    Returns dict: inter_weights[chart_id_a] = [(chart_id_b, w̃), ...]
    """
    if len(charts_k1) == 0 or len(verts_k1) == 0:
        return {}

    vert_to_chart_k1 = np.full(len(verts_k1), -1, dtype=np.int32)
    for ch in charts_k1:
        vert_to_chart_k1[ch.vert_indices] = ch.chart_id

    tree_k1 = cKDTree(verts_k1)

    result: Dict[int, List[Tuple[int, float]]] = {}

    for ch_a in charts_k:
        vid = ch_a.vert_indices
        src_verts   = verts_k[vid]
        src_normals = normals_k[vid]

        step = 0.5   # Bohr
        proj = src_verts + step * src_normals

        _, nn_idx = tree_k1.query(proj)
        chart_ids_b = vert_to_chart_k1[nn_idx]

        valid = chart_ids_b >= 0
        D_a = int(valid.sum())
        if D_a == 0:
            result[ch_a.chart_id] = []
            continue

        chart_ids_valid = chart_ids_b[valid]
        unique_b, counts = np.unique(chart_ids_valid, return_counts=True)
        raw_w = counts.astype(np.float64) / D_a
        total = raw_w.sum()
        norm_w = raw_w / total if total > 1e-12 else raw_w

        result[ch_a.chart_id] = [
            (int(b), float(w)) for b, w in zip(unique_b, norm_w)
        ]

    return result


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_fclc_levels(
    manifold_levels:  List[ManifoldLevel],
    tau_r:            float = 1.0,
    tau_2:            float = 1.5,
    min_chart_size:   int   = 5,
    compute_inter:    bool  = True,
    lam:              Tuple[float, float, float] = (0.4, 0.3, 0.3),
    alpha:            Tuple[float, float, float, float] = (0.4, 0.3, 0.3, 0.2),
    mem_thresh:       Optional[int] = None,
) -> List[FCLCLevel]:
    """Build FCLC atlas for all manifold levels of one molecule.

    Args:
        manifold_levels: output of extract_manifold_levels() (Stage 1).
        tau_r:           region compatibility threshold.
        tau_2:           two-point reference neighbourhood threshold.
        min_chart_size:  minimum vertices per chart.
        compute_inter:   whether to compute inter-layer weights.
        lam:             (λ_g, λ_s, λ_r) weights for S_R.
        alpha:           (α_U, α_D, α_C, α_F) weights for Q(e).
        mem_thresh:      dense geodesic precompute threshold. Components with
                         more than this many vertices fall back to row-wise
                         Dijkstra to cap peak memory.

    Returns:
        List[FCLCLevel], ordered by level_id (0 = innermost).
    """
    fclc_levels: List[FCLCLevel] = []

    chart_id_offset = 0   # global chart id counter (across components & levels)

    for ml in manifold_levels:
        lv = FCLCLevel(level_id=ml.level_id, threshold=ml.threshold)

        for comp in ml.components:
            verts   = comp.verts
            faces   = comp.faces
            normals = comp.normals
            scalar_f = comp.scalar_features
            vector_f = comp.vector_features

            if len(verts) < min_chart_size:
                continue

            charts = build_fclc_atlas(
                verts, faces, normals, scalar_f, vector_f,
                component_id=comp.component_id,
                level_id=ml.level_id,
                tau_r=tau_r, tau_2=tau_2,
                min_chart_size=min_chart_size,
                lam=lam, alpha=alpha,
                mem_thresh=mem_thresh,
            )

            for ch in charts:
                ch.chart_id = chart_id_offset
                ch.es_local = compute_es_local(ch)
                chart_id_offset += 1
                lv.charts.append(ch)

        fclc_levels.append(lv)

    # --- inter-layer weights ---
    if compute_inter and len(fclc_levels) >= 2:
        for ki in range(len(fclc_levels) - 1):
            lv_k  = fclc_levels[ki]
            lv_k1 = fclc_levels[ki + 1]

            ml_k  = manifold_levels[ki]
            ml_k1 = manifold_levels[ki + 1]

            if len(ml_k.components) == 0 or len(ml_k1.components) == 0:
                lv_k.inter_weights = {}
                lv_k1.inter_weights_up = {}
                continue

            from collections import defaultdict
            charts_k_by_comp:  Dict[int, List[FCLCChart]] = defaultdict(list)
            charts_k1_by_comp: Dict[int, List[FCLCChart]] = defaultdict(list)
            for ch in lv_k.charts:
                charts_k_by_comp[ch.component_id].append(ch)
            for ch in lv_k1.charts:
                charts_k1_by_comp[ch.component_id].append(ch)

            comp_verts_k: Dict[int, np.ndarray] = {
                c.component_id: c.verts for c in ml_k.components}
            comp_normals_k: Dict[int, np.ndarray] = {
                c.component_id: c.normals for c in ml_k.components}

            verts_k1_all = np.vstack([c.verts for c in ml_k1.components]) \
                if ml_k1.components else np.zeros((0, 3), dtype=np.float32)
            normals_k1_all = np.vstack([c.normals for c in ml_k1.components]) \
                if ml_k1.components else np.zeros((0, 3), dtype=np.float32)

            # Build vert_to_chart_id for k+1 in the concatenated array
            vert_to_chart_k1 = np.full(len(verts_k1_all), -1, dtype=np.int32)
            offset_k1 = 0
            for c in ml_k1.components:
                for ch in charts_k1_by_comp.get(c.component_id, []):
                    vert_to_chart_k1[offset_k1 + ch.vert_indices] = ch.chart_id
                offset_k1 += len(c.verts)

            if len(verts_k1_all) == 0:
                lv_k.inter_weights = {}
                lv_k1.inter_weights_up = {}
                continue

            tree_k1 = cKDTree(verts_k1_all)

            # --- forward direction: k ← k+1 ---
            inter: Dict[int, List[Tuple[int, float, float, float]]] = {}
            for c in ml_k.components:
                verts_c   = comp_verts_k.get(c.component_id, c.verts)
                normals_c = comp_normals_k.get(c.component_id, c.normals)
                for ch_a in charts_k_by_comp.get(c.component_id, []):
                    vid   = ch_a.vert_indices
                    src_v = verts_c[vid]
                    src_n = normals_c[vid]
                    proj  = src_v + 0.5 * src_n

                    nn_dists, nn_idx = tree_k1.query(proj)
                    chart_ids_b = vert_to_chart_k1[nn_idx]

                    valid = chart_ids_b >= 0
                    D_a = int(valid.sum())
                    if D_a == 0:
                        inter[ch_a.chart_id] = []
                        continue

                    valid_dists    = nn_dists[valid]
                    valid_idx      = nn_idx[valid]
                    valid_cids_b   = chart_ids_b[valid]
                    valid_src_n    = src_n[valid]
                    valid_tgt_n    = normals_k1_all[valid_idx]

                    unique_b, cnts = np.unique(valid_cids_b, return_counts=True)
                    raw_w  = cnts.astype(np.float64) / D_a
                    norm_w = raw_w / (raw_w.sum() + 1e-12)
                    entry: List[Tuple[int, float, float, float]] = []
                    for b_id, w in zip(unique_b, norm_w):
                        mask_b    = valid_cids_b == b_id
                        mean_d    = float(valid_dists[mask_b].mean())
                        cos_sim   = np.clip(
                            (valid_src_n[mask_b] * valid_tgt_n[mask_b]).sum(-1),
                            -1.0, 1.0,
                        )
                        mean_nd   = float((1.0 - cos_sim).mean())
                        entry.append((int(b_id), float(w), mean_d, mean_nd))
                    inter[ch_a.chart_id] = entry

            lv_k.inter_weights = inter

            # --- reverse direction: k+1 ← k ---
            verts_k_all   = np.vstack([c.verts   for c in ml_k.components]) \
                if ml_k.components else np.zeros((0, 3), dtype=np.float32)
            normals_k_all = np.vstack([c.normals for c in ml_k.components]) \
                if ml_k.components else np.zeros((0, 3), dtype=np.float32)

            vert_to_chart_k = np.full(len(verts_k_all), -1, dtype=np.int32)
            offset_k = 0
            for c in ml_k.components:
                for ch in charts_k_by_comp.get(c.component_id, []):
                    vert_to_chart_k[offset_k + ch.vert_indices] = ch.chart_id
                offset_k += len(c.verts)

            comp_verts_k1:   Dict[int, np.ndarray] = {
                c.component_id: c.verts   for c in ml_k1.components}
            comp_normals_k1: Dict[int, np.ndarray] = {
                c.component_id: c.normals for c in ml_k1.components}

            tree_k = cKDTree(verts_k_all) if len(verts_k_all) > 0 else None

            inter_up: Dict[int, List[Tuple[int, float, float, float]]] = {}
            for c in ml_k1.components:
                verts_c1   = comp_verts_k1.get(c.component_id, c.verts)
                normals_c1 = comp_normals_k1.get(c.component_id, c.normals)
                for ch_b in charts_k1_by_comp.get(c.component_id, []):
                    if tree_k is None or len(verts_k_all) == 0:
                        inter_up[ch_b.chart_id] = []
                        continue
                    vid   = ch_b.vert_indices
                    src_v = verts_c1[vid]
                    src_n = normals_c1[vid]
                    proj  = src_v - 0.5 * src_n          # project outward (toward level k)

                    nn_dists_up, nn_idx_up = tree_k.query(proj)
                    chart_ids_a = vert_to_chart_k[nn_idx_up]

                    valid = chart_ids_a >= 0
                    D_b = int(valid.sum())
                    if D_b == 0:
                        inter_up[ch_b.chart_id] = []
                        continue

                    valid_dists_up = nn_dists_up[valid]
                    valid_idx_up   = nn_idx_up[valid]
                    valid_cids_a   = chart_ids_a[valid]
                    valid_src_n    = src_n[valid]
                    valid_tgt_n    = normals_k_all[valid_idx_up]

                    unique_a, cnts = np.unique(valid_cids_a, return_counts=True)
                    raw_w  = cnts.astype(np.float64) / D_b
                    norm_w = raw_w / (raw_w.sum() + 1e-12)
                    entry_up: List[Tuple[int, float, float, float]] = []
                    for a_id, w in zip(unique_a, norm_w):
                        mask_a  = valid_cids_a == a_id
                        mean_d  = float(valid_dists_up[mask_a].mean())
                        cos_sim = np.clip(
                            (valid_src_n[mask_a] * valid_tgt_n[mask_a]).sum(-1),
                            -1.0, 1.0,
                        )
                        mean_nd = float((1.0 - cos_sim).mean())
                        entry_up.append((int(a_id), float(w), mean_d, mean_nd))
                    inter_up[ch_b.chart_id] = entry_up

            lv_k1.inter_weights_up = inter_up

    return fclc_levels


# ---------------------------------------------------------------------------
# Numba JIT warm-up (call once in the main process before forking)
# ---------------------------------------------------------------------------

def _warmup_numba_jit() -> None:
    """Trigger Numba JIT compilation of frontier kernels in the current process.

    Call this in the main process **before** creating a fork-based worker pool.
    Forked workers will then inherit the already-compiled code via copy-on-write,
    so they never need to JIT-compile themselves — preventing the semaphore
    creation that causes resource_tracker warnings at shutdown.
    """
    mask    = np.array([True, False, True, False], dtype=np.bool_)
    indptr  = np.array([0, 1, 2, 3, 4], dtype=np.int32)
    indices = np.array([1, 0, 3, 2], dtype=np.int32)
    normals = np.zeros((4, 3), dtype=np.float64)
    normals[:, 2] = 1.0
    frontier = np.array([0, 2], dtype=np.int32)

    _compute_frontier_jit(mask, indptr, indices)
    _frontier_scores_jit(frontier, mask, indptr, indices, normals, 0.4, 0.3, 0.3)


# ---------------------------------------------------------------------------
# Cache I/O
# ---------------------------------------------------------------------------

def fclc_cache_path(
    cache_dir:  str,
    mol_id:     str,
    tau_r:      float,
    tau_2:      float,
) -> str:
    import os
    return os.path.join(cache_dir, f"{mol_id}_fclc_tr{tau_r:.2f}_t2{tau_2:.2f}.pkl")


def is_fclc_bundle_path(path: str) -> bool:
    return path.endswith(".zip")


def fclc_bundle_member(mol_id: str) -> str:
    return f"molecules/{mol_id}.pkl"


def save_fclc_bundle_entry(zf: zipfile.ZipFile, mol_id: str, levels: List[FCLCLevel]) -> None:
    zf.writestr(
        fclc_bundle_member(mol_id),
        pickle.dumps(levels, protocol=pickle.HIGHEST_PROTOCOL),
        compress_type=zipfile.ZIP_STORED,
    )


def _check_inter_weights_format(levels: List["FCLCLevel"], source: str) -> None:
    """Raise RuntimeError if any inter_weights entry is the old 2-tuple format."""
    for lv in levels:
        for wlist in (lv.inter_weights or {}).values():
            if wlist and len(wlist[0]) == 2:
                raise RuntimeError(
                    f"Stage 2 cache '{source}' uses the old 2-tuple inter_weights "
                    f"format (chart_id, w̃). The new format requires 4-tuple "
                    f"(chart_id, w̃, mean_nn_dist, mean_normal_dev). "
                    f"Please rebuild Stage 2 cache with preprocess_stage2.py "
                    f"or use the unified script preprocess_stage2_to_packed.py."
                )


def load_fclc_entry(path: str, mol_id: str) -> List[FCLCLevel]:
    """Load one molecule's FCLC levels from either a bundle or a legacy dict pkl."""
    if is_fclc_bundle_path(path):
        with zipfile.ZipFile(path, "r") as zf:
            member = fclc_bundle_member(mol_id)
            with zf.open(member, "r") as f:
                levels = pickle.load(f)
        _check_inter_weights_format(levels, f"{path}::{mol_id}")
        return levels

    with open(path, "rb") as f:
        data: dict = pickle.load(f)
    levels = data[mol_id]
    _check_inter_weights_format(levels, f"{path}::{mol_id}")
    return levels


def list_fclc_bundle_ids(path: str) -> List[str]:
    """List molecule ids from a bundle or legacy dict pkl."""
    if is_fclc_bundle_path(path):
        with zipfile.ZipFile(path, "r") as zf:
            if "meta/mol_ids.pkl" in zf.namelist():
                with zf.open("meta/mol_ids.pkl", "r") as f:
                    return pickle.load(f)
            ids = []
            for name in zf.namelist():
                if name.startswith("molecules/") and name.endswith(".pkl"):
                    ids.append(name[len("molecules/"):-4])
            return ids

    with open(path, "rb") as f:
        data: dict = pickle.load(f)
    return list(data.keys())


def load_fclc_levels(path: str) -> List[FCLCLevel]:
    with open(path, "rb") as f:
        levels = pickle.load(f)
    _check_inter_weights_format(levels, path)
    return levels


def save_fclc_levels(path: str, levels: List[FCLCLevel]) -> None:
    with open(path, "wb") as f:
        pickle.dump(levels, f, protocol=pickle.HIGHEST_PROTOCOL)
