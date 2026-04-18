"""
Stage 1: Multi-Level Isodensity Manifold Extraction.

Converts a molecular electron density point cloud (from the EDBench PKL file)
into a list of ManifoldLevel objects, each representing one isodensity shell.

Pipeline per molecule:
    (coords, densities)
        → reconstruct_density_grid        — recover the original 3-D voxel grid
        → smooth_density_grid             — Gaussian pre-smoothing (optional)
        → select_thresholds_percentile    — choose K density levels
        → per level c_k:
            extract_isosurface_mesh       — Marching Cubes triangulated surface
            compute_density_derivatives_bspline  — ∇ρ, Δρ, ∂²_n ρ via B-spline
            compute_mesh_curvatures       — H, K via cotangent Laplacian
            assemble_point_features       — pack scalar/vector feature arrays
            find_mesh_components          — split into connected components
        → List[ManifoldLevel]

All spatial quantities are in Bohr atomic units.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates, spline_filter
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components as sp_connected_components


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DensityGrid:
    """3-D density grid reconstructed from a filtered PKL point cloud.

    Voxels that were absent from the PKL (density below the upstream filter
    threshold) are stored as zero.
    """
    density: np.ndarray  # (Nx, Ny, Nz) float32
    origin:  np.ndarray  # (3,) float32 — grid origin in Bohr
    spacing: np.ndarray  # (3,) float32 — voxel spacing in Bohr per axis


@dataclass
class ManifoldComponent:
    """One connected component of an isodensity manifold M_{k,ℓ}.

    Vertex features follow the notation from the ED2E proposal (§A.1.2):

        scalar_features[:, 0]  = ‖∇ρ‖
        scalar_features[:, 1]  = Δρ
        scalar_features[:, 2]  = H  (mean curvature)
        scalar_features[:, 3]  = K  (Gaussian curvature)
        scalar_features[:, 4]  = ∂²_n ρ  (2nd normal derivative)

        vector_features[:, 0, :]  = ∇_{M_k} ‖∇ρ‖
        vector_features[:, 1, :]  = ∇_{M_k} H
    """
    verts:           np.ndarray   # (V, 3) float32
    faces:           np.ndarray   # (F, 3) int32  — local vertex indices
    normals:         np.ndarray   # (V, 3) float32 — unit normals, toward ∇ρ > 0
    scalar_features: np.ndarray   # (V, 5) float32
    vector_features: np.ndarray   # (V, 2, 3) float32
    component_id:    int
    density_level:   float


@dataclass
class ManifoldLevel:
    """All connected components of one isodensity manifold M_k."""
    level_id:   int
    threshold:  float
    components: List[ManifoldComponent] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Step 1: Reconstruct the 3-D density grid from the filtered point cloud
# ---------------------------------------------------------------------------

def _infer_axis_spacing(vals: np.ndarray, decimals: int = 3) -> float:
    """Infer the uniform grid spacing along one axis.

    PKL coords originate from a regular cube-file grid, so all values along
    one axis are integer multiples of the spacing.  We recover it as the
    minimum gap between consecutive unique rounded coordinate values.
    """
    unique = np.unique(np.round(vals, decimals))
    if len(unique) < 2:
        raise ValueError(
            f"Cannot infer grid spacing: only {len(unique)} unique value(s) found."
        )
    diffs = np.diff(unique)
    diffs = diffs[diffs > 1e-6]
    if len(diffs) == 0:
        raise ValueError("All consecutive unique coordinate values are too close.")
    return float(np.min(diffs))


def reconstruct_density_grid(
    coords:    np.ndarray,  # (M, 3) float32
    densities: np.ndarray,  # (M,)   float32
) -> DensityGrid:
    """Reconstruct a regular 3-D density grid from a pre-filtered point cloud.

    The PKL stores electron density grid points that survived a density
    threshold filter.  Because the source coordinates lie on a regular lattice
    we can recover the lattice parameters (origin, spacing, dimensions) from
    the coordinate data itself and re-place the known densities into a full
    voxel array (missing voxels are filled with zero).
    """
    coords    = np.asarray(coords,    dtype=np.float64)
    densities = np.asarray(densities, dtype=np.float32)

    origin  = coords.min(axis=0).astype(np.float32)
    spacing = np.array(
        [_infer_axis_spacing(coords[:, d]) for d in range(3)],
        dtype=np.float32,
    )
    shape = (
        np.round((coords.max(axis=0) - origin) / spacing).astype(int) + 1
    )

    grid_data = np.zeros(shape, dtype=np.float32)

    # Map each point to its nearest grid index
    idx = np.round((coords - origin) / spacing).astype(int)
    for d in range(3):
        idx[:, d] = np.clip(idx[:, d], 0, shape[d] - 1)

    grid_data[idx[:, 0], idx[:, 1], idx[:, 2]] = densities

    return DensityGrid(density=grid_data, origin=origin, spacing=spacing)


# ---------------------------------------------------------------------------
# Step 2: Optional Gaussian smoothing
# ---------------------------------------------------------------------------

def smooth_density_grid(
    grid:        DensityGrid,
    sigma_bohr:  float = 0.5,
) -> DensityGrid:
    """Apply isotropic Gaussian smoothing in Bohr units.

    The zero-filled voxels (absent from the PKL) can cause step artefacts on
    the Marching-Cubes isosurface.  A small Gaussian blur removes them while
    preserving the qualitative shape of the density shells.

    Set sigma_bohr = 0 to skip smoothing.
    """
    if sigma_bohr <= 0.0:
        return grid
    sigma_vox = sigma_bohr / grid.spacing           # per-axis sigma in voxels
    smoothed = gaussian_filter(
        grid.density.astype(np.float64), sigma=sigma_vox
    ).astype(np.float32)
    return DensityGrid(
        density=smoothed,
        origin=grid.origin.copy(),
        spacing=grid.spacing.copy(),
    )


# ---------------------------------------------------------------------------
# Step 3: Percentile-based threshold selection
# ---------------------------------------------------------------------------

def select_thresholds_percentile(
    densities:   np.ndarray,
    n_levels:    int = 4,
    percentiles: Optional[List[float]] = None,
) -> np.ndarray:
    """Select n_levels isodensity thresholds from the density distribution.

    Thresholds are returned in **descending** order (level_id 0 = innermost /
    highest-density shell, level_id K-1 = outermost / lowest-density shell).

    Args:
        densities:   1-D array of density values (already filtered, all > 0.05).
        n_levels:    Number of isodensity levels K.
        percentiles: Percentiles used as thresholds.  Defaults to K equally
                     spaced values in (0, 80], e.g. [20, 40, 60, 80] for K=4.
                     The corresponding density values are used as thresholds,
                     so level_id 0 corresponds to the 80th-percentile threshold
                     (high density, inner shell) and level_id K-1 to the
                     20th-percentile threshold (low density, outer shell).

    Returns:
        (n_levels,) float32 array in descending order.
    """
    if percentiles is None:
        step = 80.0 / n_levels
        percentiles = [step * (i + 1) for i in range(n_levels)]   # [20,40,60,80] for K=4

    thresholds = np.percentile(densities, percentiles).astype(np.float32)
    return thresholds[::-1].copy()   # descending: inner → outer


# ---------------------------------------------------------------------------
# Step 4: Marching Cubes isosurface extraction
# ---------------------------------------------------------------------------

def _fd_gradient_at_voxcoords(
    density:    np.ndarray,   # (Nx, Ny, Nz)
    spacing:    np.ndarray,   # (3,)
    vox_coords: np.ndarray,   # (N, 3) — in voxel units
    delta:      float = 0.5,
) -> np.ndarray:              # (N, 3) — gradient in Bohr⁻¹ e Bohr⁻³
    """Central-difference gradient of the density field at voxel coordinates."""
    grad = np.zeros_like(vox_coords)
    for d in range(3):
        c_p = vox_coords.copy(); c_p[:, d] += delta
        c_m = vox_coords.copy(); c_m[:, d] -= delta
        kw = dict(order=1, mode="constant", cval=0.0)
        v_p = map_coordinates(density, c_p.T, **kw)
        v_m = map_coordinates(density, c_m.T, **kw)
        grad[:, d] = (v_p - v_m) / (2.0 * delta * spacing[d])
    return grad.astype(np.float32)


def extract_isosurface_mesh(
    grid:      DensityGrid,
    threshold: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract a triangulated isosurface at the given density threshold.

    Uses `skimage.measure.marching_cubes`.  Vertex normals are derived from
    the density field gradient and oriented so they point toward ∇ρ > 0
    (i.e., inward toward the molecular core, consistent with n = ∇ρ/‖∇ρ‖).

    Returns:
        verts:   (V, 3) float32 — vertex positions in Bohr (absolute coords)
        faces:   (F, 3) int32   — triangle vertex indices
        normals: (V, 3) float32 — unit vertex normals
    """
    from skimage.measure import marching_cubes

    verts_mc, faces_mc, normals_mc, _ = marching_cubes(
        grid.density,
        level=float(threshold),
        spacing=tuple(grid.spacing.tolist()),
        allow_degenerate=False,
    )

    # skimage returns coordinates relative to grid origin when spacing is given
    verts   = verts_mc.astype(np.float32) + grid.origin
    faces   = faces_mc.astype(np.int32)
    normals = normals_mc.astype(np.float32)

    # Ensure normals align with ∇ρ (toward increasing density).
    # Sample gradient at a small subset of vertices and check orientation.
    if len(verts) > 0:
        n_sample = min(32, len(verts))
        rng = np.random.default_rng(0)
        idx_s = rng.choice(len(verts), n_sample, replace=False)
        vox_s = (verts[idx_s] - grid.origin) / grid.spacing   # (n_sample, 3)
        grad_s = _fd_gradient_at_voxcoords(grid.density, grid.spacing, vox_s)
        dot_mean = (normals[idx_s] * grad_s).sum(axis=1).mean()
        if dot_mean < 0.0:
            normals = -normals

    return verts, faces, normals


# ---------------------------------------------------------------------------
# Step 5: B-spline density derivatives at mesh vertices
# ---------------------------------------------------------------------------

def compute_density_derivatives_bspline(
    grid:              DensityGrid,
    verts:             np.ndarray,   # (V, 3) float32 — vertex coords in Bohr
    normals:           np.ndarray,   # (V, 3) float32 — unit normals
    h_normal:          float = 0.1,  # step in Bohr for ∂²_n ρ finite difference
    bspline_coeffs:    Optional[np.ndarray] = None,  # precomputed; avoids recomputing per level
) -> Dict[str, np.ndarray]:
    """Compute density derivatives at mesh vertices using cubic B-spline interpolation.

    B-splines give a smooth, analytically consistent interpolant of the
    discrete density grid, enabling accurate estimation of ∇ρ and Δρ at
    arbitrary sub-voxel positions.

    Args:
        grid:           The reconstructed (and optionally smoothed) density grid.
        verts:          Mesh vertex positions in Bohr.
        normals:        Mesh vertex unit normals (for ∂²_n ρ).
        h_normal:       Finite-difference step along the normal (Bohr).
        bspline_coeffs: Pre-filtered B-spline coefficient array
                        (output of `scipy.ndimage.spline_filter`).  If None,
                        it is computed here.  Pass a precomputed array when
                        processing multiple levels of the same molecule.

    Returns dict with keys:
        "grad_rho"  : (V, 3) float32 — ∇ρ in e Bohr⁻⁴
        "grad_norm" : (V,)   float32 — ‖∇ρ‖
        "laplacian" : (V,)   float32 — Δρ
        "d2n_rho"   : (V,)   float32 — ∂²_n ρ (2nd derivative along surface normal)
    """
    if bspline_coeffs is None:
        bspline_coeffs = spline_filter(grid.density.astype(np.float64), order=3)

    def _sample(pts_bohr: np.ndarray) -> np.ndarray:
        vox = (pts_bohr.astype(np.float64) - grid.origin) / grid.spacing
        return map_coordinates(
            bspline_coeffs, vox.T, order=3, prefilter=False,
            mode="constant", cval=0.0,
        )

    V = len(verts)
    rho_c = _sample(verts)               # (V,) center values

    grad_rho  = np.zeros((V, 3), dtype=np.float64)
    laplacian = np.zeros(V,      dtype=np.float64)

    for d in range(3):
        step = 0.5 * float(grid.spacing[d])   # half-voxel in Bohr
        dv   = np.zeros(3, dtype=np.float32)
        dv[d] = step
        v_p = _sample(verts + dv)
        v_m = _sample(verts - dv)
        grad_rho[:, d]  = (v_p - v_m) / (2.0 * step)
        laplacian       += (v_p - 2.0 * rho_c + v_m) / (step ** 2)

    grad_norm = np.linalg.norm(grad_rho, axis=1)

    # ∂²_n ρ: second derivative along the vertex normal
    v_p = _sample(verts + h_normal * normals)
    v_m = _sample(verts - h_normal * normals)
    d2n_rho = (v_p - 2.0 * rho_c + v_m) / (h_normal ** 2)

    return {
        "grad_rho":  grad_rho.astype(np.float32),
        "grad_norm": grad_norm.astype(np.float32),
        "laplacian": laplacian.astype(np.float32),
        "d2n_rho":   d2n_rho.astype(np.float32),
    }


# ---------------------------------------------------------------------------
# Step 6: Mesh curvatures via cotangent Laplacian
# ---------------------------------------------------------------------------

def compute_mesh_curvatures(
    verts:   np.ndarray,  # (V, 3)
    faces:   np.ndarray,  # (F, 3)
    normals: np.ndarray,  # (V, 3)
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate per-vertex mean and Gaussian curvatures.

    Method:
      * Mean curvature H via the cotangent-Laplacian operator (Desbrun et al.
        1999):  H_normal = (1/2A) Σ_j (cot α_ij + cot β_ij)(p_j − p_i),
        where α_ij, β_ij are the angles opposite edge (i,j) in the two
        incident triangles, and A is the mixed Voronoi area.
      * Gaussian curvature K via the discrete angle-defect formula:
        K(i) = (2π − Σ_T θ_i^T) / A_i.

    All computations are vectorised over faces to avoid Python loops.

    Returns:
        H : (V,) float32 — signed mean curvature (sign from ⟨H_normal, n⟩)
        K : (V,) float32 — Gaussian curvature
    """
    V  = len(verts)
    p  = verts.astype(np.float64)
    n  = normals.astype(np.float64)

    i_f = faces[:, 0]
    j_f = faces[:, 1]
    k_f = faces[:, 2]

    pi = p[i_f]; pj = p[j_f]; pk = p[k_f]

    # Edge vectors (from each vertex to the other two)
    eij = pj - pi;  eik = pk - pi
    eji = pi - pj;  ejk = pk - pj
    eki = pi - pk;  ekj = pj - pk

    # Cotangent of the interior angle at each vertex:
    # cot(θ) = cos(θ)/sin(θ) = dot(a,b) / ‖a×b‖
    def _cot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        dot      = (a * b).sum(axis=1)
        ssin     = np.linalg.norm(np.cross(a, b), axis=1)
        safe_sin = np.where(ssin > 1e-12, ssin, 1.0)   # avoid eval-branch divide-by-zero
        return np.where(ssin > 1e-12, dot / safe_sin, 0.0)

    cot_i = _cot(eij, eik)   # angle at vertex i  (opposite edge j–k)
    cot_j = _cot(eji, ejk)   # angle at vertex j  (opposite edge i–k)
    cot_k = _cot(eki, ekj)   # angle at vertex k  (opposite edge i–j)

    # Face areas
    cross_ijk = np.cross(eij, eik)
    area2 = np.linalg.norm(cross_ijk, axis=1)   # 2 × face area
    area  = 0.5 * area2                          # (F,)

    # Mixed area (uniform triangle-area / 3 approximation)
    mixed_area = np.zeros(V, dtype=np.float64)
    np.add.at(mixed_area, i_f, area / 3.0)
    np.add.at(mixed_area, j_f, area / 3.0)
    np.add.at(mixed_area, k_f, area / 3.0)

    # Cotangent-Laplacian accumulation:
    # Contribution to vertex i from edge (i,j) uses cot at opposite vertex k,
    # and from edge (i,k) uses cot at opposite vertex j.
    lap = np.zeros((V, 3), dtype=np.float64)
    np.add.at(lap, i_f,
              cot_k[:, None] * (pj - pi) + cot_j[:, None] * (pk - pi))
    np.add.at(lap, j_f,
              cot_k[:, None] * (pi - pj) + cot_i[:, None] * (pk - pj))
    np.add.at(lap, k_f,
              cot_j[:, None] * (pi - pk) + cot_i[:, None] * (pj - pk))

    safe_area = np.maximum(mixed_area, 1e-12)
    H_normal  = lap / (2.0 * safe_area[:, None])   # (V, 3)

    # Signed mean curvature: ‖H_normal‖/2, sign from vertex normal
    H_mag = np.linalg.norm(H_normal, axis=1)
    sign  = np.sign((H_normal * n).sum(axis=1))
    H     = (sign * H_mag * 0.5).astype(np.float32)

    # Gaussian curvature: angle defect
    def _safe_angle(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        la = np.linalg.norm(a, axis=1, keepdims=True)
        lb = np.linalg.norm(b, axis=1, keepdims=True)
        cos_theta = (a * b).sum(axis=1) / (la[:, 0] * lb[:, 0] + 1e-12)
        return np.arccos(np.clip(cos_theta, -1.0, 1.0))

    angle_sum = np.zeros(V, dtype=np.float64)
    np.add.at(angle_sum, i_f, _safe_angle(eij, eik))
    np.add.at(angle_sum, j_f, _safe_angle(eji, ejk))
    np.add.at(angle_sum, k_f, _safe_angle(eki, ekj))

    K = ((2.0 * np.pi - angle_sum) / safe_area).astype(np.float32)

    return H, K


# ---------------------------------------------------------------------------
# Step 7: Tangential gradient
# ---------------------------------------------------------------------------

def compute_tangential_gradient(
    field_vals: np.ndarray,  # (V,) float32 — scalar field on vertices
    verts:      np.ndarray,  # (V, 3)
    faces:      np.ndarray,  # (F, 3)
    normals:    np.ndarray,  # (V, 3)
) -> np.ndarray:             # (V, 3) float32 — ∇_M f
    """Estimate the in-surface (tangential) gradient of a scalar field.

    For each face the gradient is solved from the 2×2 system
        ∇f · e₁ = f_j − f_i,   ∇f · e₂ = f_k − f_i
    where e₁ = p_j − p_i, e₂ = p_k − p_i.  Face gradients are then
    averaged to vertices weighted by face area, and projected onto the
    vertex tangent plane.
    """
    V = len(verts)
    p = verts.astype(np.float64)
    f = field_vals.astype(np.float64)

    i_f = faces[:, 0]; j_f = faces[:, 1]; k_f = faces[:, 2]
    pi = p[i_f]; pj = p[j_f]; pk = p[k_f]
    fi = f[i_f]; fj = f[j_f]; fk = f[k_f]

    e1 = pj - pi   # (F, 3)
    e2 = pk - pi

    E11 = (e1 * e1).sum(axis=1)  # (F,)
    E12 = (e1 * e2).sum(axis=1)
    E22 = (e2 * e2).sum(axis=1)
    df1 = fj - fi
    df2 = fk - fi

    det      = E11 * E22 - E12 * E12
    safe_det = np.where(np.abs(det) > 1e-20, det, 1.0)
    alpha    = (E22 * df1 - E12 * df2) / safe_det   # (F,)
    beta     = (E11 * df2 - E12 * df1) / safe_det

    grad_face = alpha[:, None] * e1 + beta[:, None] * e2   # (F, 3)

    # Face areas for weighting
    area = 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)   # (F,)

    vert_grad = np.zeros((V, 3), dtype=np.float64)
    vert_w    = np.zeros(V,      dtype=np.float64)
    np.add.at(vert_grad, i_f, area[:, None] * grad_face)
    np.add.at(vert_grad, j_f, area[:, None] * grad_face)
    np.add.at(vert_grad, k_f, area[:, None] * grad_face)
    np.add.at(vert_w,    i_f, area)
    np.add.at(vert_w,    j_f, area)
    np.add.at(vert_w,    k_f, area)

    g = vert_grad / np.maximum(vert_w, 1e-12)[:, None]

    # Project onto tangent plane
    n    = normals.astype(np.float64)
    dot  = (g * n).sum(axis=1, keepdims=True)
    tang = (g - dot * n).astype(np.float32)
    return tang


# ---------------------------------------------------------------------------
# Step 8: Assemble per-vertex feature arrays
# ---------------------------------------------------------------------------

def assemble_point_features(
    verts:      np.ndarray,
    faces:      np.ndarray,
    normals:    np.ndarray,
    deriv_dict: Dict[str, np.ndarray],
    H:          np.ndarray,
    K:          np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Pack all per-vertex quantities into feature arrays.

    Returns:
        scalar_features : (V, 5)    — [‖∇ρ‖, Δρ, H, K, ∂²_n ρ]
        vector_features : (V, 2, 3) — [∇_M ‖∇ρ‖, ∇_M H]
    """
    scalar_features = np.stack(
        [
            deriv_dict["grad_norm"],
            deriv_dict["laplacian"],
            H,
            K,
            deriv_dict["d2n_rho"],
        ],
        axis=1,
    ).astype(np.float32)

    tang_grad_gnorm = compute_tangential_gradient(
        deriv_dict["grad_norm"], verts, faces, normals
    )
    tang_grad_H = compute_tangential_gradient(H, verts, faces, normals)

    vector_features = np.stack(
        [tang_grad_gnorm, tang_grad_H], axis=1
    ).astype(np.float32)  # (V, 2, 3)

    return scalar_features, vector_features


# ---------------------------------------------------------------------------
# Step 9: Connected components of the mesh
# ---------------------------------------------------------------------------

def find_mesh_components(
    verts:    np.ndarray,  # (V, 3)
    faces:    np.ndarray,  # (F, 3)
    min_size: int = 10,
) -> List[Dict]:
    """Decompose the mesh into connected components via vertex-edge adjacency.

    Returns a list of dicts (one per component with ≥ min_size vertices), each
    containing:
        "vert_idx"    : (Vc,) int  — global vertex indices
        "local_faces" : (Fc, 3) int — face array with re-indexed local vertex ids
    """
    V = len(verts)
    if V == 0 or len(faces) == 0:
        return []

    # Build symmetric vertex-adjacency from all edges of all faces
    src  = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
    dst  = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])
    data = np.ones(len(src), dtype=np.float32)
    adj  = csr_matrix((data, (src, dst)), shape=(V, V))
    adj  = adj + adj.T

    n_comp, labels = sp_connected_components(adj, directed=False)

    components = []
    for c in range(n_comp):
        vert_idx = np.where(labels == c)[0]
        if len(vert_idx) < min_size:
            continue

        # Faces where all three vertices belong to this component
        vert_mask = labels == c
        face_mask = (
            vert_mask[faces[:, 0]]
            & vert_mask[faces[:, 1]]
            & vert_mask[faces[:, 2]]
        )
        comp_faces = faces[face_mask]

        # Re-index face vertex references to local (component) numbering
        remap = np.full(V, -1, dtype=np.int32)
        remap[vert_idx] = np.arange(len(vert_idx), dtype=np.int32)
        local_faces = remap[comp_faces]

        components.append({
            "vert_idx":    vert_idx,
            "local_faces": local_faces,
        })

    return components


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def extract_manifold_levels(
    coords:             np.ndarray,
    densities:          np.ndarray,
    n_levels:           int = 4,
    percentiles:        Optional[List[float]] = None,
    smooth_sigma:       float = 0.5,
    min_component_size: int = 10,
) -> List[ManifoldLevel]:
    """Convert an electron density point cloud to multi-level manifold representations.

    Full Stage-1 pipeline.  Input is a single molecule's density point cloud as
    stored in the EDBench PKL file; output is a list of ManifoldLevel objects
    ordered from innermost (highest density) to outermost (lowest density) shell.

    Args:
        coords:             (M, 3) float32 — density point positions in Bohr.
        densities:          (M,)   float32 — density values (all > 0.05 e/Bohr³).
        n_levels:           Number of isodensity layers K.
        percentiles:        Percentile values used as density thresholds
                            (see select_thresholds_percentile).  Defaults to
                            equally spaced in (0, 80].
        smooth_sigma:       Gaussian smoothing σ in Bohr before Marching Cubes.
                            Set to 0 to skip smoothing.
        min_component_size: Minimum vertex count to retain a mesh component.

    Returns:
        List[ManifoldLevel] of length n_levels, level_id 0 = innermost shell.
    """
    # 1 — Reconstruct 3-D density grid from the filtered point cloud
    grid = reconstruct_density_grid(coords, densities)

    # 2 — Optional Gaussian smoothing to suppress zero-fill artefacts
    grid = smooth_density_grid(grid, sigma_bohr=smooth_sigma)

    # 3 — Select density thresholds (descending: inner → outer)
    thresholds = select_thresholds_percentile(densities, n_levels, percentiles)

    # 4 — Precompute B-spline coefficients once; reuse across all levels
    bspline_coeffs = spline_filter(grid.density.astype(np.float64), order=3)

    levels: List[ManifoldLevel] = []

    for level_id, threshold in enumerate(thresholds):
        level = ManifoldLevel(level_id=level_id, threshold=float(threshold))

        # 5 — Marching Cubes
        try:
            verts, faces, normals = extract_isosurface_mesh(grid, threshold)
        except (ValueError, RuntimeError):
            levels.append(level)
            continue

        if len(verts) == 0 or len(faces) == 0:
            levels.append(level)
            continue

        # 6 — Density derivatives at mesh vertices via B-spline
        deriv = compute_density_derivatives_bspline(
            grid, verts, normals, bspline_coeffs=bspline_coeffs
        )

        # 7 — Curvatures
        H, K = compute_mesh_curvatures(verts, faces, normals)

        # 8 — Scalar and vector feature arrays
        scalar_feat, vector_feat = assemble_point_features(
            verts, faces, normals, deriv, H, K
        )

        # 9 — Connected components
        for comp_id, comp in enumerate(
            find_mesh_components(verts, faces, min_size=min_component_size)
        ):
            vi = comp["vert_idx"]
            level.components.append(ManifoldComponent(
                verts=verts[vi],
                faces=comp["local_faces"],
                normals=normals[vi],
                scalar_features=scalar_feat[vi],
                vector_features=vector_feat[vi],
                component_id=comp_id,
                density_level=float(threshold),
            ))

        levels.append(level)

    return levels


# ---------------------------------------------------------------------------
# Cache I/O helpers
# ---------------------------------------------------------------------------

def manifold_cache_path(
    cache_dir:    str,
    mol_id:       str,
    n_levels:     int,
    smooth_sigma: float,
) -> str:
    """Return the canonical cache path for a molecule's manifold levels.

    The hyperparameters (n_levels, smooth_sigma) are embedded in the filename
    so that changing them automatically invalidates existing caches.
    """
    import os
    tag = f"nl{n_levels}_s{smooth_sigma:.2f}"
    return os.path.join(cache_dir, f"{mol_id}_{tag}.pkl")


def save_manifold_levels(path: str, levels: List[ManifoldLevel]) -> None:
    """Pickle a List[ManifoldLevel] to *path*."""
    import os
    import pickle
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(levels, f, protocol=pickle.HIGHEST_PROTOCOL)


def _patch_legacy_pickle_modules() -> None:
    """Register 'data.manifold' in sys.modules so pickles created before the
    data/ → ed2e/data/ rename can still be deserialized."""
    import sys, types
    import ed2e.data.manifold as _this
    if "data.manifold" not in sys.modules:
        if "data" not in sys.modules:
            sys.modules["data"] = types.ModuleType("data")
        sys.modules["data.manifold"] = _this


def load_manifold_levels(path: str) -> List[ManifoldLevel]:
    """Load a cached List[ManifoldLevel] from *path*."""
    import pickle
    _patch_legacy_pickle_modules()
    with open(path, "rb") as f:
        return pickle.load(f)
