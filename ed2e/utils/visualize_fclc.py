"""
FCLC atlas visualizer.

Renders the FCLC partition of an isodensity manifold surface, showing each
chart in a distinct colour.  Supports two backends:
  - plotly   (default) — interactive 3-D HTML
  - matplotlib         — static PNG / screen window

Usage (CLI)
-----------
# Visualize one molecule, level 0, open in browser
python ed2e/utils/visualize_fclc.py \\
    --manifold-pkl data/ed_energy_5w/cache_manifold/all_nl4_s0.50.pkl \\
    --fclc-pkl     data/ed_energy_5w/cache_fclc/all_fclc_tr1.00_t21.50.zip \\
    --mol-id 308 --level 0

# Save to HTML
python ed2e/utils/visualize_fclc.py ... --save output_fclc.html

# matplotlib PNG
python ed2e/utils/visualize_fclc.py ... --backend matplotlib --save output_fclc.png

Programmatic usage
------------------
    from ed2e.utils.visualize_fclc import visualize_fclc_atlas
    visualize_fclc_atlas(manifold_level, fclc_level, backend="plotly")
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
from typing import List, Optional

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))

from ed2e.data.manifold import ManifoldLevel, _patch_legacy_pickle_modules  # noqa: E402
from ed2e.data.fclc import (                                                 # noqa: E402
    FCLCLevel, load_fclc_entry, list_fclc_bundle_ids,
)

# 20-colour qualitative palette (repeated cyclically for large atlases)
_CHART_PALETTE = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#fabed4",
    "#469990", "#dcbeff", "#9A6324", "#fffac8", "#800000",
    "#aaffc3", "#808000", "#ffd8b1", "#000075", "#a9a9a9",
]

_ELEM = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F",
         15: "P", 16: "S", 17: "Cl", 35: "Br", 53: "I"}
_ATOM_COLOUR = {
    "H": "#ffffff", "C": "#404040", "N": "#3050f8",
    "O": "#ff0d0d", "F": "#90e050", "P": "#ff8000",
    "S": "#ffff30", "Cl": "#1ff01f", "Br": "#a62929", "I": "#940094",
}
_ATOM_RADIUS = {
    "H": 3, "C": 6, "N": 6, "O": 6, "F": 5,
    "P": 8, "S": 8, "Cl": 7, "Br": 8, "I": 9,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_chart_face_sets(
    ml: ManifoldLevel,
    fl: FCLCLevel,
) -> List[dict]:
    """Return a list of dicts {verts, faces, colour, label, is_overlap} for plotting.

    For each chart in fl, we extract the triangular faces that belong
    exclusively to that chart and those shared with another chart
    (overlap region), coloured differently.
    """
    # Build a map: global_vert_idx → list of chart_ids it belongs to
    # We work per ManifoldComponent
    result = []

    for comp in ml.components:
        V = len(comp.verts)
        vert_chart_count = np.zeros(V, dtype=np.int32)
        vert_chart_first = np.full(V, -1, dtype=np.int32)

        # Collect charts for this component
        comp_charts = [ch for ch in fl.charts if ch.component_id == comp.component_id]
        if not comp_charts:
            continue

        for ch in comp_charts:
            vert_chart_count[ch.vert_indices] += 1
            for vi in ch.vert_indices:
                if vert_chart_first[vi] < 0:
                    vert_chart_first[vi] = ch.chart_id

        # For each chart, split faces into exclusive vs overlap
        for ch in comp_charts:
            vi_set = set(ch.vert_indices.tolist())
            colour = _CHART_PALETTE[ch.chart_id % len(_CHART_PALETTE)]

            # Face mask: all three vertices in this chart
            f = comp.faces
            mask = (
                np.isin(f[:, 0], ch.vert_indices) &
                np.isin(f[:, 1], ch.vert_indices) &
                np.isin(f[:, 2], ch.vert_indices)
            )
            if not mask.any():
                continue

            chart_faces = f[mask]
            # Classify faces: overlap if any vertex is in > 1 chart
            overlap_mask = (
                (vert_chart_count[chart_faces[:, 0]] > 1) |
                (vert_chart_count[chart_faces[:, 1]] > 1) |
                (vert_chart_count[chart_faces[:, 2]] > 1)
            )

            result.append({
                "verts": comp.verts,
                "faces": chart_faces[~overlap_mask],
                "colour": colour,
                "label": f"Chart {ch.chart_id}",
                "is_overlap": False,
                "chart_id": ch.chart_id,
            })
            if overlap_mask.any():
                result.append({
                    "verts": comp.verts,
                    "faces": chart_faces[overlap_mask],
                    "colour": colour,
                    "label": f"Chart {ch.chart_id} (overlap)",
                    "is_overlap": True,
                    "chart_id": ch.chart_id,
                })

    return result


# ---------------------------------------------------------------------------
# Plotly backend
# ---------------------------------------------------------------------------

def _visualize_plotly(
    ml:          ManifoldLevel,
    fl:          FCLCLevel,
    atom_coords: Optional[np.ndarray] = None,
    atom_types:  Optional[np.ndarray] = None,
    title:       str = "FCLC Atlas",
    save_path:   Optional[str] = None,
    show:        bool = True,
) -> None:
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError("plotly is required.  Install with: pip install plotly")

    fig = go.Figure()
    mesh_data = _build_chart_face_sets(ml, fl)

    seen_labels: set = set()
    for md in mesh_data:
        v = md["verts"]
        f = md["faces"]
        if len(f) == 0:
            continue
        opacity = 0.35 if md["is_overlap"] else 0.6
        label = md["label"]
        show_legend = label not in seen_labels
        seen_labels.add(label)

        fig.add_trace(go.Mesh3d(
            x=v[:, 0], y=v[:, 1], z=v[:, 2],
            i=f[:, 0], j=f[:, 1], k=f[:, 2],
            color=md["colour"],
            opacity=opacity,
            flatshading=True,
            name=label,
            showlegend=show_legend,
            hoverinfo="skip",
            lighting=dict(ambient=0.5, diffuse=0.8, specular=0.2,
                          roughness=0.5, fresnel=0.1),
            lightposition=dict(x=1000, y=1000, z=1000),
        ))

    # Chart centres as scatter markers
    if fl.charts:
        centers = np.array([ch.center for ch in fl.charts], dtype=np.float32)
        cids    = [ch.chart_id for ch in fl.charts]
        fig.add_trace(go.Scatter3d(
            x=centers[:, 0], y=centers[:, 1], z=centers[:, 2],
            mode="markers+text",
            marker=dict(size=4, color="white",
                        line=dict(color="black", width=0.5)),
            text=[str(c) for c in cids],
            textfont=dict(size=8, color="white"),
            name="Chart centres",
            hovertemplate="Chart %{text}<extra></extra>",
        ))

    # Atomic positions
    if atom_coords is not None:
        types = atom_types if atom_types is not None else np.ones(len(atom_coords), dtype=int)
        for z in np.unique(types):
            mask  = types == z
            elem   = _ELEM.get(int(z), "X")
            colour = _ATOM_COLOUR.get(elem, "#cccccc")
            size   = _ATOM_RADIUS.get(elem, 6)
            fig.add_trace(go.Scatter3d(
                x=atom_coords[mask, 0],
                y=atom_coords[mask, 1],
                z=atom_coords[mask, 2],
                mode="markers",
                marker=dict(size=size, color=colour,
                            line=dict(color="black", width=0.5)),
                name=elem,
                hovertemplate=(
                    f"{elem}<br>(%{{x:.2f}}, %{{y:.2f}}, %{{z:.2f}}) Bohr"
                    "<extra></extra>"),
            ))

    n_charts = len(fl.charts)
    fig.update_layout(
        title=dict(text=f"{title}  (Level {fl.level_id}, {n_charts} charts)", x=0.5),
        scene=dict(
            aspectmode="data",
            xaxis_title="x (Bohr)", yaxis_title="y (Bohr)", zaxis_title="z (Bohr)",
            bgcolor="rgb(20,20,20)",
        ),
        paper_bgcolor="rgb(30,30,30)",
        font_color="white",
        legend=dict(bgcolor="rgba(0,0,0,0.5)", bordercolor="gray",
                    borderwidth=1, itemsizing="constant"),
        margin=dict(l=0, r=0, t=40, b=0),
    )

    if save_path:
        fig.write_html(save_path, include_plotlyjs="cdn")
        print(f"Saved → {save_path}")
    if show:
        fig.show()


# ---------------------------------------------------------------------------
# Matplotlib backend
# ---------------------------------------------------------------------------

def _visualize_matplotlib(
    ml:          ManifoldLevel,
    fl:          FCLCLevel,
    atom_coords: Optional[np.ndarray] = None,
    atom_types:  Optional[np.ndarray] = None,
    title:       str = "FCLC Atlas",
    save_path:   Optional[str] = None,
    show:        bool = True,
) -> None:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(10, 8))
    ax  = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#141414")
    fig.patch.set_facecolor("#1e1e1e")

    mesh_data = _build_chart_face_sets(ml, fl)
    all_pts: List[np.ndarray] = []

    for md in mesh_data:
        v = md["verts"]
        f = md["faces"]
        if len(f) == 0:
            continue
        alpha = 0.25 if md["is_overlap"] else 0.5
        ec    = "white" if md["is_overlap"] else "none"
        lw    = 0.2    if md["is_overlap"] else 0.0
        tris  = v[f]   # (F, 3, 3)
        poly = Poly3DCollection(tris, alpha=alpha,
                                facecolor=md["colour"],
                                edgecolor=ec, linewidth=lw)
        ax.add_collection3d(poly)
        all_pts.append(v)

    # Chart centres
    if fl.charts:
        ctrs = np.array([ch.center for ch in fl.charts])
        ax.scatter(ctrs[:, 0], ctrs[:, 1], ctrs[:, 2],
                   c="white", s=20, edgecolors="black",
                   linewidths=0.5, zorder=10, label="centres")
        all_pts.append(ctrs)

    if atom_coords is not None:
        types = atom_types if atom_types is not None else np.ones(len(atom_coords), dtype=int)
        for z in np.unique(types):
            mask   = types == z
            elem   = _ELEM.get(int(z), "X")
            colour = _ATOM_COLOUR.get(elem, "#cccccc")
            ax.scatter(atom_coords[mask, 0], atom_coords[mask, 1],
                       atom_coords[mask, 2], c=colour, s=60,
                       edgecolors="white", linewidths=0.5,
                       zorder=10, label=elem)
        all_pts.append(atom_coords)

    if all_pts:
        pts = np.concatenate(all_pts, axis=0)
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        ctr = (lo + hi) / 2
        rad = (hi - lo).max() / 2 * 1.1
        ax.set_xlim(ctr[0] - rad, ctr[0] + rad)
        ax.set_ylim(ctr[1] - rad, ctr[1] + rad)
        ax.set_zlim(ctr[2] - rad, ctr[2] + rad)

    ax.set_xlabel("x (Bohr)", color="white")
    ax.set_ylabel("y (Bohr)", color="white")
    ax.set_zlabel("z (Bohr)", color="white")
    ax.tick_params(colors="white")
    n_charts = len(fl.charts)
    ax.set_title(f"{title}  (Level {fl.level_id}, {n_charts} charts)", color="white")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"Saved → {save_path}")
    if show:
        plt.show()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def visualize_fclc_atlas(
    manifold_level: ManifoldLevel,
    fclc_level:     FCLCLevel,
    atom_coords:    Optional[np.ndarray] = None,
    atom_types:     Optional[np.ndarray] = None,
    title:          str = "FCLC Atlas",
    backend:        str = "plotly",
    save_path:      Optional[str] = None,
    show:           bool = True,
) -> None:
    """Visualize FCLC atlas partition on an isodensity manifold level.

    Args:
        manifold_level: ManifoldLevel from Stage 1.
        fclc_level:     FCLCLevel from Stage 2 for the same density level.
        atom_coords:    (N, 3) float32 — atomic positions in Bohr (optional).
        atom_types:     (N,) int       — atomic numbers (optional).
        title:          Plot title.
        backend:        "plotly" or "matplotlib".
        save_path:      File path to save (.html for plotly, .png for matplotlib).
        show:           If True, open the viewer.
    """
    if backend == "plotly":
        _visualize_plotly(manifold_level, fclc_level, atom_coords, atom_types,
                          title, save_path, show)
    elif backend == "matplotlib":
        _visualize_matplotlib(manifold_level, fclc_level, atom_coords, atom_types,
                              title, save_path, show)
    else:
        raise ValueError(f"Unknown backend '{backend}'. Choose 'plotly' or 'matplotlib'.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Visualise Stage-2 FCLC atlas on an isodensity manifold.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--manifold-pkl", required=True,
                   help="Merged Stage-1 manifold pkl (all_nl*.pkl).")
    p.add_argument("--fclc-pkl", required=True,
                   help="Merged Stage-2 FCLC bundle/pkl (all_fclc_*).")
    p.add_argument("--mol-id", required=True,
                   help="Molecule ID key inside the merged pkls.")
    p.add_argument("--level", type=int, default=0,
                   help="Density level index to render (0 = innermost).")
    p.add_argument("--atom-pkl", default=None,
                   help="Original mol_EDthresh0.05_data.pkl for atom overlay.")
    p.add_argument("--backend", choices=["plotly", "matplotlib"], default="plotly")
    p.add_argument("--save", default=None,
                   help="Save path (.html for plotly, .png/.pdf for matplotlib).")
    p.add_argument("--no-show", action="store_true",
                   help="Do not open the viewer (useful with --save).")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # ---- Load manifold data --------------------------------------------------
    _patch_legacy_pickle_modules()
    print(f"Loading manifold PKL from {args.manifold_pkl} …")
    with open(args.manifold_pkl, "rb") as f:
        manifold_data: dict = pickle.load(f)
    if args.mol_id not in manifold_data:
        print(f"mol_id '{args.mol_id}' not found.  "
              f"Available (first 5): {list(manifold_data.keys())[:5]}")
        sys.exit(1)
    manifold_levels = manifold_data[args.mol_id]

    # ---- Load FCLC data ------------------------------------------------------
    print(f"Loading FCLC PKL from {args.fclc_pkl} …")
    try:
        fclc_levels = load_fclc_entry(args.fclc_pkl, args.mol_id)
    except KeyError:
        avail = list_fclc_bundle_ids(args.fclc_pkl)[:5]
        print(f"mol_id '{args.mol_id}' not found in FCLC data.  "
              f"Available (first 5): {avail}")
        sys.exit(1)

    # ---- Select level --------------------------------------------------------
    ml = next((l for l in manifold_levels if l.level_id == args.level), None)
    fl = next((l for l in fclc_levels     if l.level_id == args.level), None)
    if ml is None or fl is None:
        avail = [l.level_id for l in manifold_levels]
        print(f"Level {args.level} not found.  Available: {avail}")
        sys.exit(1)

    # ---- Print statistics ----------------------------------------------------
    print(f"Molecule  : {args.mol_id}")
    print(f"Level {ml.level_id}  ρ={ml.threshold:.4f}  "
          f"components={len(ml.components)}")
    for comp in ml.components:
        print(f"  Component {comp.component_id}: {len(comp.verts)} verts")
    print(f"FCLC charts: {len(fl.charts)}")
    sizes = [len(ch.vert_indices) for ch in fl.charts]
    if sizes:
        print(f"  Chart sizes — mean={np.mean(sizes):.1f}  "
              f"median={np.median(sizes):.0f}  "
              f"[{min(sizes)}, {max(sizes)}]")

    # ---- Atomic positions (optional) -----------------------------------------
    atom_coords = atom_types = None
    if args.atom_pkl:
        with open(args.atom_pkl, "rb") as f:
            raw = pickle.load(f)
        if args.mol_id in raw:
            mol = raw[args.mol_id]["mol"]
            atom_coords = mol["coords"]
            atom_types  = mol["x"]

    # ---- Visualise -----------------------------------------------------------
    title = f"Molecule {args.mol_id}"
    visualize_fclc_atlas(
        ml, fl,
        atom_coords=atom_coords,
        atom_types=atom_types,
        title=title,
        backend=args.backend,
        save_path=args.save,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
