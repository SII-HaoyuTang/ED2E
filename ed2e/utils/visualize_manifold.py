
"""
Isodensity manifold visualizer.

Supports two backends:
  - plotly   (default) — interactive 3-D HTML, can be saved and opened in any browser
  - matplotlib         — static PNG / screen window, no extra heavy dependencies

Usage (CLI)
-----------
# Visualize one molecule from the merged pkl, open in browser
python ed2e/utils/visualize_manifold.py \
    --manifold-pkl data/ed_energy_5w/cache_manifold/all_nl4_s0.50.pkl \
    --mol-id <mol_id> \
    --atom-pkl data/ed_energy_5w/processed/mol_EDthresh0.05_data.pkl

# Save to HTML instead of opening
python ed2e/utils/visualize_manifold.py ... --save output.html

# Use matplotlib and save PNG
python ed2e/utils/visualize_manifold.py ... --backend matplotlib --save output.png

# Load from per-molecule cache file (before merging)
python ed2e/utils/visualize_manifold.py \
    --manifold-pkl data/ed_energy_5w/cache_manifold/<mol_id>_nl4_s0.50.pkl \
    --mol-id <mol_id>

Programmatic usage
------------------
    from ed2e.utils.visualize_manifold import visualize_manifolds
    visualize_manifolds(levels, atom_coords=coords, backend="plotly")
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

from ed2e.data.manifold import ManifoldLevel  # noqa: E402
from ed2e.data.manifold import _patch_legacy_pickle_modules  # noqa: E402

# Colour palette: inner (level 0) → outer (level K-1)
# Warm → cool so inner shells look "hotter"
_LEVEL_COLOURS = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4",
                  "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]

# Atomic number → element symbol (subset used in organic chemistry)
_ELEM = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F", 15: "P",
         16: "S", 17: "Cl", 35: "Br", 53: "I"}
_ATOM_COLOUR = {"H": "#ffffff", "C": "#404040", "N": "#3050f8",
                "O": "#ff0d0d", "F": "#90e050", "P": "#ff8000",
                "S": "#ffff30", "Cl": "#1ff01f", "Br": "#a62929",
                "I": "#940094"}
_ATOM_RADIUS = {"H": 3, "C": 6, "N": 6, "O": 6, "F": 5,
                "P": 8, "S": 8, "Cl": 7, "Br": 8, "I": 9}


# ---------------------------------------------------------------------------
# Plotly backend
# ---------------------------------------------------------------------------

def _visualize_plotly(
    levels:      List[ManifoldLevel],
    atom_coords: Optional[np.ndarray] = None,
    atom_types:  Optional[np.ndarray] = None,
    title:       str = "Isodensity Manifolds",
    save_path:   Optional[str] = None,
    show:        bool = True,
) -> None:
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError("plotly is required for the plotly backend.  "
                          "Install it with: pip install plotly")

    fig = go.Figure()
    n_levels = len(levels)

    for lv in levels:
        colour  = _LEVEL_COLOURS[lv.level_id % len(_LEVEL_COLOURS)]
        # Inner shells more opaque so the layered structure is visible
        opacity = max(0.15, 0.55 - 0.1 * lv.level_id)
        label   = f"Level {lv.level_id}  ρ={lv.threshold:.4f} e/Bohr³"

        for ci, comp in enumerate(lv.components):
            v = comp.verts   # (V, 3)
            f = comp.faces   # (F, 3)

            # Vertex normals for smooth shading
            nx, ny, nz = comp.normals[:, 0], comp.normals[:, 1], comp.normals[:, 2]

            fig.add_trace(go.Mesh3d(
                x=v[:, 0], y=v[:, 1], z=v[:, 2],
                i=f[:, 0], j=f[:, 1], k=f[:, 2],
                color=colour,
                opacity=opacity,
                flatshading=False,
                # Pass normals for Phong-like shading
                vertexcolor=None,
                name=label if ci == 0 else None,
                showlegend=(ci == 0),
                hoverinfo="skip",
                lighting=dict(ambient=0.4, diffuse=0.8, specular=0.3,
                              roughness=0.5, fresnel=0.2),
                lightposition=dict(x=1000, y=1000, z=1000),
            ))

    # Atomic positions
    if atom_coords is not None:
        types = atom_types if atom_types is not None else np.ones(len(atom_coords), dtype=int)
        for z in np.unique(types):
            mask = types == z
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
                hovertemplate=f"{elem}<br>(%{{x:.2f}}, %{{y:.2f}}, %{{z:.2f}}) Bohr<extra></extra>",
            ))

    fig.update_layout(
        title=dict(text=title, x=0.5),
        scene=dict(
            aspectmode="data",
            xaxis_title="x (Bohr)",
            yaxis_title="y (Bohr)",
            zaxis_title="z (Bohr)",
            bgcolor="rgb(20,20,20)",
        ),
        paper_bgcolor="rgb(30,30,30)",
        font_color="white",
        legend=dict(bgcolor="rgba(0,0,0,0.5)", bordercolor="gray", borderwidth=1),
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
    levels:      List[ManifoldLevel],
    atom_coords: Optional[np.ndarray] = None,
    atom_types:  Optional[np.ndarray] = None,
    title:       str = "Isodensity Manifolds",
    save_path:   Optional[str] = None,
    show:        bool = True,
) -> None:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    _MPL_COLOURS = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4"]

    fig = plt.figure(figsize=(10, 8))
    ax  = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#141414")
    fig.patch.set_facecolor("#1e1e1e")

    all_verts: List[np.ndarray] = []

    for lv in levels:
        colour  = _MPL_COLOURS[lv.level_id % len(_MPL_COLOURS)]
        alpha   = max(0.10, 0.45 - 0.08 * lv.level_id)

        for comp in lv.components:
            tris = comp.verts[comp.faces]   # (F, 3, 3)
            poly = Poly3DCollection(
                tris, alpha=alpha,
                facecolor=colour, edgecolor="none",
                label=f"Level {lv.level_id} ρ={lv.threshold:.4f}",
            )
            ax.add_collection3d(poly)
            all_verts.append(comp.verts)

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
        all_verts.append(atom_coords)

    # Equal-aspect bounding box
    if all_verts:
        pts = np.concatenate(all_verts, axis=0)
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
    ax.set_title(title, color="white")

    # Deduplicate legend entries
    handles, labels = ax.get_legend_handles_labels()
    seen: dict = {}
    for h, l in zip(handles, labels):
        if l not in seen:
            seen[l] = h
    ax.legend(seen.values(), seen.keys(),
              facecolor="#333333", labelcolor="white", loc="upper left")

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

def visualize_manifolds(
    levels:      List[ManifoldLevel],
    atom_coords: Optional[np.ndarray] = None,
    atom_types:  Optional[np.ndarray] = None,
    title:       str = "Isodensity Manifolds",
    backend:     str = "plotly",
    save_path:   Optional[str] = None,
    show:        bool = True,
) -> None:
    """Visualize isodensity manifold levels for one molecule.

    Args:
        levels:      List[ManifoldLevel] from extract_manifold_levels().
        atom_coords: (N, 3) float32 — atomic positions in Bohr (optional).
        atom_types:  (N,) int       — atomic numbers (optional, for colouring).
        title:       Plot title.
        backend:     "plotly" (interactive HTML) or "matplotlib" (static image).
        save_path:   If given, save to this path (.html for plotly, .png for matplotlib).
        show:        If True, open the viewer / display the figure.
    """
    if backend == "plotly":
        _visualize_plotly(levels, atom_coords, atom_types, title, save_path, show)
    elif backend == "matplotlib":
        _visualize_matplotlib(levels, atom_coords, atom_types, title, save_path, show)
    else:
        raise ValueError(f"Unknown backend '{backend}'. Choose 'plotly' or 'matplotlib'.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Visualise Stage-1 isodensity manifolds.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--manifold-pkl", required=True,
                   help="Path to the manifold pickle file — either the merged "
                        "all_nl*.pkl dict or a single-molecule .pkl list.")
    p.add_argument("--mol-id", default=None,
                   help="Molecule ID key inside the merged pkl.  "
                        "Not needed for single-molecule pkl files.")
    p.add_argument("--atom-pkl", default=None,
                   help="Path to the original mol_EDthresh0.05_data.pkl to "
                        "overlay atomic positions.")
    p.add_argument("--levels", type=int, nargs="+", default=None,
                   help="Subset of level_ids to render, e.g. --levels 0 1.  "
                        "Default: all levels.")
    p.add_argument("--backend", choices=["plotly", "matplotlib"], default="plotly",
                   help="Visualisation backend.")
    p.add_argument("--save", default=None,
                   help="Save path (.html for plotly, .png / .pdf for matplotlib).")
    p.add_argument("--no-show", action="store_true",
                   help="Do not open the viewer (useful with --save).")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # ---- Load manifold levels -----------------------------------------------
    _patch_legacy_pickle_modules()
    with open(args.manifold_pkl, "rb") as f:
        data = pickle.load(f)

    if isinstance(data, dict):
        # Merged pkl: {mol_id: List[ManifoldLevel]}
        if args.mol_id is None:
            keys = list(data.keys())
            print(f"--mol-id not specified.  Available IDs (first 5): {keys[:5]}")
            print(f"Total molecules in file: {len(keys)}")
            sys.exit(1)
        levels: List[ManifoldLevel] = data[args.mol_id]
    else:
        # Single-molecule pkl: List[ManifoldLevel]
        levels = data

    # Filter levels if requested
    if args.levels is not None:
        levels = [lv for lv in levels if lv.level_id in args.levels]

    # ---- Load atomic positions (optional) -----------------------------------
    atom_coords = atom_types = None
    if args.atom_pkl and args.mol_id:
        with open(args.atom_pkl, "rb") as f:
            raw = pickle.load(f)
        if args.mol_id in raw:
            mol = raw[args.mol_id]["mol"]
            atom_coords = mol["coords"]    # (N, 3) float32
            atom_types  = mol["x"]         # (N,) int

    # ---- Statistics ---------------------------------------------------------
    print(f"Molecule : {args.mol_id or '(single-molecule file)'}")
    for lv in levels:
        n_verts = sum(len(c.verts) for c in lv.components)
        n_faces = sum(len(c.faces) for c in lv.components)
        print(f"  Level {lv.level_id}  ρ={lv.threshold:.4f}  "
              f"components={len(lv.components)}  verts={n_verts}  faces={n_faces}")

    title = f"Molecule {args.mol_id}" if args.mol_id else "Isodensity Manifolds"

    # ---- Visualize ----------------------------------------------------------
    visualize_manifolds(
        levels,
        atom_coords=atom_coords,
        atom_types=atom_types,
        title=title,
        backend=args.backend,
        save_path=args.save,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
