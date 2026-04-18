from .cube_parser import CubeData, get_grid_coords, parse_cube
from .clustering import cluster_pointcloud, extract_representative_points
from .dataset import EDBenchDataset, EDBenchPKLDataset, collate_fn
from .stage3_local import (
    Stage3Sample,
    Stage3TensorBatch,
    build_stage3_sample,
    check_chart_normal_alignment,
    check_chart_plane_residual,
    check_chart_vector_projection,
    check_membership_weights,
    check_overlap_jaccard,
    collate_stage3_samples,
    list_stage3_bundle_ids,
    load_stage3_entry,
    load_stage3_sample,
    save_stage3_bundle_entry,
    save_stage3_sample,
    stage3_cache_path,
    summarize_stage3_sample,
    validate_stage3_sample,
)
from .stage3_packed import (
    Stage3PackedDataset,
    collate_stage3_packed_samples,
    is_stage3_packed_dir,
    list_stage3_packed_ids,
    load_stage3_packed_meta,
    load_stage3_packed_sample,
    pack_stage3_cache,
    stage3_packed_index_path,
    stage3_packed_meta_path,
)

__all__ = [
    "parse_cube",
    "get_grid_coords",
    "CubeData",
    "cluster_pointcloud",
    "extract_representative_points",
    "EDBenchPKLDataset",
    "EDBenchDataset",
    "collate_fn",
    "Stage3Sample",
    "Stage3TensorBatch",
    "build_stage3_sample",
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
    "stage3_packed_meta_path",
    "stage3_packed_index_path",
    "is_stage3_packed_dir",
    "list_stage3_packed_ids",
    "load_stage3_packed_meta",
    "pack_stage3_cache",
    "Stage3PackedDataset",
    "load_stage3_packed_sample",
    "collate_stage3_packed_samples",
]

try:
    from .fclc import (
        FCLCChart,
        FCLCLevel,
        build_fclc_levels,
        fclc_cache_path,
        list_fclc_bundle_ids,
        load_fclc_entry,
        load_fclc_levels,
        save_fclc_levels,
    )
except Exception:
    pass
else:
    __all__.extend(
        [
            "build_fclc_levels",
            "save_fclc_levels",
            "load_fclc_levels",
            "load_fclc_entry",
            "list_fclc_bundle_ids",
            "fclc_cache_path",
            "FCLCChart",
            "FCLCLevel",
        ]
    )

try:
    from .manifold import (
        DensityGrid,
        ManifoldComponent,
        ManifoldLevel,
        extract_manifold_levels,
        load_manifold_levels,
        manifold_cache_path,
        save_manifold_levels,
    )
except ModuleNotFoundError:
    # Stage 3 cache loading and smoke checks should still work even when the
    # Stage 1 extraction dependency stack (e.g. skimage) is absent.
    pass
else:
    __all__.extend(
        [
            "extract_manifold_levels",
            "save_manifold_levels",
            "load_manifold_levels",
            "manifold_cache_path",
            "DensityGrid",
            "ManifoldComponent",
            "ManifoldLevel",
        ]
    )
