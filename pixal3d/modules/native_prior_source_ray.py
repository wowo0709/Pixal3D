"""Retarget source rays while preserving the pretrained native MV4 carrier.

The source feature being subtracted is present in this carrier. This differs
from the historical target-only carrier, where subtracting it was invalid.
Missing matching evidence leaves the native multiview prior unchanged.
"""
import torch


def source_ray_delta(numerator, semantic_mass, original_source):
    if (numerator.ndim != 4 or numerator.shape[1:] != (3, 1024, 1024)
            or semantic_mass.shape != numerator.shape[:-1] + (1,)
            or original_source.shape != numerator.shape):
        raise ValueError('native SS requires three source32patch maps and scalar ray mass')
    # Subtract before projection, preserving exact zero when E == original S
    # and R == 0, including arbitrary nonzero use/semantic weights.
    return numerator - semantic_mass * original_source.float()


def project_native_prior_source_delta(grid, numerator, semantic_mass, original_source, camera):
    """sum_v Proj_v[u*g*(E-S_v)+u*R]/4; no target/visibility write mask."""
    from ..trainers.flow_matching.mixins.image_conditioned_proj import compute_multiview_projection_matrices
    delta = source_ray_delta(numerator, semantic_mass, original_source)
    matrices, _ = compute_multiview_projection_matrices(camera['transform_matrix'],
        camera['distance'], grid.front_view_transform_matrix)
    projected = None
    for source in range(3):
        view = source + 1
        value = grid(delta[:, source].float().reshape(delta.shape[0], 32, 32, 1024),
            camera['camera_angle_x'][:, view], camera['distance'][:, view],
            camera['mesh_scale'], matrices[:, view]).float()
        projected = value if projected is None else projected + value
    return projected / 4
