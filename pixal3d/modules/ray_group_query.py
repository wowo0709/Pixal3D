"""One query per camera/patch ray, without changing native feature projection."""
from dataclasses import dataclass

import torch


@dataclass
class RayQueries:
    values: torch.Tensor  # [B,V,R,D]; exactly one vector per camera ray
    valid: torch.Tensor  # [B,V,R]; empty rays must not enter attention
    voxel_count: torch.Tensor  # [B,V,R]


def pool_ray_queries(features, ray_ids, *, num_rays):
    """Mean voxel inputs within each explicitly declared camera ray group.

    Feature provenance is supplied by the caller. This function does not read
    geometry GT, choose correspondence, or replace ProjGrid with a voxel splice.
    An index of -1 excludes a voxel; a zero-valued empty-ray slot is explicitly
    invalid rather than an observed zero feature.
    """
    if (features.ndim != 3 or not features.is_floating_point() or ray_ids.ndim != 3
            or ray_ids.shape[0] != features.shape[0] or ray_ids.shape[2] != features.shape[1]
            or ray_ids.dtype != torch.long or ray_ids.device != features.device
            or not isinstance(num_rays, int) or isinstance(num_rays, bool) or num_rays < 1
            or ray_ids.shape[1] < 1):
        raise ValueError('explicit floating[B,N,D],long[B,V,N],positive ray count required')
    if not torch.isfinite(features).all() or ((ray_ids < -1) | (ray_ids >= num_rays)).any():
        raise ValueError('finite features and ray IDs in[-1,num_rays) required')
    batch, _, width = features.shape
    values, counts = [], []
    offsets = torch.arange(batch, device=features.device)[:, None] * num_rays
    # Loop cameras to avoid materializing[B,V,N,D] for dense SS microbatches.
    for view in range(ray_ids.shape[1]):
        ids = ray_ids[:, view]
        valid = ids >= 0
        index = (offsets + ids)[valid]
        count = torch.zeros(batch * num_rays, dtype=torch.long, device=features.device)
        count.index_add_(0, index, torch.ones_like(index))
        total = torch.zeros(batch * num_rays, width, dtype=torch.float32, device=features.device)
        total.index_add_(0, index, features[valid].float())
        values.append((total / count.clamp_min(1)[:, None]).reshape(batch, num_rays, width))
        counts.append(count.reshape(batch, num_rays))
    count = torch.stack(counts, dim=1)
    return RayQueries(torch.stack(values, dim=1), count > 0, count)


def native_ray_membership(grid, *, transform_matrix, distance, camera_angle_x,
                          mesh_scale, views, patch_resolution=32):
    """Historical native camera/nearest-patch ray bins; no depth observation.

    Query bins follow target_proxy_ray_summary's nearest patch assignment.
    Native ProjGrid's bilinear feature sampling remains a separate operation.
    """
    from ..trainers.flow_matching.mixins.image_conditioned_proj import (
        compute_multiview_projection_matrices, project_points_to_image_batch,
    )
    if (not isinstance(views, (list, tuple)) or not views or len(set(views)) != len(views)
            or any(isinstance(v, bool) or not isinstance(v, int) for v in views)
            or any(v < 0 or v >= transform_matrix.shape[1] for v in views)
            or patch_resolution != 32):
        raise ValueError('distinct explicit camera indices and native32patch ray bins required')
    batch = transform_matrix.shape[0]
    if (camera_angle_x.shape != transform_matrix.shape[:2] or mesh_scale.shape != (batch,)
            or not torch.isfinite(mesh_scale).all() or (mesh_scale <= 0).any()):
        raise ValueError('finite positive mesh scale and camera vectors required')
    matrices, _ = compute_multiview_projection_matrices(transform_matrix, distance, grid.front_view_transform_matrix)
    points = grid.grid_points.expand(batch, -1, -1) / mesh_scale[:, None, None] / 2
    groups = []
    for view in views:
        pixels, _, in_frustum = project_points_to_image_batch(
            points, matrices[:, view], camera_angle_x[:, view], grid.image_resolution)
        xy = torch.round((pixels + .5) * (patch_resolution / grid.image_resolution) - .5).long()
        x, y = xy[..., 0], xy[..., 1]
        valid = in_frustum & (x >= 0) & (x < patch_resolution) & (y >= 0) & (y < patch_resolution)
        groups.append(torch.where(valid, y * patch_resolution + x, torch.full_like(x, -1)))
    return torch.stack(groups, dim=1)
