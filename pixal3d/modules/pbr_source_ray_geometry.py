"""Sparse PBR source-ray geometry using the native camera and sampler rules.

The packed coordinate order belongs to the native sparse consumer. Query
pooling uses one group per source-camera patch; delivery still samples the
correction at every native sparse coordinate, without a visibility write gate.
This module does not change any running SS or historical PBR implementation.
"""
import torch

from .ray_group_query import RayQueries


def _validated_geometry(grid, coords, camera):
    required = {'transform_matrix', 'distance', 'camera_angle_x', 'mesh_scale'}
    if not required <= camera.keys():
        raise ValueError('explicit native four-view camera fields required')
    transform = camera['transform_matrix']
    batch = len(transform)
    resolution = grid.grid_resolution
    if (transform.shape != (batch, 4, 4, 4) or batch < 1
            or coords.ndim != 2 or coords.shape[1] != 4
            or coords.dtype not in (torch.int32, torch.int64)
            or coords.device != grid.grid_points.device
            or grid.grid_points.shape != (resolution ** 3, 3)
            or any(camera[key].device != coords.device for key in required)
            or camera['distance'].shape != (batch, 4)
            or camera['camera_angle_x'].shape != (batch, 4)
            or camera['mesh_scale'].shape != (batch,)
            or any(not torch.isfinite(camera[key]).all() for key in required)
            or (camera['mesh_scale'] <= 0).any()
            or (camera['camera_angle_x'] <= 0).any()
            or (camera['camera_angle_x'] >= torch.pi).any()):
        raise ValueError('native grid, integer packed coordinates and finite aligned cameras required')
    if (len(coords) and ((coords[:, 0] < 0).any() or (coords[:, 0] >= batch).any()
            or (coords[:, 1:] < 0).any() or (coords[:, 1:] >= resolution).any())):
        raise ValueError('sparse coordinate outside its declared batch or native grid')
    from ..trainers.flow_matching.mixins.image_conditioned_proj import compute_multiview_projection_matrices
    matrices, _ = compute_multiview_projection_matrices(transform, camera['distance'],
        grid.front_view_transform_matrix)
    return batch, matrices


def _points(grid, coords, camera, batch):
    rows = (coords[:, 0] == batch).nonzero().flatten()
    xyz = coords[rows, 1:].long()
    resolution = grid.grid_resolution
    flat = (xyz[:, 0] * resolution + xyz[:, 1]) * resolution + xyz[:, 2]
    # Use the stored rotated native grid, never reconstruct an unrotated cube.
    points = grid.grid_points[flat][None] / camera['mesh_scale'][batch:batch+1, None, None] / 2
    return rows, points


def sparse_source_ray_membership(grid, coords, camera):
    """Return long[N,3] source-camera nearest32patch groups in packed order.

    Out-of-frustum query members are -1. This excludes them from query pooling,
    not from correction delivery, whose border sampling follows native ProjGrid.
    No surface depth, GT correspondence or visibility mask is consumed.
    """
    from ..trainers.flow_matching.mixins.image_conditioned_proj import project_points_to_image_batch
    batch_size, matrices = _validated_geometry(grid, coords, camera)
    result = torch.full((len(coords), 3), -1, dtype=torch.long, device=coords.device)
    for batch in range(batch_size):
        rows, points = _points(grid, coords, camera, batch)
        if not len(rows):
            continue
        for source in range(3):
            view = source + 1
            pixels, _, frustum = project_points_to_image_batch(points,
                matrices[batch:batch+1, view], camera['camera_angle_x'][batch:batch+1, view],
                grid.image_resolution)
            xy = torch.round((pixels + .5) * (32 / grid.image_resolution) - .5).long()[0]
            valid = frustum[0] & (xy >= 0).all(-1) & (xy < 32).all(-1)
            ids = xy[:, 1] * 32 + xy[:, 0]
            result[rows, source] = torch.where(valid, ids, -1)
    return result


def pool_sparse_source_ray_queries(features, ray_ids, batch_ids, *, batch_size):
    """Mean only real packed consumer rows; no padded zero observations."""
    if (features.ndim != 2 or not features.is_floating_point() or features.shape[1] < 1
            or ray_ids.shape != (len(features), 3) or ray_ids.dtype != torch.long
            or batch_ids.shape != (len(features),)
            or batch_ids.dtype not in (torch.int32, torch.int64)
            or features.device != ray_ids.device or features.device != batch_ids.device
            or isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1
            or not torch.isfinite(features).all()
            or ((ray_ids < -1) | (ray_ids >= 1024)).any()
            or ((batch_ids < 0) | (batch_ids >= batch_size)).any()):
        raise ValueError('finite packed features, native batch IDs and three32patch ray groups required')
    values, counts = [], []
    for source in range(3):
        valid = ray_ids[:, source] >= 0
        index = batch_ids[valid].long() * 1024 + ray_ids[valid, source]
        count = torch.zeros(batch_size * 1024, dtype=torch.long, device=features.device)
        count.index_add_(0, index, torch.ones_like(index))
        total = torch.zeros(batch_size * 1024, features.shape[1], dtype=torch.float32,
            device=features.device)
        total.index_add_(0, index, features[valid].float())
        values.append((total / count.clamp_min(1)[:, None]).reshape(batch_size, 1024, -1))
        counts.append(count.reshape(batch_size, 1024))
    count = torch.stack(counts, 1)
    return RayQueries(torch.stack(values, 1), count > 0, count)


def project_sparse_pbr_source_delta(grid, numerator, semantic_mass, original_source,
                                    coords, camera, *, channel_chunk=256):
    """Native bilinear sum_v Proj_v[u*g*(E-S_v)+u*R]/4 at packed rows.

    Both DINO1024 and NAF1024 are delivered. Channel chunks bound the temporary
    interpolation output, not the scientific microbatch or its accumulation.
    """
    from ..trainers.flow_matching.mixins.image_conditioned_proj import (
        project_points_to_image_batch, sample_features,
    )
    batch_size, matrices = _validated_geometry(grid, coords, camera)
    if (numerator.shape != (batch_size, 3, 1024, 2048)
            or semantic_mass.shape != (batch_size, 3, 1024, 1)
            or original_source.shape != numerator.shape
            or any(x.device != coords.device or not x.is_floating_point()
                for x in (numerator, semantic_mass, original_source))
            or any(not torch.isfinite(x).all() for x in (numerator, semantic_mass, original_source))
            or isinstance(channel_chunk, bool) or not isinstance(channel_chunk, int) or channel_chunk < 1):
        raise ValueError('actual PBR DINO1024+NAF1024 source maps, scalar mass and channel chunk required')
    # Own source is already present in the unchanged native MV4 carrier.
    delta = numerator.float() - semantic_mass.float() * original_source.float()
    result = delta.new_zeros(len(coords), 2048)
    for batch in range(batch_size):
        rows, points = _points(grid, coords, camera, batch)
        if not len(rows):
            continue
        projected = None
        for source in range(3):
            view = source + 1
            pixels, _, _ = project_points_to_image_batch(points,
                matrices[batch:batch+1, view], camera['camera_angle_x'][batch:batch+1, view],
                grid.image_resolution)
            normalized = (pixels + .5) / grid.image_resolution * 2 - 1
            maps = delta[batch, source].reshape(32, 32, 2048).permute(2, 0, 1)[None]
            parts = [sample_features(maps[:, start:start+channel_chunk], normalized)
                for start in range(0, 2048, channel_chunk)]
            value = torch.cat(parts, 1)[0].transpose(0, 1)
            projected = value if projected is None else projected + value
        result = result.index_copy(0, rows, projected / 4)
    # Retain finite zero gradients for a declared batch with no sparse rows.
    return result + (delta.sum() * 0 if not len(coords) else 0)
