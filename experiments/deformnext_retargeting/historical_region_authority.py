"""Original native SS support/cube-ray authority functions, copied by AST."""
from __future__ import annotations
from collections.abc import Mapping
from typing import Any
import numpy as np
from .historical_geometry import _apply_affine_points, target_anchor_affine
FREE, SURFACE, UNKNOWN = 0, 1, 2


def _surface_support(vertices: np.ndarray, faces: np.ndarray, *, resolution: int) -> np.ndarray:
    """Use the production native mesh-to-flexible-grid encoder input at 64³."""
    import torch
    import o_voxel

    indices, _dual, _intersected = o_voxel.convert.mesh_to_flexible_dual_grid(
        torch.from_numpy(np.asarray(vertices, np.float32)).cpu(),
        torch.from_numpy(np.asarray(faces, np.int64)).cpu(),
        grid_size=int(resolution), aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        face_weight=1.0, boundary_weight=0.2, regularization_weight=1e-2, timing=False,
    )
    result = indices.detach().cpu().numpy().astype(np.int16, copy=False)
    if not len(result):
        raise RuntimeError("native flexible-grid support is empty")
    if (result < 0).any() or (result >= int(resolution)).any():
        raise RuntimeError("native flexible-grid support exceeds resolution")
    return np.unique(result, axis=0)


def _camera_clip(vertices: np.ndarray, camera: Mapping[str, Any]) -> np.ndarray:
    if "world_to_clip" in camera:
        raise RuntimeError("locked Blender camera record required; world_to_clip is forbidden")
    c2w = np.asarray(camera.get("transform_matrix"), dtype=np.float64)
    try:
        fov_x = float(camera["camera_angle_x"])
        blender_w2c = np.linalg.inv(c2w)
    except (KeyError, TypeError, ValueError, np.linalg.LinAlgError) as error:
        raise RuntimeError("invalid camera clip conversion") from error
    if c2w.shape != (4, 4) or not np.isfinite(c2w).all() or not np.isfinite(fov_x) or not 0 < fov_x < np.pi:
        raise RuntimeError("invalid camera clip conversion")
    focal = 1.0 / np.tan(fov_x / 2.0)
    near, far = 0.01, 100.0
    # Match the established Blender-native nvdiffrast convention exactly:
    # direct inverse c2w, negative-Z forward, and OpenGL clip depth.
    projection = np.asarray([[focal, 0, 0, 0], [0, focal, 0, 0], [0, 0, (far + near) / (near - far), 2 * far * near / (near - far)], [0, 0, -1, 0]], dtype=np.float64)
    matrix = (projection @ blender_w2c).astype(np.float32)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise RuntimeError("invalid camera clip conversion")
    homogeneous = np.concatenate((vertices.astype(np.float32), np.ones((len(vertices), 1), np.float32)), axis=1)
    clip = homogeneous @ matrix.T
    if not np.isfinite(clip).all() or np.any(np.abs(clip[:, 3]) < 1e-12):
        raise RuntimeError("invalid camera clip conversion")
    return clip


def target_anchor_to_world_points(
    points: np.ndarray, frame: Mapping[str, Any], *, scale: float = 1.0,
) -> np.ndarray:
    """Map native target-view SS coordinates back to normalized world space."""

    value = float(scale)
    if not np.isfinite(value) or value <= 0:
        raise ValueError("target-anchor scale must be finite and positive")
    return _apply_affine_points(
        np.asarray(points, dtype=np.float64) / value,
        np.linalg.inv(target_anchor_affine(frame)),
    )


def voxel_grid_center_corner_points(resolution: int) -> tuple[np.ndarray, np.ndarray]:
    """Return canonical indices and center-plus-eight-corner samples per cell."""

    value = int(resolution)
    if value < 1:
        raise ValueError("resolution must be positive")
    indices = np.stack(
        np.meshgrid(np.arange(value), np.arange(value), np.arange(value), indexing="ij"),
        axis=-1,
    ).reshape(-1, 3).astype(np.int32)
    centers = (indices.astype(np.float64) + 0.5) / float(value) - 0.5
    half = 0.5 / float(value)
    offsets = np.asarray(
        [(x, y, z) for x in (-half, half) for y in (-half, half) for z in (-half, half)],
        dtype=np.float64,
    )
    samples = np.concatenate((centers[:, None, :], centers[:, None, :] + offsets[None]), axis=1)
    return indices, samples.astype(np.float32)


def classify_projected_voxel_cubes(
    sample_ndc: np.ndarray,
    *,
    target_support: np.ndarray,
    visible_target_support: np.ndarray,
    raster_depth: np.ndarray,
    raster_mask: np.ndarray,
    depth_tolerance: float,
) -> np.ndarray:
    """Conservatively classify physical voxel cubes against an observed z-buffer.

    ``sample_ndc`` contains the projected center and/or corners of each voxel.
    A non-target voxel is FREE only when every sample is in-frustum and either
    every sampled ray is background or the complete cube is in front of every
    sampled foreground depth.  Mixed silhouette-boundary cells stay UNKNOWN.
    Visible GT target-support cells are SURFACE by oracle construction, and no
    target-support cell may be labelled FREE.
    """

    samples = np.asarray(sample_ndc, dtype=np.float32)
    target = np.asarray(target_support, dtype=bool)
    visible = np.asarray(visible_target_support, dtype=bool)
    depth = np.asarray(raster_depth, dtype=np.float32)
    mask = np.asarray(raster_mask, dtype=bool)
    tolerance = float(depth_tolerance)
    if samples.ndim != 3 or samples.shape[2] != 3 or samples.shape[1] < 1:
        raise ValueError("sample_ndc must have shape [N,S,3]")
    if target.shape != (len(samples),) or visible.shape != target.shape:
        raise ValueError("target support masks must have shape [N]")
    if np.any(visible & ~target):
        raise ValueError("visible target support must be a target-support subset")
    if depth.ndim != 2 or mask.shape != depth.shape or not np.isfinite(depth).all():
        raise ValueError("raster depth/mask must share finite [H,W] shape")
    if not np.isfinite(samples).all() or not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("projected samples and tolerance must be finite")

    output = np.full(len(samples), UNKNOWN, dtype=np.uint8)
    inside = np.all((samples >= -1.0) & (samples <= 1.0), axis=2)
    complete = inside.all(axis=1)
    height, width = depth.shape
    x = np.rint((samples[..., 0] + 1.0) * 0.5 * max(width - 1, 0)).astype(np.int64)
    y = np.rint((samples[..., 1] + 1.0) * 0.5 * max(height - 1, 0)).astype(np.int64)
    x = np.clip(x, 0, max(width - 1, 0))
    y = np.clip(y, 0, max(height - 1, 0))
    hit = mask[y, x]
    sampled_depth = depth[y, x]

    all_background = (~hit).all(axis=1)
    all_foreground = hit.all(axis=1)
    wholly_before = np.all(samples[..., 2] < sampled_depth - tolerance, axis=1)
    free = complete & ~target & (all_background | (all_foreground & wholly_before))
    output[free] = FREE
    output[visible] = SURFACE
    return output


def pool_support(coords: np.ndarray, *, source_resolution: int, target_resolution: int) -> np.ndarray:
    """Max-pool binary sparse support using exact integer parent cells."""
    source = _coords(coords, name="source")
    if source_resolution < 1 or target_resolution < 1 or source_resolution % target_resolution:
        raise ValueError("support resolutions must have an integer pooling ratio")
    if len(source) and ((source < 0).any() or (source >= source_resolution).any()):
        raise ValueError("support coordinates exceed source resolution")
    return np.unique(source // (source_resolution // target_resolution), axis=0).astype(np.int16, copy=False)


def _coords(value: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.int32)
    if array.ndim != 2 or array.shape[1:] != (3,):
        raise ValueError(f"{name} coordinates must have shape [N, 3]")
    return array
