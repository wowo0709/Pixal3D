"""Exact historical GT surface-route geometry, isolated from runtime worktrees.

Teacher-only utilities: they never enter deployment matching caches.
"""
from __future__ import annotations
from collections.abc import Mapping
from typing import Any
import numpy as np

IMAGE_RESOLUTION = 512
PATCH_GRID = 32
VISIBILITY_TOLERANCE = 2e-3


def project_points(
    points: np.ndarray, camera_to_world: np.ndarray, fov_x: float,
    *, resolution: int = 512,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project Blender-aligned world points using the production camera contract."""
    points = np.asarray(points, dtype=np.float64)
    c2w = np.asarray(camera_to_world, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or c2w.shape != (4, 4):
        raise ValueError("points/camera must be [N,3] and [4,4]")
    homogeneous = np.concatenate((points, np.ones((len(points), 1))), axis=-1)
    camera = homogeneous @ np.linalg.inv(c2w).T
    depth = -camera[:, 2]
    focal = 1 / np.tan(float(fov_x) / 2)
    ndc_x = camera[:, 0] * focal / np.maximum(depth, 1e-12)
    ndc_y = camera[:, 1] * focal / np.maximum(depth, 1e-12)
    pixel = np.stack((
        (ndc_x + 1) * resolution / 2 - .5,
        (1 - ndc_y) * resolution / 2 - .5,
    ), axis=-1)
    valid = (
        (depth > 0)
        & (pixel[:, 0] >= 0) & (pixel[:, 0] < resolution)
        & (pixel[:, 1] >= 0) & (pixel[:, 1] < resolution)
    )
    return pixel.astype(np.float32), depth.astype(np.float32), valid


def barycentric_points(vertices: np.ndarray, faces: np.ndarray, face: np.ndarray, barycentric: np.ndarray) -> np.ndarray:
    return np.einsum("ni,nic->nc", barycentric, vertices[faces[face]]).astype(np.float32)


def patch_center_pixels(grid: int, *, image_resolution: int) -> np.ndarray:
    if grid < 1 or image_resolution < 1 or image_resolution % grid:
        raise ValueError("image_resolution must be a positive multiple of grid")
    stride = image_resolution / grid
    yy, xx = np.meshgrid(np.arange(grid), np.arange(grid), indexing="ij")
    return np.stack(
        ((xx + 0.5) * stride - 0.5, (yy + 0.5) * stride - 0.5), axis=-1,
    ).astype(np.float32)


def teacher_pair_mask(
    source_visible: np.ndarray, target_counterfactual_visible: np.ndarray
) -> np.ndarray:
    """Keep only exact surface pairs visible in both same-camera states."""

    source = np.asarray(source_visible, dtype=bool)
    target = np.asarray(target_counterfactual_visible, dtype=bool)
    if source.shape != target.shape or source.ndim != 1:
        raise ValueError("visibility masks must share one-dimensional shape")
    return source & target


def camera_rays_for_pixels(
    pixel: np.ndarray,
    camera_to_world: np.ndarray,
    fov_x: float,
    *,
    resolution: int,
) -> np.ndarray:
    """Return Open3D-format world rays through Blender pixel centers."""
    pixel = np.asarray(pixel, dtype=np.float32)
    if pixel.ndim != 2 or pixel.shape[1] != 2:
        raise ValueError("pixel must be [N,2] in (x,y) order")
    focal = np.float32(1.0 / np.tan(float(fov_x) / 2.0))
    ndc_x = (pixel[:, 0] + 0.5) * 2.0 / resolution - 1.0
    ndc_y = 1.0 - (pixel[:, 1] + 0.5) * 2.0 / resolution
    camera_direction = np.stack((
        ndc_x / focal,
        ndc_y / focal,
        -np.ones(len(pixel), dtype=np.float32),
    ), axis=-1).astype(np.float32)
    camera_to_world = np.asarray(camera_to_world, dtype=np.float32)
    direction = camera_direction @ camera_to_world[:3, :3].T
    origin = np.broadcast_to(camera_to_world[:3, 3], direction.shape).copy()
    return np.concatenate((origin, direction), axis=-1).astype(np.float32)


def raycast_mesh_pixels(
    vertices: np.ndarray,
    faces: np.ndarray,
    pixel: np.ndarray,
    camera_to_world: np.ndarray,
    fov_x: float,
    *,
    resolution: int,
) -> dict[str, np.ndarray]:
    """Intersect selected image rays and return exact face/bary coordinates."""
    try:
        import open3d as o3d
    except ImportError as error:  # pragma: no cover - production dependency
        raise RuntimeError("coordinate oracle requires open3d") from error
    vertices = np.asarray(vertices, dtype=np.float32)
    faces_u32 = np.asarray(faces, dtype=np.uint32)
    mesh = o3d.t.geometry.TriangleMesh(
        o3d.core.Tensor(vertices), o3d.core.Tensor(faces_u32),
    )
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh)
    rays = camera_rays_for_pixels(
        pixel, camera_to_world, fov_x, resolution=resolution,
    )
    hit = scene.cast_rays(o3d.core.Tensor(rays))
    distance = hit["t_hit"].numpy()
    face_index = hit["primitive_ids"].numpy().astype(np.int64)
    uv = hit["primitive_uvs"].numpy().astype(np.float32)
    valid = np.isfinite(distance) & (face_index >= 0) & (face_index < len(faces_u32))
    barycentric = np.stack((1.0 - uv[:, 0] - uv[:, 1], uv[:, 0], uv[:, 1]), axis=-1)
    points = np.full((len(pixel), 3), np.nan, dtype=np.float32)
    if valid.any():
        points[valid] = barycentric_points(
            vertices, faces_u32.astype(np.int64), face_index[valid], barycentric[valid],
        )
    return {
        "points": points,
        "face_index": face_index,
        "barycentric": barycentric.astype(np.float32),
        "valid": valid,
        "t_hit": distance.astype(np.float32),
    }


def _visible_points(
    points: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    camera: dict,
) -> tuple[np.ndarray, np.ndarray]:
    pixel, _, projected = project_points(
        points,
        np.asarray(camera["transform_matrix"], dtype=np.float32),
        float(camera["camera_angle_x"]),
        resolution=IMAGE_RESOLUTION,
    )
    hit = raycast_mesh_pixels(
        vertices,
        faces,
        pixel,
        np.asarray(camera["transform_matrix"], dtype=np.float32),
        float(camera["camera_angle_x"]),
        resolution=IMAGE_RESOLUTION,
    )
    visible = projected & hit["valid"] & (
        np.linalg.norm(hit["points"] - points, axis=-1) <= VISIBILITY_TOLERANCE
    )
    return pixel.astype(np.float32), visible


def _surface_route(
    source_vertices: np.ndarray,
    target_vertices: np.ndarray,
    faces: np.ndarray,
    camera: dict,
    anchor_camera: dict,
) -> dict[str, np.ndarray]:
    pixels = patch_center_pixels(PATCH_GRID, image_resolution=IMAGE_RESOLUTION).reshape(-1, 2)
    c2w = np.asarray(camera["transform_matrix"], dtype=np.float32)
    fov = float(camera["camera_angle_x"])
    target_hit = raycast_mesh_pixels(
        target_vertices, faces, pixels, c2w, fov, resolution=IMAGE_RESOLUTION,
    )
    target_valid = target_hit["valid"]
    face = target_hit["face_index"]
    barycentric = target_hit["barycentric"]
    source_points = np.full_like(target_hit["points"], np.nan)
    if target_valid.any():
        source_points[target_valid] = barycentric_points(
            source_vertices,
            faces,
            face[target_valid],
            barycentric[target_valid],
        )
    source_pixel, source_visible = _visible_points(
        np.nan_to_num(source_points), source_vertices, faces, camera,
    )
    _, target_anchor_visible = _visible_points(
        np.nan_to_num(target_hit["points"]), target_vertices, faces, anchor_camera,
    )
    valid = target_valid & teacher_pair_mask(source_visible, target_valid)
    target_anchor_visible &= valid
    return {
        "target_pixel": pixels.astype(np.float32),
        "source_pixel": source_pixel,
        "target_point": target_hit["points"].astype(np.float32),
        "source_point": source_points.astype(np.float32),
        "face_index": face.astype(np.int32),
        "barycentric": barycentric.astype(np.float32),
        "valid": valid,
        "target_anchor_visible": target_anchor_visible,
    }


def _coord64(points: np.ndarray, target_camera: dict, mesh_scale: float) -> np.ndarray:
    canonical = world_to_target_anchor_points(points, target_camera, scale=mesh_scale)
    output = np.full(canonical.shape, -1, dtype=np.int32)
    finite = np.isfinite(canonical).all(axis=1)
    output[finite] = np.clip(
        np.floor((canonical[finite] + 0.5) * 64), 0, 63,
    ).astype(np.int32)
    return output


def target_anchor_affine(frame: Mapping[str, Any]) -> np.ndarray:
    """Return the episode camera's world-to-target-anchor affine.

    Pixal3D conditions its native SS grid in the first (target) view's
    canonical frame.  This transform is therefore episode-specific; the fixed
    ProjGrid axis rotation alone is not an output-to-world transform.
    """

    import torch
    from data_toolkit.utils import transform_mesh

    basis = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    mapped = transform_mesh(basis, dict(frame)).cpu().numpy().astype(np.float64)
    affine = np.eye(4, dtype=np.float64)
    affine[:3, 3] = mapped[0]
    affine[:3, :3] = np.stack(
        (mapped[1] - mapped[0], mapped[2] - mapped[0], mapped[3] - mapped[0]),
        axis=1,
    )
    if not np.isfinite(affine).all() or abs(np.linalg.det(affine[:3, :3])) < 1e-8:
        raise ValueError("target-anchor affine must be finite and invertible")
    return affine


def _apply_affine_points(points: np.ndarray, affine: np.ndarray) -> np.ndarray:
    value = np.asarray(points, dtype=np.float64)
    matrix = np.asarray(affine, dtype=np.float64)
    if value.ndim < 1 or value.shape[-1] != 3:
        raise ValueError("points must end in an xyz axis")
    if matrix.shape != (4, 4):
        raise ValueError("affine must have shape [4,4]")
    flat = value.reshape(-1, 3)
    homogeneous = np.concatenate((flat, np.ones((len(flat), 1))), axis=1)
    mapped = (homogeneous @ matrix.T)[:, :3]
    return mapped.reshape(value.shape).astype(np.float32)


def world_to_target_anchor_points(
    points: np.ndarray, frame: Mapping[str, Any], *, scale: float = 1.0,
) -> np.ndarray:
    """Map normalized world points into the target-view SS canonical frame."""

    value = float(scale)
    if not np.isfinite(value) or value <= 0:
        raise ValueError("target-anchor scale must be finite and positive")
    output = _apply_affine_points(points, target_anchor_affine(frame))
    return (output * value).astype(np.float32)
