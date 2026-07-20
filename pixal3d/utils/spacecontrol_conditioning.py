"""Contracts for conditioning SpaceControl with a prealigned mesh surface."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
from typing import Callable

import numpy as np
import torch


_RESOLUTION = 64
_LATENT_SHAPE = (1, 8, 16, 16, 16)
_OCCUPANCY_SHAPE = (1, 1, _RESOLUTION, _RESOLUTION, _RESOLUTION)
_CLIP_EPSILON = 1e-6


@dataclass(frozen=True)
class SurfaceCondition:
    mesh_path: Path
    last_to_pixal_canonical: np.ndarray
    encoder_path: str
    step_index: int = 6
    resolution: int = _RESOLUTION

    def __post_init__(self) -> None:
        object.__setattr__(self, "mesh_path", Path(self.mesh_path))
        object.__setattr__(
            self,
            "last_to_pixal_canonical",
            _validated_transform(self.last_to_pixal_canonical),
        )
        _validated_resolution(self.resolution)
        if not 0 <= self.step_index < 12:
            raise ValueError("SpaceControl step_index must be in [0, 12)")


@dataclass(frozen=True)
class SurfaceVoxelization:
    occupancy: torch.Tensor
    transformed_vertices: np.ndarray
    clipped_vertices: np.ndarray
    faces: np.ndarray
    active_indices: np.ndarray
    diagnostics: dict[str, object]


def voxelize_surface_condition(
    mesh_path: str | Path,
    last_to_pixal_canonical: np.ndarray,
    *,
    resolution: int = _RESOLUTION,
) -> SurfaceVoxelization:
    """Voxelize only the surface of an already canonicalized mesh."""
    _validated_resolution(resolution)
    vertices, faces = _load_triangle_arrays(Path(mesh_path))
    matrix = _validated_transform(last_to_pixal_canonical)

    homogeneous = np.column_stack([vertices, np.ones(len(vertices))])
    transformed = (homogeneous @ matrix.T)[:, :3]
    outside = np.any(np.abs(transformed) > 0.5 - _CLIP_EPSILON, axis=1)
    clipped = np.clip(transformed, -0.5 + _CLIP_EPSILON, 0.5 - _CLIP_EPSILON)
    active = _open3d_surface_indices(clipped, faces, resolution)
    if active.size == 0:
        raise ValueError("SpaceControl surface voxelization is empty")
    if active.ndim != 2 or active.shape[1] != 3:
        raise ValueError("SpaceControl surface voxelization returned invalid indices")
    if np.any(active < 0) or np.any(active >= resolution):
        raise ValueError("SpaceControl surface voxelization returned out-of-bounds indices")

    occupancy = torch.zeros(_OCCUPANCY_SHAPE, dtype=torch.float32)
    occupancy[0, 0, active[:, 0], active[:, 1], active[:, 2]] = 1.0
    boundary = np.any((active == 0) | (active == resolution - 1), axis=1)
    diagnostics = {
        "resolution": resolution,
        "active_voxel_count": int(len(active)),
        "active_voxel_bounds": [active.min(0).tolist(), active.max(0).tolist()],
        "transformed_bounds": [transformed.min(0).tolist(), transformed.max(0).tolist()],
        "clipped_vertex_fraction": float(outside.mean()),
        "clipped_voxel_fraction": float(boundary.mean()),
    }
    return SurfaceVoxelization(
        occupancy=occupancy,
        transformed_vertices=transformed,
        clipped_vertices=clipped,
        faces=faces,
        active_indices=active,
        diagnostics=diagnostics,
    )


def encode_surface_condition(
    occupancy: torch.Tensor, encoder_path: str | Path, device: torch.device
) -> torch.Tensor:
    """Encode a fixed-size surface occupancy grid using a local checkpoint."""
    if not isinstance(occupancy, torch.Tensor):
        raise TypeError("SpaceControl occupancy must be a torch.Tensor")
    if tuple(occupancy.shape) != _OCCUPANCY_SHAPE:
        raise ValueError(
            "Unexpected SpaceControl occupancy shape: "
            f"{tuple(occupancy.shape)}"
        )
    if not torch.isfinite(occupancy).all():
        raise ValueError("SpaceControl occupancy contains non-finite values")

    base = Path(encoder_path).expanduser().resolve()
    for suffix in (".json", ".safetensors"):
        sidecar = base.with_suffix(suffix)
        if not sidecar.is_file():
            raise FileNotFoundError(sidecar)

    encoder = _load_encoder(str(base)).eval().to(device)
    with torch.inference_mode():
        latent = encoder(occupancy.to(device=device, dtype=torch.float32))
    if tuple(latent.shape) != _LATENT_SHAPE:
        raise ValueError(f"Unexpected SpaceControl latent shape: {tuple(latent.shape)}")
    if not torch.isfinite(latent).all():
        raise ValueError("SpaceControl encoder produced non-finite values")
    return latent


def write_surface_condition_artifacts(
    voxelization: SurfaceVoxelization, output_dir: str | Path
) -> None:
    """Write inspectable surface-conditioning artifacts without partial files."""
    import trimesh

    output_dir = Path(output_dir)
    resolution = voxelization.diagnostics.get("resolution")
    _validated_resolution(resolution)
    active = np.asarray(voxelization.active_indices, dtype=np.int64)
    if active.ndim != 2 or active.shape[1] != 3 or active.size == 0:
        raise ValueError("SpaceControl surface voxelization has invalid active indices")
    if np.any(active < 0) or np.any(active >= resolution):
        raise ValueError("SpaceControl surface voxelization has out-of-bounds indices")

    clipped_mesh = trimesh.Trimesh(
        vertices=voxelization.clipped_vertices,
        faces=voxelization.faces,
        process=False,
    )
    voxel_centers = (active.astype(np.float64) + 0.5) / resolution - 0.5

    _atomic_write(
        output_dir / "encoder_input_last_canonical.ply",
        lambda path: clipped_mesh.export(path),
    )
    _atomic_write(
        output_dir / "last_voxels_64.ply",
        lambda path: trimesh.PointCloud(voxel_centers).export(path),
    )
    _atomic_write(
        output_dir / "voxelization.json",
        lambda path: path.write_text(json.dumps(voxelization.diagnostics, indent=2) + "\n"),
    )


def _validated_resolution(resolution: object) -> int:
    if resolution != _RESOLUTION:
        raise ValueError("SpaceControl requires resolution 64")
    return _RESOLUTION


def _validated_transform(last_to_pixal_canonical: np.ndarray) -> np.ndarray:
    matrix = np.asarray(last_to_pixal_canonical, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError("last_to_pixal_canonical must have shape (4, 4)")
    if not np.isfinite(matrix).all():
        raise ValueError("last_to_pixal_canonical must contain only finite values")
    if np.isclose(np.linalg.det(matrix), 0.0):
        raise ValueError("last_to_pixal_canonical must be invertible")
    return matrix


def _load_triangle_arrays(mesh_path: Path) -> tuple[np.ndarray, np.ndarray]:
    import trimesh

    mesh = trimesh.load_mesh(mesh_path, process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("SpaceControl input must be a single triangle mesh")

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
        raise ValueError("SpaceControl mesh must contain vertices with shape (N, 3)")
    if not np.isfinite(vertices).all():
        raise ValueError("SpaceControl mesh vertices must be finite")
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
        raise ValueError("SpaceControl mesh must contain triangle faces")
    if np.any(faces < 0) or np.any(faces >= len(vertices)):
        raise ValueError("SpaceControl mesh faces contain invalid vertex indices")
    return vertices, faces


def _open3d_surface_indices(
    vertices: np.ndarray, faces: np.ndarray, resolution: int
) -> np.ndarray:
    import open3d as o3d

    mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(vertices),
        triangles=o3d.utility.Vector3iVector(faces),
    )
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(
        mesh,
        voxel_size=1.0 / resolution,
        min_bound=(-0.5, -0.5, -0.5),
        max_bound=(0.5, 0.5, 0.5),
    )
    return np.asarray(
        [voxel.grid_index for voxel in voxel_grid.get_voxels()], dtype=np.int64
    ).reshape(-1, 3)


def _load_encoder(encoder_path: str) -> torch.nn.Module:
    from pixal3d import models

    return models.from_pretrained(encoder_path)


def _atomic_write(path: Path, writer: Callable[[Path], object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, suffix=path.suffix, delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        writer(temporary_path)
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
