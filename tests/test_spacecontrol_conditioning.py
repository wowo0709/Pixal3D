import json

import numpy as np
import pytest
import torch
import trimesh

import pixal3d.utils.spacecontrol_conditioning as conditioning
from pixal3d.utils.spacecontrol_conditioning import (
    SurfaceCondition,
    encode_surface_condition,
    voxelize_surface_condition,
    write_surface_condition_artifacts,
)


class FakeEncoder(torch.nn.Module):
    def forward(self, occupancy):
        assert occupancy.shape == (1, 1, 64, 64, 64)
        return torch.ones((1, 8, 16, 16, 16), device=occupancy.device)


def test_transform_is_applied_without_extent_renormalization(tmp_path):
    mesh_path = tmp_path / "box.ply"
    trimesh.creation.box(extents=(2.0, 1.0, 0.5)).export(mesh_path)
    transform = np.eye(4)
    transform[:3, :3] *= 0.1
    transform[:3, 3] = (0.12, -0.05, 0.08)

    result = voxelize_surface_condition(mesh_path, transform, resolution=64)

    np.testing.assert_allclose(
        np.ptp(result.transformed_vertices, axis=0), (0.2, 0.1, 0.05), atol=1e-6
    )
    assert result.occupancy.shape == (1, 1, 64, 64, 64)
    assert result.active_indices.shape[0] > 0
    assert float(result.occupancy.mean()) < 0.10


def test_artifacts_record_preclip_bounds_and_fractions(tmp_path):
    mesh_path = tmp_path / "box.ply"
    trimesh.creation.box(extents=(1.2, 0.2, 0.2)).export(mesh_path)
    result = voxelize_surface_condition(mesh_path, np.eye(4), resolution=64)
    write_surface_condition_artifacts(result, tmp_path / "artifacts")
    payload = json.loads((tmp_path / "artifacts/voxelization.json").read_text())
    assert payload["clipped_vertex_fraction"] > 0.0
    assert payload["resolution"] == 64
    assert (tmp_path / "artifacts/encoder_input_last_canonical.ply").is_file()
    assert (tmp_path / "artifacts/last_voxels_64.ply").is_file()


def test_fake_encoder_has_exact_finite_shape(tmp_path, monkeypatch):
    base = tmp_path / "encoder"
    base.with_suffix(".json").write_text("{}")
    base.with_suffix(".safetensors").write_bytes(b"weights")
    monkeypatch.setattr(conditioning, "_load_encoder", lambda _: FakeEncoder())
    latent = encode_surface_condition(
        torch.ones((1, 1, 64, 64, 64)), str(base), torch.device("cpu")
    )
    assert latent.shape == (1, 8, 16, 16, 16)
    assert torch.isfinite(latent).all()


@pytest.mark.parametrize("resolution", [32, 128])
def test_only_resolution_64_is_accepted(tmp_path, resolution):
    mesh_path = tmp_path / "box.ply"
    trimesh.creation.box().export(mesh_path)
    with pytest.raises(ValueError, match="resolution 64"):
        voxelize_surface_condition(mesh_path, np.eye(4), resolution=resolution)


@pytest.mark.parametrize("matrix", [
    np.full((4, 4), np.nan),
    np.diag([1.0, 1.0, 0.0, 1.0]),
])
def test_invalid_transform_is_rejected(tmp_path, matrix):
    mesh_path = tmp_path / "box.ply"
    trimesh.creation.box().export(mesh_path)
    with pytest.raises(ValueError):
        voxelize_surface_condition(mesh_path, matrix, resolution=64)


def test_empty_surface_occupancy_is_rejected(tmp_path, monkeypatch):
    mesh_path = tmp_path / "box.ply"
    trimesh.creation.box().export(mesh_path)
    monkeypatch.setattr(
        conditioning,
        "_open3d_surface_indices",
        lambda vertices, faces, resolution: np.empty((0, 3), np.int64),
    )
    with pytest.raises(ValueError, match="voxelization is empty"):
        voxelize_surface_condition(mesh_path, np.eye(4), resolution=64)


def test_missing_checkpoint_sidecar_is_rejected(tmp_path):
    base = tmp_path / "encoder"
    base.with_suffix(".json").write_text("{}")
    with pytest.raises(FileNotFoundError):
        encode_surface_condition(
            torch.ones((1, 1, 64, 64, 64)), str(base), torch.device("cpu")
        )


@pytest.mark.parametrize("latent", [
    torch.zeros((1, 7, 16, 16, 16)),
    torch.full((1, 8, 16, 16, 16), float("inf")),
])
def test_invalid_encoder_output_is_rejected(tmp_path, monkeypatch, latent):
    class FixedEncoder(torch.nn.Module):
        def forward(self, occupancy):
            return latent

    base = tmp_path / "encoder"
    base.with_suffix(".json").write_text("{}")
    base.with_suffix(".safetensors").write_bytes(b"weights")
    monkeypatch.setattr(conditioning, "_load_encoder", lambda _: FixedEncoder())
    with pytest.raises(ValueError):
        encode_surface_condition(
            torch.ones((1, 1, 64, 64, 64)), str(base), torch.device("cpu")
        )


def test_surface_condition_rejects_invalid_fixed_contracts(tmp_path):
    mesh_path = tmp_path / "box.ply"
    trimesh.creation.box().export(mesh_path)
    kwargs = {
        "mesh_path": mesh_path,
        "last_to_pixal_canonical": np.eye(4),
        "encoder_path": "encoder",
    }

    with pytest.raises(ValueError, match="resolution 64"):
        SurfaceCondition(**kwargs, resolution=32)
    with pytest.raises(ValueError, match="step_index"):
        SurfaceCondition(**kwargs, step_index=12)
    with pytest.raises(ValueError):
        SurfaceCondition(
            mesh_path=mesh_path,
            last_to_pixal_canonical=np.eye(3),
            encoder_path="encoder",
        )
