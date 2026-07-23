import json

import numpy as np
import pytest
import torch
from PIL import Image

from pixal3d.datasets.components import (
    MultiViewImageConditionedMixin,
    load_anchor_first_conditions,
)


def write_render_fixture(root, num_views=8):
    frames = []
    for index in range(num_views):
        rgba = np.zeros((4, 4, 4), dtype=np.uint8)
        rgba[..., index % 3] = 10 + index
        rgba[..., 3] = 255
        rgba[0, 0, 3] = 0
        Image.fromarray(rgba, mode="RGBA").save(root / f"{index:03d}.png")
        transform = np.eye(4, dtype=np.float32)
        transform[0, 3] = index
        transform[2, 3] = 2.0
        frames.append({
            "file_path": f"{index:03d}.png",
            "camera_angle_x": 0.5 + index * 0.01,
            "transform_matrix": transform.tolist(),
        })
    (root / "transforms.json").write_text(json.dumps({"frames": frames}))


def test_loader_keeps_anchor_first_and_all_views_unique(tmp_path):
    write_render_fixture(tmp_path)
    result = load_anchor_first_conditions(
        tmp_path,
        anchor_index=1,
        image_size=4,
        other_view_indices=[7, 4, 0, 2, 3, 5, 6],
    )
    assert result["view_indices"].tolist() == [1, 7, 4, 0, 2, 3, 5, 6]
    assert result["cond"].shape == (8, 3, 4, 4)
    assert result["camera_angle_x"].shape == (8,)
    assert result["camera_distance"].shape == (8,)
    assert result["transform_matrix"].shape == (8, 4, 4)
    assert result["cond"].dtype == torch.float32
    assert result["camera_angle_x"].dtype == torch.float32
    assert result["camera_distance"].dtype == torch.float32
    assert result["transform_matrix"].dtype == torch.float32
    assert result["view_indices"].dtype == torch.int64
    assert torch.equal(result["cond"][0, :, 0, 0], torch.zeros(3))
    assert torch.equal(
        result["transform_matrix"][0, :3, 3], torch.tensor([1.0, 0.0, 2.0])
    )


def test_loader_requires_exactly_eight_training_frames(tmp_path):
    write_render_fixture(tmp_path, num_views=7)
    with pytest.raises(ValueError, match="exactly eight"):
        load_anchor_first_conditions(
            tmp_path,
            anchor_index=1,
            image_size=4,
            other_view_indices=[0, 2, 3, 4, 5, 6],
        )


def test_loader_rejects_duplicate_anchor(tmp_path):
    write_render_fixture(tmp_path)
    with pytest.raises(ValueError, match="anchor"):
        load_anchor_first_conditions(
            tmp_path,
            anchor_index=1,
            image_size=4,
            other_view_indices=[1, 0, 2, 3, 4, 5, 6],
        )


def test_loader_rejects_missing_or_nonfinite_camera(tmp_path):
    write_render_fixture(tmp_path)
    manifest = json.loads((tmp_path / "transforms.json").read_text())
    manifest["frames"][3]["transform_matrix"][0][0] = float("nan")
    (tmp_path / "transforms.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="finite"):
        load_anchor_first_conditions(
            tmp_path,
            anchor_index=0,
            image_size=4,
            other_view_indices=[1, 2, 3, 4, 5, 6, 7],
        )


def test_loader_rejects_camera_angle_that_overflows_float32(tmp_path):
    write_render_fixture(tmp_path)
    manifest = json.loads((tmp_path / "transforms.json").read_text())
    manifest["frames"][0]["camera_angle_x"] = 1e39
    (tmp_path / "transforms.json").write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="camera_angle_x must be finite"):
        load_anchor_first_conditions(
            tmp_path,
            anchor_index=0,
            image_size=4,
            other_view_indices=[1, 2, 3, 4, 5, 6, 7],
        )


def test_loader_rejects_existing_image_path_that_escapes_root(tmp_path):
    render_root = tmp_path / "render"
    render_root.mkdir()
    write_render_fixture(render_root)
    outside_image = tmp_path / "outside.png"
    Image.new("RGBA", (4, 4), (255, 255, 255, 255)).save(outside_image)
    manifest = json.loads((render_root / "transforms.json").read_text())
    manifest["frames"][0]["file_path"] = "../outside.png"
    (render_root / "transforms.json").write_text(json.dumps(manifest))

    with pytest.raises(FileNotFoundError, match="condition image for view 0"):
        load_anchor_first_conditions(
            render_root,
            anchor_index=0,
            image_size=4,
            other_view_indices=[1, 2, 3, 4, 5, 6, 7],
        )


class _ParentFailsBeforeAnchor:
    def get_instance(self, root, instance):
        raise FileNotFoundError("latent missing")


class _ParentFailsAfterAnchor:
    def get_instance(self, root, instance):
        self._current_view_idx = 1
        raise FileNotFoundError("view latent missing")


class _ParentLoadsAnchor:
    def get_instance(self, root, instance):
        self._current_view_idx = 0
        self._current_latent_dir = self.latent_dir
        return {"x_0": torch.zeros(1)}


class _FailsBeforeAnchorDataset(
    MultiViewImageConditionedMixin, _ParentFailsBeforeAnchor
):
    pass


class _FailsAfterAnchorDataset(MultiViewImageConditionedMixin, _ParentFailsAfterAnchor):
    pass


class _LoadsAnchorDataset(MultiViewImageConditionedMixin, _ParentLoadsAnchor):
    pass


def test_mixin_contextualizes_parent_failure_before_anchor_selection():
    dataset = _FailsBeforeAnchorDataset.__new__(_FailsBeforeAnchorDataset)
    dataset._current_dataset_name = "source-a"
    dataset._current_view_idx = 0  # stale state from a previous sample

    with pytest.raises(
        RuntimeError,
        match=r"source=source-a asset=asset-a anchor=unknown: latent missing",
    ) as caught:
        dataset.get_instance({}, "asset-a")

    assert isinstance(caught.value.__cause__, FileNotFoundError)


def test_mixin_contextualizes_parent_failure_after_anchor_selection():
    dataset = _FailsAfterAnchorDataset.__new__(_FailsAfterAnchorDataset)
    dataset._current_dataset_name = "source-b"

    with pytest.raises(
        RuntimeError,
        match=r"source=source-b asset=asset-b anchor=view01: view latent missing",
    ) as caught:
        dataset.get_instance({}, "asset-b")

    assert isinstance(caught.value.__cause__, FileNotFoundError)


def test_mixin_rejects_mesh_scale_that_underflows_float32(tmp_path):
    render_root = tmp_path / "renders" / "asset-c"
    render_root.mkdir(parents=True)
    write_render_fixture(render_root)
    latent_dir = tmp_path / "latents"
    latent_dir.mkdir()
    (latent_dir / "view00_scale.json").write_text(json.dumps({"total_scale": 1e-46}))

    dataset = _LoadsAnchorDataset.__new__(_LoadsAnchorDataset)
    dataset.image_size = 4
    dataset.latent_dir = latent_dir
    dataset._current_dataset_name = "source-c"

    with pytest.raises(
        RuntimeError,
        match=r"source=source-c asset=asset-c anchor=view00: total_scale",
    ):
        dataset.get_instance({"render_cond": str(tmp_path / "renders")}, "asset-c")


def test_mixin_returns_scalar_float32_mesh_scale(tmp_path):
    render_root = tmp_path / "renders" / "asset-d"
    render_root.mkdir(parents=True)
    write_render_fixture(render_root)
    latent_dir = tmp_path / "latents"
    latent_dir.mkdir()
    (latent_dir / "view00_scale.json").write_text(json.dumps({"total_scale": 0.25}))

    dataset = _LoadsAnchorDataset.__new__(_LoadsAnchorDataset)
    dataset.image_size = 4
    dataset.latent_dir = latent_dir
    dataset._current_dataset_name = "source-d"

    result = dataset.get_instance(
        {"render_cond": str(tmp_path / "renders")}, "asset-d"
    )

    assert result["mesh_scale"].shape == torch.Size([])
    assert result["mesh_scale"].dtype == torch.float32
    assert torch.isfinite(result["mesh_scale"])
    assert result["mesh_scale"] == torch.tensor(0.25, dtype=torch.float32)
