import json

import numpy as np
import pytest
import torch
from PIL import Image

from pixal3d.datasets.components import load_anchor_first_conditions


def write_render_fixture(root, num_views=8):
    frames = []
    for index in range(num_views):
        rgba = np.zeros((4, 4, 4), dtype=np.uint8)
        rgba[..., index % 3] = 10 + index
        rgba[..., 3] = 255
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
