import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from pixal3d.datasets.deformnext import DeformNeXtPBR1024, DeformNeXtSS64


def _image(path: Path, value: int):
    rgba = np.full((8, 8, 4), value, dtype=np.uint8)
    rgba[..., 3] = 255
    Image.fromarray(rgba, "RGBA").save(path)


def _fixture(tmp_path: Path):
    anchor = tmp_path / "geometry_anchor.png"
    sources = [tmp_path / f"source_{index}.png" for index in range(3)]
    _image(anchor, 16)
    for index, path in enumerate(sources):
        _image(path, 64 + index)
    np.savez(
        tmp_path / "transport.npz",
        target_keys=np.zeros((4, 2), np.float32),
        source_keys=np.zeros((3, 5, 2), np.float32),
        source_values=np.zeros((3, 5, 6), np.float32),
        source_valid=np.ones((3, 5), bool),
        patch_grid=np.arange(8).reshape(2, 2, 2) % 4,
        spatial_indices=np.zeros((3, 4), np.int64),
        target_query=np.zeros((4, 3), np.float32),
        deformation=np.zeros((4, 3), np.float32),
    )
    np.savez(
        tmp_path / "transport_pbr.npz",
        source_values=np.full((3, 5, 6), 9, np.float32),
        target_query=np.full((4, 3), 7, np.float32),
        patch_ids=np.array([0, 1], np.int64),
    )
    np.savez(tmp_path / "ss.npz", z=np.ones((8, 2, 2, 2), np.float32))
    coords = np.array([[0, 1, 1], [1, 0, 1]], np.int32)
    np.savez(tmp_path / "pbr.npz", coords=coords, feats=np.ones((2, 32), np.float32))
    np.savez(tmp_path / "shape.npz", coords=coords, feats=np.ones((2, 32), np.float32))
    transform = np.eye(4)
    transform[1, 3] = -2.0
    row = {
        "sample_id": "sample-a", "split": "train",
        "target_anchor": anchor.name, "target_anchor_kind": "geometry_depth",
        "source_images": [path.name for path in sources],
        "camera_angle_x": [1.0] * 4,
        "transform_matrix": [transform.tolist()] * 4,
        "mesh_scale": 0.96, "transport_cache": "transport.npz",
        "transport_cache_pbr": "transport_pbr.npz",
        "ss_latent": "ss.npz", "pbr_latent": "pbr.npz",
        "shape_latent": "shape.npz", "pbr_tokens": 2,
    }
    manifest = tmp_path / "train.jsonl"
    manifest.write_text(json.dumps(row) + "\n")
    return manifest, row


def test_ss_loader_preserves_target_then_three_source_roles(tmp_path):
    manifest, _ = _fixture(tmp_path)
    dataset = DeformNeXtSS64(str(manifest), image_size=8, expected_split="train")
    item = dataset[0]
    assert item["cond"].shape == (4, 3, 8, 8)
    assert item["cond"][:, 0, 0, 0].tolist() == pytest.approx(
        [16 / 255, 64 / 255, 65 / 255, 66 / 255]
    )
    assert item["x_0"].shape == (8, 2, 2, 2)
    assert item["transport_source_valid"].dtype == torch.bool
    batch = dataset.collate_fn([item, item])
    assert batch["cond"].shape == (2, 4, 3, 8, 8)
    assert batch["transport_target_keys"].shape == (2, 4, 2)


def test_v1_loader_reads_only_spatial_transport_fields(tmp_path):
    manifest, _ = _fixture(tmp_path)
    dataset = DeformNeXtSS64(
        str(manifest), image_size=8, expected_split="train", transport_arm="v1"
    )
    item = dataset[0]
    assert "transport_source_values" in item
    assert "transport_spatial_indices" in item
    assert "transport_target_keys" not in item
    assert "transport_source_keys" not in item
    assert "transport_target_query" not in item


def test_match_cache_avoids_loading_full_omega_keys(tmp_path):
    manifest, _ = _fixture(tmp_path)
    matches = tmp_path / "matches"
    matches.mkdir()
    np.savez(
        matches / "sample-a.npz",
        semantic_indices=np.zeros((4, 2), np.uint16),
        matching_logits=np.zeros((4, 2), np.float32),
        candidate_valid=np.ones((4, 2), bool),
    )
    dataset = DeformNeXtSS64(
        str(manifest), image_size=8, expected_split="train",
        transport_arm="b3", match_root=str(matches),
    )
    item = dataset[0]
    assert item["transport_semantic_indices"].shape == (4, 2)
    assert "transport_target_keys" not in item
    assert "transport_source_keys" not in item


def test_textured_target_render_is_rejected(tmp_path):
    manifest, row = _fixture(tmp_path)
    render = tmp_path / "render_cond"
    render.mkdir()
    target = render / "view00.png"
    _image(target, 99)
    row["target_anchor"] = str(target)
    manifest.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="textured render_cond"):
        DeformNeXtSS64(str(manifest), image_size=8)


def test_pbr_loader_and_sparse_collation(tmp_path):
    manifest, _ = _fixture(tmp_path)
    dataset = DeformNeXtPBR1024(
        str(manifest), image_size=8, expected_split="train", max_tokens=8
    )
    item = dataset[0]
    assert torch.all(item["transport_source_values"] == 9)
    assert torch.all(item["transport_target_query"] == 7)
    batches = dataset.collate_fn([item, item], split_size=1)
    assert len(batches) == 1
    assert batches[0]["x_0"].coords.shape == (4, 4)
    assert batches[0]["x_0"].feats.shape == (4, 32)
