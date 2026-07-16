import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from data_toolkit.pipeline.validation import (
    ValidationError,
    validate_render_dir,
    validate_scale,
    validate_sparse_latent,
    validate_ss_latent,
)


def _write_render_directory(
    path: Path, expected_views: int = 1, resolution: int = 16
) -> list[dict]:
    frames = []
    for index in range(expected_views):
        name = f"{index:03d}.png"
        rgba = np.zeros((resolution, resolution, 4), np.uint8)
        rgba[
            resolution // 4 : 3 * resolution // 4,
            resolution // 4 : 3 * resolution // 4,
            3,
        ] = 255
        Image.fromarray(rgba, "RGBA").save(path / name)
        frames.append(
            {
                "file_path": name,
                "camera_angle_x": 0.7,
                "transform_matrix": np.eye(4).tolist(),
                "radius": 2.0,
            }
        )
    (path / "transforms.json").write_text(json.dumps({"frames": frames}))
    return frames


def _rewrite_frames(path: Path, frames: list[dict]) -> None:
    (path / "transforms.json").write_text(json.dumps({"frames": frames}))


def test_valid_render_directory(tmp_path):
    _write_render_directory(tmp_path, expected_views=8, resolution=512)

    validate_render_dir(tmp_path, 8, 512)


@pytest.mark.parametrize("metadata", [None, "not-json", "{}"])
def test_invalid_render_metadata_is_translated(metadata, tmp_path):
    if metadata is not None:
        (tmp_path / "transforms.json").write_text(metadata)

    with pytest.raises(ValidationError, match="transforms"):
        validate_render_dir(tmp_path, 1, 16)


def test_render_directory_requires_exact_image_count(tmp_path):
    _write_render_directory(tmp_path)
    Image.new("RGBA", (16, 16)).save(tmp_path / "001.png")

    with pytest.raises(ValidationError, match="expected 1 render images, found 2"):
        validate_render_dir(tmp_path, 1, 16)


def test_render_directory_requires_exact_frame_count(tmp_path):
    _write_render_directory(tmp_path)
    _rewrite_frames(tmp_path, [])

    with pytest.raises(ValidationError, match="expected 1 frames, found 0"):
        validate_render_dir(tmp_path, 1, 16)


@pytest.mark.parametrize(
    ("mode", "size", "match"),
    [
        ("RGB", (16, 16), "RGBA"),
        ("RGBA", (8, 16), "resolution"),
    ],
)
def test_render_images_require_rgba_at_exact_resolution(
    mode, size, match, tmp_path
):
    _write_render_directory(tmp_path)
    Image.new(mode, size).save(tmp_path / "000.png")

    with pytest.raises(ValidationError, match=match):
        validate_render_dir(tmp_path, 1, 16)


@pytest.mark.parametrize("opaque_pixels", [0, 256])
def test_render_images_reject_alpha_outside_bounds(opaque_pixels, tmp_path):
    _write_render_directory(tmp_path)
    rgba = np.zeros((16, 16, 4), np.uint8)
    rgba.reshape(-1, 4)[:opaque_pixels, 3] = 255
    Image.fromarray(rgba, "RGBA").save(tmp_path / "000.png")

    with pytest.raises(ValidationError, match="alpha fraction"):
        validate_render_dir(tmp_path, 1, 16)


def test_render_image_decode_errors_are_translated(tmp_path):
    _write_render_directory(tmp_path)
    (tmp_path / "000.png").write_bytes(b"not a png")

    with pytest.raises(ValidationError, match="render image"):
        validate_render_dir(tmp_path, 1, 16)


@pytest.mark.parametrize(
    "matrix",
    [
        [[1.0, 0.0], [0.0, 1.0]],
        np.diag([1.0, 1.0, 1.0, 0.0]).tolist(),
        np.diag([1.0, 1.0, 1.0, np.nan]).tolist(),
    ],
)
def test_render_rejects_invalid_camera_matrices(matrix, tmp_path):
    frames = _write_render_directory(tmp_path)
    frames[0]["transform_matrix"] = matrix
    _rewrite_frames(tmp_path, frames)

    with pytest.raises(ValidationError, match="camera frame 0"):
        validate_render_dir(tmp_path, 1, 16)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("camera_angle_x", np.inf, "camera angle"),
        ("radius", np.nan, "camera radius"),
        ("radius", -1.0, "camera radius"),
    ],
)
def test_render_rejects_invalid_camera_metadata(field, value, match, tmp_path):
    frames = _write_render_directory(tmp_path)
    frames[0][field] = value
    _rewrite_frames(tmp_path, frames)

    with pytest.raises(ValidationError, match=match):
        validate_render_dir(tmp_path, 1, 16)


def test_render_frame_key_errors_are_translated(tmp_path):
    frames = _write_render_directory(tmp_path)
    frames[0].pop("transform_matrix")
    _rewrite_frames(tmp_path, frames)

    with pytest.raises(ValidationError, match="camera frame 0"):
        validate_render_dir(tmp_path, 1, 16)


def test_valid_sparse_latent(tmp_path):
    path = tmp_path / "valid.npz"
    np.savez(
        path,
        feats=np.ones((2, 4), np.float32),
        coords=np.asarray([[0, 1, 2], [15, 15, 15]], np.uint8),
    )

    validate_sparse_latent(path, 16, 8192)


def test_non_finite_latent_is_rejected(tmp_path):
    path = tmp_path / "bad.npz"
    np.savez(
        path,
        feats=np.array([[np.nan]], np.float32),
        coords=np.zeros((1, 3), np.uint8),
    )

    with pytest.raises(ValidationError, match="non-finite"):
        validate_sparse_latent(path, 16, 8192)


@pytest.mark.parametrize(
    ("feats", "coords"),
    [
        (np.ones((1,), np.float32), np.zeros((1, 3), np.uint8)),
        (np.ones((1, 2), np.float32), np.zeros((3,), np.uint8)),
        (np.ones((1, 2), np.float32), np.zeros((1, 2), np.uint8)),
        (np.ones((2, 2), np.float32), np.zeros((1, 3), np.uint8)),
    ],
)
def test_sparse_latent_rejects_shape_mismatches(feats, coords, tmp_path):
    path = tmp_path / "bad-shape.npz"
    np.savez(path, feats=feats, coords=coords)

    with pytest.raises(ValidationError, match="shape"):
        validate_sparse_latent(path, 16, 8192)


def test_sparse_latent_rejects_too_many_tokens(tmp_path):
    path = tmp_path / "too-many.npz"
    np.savez(
        path,
        feats=np.ones((3, 1), np.float32),
        coords=np.asarray([[0, 0, 0], [0, 0, 1], [0, 0, 2]]),
    )

    with pytest.raises(ValidationError, match="token limit"):
        validate_sparse_latent(path, 16, 2)


@pytest.mark.parametrize(
    ("coords", "match"),
    [
        (np.asarray([[0.0, np.nan, 1.0]]), "non-finite coordinates"),
        (np.asarray([[0.0, 0.5, 1.0]]), "integral"),
        (np.asarray([[-1, 0, 0]]), "outside grid"),
        (np.asarray([[16, 0, 0]]), "outside grid"),
        (np.asarray([[1, 2, 3], [1, 2, 3]]), "unique"),
    ],
)
def test_sparse_latent_rejects_invalid_coordinates(coords, match, tmp_path):
    path = tmp_path / "bad-coordinates.npz"
    np.savez(path, feats=np.ones((len(coords), 1)), coords=coords)

    with pytest.raises(ValidationError, match=match):
        validate_sparse_latent(path, 16, 8192)


@pytest.mark.parametrize("case", ["missing", "corrupt", "missing-key"])
def test_sparse_latent_load_errors_are_translated(case, tmp_path):
    path = tmp_path / "bad.npz"
    if case == "corrupt":
        path.write_bytes(b"not an npz")
    elif case == "missing-key":
        np.savez(path, feats=np.ones((1, 1)))

    with pytest.raises(ValidationError, match="sparse latent"):
        validate_sparse_latent(path, 16, 8192)


def test_ss_latent_requires_finite_z(tmp_path):
    valid = tmp_path / "valid.npz"
    invalid = tmp_path / "invalid.npz"
    np.savez(valid, z=np.ones((1, 2), np.float32))
    np.savez(invalid, z=np.asarray([np.inf], np.float32))

    validate_ss_latent(valid)
    with pytest.raises(ValidationError, match="non-finite SS latent"):
        validate_ss_latent(invalid)


@pytest.mark.parametrize("case", ["missing", "corrupt", "missing-key"])
def test_ss_latent_load_errors_are_translated(case, tmp_path):
    path = tmp_path / "bad.npz"
    if case == "corrupt":
        path.write_bytes(b"not an npz")
    elif case == "missing-key":
        np.savez(path, value=np.ones((1, 1)))

    with pytest.raises(ValidationError, match="SS latent"):
        validate_ss_latent(path)


def test_scale_requires_nonempty_finite_metadata(tmp_path):
    valid = tmp_path / "valid.json"
    valid.write_text('{"total_scale": 1.25, "sphere_radius": 2.0}')
    validate_scale(valid)

    for index, value in enumerate(({}, {"scale": np.nan}, [])):
        path = tmp_path / f"invalid-{index}.json"
        path.write_text(json.dumps(value))
        with pytest.raises(ValidationError, match="scale metadata"):
            validate_scale(path)


@pytest.mark.parametrize("case", ["missing", "malformed"])
def test_scale_load_errors_are_translated(case, tmp_path):
    path = tmp_path / "bad.json"
    if case == "malformed":
        path.write_text("not-json")

    with pytest.raises(ValidationError, match="scale metadata"):
        validate_scale(path)
