import json
import math
from hashlib import sha256

import pytest
import torch
from PIL import Image

from pixal3d.experiments.correspondence import (
    load_calibrated_views,
    resolve_foreground_mask,
)


def _transform(translation=(0.0, 0.0, 2.0)):
    matrix = torch.eye(4, dtype=torch.float64)
    matrix[:3, 3] = torch.tensor(translation, dtype=torch.float64)
    return matrix.tolist()


def _write_manifest(root, frames, **top_level):
    path = root / "transforms.json"
    path.write_text(json.dumps({**top_level, "frames": frames}))
    return path


def _write_image(path, color, *, mode="RGBA"):
    Image.new(mode, (4, 3), color=color).save(path)


def test_calibrated_views_load_ordered_first_k_with_camera_provenance(tmp_path):
    frames = []
    translations = [(0, 0, 2), (3, 4, 0), (1, 2, 2), (4, 4, 7), (9, 9, 9)]
    for index, translation in enumerate(translations):
        image_path = tmp_path / f"view_{index}.png"
        _write_image(image_path, (index * 20, 1, 2, 255))
        frame = {
            "file_path": image_path.name,
            "transform_matrix": _transform(translation),
        }
        if index % 2 == 0:
            frame["camera_angle_x"] = 0.7 + index / 100
        frames.append(frame)

    views, mesh_scale = load_calibrated_views(
        _write_manifest(tmp_path, frames, camera_angle_x=0.65),
        mesh_scale=1.25,
    )

    assert mesh_scale == 1.25
    assert [view.frame_index for view in views] == [0, 1, 2, 3]
    assert [view.image.getpixel((0, 0))[0] for view in views] == [0, 20, 40, 60]
    assert [view.camera_angle_x for view in views] == pytest.approx(
        [0.7, 0.65, 0.72, 0.65]
    )
    assert [view.distance for view in views] == pytest.approx([2.0, 5.0, 3.0, 9.0])
    assert all(view.transform_matrix.shape == (4, 4) for view in views)
    assert all(torch.isfinite(view.transform_matrix).all() for view in views)
    assert [view.source_sha256 for view in views] == [
        sha256((tmp_path / f"view_{index}.png").read_bytes()).hexdigest()
        for index in range(4)
    ]


def test_calibrated_distance_uses_float64_translation_norm(tmp_path):
    _write_image(tmp_path / "view.png", (0, 0, 0, 255))
    translation = (100_000_000.0, 10_000.0, 3.0)
    frame = {
        "file_path": "view.png",
        "camera_angle_x": 0.7,
        "transform_matrix": _transform(translation),
    }

    views, _ = load_calibrated_views(
        _write_manifest(tmp_path, [frame]), mesh_scale=1.0, num_views=1
    )

    assert views[0].distance == math.sqrt(sum(value * value for value in translation))


@pytest.mark.parametrize("mesh_scale", [0.0, -1.0, float("nan"), float("inf")])
def test_calibrated_views_require_finite_positive_mesh_scale(tmp_path, mesh_scale):
    with pytest.raises(ValueError, match="mesh_scale must be finite and positive"):
        load_calibrated_views(tmp_path / "transforms.json", mesh_scale=mesh_scale)


@pytest.mark.parametrize(
    "transform",
    [
        [[1.0]],
        [[1.0, 0.0, 0.0, float("nan")], *torch.eye(4).tolist()[1:]],
        [[1e39, 0.0, 0.0, 0.0], *torch.eye(4).tolist()[1:]],
    ],
)
def test_calibrated_views_reject_malformed_or_nonfinite_transforms(
    tmp_path, transform
):
    _write_image(tmp_path / "view.png", (0, 0, 0, 255))
    frame = {
        "file_path": "view.png",
        "camera_angle_x": 0.7,
        "transform_matrix": transform,
    }

    with pytest.raises(ValueError, match=r"transform_matrix must be finite \[4, 4\]"):
        load_calibrated_views(
            _write_manifest(tmp_path, [frame]), mesh_scale=1.0, num_views=1
        )


@pytest.mark.parametrize("camera_angle_x", [None, float("nan"), float("inf")])
def test_calibrated_views_require_finite_frame_or_top_level_fov(
    tmp_path, camera_angle_x
):
    _write_image(tmp_path / "view.png", (0, 0, 0, 255))
    frame = {"file_path": "view.png", "transform_matrix": _transform()}
    top_level = {} if camera_angle_x is None else {"camera_angle_x": camera_angle_x}

    with pytest.raises(ValueError, match="camera_angle_x must be finite"):
        load_calibrated_views(
            _write_manifest(tmp_path, [frame], **top_level),
            mesh_scale=1.0,
            num_views=1,
        )


def test_calibrated_views_reject_frame_path_escape(tmp_path):
    outside = tmp_path.parent / f"{tmp_path.name}-outside.png"
    _write_image(outside, (0, 0, 0, 255))
    frame = {
        "file_path": f"../{outside.name}",
        "camera_angle_x": 0.7,
        "transform_matrix": _transform(),
    }

    with pytest.raises(ValueError, match="inside the manifest directory"):
        load_calibrated_views(
            _write_manifest(tmp_path, [frame]), mesh_scale=1.0, num_views=1
        )


def test_calibrated_views_reject_symlink_target_escape(tmp_path):
    outside = tmp_path.parent / f"{tmp_path.name}-outside.png"
    _write_image(outside, (0, 0, 0, 255))
    (tmp_path / "linked.png").symlink_to(outside)
    frame = {
        "file_path": "linked.png",
        "camera_angle_x": 0.7,
        "transform_matrix": _transform(),
    }

    with pytest.raises(ValueError, match="inside the manifest directory"):
        load_calibrated_views(
            _write_manifest(tmp_path, [frame]), mesh_scale=1.0, num_views=1
        )


def test_calibrated_views_reject_insufficient_frames(tmp_path):
    frames = [
        {
            "file_path": f"view_{index}.png",
            "camera_angle_x": 0.7,
            "transform_matrix": _transform(),
        }
        for index in range(3)
    ]

    with pytest.raises(ValueError, match="at least 4 frames"):
        load_calibrated_views(_write_manifest(tmp_path, frames), mesh_scale=1.0)


def _load_single_view(tmp_path, *, color, mode="RGBA", file_path="images/view.png"):
    image_path = tmp_path / file_path
    image_path.parent.mkdir(parents=True, exist_ok=True)
    _write_image(image_path, color, mode=mode)
    frame = {
        "file_path": file_path,
        "camera_angle_x": 0.7,
        "transform_matrix": _transform(),
    }
    views, _ = load_calibrated_views(
        _write_manifest(tmp_path, [frame]), mesh_scale=1.0, num_views=1
    )
    return views[0], frame


def test_mask_explicit_path_has_priority_and_hashes_original_mask_file(tmp_path):
    view, frame = _load_single_view(tmp_path, color=(10, 20, 30, 0))
    mask_path = tmp_path / "masks" / "foreground.png"
    mask_path.parent.mkdir()
    Image.new("L", view.image.size, 0).save(mask_path)
    with Image.open(mask_path) as source:
        mask_image = source.copy()
    mask_image.putpixel((1, 1), 255)
    mask_image.save(mask_path)
    frame["foreground_mask_path"] = "masks/foreground.png"

    def forbidden_provider(_image):
        raise AssertionError("explicit masks must not invoke rembg")

    foreground = resolve_foreground_mask(
        view, frame, rembg_provider=forbidden_provider
    )

    assert foreground.provenance == "explicit"
    assert foreground.mask.dtype == torch.bool
    assert foreground.mask.shape == (3, 4)
    assert foreground.mask.sum().item() == 1
    assert foreground.source_sha256 == sha256(mask_path.read_bytes()).hexdigest()


def test_mask_explicit_path_cannot_escape_manifest_directory(tmp_path):
    view, frame = _load_single_view(tmp_path, color=(10, 20, 30, 255))
    outside = tmp_path.parent / f"{tmp_path.name}-mask.png"
    Image.new("L", view.image.size, 255).save(outside)
    frame["foreground_mask_path"] = f"../{outside.name}"

    with pytest.raises(ValueError, match="inside the manifest directory"):
        resolve_foreground_mask(view, frame, rembg_provider=lambda image: image)


def test_mask_rejects_empty_explicit_source_without_falling_back(tmp_path):
    view, frame = _load_single_view(tmp_path, color=(10, 20, 30, 255))
    mask_path = tmp_path / "empty.png"
    Image.new("L", view.image.size, 0).save(mask_path)
    frame["foreground_mask_path"] = mask_path.name

    def forbidden_provider(_image):
        raise AssertionError("an explicit mask must not fall through to rembg")

    with pytest.raises(ValueError, match="foreground mask is empty"):
        resolve_foreground_mask(view, frame, rembg_provider=forbidden_provider)


def test_mask_uses_meaningful_rgba_alpha_and_source_image_hash(tmp_path):
    view, frame = _load_single_view(tmp_path, color=(10, 20, 30, 0))
    image = view.image.copy()
    image.putpixel((2, 1), (10, 20, 30, 128))
    image.save(view.image_path)
    views, _ = load_calibrated_views(
        tmp_path / "transforms.json", mesh_scale=1.0, num_views=1
    )
    view = views[0]

    def forbidden_provider(_image):
        raise AssertionError("meaningful alpha must not invoke rembg")

    foreground = resolve_foreground_mask(
        view, frame, rembg_provider=forbidden_provider
    )

    assert foreground.provenance == "alpha"
    assert foreground.mask.sum().item() == 1
    assert foreground.mask[1, 2]
    assert foreground.source_sha256 == sha256(view.image_path.read_bytes()).hexdigest()
    assert foreground.source_sha256 == view.source_sha256


def test_mask_rejects_empty_alpha_without_invoking_rembg(tmp_path):
    view, frame = _load_single_view(tmp_path, color=(10, 20, 30, 0))

    def forbidden_provider(_image):
        raise AssertionError("empty alpha must fail before rembg")

    with pytest.raises(ValueError, match="foreground mask is empty"):
        resolve_foreground_mask(view, frame, rembg_provider=forbidden_provider)


@pytest.mark.parametrize(
    ("mode", "color"),
    [("RGB", (10, 20, 30)), ("RGBA", (10, 20, 30, 255))],
)
def test_mask_opaque_images_invoke_injected_rembg_once_without_mutating_view(
    tmp_path, mode, color
):
    view, frame = _load_single_view(tmp_path, color=color, mode=mode)
    original_mode = view.image.mode
    original_bytes = view.image.tobytes()
    calls = []
    input_modes = []

    def provider(image):
        calls.append(image)
        input_modes.append(image.mode)
        image.putalpha(Image.new("L", image.size, 0))
        image.putpixel((3, 2), (*image.getpixel((3, 2))[:3], 255))
        return image

    foreground = resolve_foreground_mask(view, frame, rembg_provider=provider)

    assert len(calls) == 1
    assert input_modes == ["RGB"]
    assert foreground.provenance == "rembg"
    assert foreground.mask.sum().item() == 1
    assert foreground.mask[2, 3]
    assert foreground.source_sha256 == view.source_sha256
    assert view.image.mode == original_mode
    assert view.image.tobytes() == original_bytes


def test_mask_rejects_empty_rembg_result(tmp_path):
    view, frame = _load_single_view(tmp_path, color=(10, 20, 30), mode="RGB")

    def provider(image):
        image.putalpha(Image.new("L", image.size, 0))
        return image

    with pytest.raises(ValueError, match="foreground mask is empty"):
        resolve_foreground_mask(view, frame, rembg_provider=provider)
