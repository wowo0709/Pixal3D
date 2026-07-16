import json
import math
from numbers import Real
from pathlib import Path

import numpy as np
from PIL import Image


class ValidationError(ValueError):
    pass


def _load_json(path: Path, description: str):
    try:
        return json.loads(path.read_text())
    except Exception as error:
        raise ValidationError(
            f"invalid {description}: {path}: {error}"
        ) from error


def _load_npz(path: Path, keys: tuple[str, ...], description: str):
    try:
        with np.load(path, allow_pickle=False) as data:
            return tuple(np.asarray(data[key]) for key in keys)
    except Exception as error:
        raise ValidationError(
            f"invalid {description}: {path}: {error}"
        ) from error


def _is_finite(array: np.ndarray) -> bool:
    try:
        return bool(np.isfinite(array).all())
    except (TypeError, ValueError):
        return False


def _is_finite_number(value) -> bool:
    return (
        isinstance(value, Real)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def validate_render_dir(
    path: Path, expected_views: int, resolution: int
) -> None:
    path = Path(path)
    transforms = _load_json(path / "transforms.json", "transforms")
    if not isinstance(transforms, dict) or not isinstance(
        transforms.get("frames"), list
    ):
        raise ValidationError(f"invalid transforms: {path / 'transforms.json'}")

    frames = transforms["frames"]
    if len(frames) != expected_views:
        raise ValidationError(
            f"expected {expected_views} frames, found {len(frames)}"
        )

    expected_names = [f"{index:03d}.png" for index in range(expected_views)]
    image_names = sorted(
        item.name for item in path.glob("*.png") if item.is_file()
    )
    if len(image_names) != expected_views:
        raise ValidationError(
            f"expected {expected_views} render images, found {len(image_names)}"
        )
    if image_names != expected_names:
        raise ValidationError(f"invalid render image names: {image_names}")

    for index, (frame, image_name) in enumerate(zip(frames, expected_names)):
        if not isinstance(frame, dict) or frame.get("file_path") != image_name:
            raise ValidationError(f"invalid render frame {index}")

        angle = frame.get("camera_angle_x")
        if not _is_finite_number(angle):
            raise ValidationError(f"invalid camera angle for frame {index}")
        radius = frame.get("radius")
        if not _is_finite_number(radius) or radius <= 0:
            raise ValidationError(f"invalid camera radius for frame {index}")

        try:
            matrix = np.asarray(frame["transform_matrix"], np.float64)
        except Exception as error:
            raise ValidationError(f"invalid camera frame {index}: {error}") from error
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            raise ValidationError(f"invalid camera frame {index}")
        with np.errstate(over="ignore", invalid="ignore"):
            determinant = float(np.linalg.det(matrix))
        if not math.isfinite(determinant) or abs(determinant) < 1e-8:
            raise ValidationError(f"invalid camera frame {index}")

        image_path = path / image_name
        try:
            with Image.open(image_path) as image:
                image.load()
                image_format = image.format
                image_mode = image.mode
                image_size = image.size
                rgba = np.asarray(image) if image_mode == "RGBA" else None
        except Exception as error:
            raise ValidationError(
                f"invalid render image: {image_path}: {error}"
            ) from error
        if image_format != "PNG" or image_mode != "RGBA":
            raise ValidationError(f"render image must be PNG RGBA: {image_path}")
        if image_size != (resolution, resolution):
            raise ValidationError(
                f"invalid render image resolution: {image_path}: {image_size}"
            )

        alpha_fraction = float((rgba[..., 3] > 0).mean())
        if not 0.01 <= alpha_fraction <= 0.95:
            raise ValidationError(
                f"invalid alpha fraction: {alpha_fraction}: {image_path}"
            )


def validate_sparse_latent(
    path: Path, grid_resolution: int, max_tokens: int
) -> None:
    path = Path(path)
    feats, coords = _load_npz(path, ("feats", "coords"), "sparse latent")
    if (
        feats.ndim != 2
        or coords.ndim != 2
        or coords.shape[1] != 3
        or feats.shape[0] != coords.shape[0]
    ):
        raise ValidationError(f"shape mismatch: {path}")
    if len(coords) > max_tokens:
        raise ValidationError(f"token limit exceeded: {len(coords)}")
    if not _is_finite(feats):
        raise ValidationError(f"non-finite features: {path}")
    if not _is_finite(coords):
        raise ValidationError(f"non-finite coordinates: {path}")

    coordinate_dtype = coords.dtype
    if not (
        np.issubdtype(coordinate_dtype, np.integer)
        or np.issubdtype(coordinate_dtype, np.floating)
    ) or not np.equal(coords, np.trunc(coords)).all():
        raise ValidationError(f"coordinates must be integral: {path}")
    if (coords < 0).any() or (coords >= grid_resolution).any():
        raise ValidationError(f"coordinates outside grid: {path}")
    if len(np.unique(coords, axis=0)) != len(coords):
        raise ValidationError(f"coordinates must be unique: {path}")


def validate_ss_latent(path: Path) -> None:
    path = Path(path)
    (z,) = _load_npz(path, ("z",), "SS latent")
    if not _is_finite(z):
        raise ValidationError(f"non-finite SS latent: {path}")


def validate_scale(path: Path) -> None:
    path = Path(path)
    value = _load_json(path, "scale metadata")
    if (
        not isinstance(value, dict)
        or not value
        or not all(_is_finite_number(item) for item in value.values())
    ):
        raise ValidationError(f"invalid scale metadata: {path}")
