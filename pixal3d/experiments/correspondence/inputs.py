"""Validated calibrated inputs for correspondence experiments."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


@dataclass(frozen=True)
class CalibratedView:
    frame_index: int
    manifest_root: Path
    frame_file_path: str
    image_path: Path
    image: Image.Image
    camera_angle_x: float
    distance: float
    transform_matrix: torch.Tensor
    source_sha256: str


@dataclass(frozen=True)
class ForegroundMask:
    mask: torch.Tensor
    provenance: str
    source_sha256: str


def _contained_file(root: Path, value: Any, *, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must name a file inside the manifest directory")
    path = (root / value).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"{field} must name a file inside the manifest directory")
    return path


def _finite_float(value: Any, *, field: str, frame_index: int) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be finite for frame {frame_index}") from exc
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite for frame {frame_index}")
    return result


def load_calibrated_views(
    path: Path, *, mesh_scale: float, num_views: int = 4
) -> tuple[list[CalibratedView], float]:
    """Load the first ordered calibrated views from a transforms manifest."""

    try:
        scale = float(mesh_scale)
    except (TypeError, ValueError) as exc:
        raise ValueError("mesh_scale must be finite and positive") from exc
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("mesh_scale must be finite and positive")
    if not isinstance(num_views, int) or isinstance(num_views, bool) or num_views <= 0:
        raise ValueError("num_views must be a positive integer")

    manifest_path = Path(path).resolve()
    if not manifest_path.is_file():
        raise ValueError("transforms path must name a manifest file")
    manifest_root = manifest_path.parent
    try:
        metadata = json.loads(manifest_path.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("transforms path must contain valid JSON") from exc
    if not isinstance(metadata, dict):
        raise ValueError("transforms manifest must be a JSON object")
    frames = metadata.get("frames")
    if not isinstance(frames, list) or len(frames) < num_views:
        raise ValueError(f"transforms manifest must contain at least {num_views} frames")

    views: list[CalibratedView] = []
    for frame_index, frame in enumerate(frames[:num_views]):
        if not isinstance(frame, dict):
            raise ValueError(f"frame {frame_index} must be a JSON object")
        frame_file_path = frame.get("file_path")
        image_path = _contained_file(
            manifest_root, frame_file_path, field="file_path"
        )
        camera_angle_x = _finite_float(
            frame.get("camera_angle_x", metadata.get("camera_angle_x")),
            field="camera_angle_x",
            frame_index=frame_index,
        )
        try:
            transform64 = torch.as_tensor(
                frame.get("transform_matrix"), dtype=torch.float64
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"transform_matrix must be finite [4, 4] for frame {frame_index}"
            ) from exc
        if transform64.shape != (4, 4) or not torch.isfinite(transform64).all():
            raise ValueError(
                f"transform_matrix must be finite [4, 4] for frame {frame_index}"
            )
        transform = transform64.to(dtype=torch.float32)
        if not torch.isfinite(transform).all():
            raise ValueError(
                f"transform_matrix must be finite [4, 4] for frame {frame_index}"
            )
        translation = transform64[:3, 3].tolist()
        distance = math.sqrt(math.fsum(value * value for value in translation))
        if not math.isfinite(distance):
            raise ValueError(f"distance must be finite for frame {frame_index}")

        raw = image_path.read_bytes()
        try:
            with Image.open(image_path) as source:
                image = source.copy()
        except (OSError, ValueError) as exc:
            raise ValueError(f"file_path must name a readable image for frame {frame_index}") from exc
        views.append(
            CalibratedView(
                frame_index=frame_index,
                manifest_root=manifest_root,
                frame_file_path=frame_file_path,
                image_path=image_path,
                image=image,
                camera_angle_x=camera_angle_x,
                distance=distance,
                transform_matrix=transform,
                source_sha256=sha256(raw).hexdigest(),
            )
        )
    return views, scale


def resolve_foreground_mask(
    view: CalibratedView, frame: dict, *, rembg_provider=None
) -> ForegroundMask:
    """Resolve an explicit, alpha-derived, or lazily segmented foreground."""

    if not isinstance(frame, dict):
        raise ValueError("frame must be a JSON object")

    if frame.get("file_path") != view.frame_file_path:
        raise ValueError("frame file_path does not match loaded view provenance")

    explicit_value = frame.get("foreground_mask_path")
    if explicit_value is not None:
        mask_path = _contained_file(
            view.manifest_root, explicit_value, field="foreground_mask_path"
        )
        raw = mask_path.read_bytes()
        try:
            with Image.open(mask_path) as source:
                mask_image = source.convert("L")
        except (OSError, ValueError) as exc:
            raise ValueError("foreground_mask_path must name a readable image") from exc
        mask = _mask_tensor(mask_image, expected_size=view.image.size)
        _require_nonempty(mask)
        return ForegroundMask(
            mask=mask,
            provenance="explicit",
            source_sha256=sha256(raw).hexdigest(),
        )

    if view.image.mode == "RGBA":
        alpha = view.image.getchannel("A")
        alpha_values = np.asarray(alpha)
        if not np.any(alpha_values):
            raise ValueError("foreground mask is empty")
        if np.any(alpha_values < 255):
            mask = _mask_tensor(alpha, expected_size=view.image.size)
            _require_nonempty(mask)
            return ForegroundMask(
                mask=mask,
                provenance="alpha",
                source_sha256=view.source_sha256,
            )

    provider = rembg_provider
    if provider is None:
        provider = _default_rembg_provider()
    segmented = provider(view.image.convert("RGB"))
    if not isinstance(segmented, Image.Image):
        raise ValueError("rembg provider must return a PIL image")
    if segmented.mode == "RGBA":
        rembg_mask = segmented.getchannel("A")
    elif segmented.mode == "L":
        rembg_mask = segmented
    else:
        raise ValueError("rembg provider must return an RGBA image or grayscale mask")
    mask = _mask_tensor(rembg_mask, expected_size=view.image.size)
    _require_nonempty(mask)
    return ForegroundMask(
        mask=mask,
        provenance="rembg",
        source_sha256=view.source_sha256,
    )


def _mask_tensor(image: Image.Image, *, expected_size: tuple[int, int]) -> torch.Tensor:
    if image.size != expected_size:
        raise ValueError("foreground mask dimensions must match the calibrated image")
    return torch.from_numpy(np.asarray(image, dtype=np.uint8).copy()) > 0


def _require_nonempty(mask: torch.Tensor) -> None:
    if not mask.any():
        raise ValueError("foreground mask is empty")


def _default_rembg_provider():
    from pixal3d.pipelines.rembg import BiRefNet

    provider = BiRefNet()
    provider.cuda()
    return provider
