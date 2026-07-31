"""Read-only projection diagnostics for multi-view correspondence experiments."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class MultiviewProjection:
    """Projected point diagnostics for every batch item and input view."""

    pixel_coordinates: torch.Tensor
    normalized_coordinates: torch.Tensor
    depth: torch.Tensor
    valid_mask: torch.Tensor


def _require_tensor(value: torch.Tensor, *, name: str) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")


def _validate_floating_finite(value: torch.Tensor, *, name: str) -> None:
    if not value.is_floating_point():
        raise ValueError(f"{name} must be floating point")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must be finite")


def project_points_for_views(
    points_3d: torch.Tensor,
    projection_transforms: torch.Tensor,
    camera_angle_x: torch.Tensor,
    *,
    resolution: int,
) -> MultiviewProjection:
    """Project shared or batched points independently into each camera view."""
    _require_tensor(points_3d, name="points_3d")
    _require_tensor(projection_transforms, name="projection_transforms")
    _require_tensor(camera_angle_x, name="camera_angle_x")
    _validate_floating_finite(points_3d, name="points_3d")
    _validate_floating_finite(
        projection_transforms, name="projection_transforms"
    )
    _validate_floating_finite(camera_angle_x, name="camera_angle_x")
    if points_3d.ndim not in (2, 3) or points_3d.shape[-1] != 3:
        raise ValueError("points_3d must have shape [N, 3] or [B, N, 3]")
    if projection_transforms.ndim != 4 or projection_transforms.shape[-2:] != (4, 4):
        raise ValueError("projection_transforms must have shape [B, K, 4, 4]")
    batch_size, view_count = projection_transforms.shape[:2]
    if camera_angle_x.shape != (batch_size, view_count):
        raise ValueError("camera_angle_x must have shape [B, K]")
    if not (camera_angle_x > 0).all():
        raise ValueError("camera_angle_x FOV must be positive")
    if isinstance(resolution, bool) or not isinstance(resolution, int) or resolution < 1:
        raise ValueError("resolution must be an integer at least 1")
    if points_3d.ndim == 3 and points_3d.shape[0] != batch_size:
        raise ValueError("batched points_3d must have matching B")

    # This import must remain local: the existing projector imports correspondence
    # aggregation during its module initialization.
    from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import (
        project_points_to_image_batch,
    )

    point_count = points_3d.shape[-2]
    if points_3d.ndim == 2:
        points_per_view = points_3d.unsqueeze(0).expand(
            batch_size * view_count, -1, -1
        )
    else:
        points_per_view = (
            points_3d.unsqueeze(1)
            .expand(-1, view_count, -1, -1)
            .reshape(batch_size * view_count, point_count, 3)
        )
    pixels, depth, valid_mask = project_points_to_image_batch(
        points_per_view,
        projection_transforms.reshape(batch_size * view_count, 4, 4),
        camera_angle_x.reshape(batch_size * view_count),
        resolution=resolution,
    )
    pixels = pixels.reshape(batch_size, view_count, point_count, 2)
    depth = depth.reshape(batch_size, view_count, point_count)
    valid_mask = valid_mask.reshape(batch_size, view_count, point_count)
    normalized = (pixels + 0.5) / resolution * 2.0 - 1.0
    return MultiviewProjection(
        pixel_coordinates=pixels,
        normalized_coordinates=normalized,
        depth=depth,
        valid_mask=valid_mask,
    )


def _validate_mask_values(values: torch.Tensor, *, name: str) -> None:
    _require_tensor(values, name=name)
    if not values.is_floating_point():
        raise TypeError(f"{name} must be floating point")
    if not torch.isfinite(values).all():
        raise ValueError(f"{name} must be finite")
    if (values < 0).any() or (values > 1).any():
        raise ValueError(f"{name} values must be between 0 and 1")


def sample_oracle_masks(
    masks: torch.Tensor,
    normalized_coordinates: torch.Tensor,
) -> torch.Tensor:
    """Bilinearly sample per-view oracle masks at normalized point coordinates."""
    _validate_mask_values(masks, name="masks")
    _require_tensor(normalized_coordinates, name="normalized_coordinates")
    if masks.ndim != 4:
        raise ValueError("masks must have shape [B, K, H, W]")
    if normalized_coordinates.ndim != 4 or normalized_coordinates.shape[-1] != 2:
        raise ValueError("normalized_coordinates must have shape [B, K, N, 2]")
    if normalized_coordinates.shape[:2] != masks.shape[:2]:
        raise ValueError("masks and normalized_coordinates must have matching [B, K]")
    if not normalized_coordinates.is_floating_point():
        raise TypeError("normalized_coordinates must be floating point")
    if not torch.isfinite(normalized_coordinates).all():
        raise ValueError("normalized_coordinates must be finite")
    if normalized_coordinates.device != masks.device:
        raise ValueError("masks and normalized_coordinates must share a device")

    batch_size, view_count, height, width = masks.shape
    point_count = normalized_coordinates.shape[2]
    sampled = F.grid_sample(
        masks.reshape(batch_size * view_count, 1, height, width),
        normalized_coordinates.to(dtype=masks.dtype).reshape(
            batch_size * view_count, point_count, 1, 2
        ),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    return sampled.reshape(batch_size, view_count, point_count)


def oracle_reliability(sampled_masks: torch.Tensor) -> torch.Tensor:
    """Return an oracle reliability score where zero mask means fully reliable."""
    _validate_mask_values(sampled_masks, name="sampled_masks")
    return 1.0 - sampled_masks
