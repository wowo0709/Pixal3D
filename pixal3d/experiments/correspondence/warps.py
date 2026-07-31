"""Controlled image warps with explicit forward-field geometry."""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torchvision.transforms.functional import gaussian_blur


@dataclass(frozen=True)
class ControlledWarp:
    """An image artifact paired with its forward oracle geometry."""

    warped_image: torch.Tensor
    affected_mask: torch.Tensor
    forward_field_px: torch.Tensor
    inverse_grid_norm: torch.Tensor
    invalid_mask: torch.Tensor


def _validate_points(points: torch.Tensor) -> None:
    if not isinstance(points, torch.Tensor):
        raise TypeError("points must be a torch.Tensor")
    if not points.is_floating_point():
        raise TypeError("points must be floating point")
    if points.ndim < 1 or points.shape[-1] != 2:
        raise ValueError("points must have shape [..., 2]")


def _validate_size(value: int, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _validate_floating_finite(value: torch.Tensor, *, name: str) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not value.is_floating_point():
        raise TypeError(f"{name} must be floating point")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must be finite")


def _pixel_mesh(
    *,
    height: int,
    width: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    y, x = torch.meshgrid(
        torch.arange(height, dtype=dtype, device=device),
        torch.arange(width, dtype=dtype, device=device),
        indexing="ij",
    )
    return torch.stack((x, y), dim=-1)


def _validate_affine_matrix(matrix: torch.Tensor) -> None:
    _validate_floating_finite(matrix, name="matrix")
    if matrix.shape != (2, 3):
        raise ValueError("matrix must have shape [2, 3]")


def pixel_to_normalized(
    points_px: torch.Tensor,
    *,
    height: int,
    width: int,
) -> torch.Tensor:
    """Convert pixel-center coordinates to ``align_corners=False`` coordinates."""
    _validate_points(points_px)
    _validate_size(height, name="height")
    _validate_size(width, name="width")
    x_norm = 2.0 * (points_px[..., 0] + 0.5) / width - 1.0
    y_norm = 2.0 * (points_px[..., 1] + 0.5) / height - 1.0
    return torch.stack((x_norm, y_norm), dim=-1)


def normalized_to_pixel(
    points_norm: torch.Tensor,
    *,
    height: int,
    width: int,
) -> torch.Tensor:
    """Convert ``align_corners=False`` coordinates to pixel centers."""
    _validate_points(points_norm)
    _validate_size(height, name="height")
    _validate_size(width, name="width")
    x_px = (points_norm[..., 0] + 1.0) * width / 2.0 - 0.5
    y_px = (points_norm[..., 1] + 1.0) * height / 2.0 - 0.5
    return torch.stack((x_px, y_px), dim=-1)


def affine_forward_field(
    matrix: torch.Tensor,
    *,
    height: int,
    width: int,
) -> torch.Tensor:
    """Evaluate a clean-to-destination affine displacement field."""
    _validate_affine_matrix(matrix)
    _validate_size(height, name="height")
    _validate_size(width, name="width")
    clean_points = _pixel_mesh(
        height=height,
        width=width,
        dtype=matrix.dtype,
        device=matrix.device,
    )
    homogeneous = torch.cat(
        (clean_points, torch.ones_like(clean_points[..., :1])),
        dim=-1,
    )
    destination_points = homogeneous @ matrix.transpose(0, 1)
    return destination_points - clean_points


def affine_inverse_grid(
    matrix: torch.Tensor,
    *,
    height: int,
    width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the exact matrix-derived destination-to-source sampling grid."""
    _validate_affine_matrix(matrix)
    _validate_size(height, name="height")
    _validate_size(width, name="width")
    matrix_fp32 = matrix.to(dtype=torch.float32)
    homogeneous_matrix = torch.cat(
        (
            matrix_fp32,
            matrix_fp32.new_tensor([[0.0, 0.0, 1.0]]),
        ),
        dim=0,
    )
    try:
        inverse_matrix = torch.linalg.inv(homogeneous_matrix)
    except RuntimeError as error:
        raise ValueError("matrix must not be singular") from error

    destination_points = _pixel_mesh(
        height=height,
        width=width,
        dtype=torch.float32,
        device=matrix.device,
    )
    homogeneous_destination = torch.cat(
        (
            destination_points,
            torch.ones_like(destination_points[..., :1]),
        ),
        dim=-1,
    )
    homogeneous_source = (
        homogeneous_destination @ inverse_matrix.transpose(0, 1)
    )
    source_points = (
        homogeneous_source[..., :2] / homogeneous_source[..., 2:].clone()
    )
    invalid_mask = _outside_image(
        source_points,
        height=height,
        width=width,
    )
    return (
        pixel_to_normalized(source_points, height=height, width=width),
        invalid_mask,
    )


def _outside_image(
    points_px: torch.Tensor,
    *,
    height: int,
    width: int,
) -> torch.Tensor:
    return (
        (points_px[..., 0] < 0)
        | (points_px[..., 0] > width - 1)
        | (points_px[..., 1] < 0)
        | (points_px[..., 1] > height - 1)
    )


def _sample_forward_field(
    forward_field_px: torch.Tensor,
    source_points_px: torch.Tensor,
) -> torch.Tensor:
    height, width = forward_field_px.shape[:2]
    sampling_grid = pixel_to_normalized(
        source_points_px,
        height=height,
        width=width,
    )
    sampled = F.grid_sample(
        forward_field_px.permute(2, 0, 1).unsqueeze(0),
        sampling_grid.unsqueeze(0),
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )
    return sampled.squeeze(0).permute(1, 2, 0)


def invert_forward_field(
    forward_field_px: torch.Tensor,
    *,
    iterations: int = 12,
    convergence_tolerance_px: float = 1e-3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Numerically invert a forward field with border-extended fixed points."""
    _validate_floating_finite(forward_field_px, name="forward_field_px")
    if (
        forward_field_px.ndim != 3
        or forward_field_px.shape[-1] != 2
        or min(forward_field_px.shape[:2]) < 1
    ):
        raise ValueError("forward_field_px must have shape [H, W, 2]")
    if (
        isinstance(iterations, bool)
        or not isinstance(iterations, int)
        or iterations < 1
    ):
        raise ValueError("iterations must be an integer at least 1")
    if (
        isinstance(convergence_tolerance_px, bool)
        or not isinstance(convergence_tolerance_px, (int, float))
        or not math.isfinite(convergence_tolerance_px)
        or convergence_tolerance_px <= 0
    ):
        raise ValueError("convergence tolerance must be finite and positive")

    height, width = forward_field_px.shape[:2]
    destination_points = _pixel_mesh(
        height=height,
        width=width,
        dtype=forward_field_px.dtype,
        device=forward_field_px.device,
    )
    source_points = destination_points
    for _ in range(iterations):
        sampled_field = _sample_forward_field(forward_field_px, source_points)
        source_points = destination_points - sampled_field

    final_field = _sample_forward_field(forward_field_px, source_points)
    residual = torch.linalg.vector_norm(
        source_points + final_field - destination_points,
        dim=-1,
    )
    invalid_mask = _outside_image(
        source_points,
        height=height,
        width=width,
    ) | (residual > convergence_tolerance_px)
    return (
        pixel_to_normalized(source_points, height=height, width=width),
        invalid_mask,
    )


def _validate_warp_inputs(
    image: torch.Tensor,
    affected_mask: torch.Tensor,
) -> tuple[int, int]:
    _validate_floating_finite(image, name="image")
    if image.ndim != 3 or min(image.shape) < 1:
        raise ValueError("image must have shape [C, H, W]")
    _, height, width = image.shape
    if not isinstance(affected_mask, torch.Tensor):
        raise TypeError("affected_mask must be a torch.Tensor")
    if affected_mask.dtype != torch.bool:
        raise TypeError("affected_mask must have bool dtype")
    if affected_mask.shape != (height, width):
        raise ValueError("affected_mask shape must match image [H, W]")
    if affected_mask.device != image.device:
        raise ValueError("image and affected_mask must share a device")
    return height, width


def _make_controlled_warp(
    image: torch.Tensor,
    affected_mask: torch.Tensor,
    forward_field_px: torch.Tensor,
    inverse_grid_norm: torch.Tensor,
    invalid_mask: torch.Tensor,
) -> ControlledWarp:
    sampling_grid = inverse_grid_norm.to(dtype=image.dtype)
    sampled = F.grid_sample(
        image.unsqueeze(0),
        sampling_grid.unsqueeze(0),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    ).squeeze(0)
    warped = torch.where(affected_mask.unsqueeze(0), sampled, image)
    affected_invalid = affected_mask & invalid_mask
    warped = torch.where(
        affected_invalid.unsqueeze(0),
        torch.zeros_like(warped),
        warped,
    )
    return ControlledWarp(
        warped_image=warped,
        affected_mask=affected_mask,
        forward_field_px=forward_field_px,
        inverse_grid_norm=inverse_grid_norm,
        invalid_mask=invalid_mask,
    )


def warp_affine(
    image: torch.Tensor,
    affected_mask: torch.Tensor,
    matrix: torch.Tensor,
) -> ControlledWarp:
    """Warp an image with an exact affine inverse and a forward oracle field."""
    height, width = _validate_warp_inputs(image, affected_mask)
    _validate_affine_matrix(matrix)
    if matrix.device != image.device:
        raise ValueError("image and matrix must share a device")
    forward_field = affine_forward_field(matrix, height=height, width=width)
    inverse_grid, invalid_mask = affine_inverse_grid(
        matrix,
        height=height,
        width=width,
    )
    return _make_controlled_warp(
        image,
        affected_mask,
        forward_field,
        inverse_grid,
        invalid_mask,
    )


def warp_with_forward_field(
    image: torch.Tensor,
    affected_mask: torch.Tensor,
    forward_field_px: torch.Tensor,
    *,
    inverse_iterations: int = 12,
    convergence_tolerance_px: float = 1e-3,
) -> ControlledWarp:
    """Warp an affected destination region from a forward oracle field."""
    height, width = _validate_warp_inputs(image, affected_mask)
    _validate_floating_finite(forward_field_px, name="forward_field_px")
    if forward_field_px.shape != (height, width, 2):
        raise ValueError("forward_field_px shape must be [H, W, 2]")
    if forward_field_px.device != image.device:
        raise ValueError("image and forward_field_px must share a device")
    inverse_grid, invalid_mask = invert_forward_field(
        forward_field_px,
        iterations=inverse_iterations,
        convergence_tolerance_px=convergence_tolerance_px,
    )
    return _make_controlled_warp(
        image,
        affected_mask,
        forward_field_px,
        inverse_grid,
        invalid_mask,
    )


def smooth_random_forward_field(
    affected_mask: torch.Tensor,
    *,
    seed: int,
    max_displacement_px: float,
    coarse_size: int = 5,
    blur_kernel_size: int = 9,
    max_gradient: float = 0.5,
) -> torch.Tensor:
    """Generate a deterministic masked field with global magnitude bounds."""
    if not isinstance(affected_mask, torch.Tensor):
        raise TypeError("affected_mask must be a torch.Tensor")
    if affected_mask.dtype != torch.bool:
        raise TypeError("affected_mask must have bool dtype")
    if affected_mask.ndim != 2 or min(affected_mask.shape) < 1:
        raise ValueError("affected_mask must have shape [H, W]")
    if not affected_mask.any():
        raise ValueError("affected_mask must not be empty")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if (
        isinstance(max_displacement_px, bool)
        or not isinstance(max_displacement_px, (int, float))
        or not math.isfinite(max_displacement_px)
        or max_displacement_px < 0
    ):
        raise ValueError("max_displacement_px must be finite and non-negative")
    if (
        isinstance(coarse_size, bool)
        or not isinstance(coarse_size, int)
        or coarse_size < 2
    ):
        raise ValueError("coarse_size must be an integer at least 2")
    if (
        isinstance(blur_kernel_size, bool)
        or not isinstance(blur_kernel_size, int)
        or blur_kernel_size < 1
        or blur_kernel_size % 2 == 0
    ):
        raise ValueError("blur_kernel_size must be a positive odd integer")
    if (
        isinstance(max_gradient, bool)
        or not isinstance(max_gradient, (int, float))
        or not math.isfinite(max_gradient)
        or max_gradient <= 0
    ):
        raise ValueError("max_gradient must be finite and positive")

    height, width = affected_mask.shape
    if max_displacement_px == 0:
        return torch.zeros(
            (height, width, 2),
            dtype=torch.float32,
            device=affected_mask.device,
        )

    generator = torch.Generator(device="cpu").manual_seed(seed)
    coarse_field = torch.randn(
        (1, 2, coarse_size, coarse_size),
        dtype=torch.float32,
        generator=generator,
        device="cpu",
    )
    resized_field = F.interpolate(
        coarse_field,
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )
    smoothed_field = gaussian_blur(
        resized_field,
        kernel_size=[blur_kernel_size, blur_kernel_size],
    )
    field = smoothed_field.squeeze(0).permute(1, 2, 0).to(
        device=affected_mask.device
    )
    field = field * affected_mask.unsqueeze(-1)

    max_magnitude = torch.linalg.vector_norm(field, dim=-1).amax()
    max_neighbor_gradient = field.new_zeros(())
    if width > 1:
        max_neighbor_gradient = torch.maximum(
            max_neighbor_gradient,
            torch.linalg.vector_norm(
                field[:, 1:] - field[:, :-1],
                dim=-1,
            ).amax(),
        )
    if height > 1:
        max_neighbor_gradient = torch.maximum(
            max_neighbor_gradient,
            torch.linalg.vector_norm(
                field[1:] - field[:-1],
                dim=-1,
            ).amax(),
        )
    denominator_floor = torch.finfo(field.dtype).tiny
    displacement_scale = max_displacement_px / max_magnitude.clamp_min(
        denominator_floor
    )
    gradient_scale = max_gradient / max_neighbor_gradient.clamp_min(
        denominator_floor
    )
    global_scale = torch.minimum(
        field.new_tensor(1.0),
        torch.minimum(displacement_scale, gradient_scale),
    )
    return field * global_scale
