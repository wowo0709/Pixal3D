"""Pure CPU tensor generators for controlled local image corruptions."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torchvision.transforms import functional as tv_functional


@dataclass(frozen=True)
class ControlledCorruption:
    image: torch.Tensor
    oracle_mask: torch.Tensor
    kind: str
    parameters: dict


def _validate_mask(mask: torch.Tensor, *, name: str) -> None:
    if not isinstance(mask, torch.Tensor) or mask.ndim != 2 or mask.dtype != torch.bool:
        raise ValueError(f"{name} must be a bool [H,W] tensor")


def _validate_corruption_inputs(
    image: torch.Tensor,
    foreground: torch.Tensor,
    region: torch.Tensor,
) -> None:
    if (
        not isinstance(image, torch.Tensor)
        or image.ndim != 3
        or image.shape[0] != 3
        or image.dtype != torch.float32
    ):
        raise ValueError("image must be a float32 [3,H,W] tensor")
    if not torch.isfinite(image).all() or image.min() < 0 or image.max() > 1:
        raise ValueError("image values must be finite and in [0,1]")
    _validate_mask(foreground, name="foreground")
    _validate_mask(region, name="region")
    if image.shape[1:] != foreground.shape or region.shape != foreground.shape:
        raise ValueError("image, foreground, and region spatial shapes must match")
    if image.device != foreground.device or region.device != foreground.device:
        raise ValueError("image, foreground, and region must share a device")
    if (region & ~foreground).any():
        raise ValueError("region must be inside foreground")


def sample_foreground_region(
    foreground: torch.Tensor,
    *,
    seed: int,
    area_fraction: float = 0.12,
) -> torch.Tensor:
    """Select a deterministic rectangular subset of a foreground mask."""
    _validate_mask(foreground, name="foreground")
    if not math.isfinite(area_fraction) or not 0.0 < area_fraction <= 1.0:
        raise ValueError("area_fraction must be finite and in (0, 1]")

    coordinates = foreground.nonzero(as_tuple=False).cpu()
    if coordinates.shape[0] == 0:
        raise ValueError("foreground is empty")
    if coordinates.shape[0] < 2:
        raise ValueError("foreground is too small")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    center_index = torch.randint(
        coordinates.shape[0], (1,), generator=generator
    ).item()
    center_y, center_x = coordinates[center_index].tolist()
    target_area = max(1, math.ceil(coordinates.shape[0] * area_fraction))
    side = max(1, math.ceil(math.sqrt(target_area)))

    height, width = foreground.shape
    top = max(0, min(height - side, center_y - side // 2))
    left = max(0, min(width - side, center_x - side // 2))
    rectangle = torch.zeros_like(foreground)
    rectangle[top : min(top + side, height), left : min(left + side, width)] = True
    region = rectangle & foreground
    if not region.any():  # Defensive: the selected foreground center must survive.
        raise ValueError("unable to sample a nonempty foreground region")
    return region


def corrupt_local_color(
    image: torch.Tensor,
    foreground: torch.Tensor,
    region: torch.Tensor,
    *,
    hue: float,
    saturation: float,
    brightness: float,
) -> ControlledCorruption:
    """Apply torchvision color adjustments only inside ``region``."""
    _validate_corruption_inputs(image, foreground, region)
    if not math.isfinite(hue) or not -0.5 <= hue <= 0.5:
        raise ValueError("hue must be finite and in [-0.5, 0.5]")
    if not math.isfinite(saturation) or saturation < 0:
        raise ValueError("saturation must be finite and nonnegative")
    if not math.isfinite(brightness) or brightness < 0:
        raise ValueError("brightness must be finite and nonnegative")

    adjusted = tv_functional.adjust_hue(image, hue)
    adjusted = tv_functional.adjust_saturation(adjusted, saturation)
    adjusted = tv_functional.adjust_brightness(adjusted, brightness)
    output = image.clone()
    output[:, region] = adjusted[:, region]
    output.clamp_(0.0, 1.0)
    oracle = region & (output - image).abs().gt(1.0 / 255.0).any(dim=0)
    return ControlledCorruption(
        image=output,
        oracle_mask=oracle,
        kind="c1_color",
        parameters={
            "hue": hue,
            "saturation": saturation,
            "brightness": brightness,
        },
    )


def corrupt_procedural_pattern(
    image: torch.Tensor,
    foreground: torch.Tensor,
    region: torch.Tensor,
    *,
    seed: int,
    pattern: str = "sole",
) -> ControlledCorruption:
    """Alpha-composite a deterministic asset-free pattern within ``region``."""
    _validate_corruption_inputs(image, foreground, region)
    if pattern not in {"sole", "stripes", "logo"}:
        raise ValueError("pattern must be one of: sole, stripes, logo")
    if not region.any():
        raise ValueError("region is empty")

    coordinates = region.nonzero(as_tuple=False)
    top, left = coordinates.amin(dim=0).tolist()
    bottom, right = coordinates.amax(dim=0).tolist()
    height = bottom - top + 1
    width = right - left + 1
    y = torch.arange(height, device=image.device).view(height, 1)
    x = torch.arange(width, device=image.device).view(1, width)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    phase = int(torch.randint(0, 12, (1,), generator=generator).item())
    if pattern == "sole":
        local_mark = (((x + phase) // 3 + y // 4) % 2 == 0) & (
            (x + 2 * y + phase) % 5 != 0
        )
    elif pattern == "stripes":
        local_mark = (x + 2 * y + phase) % 7 < 3
    else:
        normalized_x = (x - (width - 1) / 2) / max(width, 1)
        normalized_y = (y - (height - 1) / 2) / max(height, 1)
        radius = torch.sqrt(normalized_x.square() + normalized_y.square())
        ring = (radius > 0.22) & (radius < 0.36)
        slash = (normalized_x + normalized_y).abs() < 0.055
        local_mark = ring | slash

    mark = torch.zeros_like(region)
    mark[top : bottom + 1, left : right + 1] = local_mark
    mark &= region
    color = 0.2 + 0.6 * torch.rand((3, 1), generator=generator)
    color = color.to(device=image.device, dtype=image.dtype)
    alpha = 0.30
    output = image.clone()
    output[:, mark] = (1.0 - alpha) * image[:, mark] + alpha * color
    output.clamp_(0.0, 1.0)
    oracle = region & (output - image).abs().gt(1.0 / 255.0).any(dim=0)
    return ControlledCorruption(
        image=output,
        oracle_mask=oracle,
        kind="c2_pattern",
        parameters={"seed": seed, "pattern": pattern},
    )


def corrupt_local_deletion(
    image: torch.Tensor,
    foreground: torch.Tensor,
    region: torch.Tensor,
) -> ControlledCorruption:
    """Replace a region with a robust background or border color estimate."""
    _validate_corruption_inputs(image, foreground, region)
    background = ~foreground
    if background.any():
        samples = image[:, background]
        fill_source = "background"
    else:
        border = torch.zeros_like(foreground)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        samples = image[:, border]
        fill_source = "border"

    fill = samples.median(dim=1).values
    output = image.clone()
    output[:, region] = fill.view(3, 1)
    oracle = region & (output - image).abs().gt(1.0 / 255.0).any(dim=0)
    return ControlledCorruption(
        image=output,
        oracle_mask=oracle,
        kind="c3_deletion",
        parameters={
            "fill_source": fill_source,
            "fill_rgb": fill.tolist(),
        },
    )
