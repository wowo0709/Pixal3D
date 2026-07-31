"""Pure tensor metrics for multi-view correspondence experiments."""

import math

import torch
import torch.nn.functional as F


def _validate_features(features: torch.Tensor, *, name: str) -> None:
    if not isinstance(features, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if features.ndim != 3:
        raise ValueError(f"{name} must have shape [B, N, C]")
    if not features.is_floating_point():
        raise TypeError(f"{name} must be floating point")
    if not torch.isfinite(features).all():
        raise ValueError(f"{name} must be finite")


def _selected_mask(
    mask: torch.Tensor | None,
    *,
    batch_size: int,
    voxel_count: int,
    device: torch.device,
    name: str = "mask",
) -> torch.Tensor:
    if mask is None:
        selected = torch.ones((batch_size, voxel_count), dtype=torch.bool, device=device)
    else:
        if not isinstance(mask, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if mask.dtype != torch.bool:
            raise TypeError(f"{name} must be boolean")
        if mask.shape != (batch_size, voxel_count):
            raise ValueError(f"{name} must have shape [B, N]")
        if mask.device != device:
            raise ValueError(f"{name} must be on the same device as the input")
        selected = mask
    if not selected.any():
        raise ValueError(f"{name} must select at least one voxel")
    return selected


def _validate_feature_pair(
    candidate: torch.Tensor,
    reference: torch.Tensor,
    mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _validate_features(candidate, name="candidate")
    _validate_features(reference, name="reference")
    if candidate.shape != reference.shape:
        raise ValueError("candidate and reference must have the same shape")
    selected = _selected_mask(
        mask,
        batch_size=candidate.shape[0],
        voxel_count=candidate.shape[1],
        device=candidate.device,
    )
    return candidate.to(torch.float32), reference.to(torch.float32), selected


def feature_cosine_error(
    candidate: torch.Tensor,
    reference: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return mean cosine error across the selected feature voxels."""
    candidate_fp32, reference_fp32, selected = _validate_feature_pair(
        candidate, reference, mask
    )
    errors = 1.0 - F.cosine_similarity(candidate_fp32, reference_fp32, dim=-1)
    return errors[selected].mean(dtype=torch.float32)


def feature_l2_drift(
    candidate: torch.Tensor,
    reference: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return mean per-voxel Euclidean drift across selected feature voxels."""
    candidate_fp32, reference_fp32, selected = _validate_feature_pair(
        candidate, reference, mask
    )
    drift = torch.linalg.vector_norm(candidate_fp32 - reference_fp32, dim=-1)
    return drift[selected].mean(dtype=torch.float32)


def branch_norms(features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return mean L2 norms for the low- and high-channel feature branches."""
    _validate_features(features, name="features")
    channel_count = features.shape[-1]
    if channel_count == 0 or channel_count % 2:
        raise ValueError("features must have a nonzero even channel count")
    features_fp32 = features.to(torch.float32)
    split = channel_count // 2
    return (
        torch.linalg.vector_norm(features_fp32[..., :split], dim=-1).mean(
            dtype=torch.float32
        ),
        torch.linalg.vector_norm(features_fp32[..., split:], dim=-1).mean(
            dtype=torch.float32
        ),
    )


def weight_diagnostics(
    weights: torch.Tensor,
    *,
    active_mask: torch.Tensor | None = None,
    corrupted_region_mask: torch.Tensor | None = None,
    corrupted_view_index: int | None = None,
) -> dict[str, torch.Tensor]:
    """Summarize normalized view weights over active feature voxels."""
    if not isinstance(weights, torch.Tensor):
        raise TypeError("weights must be a torch.Tensor")
    if weights.ndim != 3:
        raise ValueError("weights must have shape [B, K, N]")
    if not weights.is_floating_point():
        raise TypeError("weights must be floating point")
    if not torch.isfinite(weights).all():
        raise ValueError("weights must be finite")
    batch_size, view_count, voxel_count = weights.shape
    if view_count < 1:
        raise ValueError("weights must include at least one view")

    weights_fp32 = weights.to(torch.float32)
    if (weights_fp32 < 0).any():
        raise ValueError("weights must be non-negative")
    if not torch.allclose(
        weights_fp32.sum(dim=1),
        torch.ones((batch_size, voxel_count), device=weights.device),
        atol=1e-5,
        rtol=1e-5,
    ):
        raise ValueError("weights must sum to one over views")

    active = _selected_mask(
        active_mask,
        batch_size=batch_size,
        voxel_count=voxel_count,
        device=weights.device,
        name="active_mask",
    )
    selected_weights = weights_fp32.permute(0, 2, 1)[active]
    entropy_per_voxel = -(
        selected_weights
        * torch.log(selected_weights.clamp_min(torch.finfo(torch.float32).tiny))
    ).sum(dim=-1)
    entropy = entropy_per_voxel.mean(dtype=torch.float32)
    normalized_entropy = (
        torch.zeros((), dtype=torch.float32, device=weights.device)
        if view_count == 1
        else entropy / math.log(view_count)
    )
    uniform_weight = 1.0 / view_count
    diagnostics = {
        "entropy": entropy,
        "normalized_entropy": normalized_entropy,
        "mean_weight_per_view": selected_weights.mean(dim=0, dtype=torch.float32),
        "mean_abs_deviation_from_uniform": (
            (selected_weights - uniform_weight).abs().mean(dtype=torch.float32)
        ),
    }

    if (corrupted_region_mask is None) != (corrupted_view_index is None):
        raise ValueError(
            "corrupted_region_mask and corrupted_view_index must be supplied together"
        )
    if corrupted_region_mask is not None:
        if isinstance(corrupted_view_index, bool) or not isinstance(
            corrupted_view_index, int
        ):
            raise TypeError("corrupted_view_index must be an integer")
        if not 0 <= corrupted_view_index < view_count:
            raise ValueError("corrupted_view_index must identify an input view")
        corrupted = _selected_mask(
            corrupted_region_mask,
            batch_size=batch_size,
            voxel_count=voxel_count,
            device=weights.device,
            name="corrupted_region_mask",
        )
        inside = active & corrupted
        outside = active & ~corrupted
        if not inside.any() or not outside.any():
            raise ValueError(
                "corrupted_region_mask must include and exclude at least one active voxel"
            )
        corrupt_view_weights = weights_fp32[:, corrupted_view_index]
        diagnostics["corrupt_view_mass_inside_region"] = corrupt_view_weights[inside].mean(
            dtype=torch.float32
        )
        diagnostics["corrupt_view_mass_outside_region"] = corrupt_view_weights[
            outside
        ].mean(dtype=torch.float32)
    return diagnostics
