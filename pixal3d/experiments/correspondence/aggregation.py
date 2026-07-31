"""Pure tensor kernels for multi-view projection consensus."""

from dataclasses import dataclass
import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class ProjectionAggregationDiagnostics:
    scores: torch.Tensor
    weights: torch.Tensor
    entropy: torch.Tensor
    fallback_mask: torch.Tensor


def residual_to_uniform_weights(
    scores: torch.Tensor,
    *,
    alpha: float,
    temperature: float,
) -> ProjectionAggregationDiagnostics:
    """Turn agreement scores into residual-to-uniform view weights."""
    if scores.ndim != 3:
        raise ValueError("scores must have shape [B, K, N]")
    if not scores.is_floating_point():
        raise TypeError("scores must be floating point")
    if scores.shape[1] < 1:
        raise ValueError("scores must include at least one view")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")

    scores_fp32 = scores.to(dtype=torch.float32)
    view_count = scores.shape[1]
    uniform = torch.full_like(scores_fp32, 1.0 / view_count)
    fallback_mask = ~torch.isfinite(scores_fp32).all(dim=1)
    safe_scores = torch.where(
        fallback_mask.unsqueeze(1), torch.zeros_like(scores_fp32), scores_fp32
    )
    softmax_weights = torch.softmax(safe_scores / temperature, dim=1)
    weights = alpha * softmax_weights + (1.0 - alpha) * uniform
    weights = torch.where(fallback_mask.unsqueeze(1), uniform, weights)
    entropy = -(
        weights * torch.log(weights.clamp_min(torch.finfo(weights.dtype).tiny))
    ).sum(dim=1)
    return ProjectionAggregationDiagnostics(
        scores=scores_fp32,
        weights=weights,
        entropy=entropy,
        fallback_mask=fallback_mask,
    )


def aggregate_consensus_projection_naive(
    stacked_features: torch.Tensor,
    *,
    alpha: float,
    temperature: float,
) -> tuple[torch.Tensor, ProjectionAggregationDiagnostics]:
    """Aggregate view features using a full leave-one-out reference kernel."""
    if stacked_features.ndim != 4:
        raise ValueError("stacked_features must have shape [B, K, N, C]")
    if not stacked_features.is_floating_point():
        raise TypeError("stacked_features must be floating point")
    if not torch.isfinite(stacked_features).all():
        raise ValueError("stacked_features must be finite")
    batch_size, view_count, voxel_count, channel_count = stacked_features.shape
    if view_count < 1:
        raise ValueError("stacked_features must include at least one view")
    if channel_count % 2:
        raise ValueError("stacked_features must have an even channel count")

    if view_count == 1:
        diagnostics = residual_to_uniform_weights(
            torch.zeros(
                (batch_size, 1, voxel_count),
                dtype=torch.float32,
                device=stacked_features.device,
            ),
            alpha=alpha,
            temperature=temperature,
        )
        return stacked_features[:, 0], diagnostics

    feature_dim = channel_count // 2
    low_features, high_features = (
        stacked_features[..., :feature_dim].to(dtype=torch.float32),
        stacked_features[..., feature_dim:].to(dtype=torch.float32),
    )
    degenerate_view_mask = (
        (torch.linalg.vector_norm(low_features, dim=-1) <= 1e-6)
        | (torch.linalg.vector_norm(high_features, dim=-1) <= 1e-6)
    ).any(dim=1)
    normalized_low = F.normalize(low_features, dim=-1, eps=1e-6)
    normalized_high = F.normalize(high_features, dim=-1, eps=1e-6)

    low_leave_one_out = (
        normalized_low.sum(dim=1, keepdim=True) - normalized_low
    )
    high_leave_one_out = (
        normalized_high.sum(dim=1, keepdim=True) - normalized_high
    )
    degenerate_prototype_mask = (
        (torch.linalg.vector_norm(low_leave_one_out, dim=-1) <= 1e-6)
        | (torch.linalg.vector_norm(high_leave_one_out, dim=-1) <= 1e-6)
    ).any(dim=1)
    degeneracy_mask = degenerate_view_mask | degenerate_prototype_mask

    low_prototypes = F.normalize(
        low_leave_one_out / (view_count - 1),
        dim=-1,
        eps=1e-6,
    )
    high_prototypes = F.normalize(
        high_leave_one_out / (view_count - 1),
        dim=-1,
        eps=1e-6,
    )
    scores = (
        0.5 * (normalized_low * low_prototypes).sum(dim=-1)
        + 0.5 * (normalized_high * high_prototypes).sum(dim=-1)
    )
    diagnostics = residual_to_uniform_weights(
        scores, alpha=alpha, temperature=temperature
    )
    uniform_weights = torch.full_like(
        diagnostics.weights, 1.0 / view_count
    )
    weights = torch.where(
        degeneracy_mask.unsqueeze(1), uniform_weights, diagnostics.weights
    )
    uniform_entropy = -(
        uniform_weights
        * torch.log(
            uniform_weights.clamp_min(torch.finfo(uniform_weights.dtype).tiny)
        )
    ).sum(dim=1)
    diagnostics = ProjectionAggregationDiagnostics(
        scores=diagnostics.scores,
        weights=weights,
        entropy=torch.where(
            degeneracy_mask, uniform_entropy, diagnostics.entropy
        ),
        fallback_mask=diagnostics.fallback_mask | degeneracy_mask,
    )
    fused = (stacked_features * diagnostics.weights.unsqueeze(-1)).sum(dim=1)
    return fused.to(dtype=stacked_features.dtype), diagnostics


def aggregate_consensus_projection(
    view_features: Sequence[torch.Tensor],
    *,
    alpha: float,
    temperature: float,
    chunk_size: int,
    compute_device: torch.device | str,
    output_device: torch.device | str,
) -> tuple[torch.Tensor, ProjectionAggregationDiagnostics]:
    """Chunked consensus aggregation without stacking the full input sequence."""
    if not view_features:
        raise ValueError("view_features must include at least one view")
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1")

    first_feature = view_features[0]
    if first_feature.ndim != 3:
        raise ValueError("each view feature must have shape [B, N, C]")
    if not first_feature.is_floating_point():
        raise TypeError("view features must be floating point")
    if not torch.isfinite(first_feature).all():
        raise ValueError("view features must be finite")
    batch_size, voxel_count, channel_count = first_feature.shape
    if channel_count % 2:
        raise ValueError("view features must have an even channel count")

    for feature in view_features[1:]:
        if feature.shape != first_feature.shape:
            raise ValueError("all view features must have the same shape")
        if not feature.is_floating_point():
            raise TypeError("view features must be floating point")
        if not torch.isfinite(feature).all():
            raise ValueError("view features must be finite")

    compute = torch.device(compute_device)
    output = torch.device(output_device)
    view_count = len(view_features)
    fused = torch.empty(
        (batch_size, voxel_count, channel_count),
        dtype=first_feature.dtype,
        device=output,
    )
    scores = torch.empty((batch_size, view_count, voxel_count), dtype=torch.float32)
    weights = torch.empty_like(scores)
    entropy = torch.empty((batch_size, voxel_count), dtype=torch.float32)
    fallback_mask = torch.empty((batch_size, voxel_count), dtype=torch.bool)

    for start in range(0, voxel_count, chunk_size):
        end = min(start + chunk_size, voxel_count)
        stacked_chunk = torch.stack(
            [feature[:, start:end].to(compute) for feature in view_features], dim=1
        )
        fused_chunk, diagnostics = aggregate_consensus_projection_naive(
            stacked_chunk, alpha=alpha, temperature=temperature
        )
        fused[:, start:end] = fused_chunk.to(device=output, dtype=first_feature.dtype)
        scores[:, :, start:end] = diagnostics.scores.cpu()
        weights[:, :, start:end] = diagnostics.weights.cpu()
        entropy[:, start:end] = diagnostics.entropy.cpu()
        fallback_mask[:, start:end] = diagnostics.fallback_mask.cpu()

    return fused, ProjectionAggregationDiagnostics(
        scores=scores,
        weights=weights,
        entropy=entropy,
        fallback_mask=fallback_mask,
    )
