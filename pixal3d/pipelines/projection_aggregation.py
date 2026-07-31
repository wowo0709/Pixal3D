from dataclasses import dataclass
from typing import Literal, Optional

import torch


AggregationMode = Literal["mean", "consensus", "oracle"]
GlobalAggregationMode = Literal["mean", "projection_weights"]


@dataclass(frozen=True)
class ProjectionAggregationConfig:
    mode: AggregationMode = "mean"
    alpha: float = 0.5
    temperature: float = 0.1
    chunk_size: int = 4096
    global_mode: GlobalAggregationMode = "mean"

    def __post_init__(self) -> None:
        if self.mode not in {"mean", "consensus", "oracle"}:
            raise ValueError("mode must be mean, consensus, or oracle")
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        if self.temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.global_mode not in {"mean", "projection_weights"}:
            raise ValueError("global_mode must be mean or projection_weights")


@dataclass(frozen=True)
class ProjectionAggregationDiagnostics:
    scores: Optional[torch.Tensor]
    weights: torch.Tensor
    projected_corruption: Optional[torch.Tensor]


def _uniform_weights(
    num_views: int,
    num_tokens: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    return torch.full(
        (num_views, num_tokens),
        1.0 / num_views,
        dtype=torch.float32,
        device=device,
    )


def _safe_normalize(features: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    norms = torch.linalg.vector_norm(features, dim=-1, keepdim=True)
    return features / norms.clamp_min(eps)


def _consensus_scores(features: torch.Tensor) -> torch.Tensor:
    if features.shape[-1] % 2:
        raise ValueError("projected feature width must split evenly into L/H")
    split = features.shape[-1] // 2
    low = _safe_normalize(features[..., :split].float())
    high = _safe_normalize(features[..., split:].float())

    def branch_scores(branch: torch.Tensor) -> torch.Tensor:
        prototype = _safe_normalize(branch.sum(dim=0, keepdim=True) - branch)
        return (branch * prototype).sum(dim=-1)

    scores = 0.5 * branch_scores(low) + 0.5 * branch_scores(high)
    return torch.where(torch.isfinite(scores), scores, torch.zeros_like(scores))


def _residual_weights(
    routing: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    uniform = torch.full_like(routing, 1.0 / routing.shape[0])
    return (1.0 - alpha) * uniform + alpha * routing


def aggregate_projected_features(
    features: torch.Tensor,
    config: ProjectionAggregationConfig,
    *,
    projected_corruption: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, ProjectionAggregationDiagnostics]:
    if features.ndim != 3:
        raise ValueError("features must have shape [K, N, C]")
    num_views, num_tokens, _ = features.shape
    if num_views < 1:
        raise ValueError("at least one view is required")
    if config.mode == "oracle":
        if projected_corruption is None:
            raise ValueError("oracle mode requires projected_corruption")
        if projected_corruption.shape != (num_views, num_tokens):
            raise ValueError("projected_corruption must have shape [K, N]")

    fused_chunks = []
    score_chunks = []
    weight_chunks = []
    for start in range(0, num_tokens, config.chunk_size):
        stop = min(start + config.chunk_size, num_tokens)
        chunk = features[:, start:stop]
        uniform = _uniform_weights(
            num_views, stop - start, device=features.device
        )
        scores = None
        exact_mean = (
            num_views == 1 or config.mode == "mean" or config.alpha == 0.0
        )
        if exact_mean:
            weights = uniform
        elif config.mode == "consensus":
            scores = _consensus_scores(chunk)
            routing = torch.softmax(scores / config.temperature, dim=0)
            finite = torch.isfinite(routing).all(dim=0, keepdim=True)
            routing = torch.where(finite, routing, uniform)
            weights = _residual_weights(routing, config.alpha)
        else:
            reliability = (
                1.0 - projected_corruption[:, start:stop].float()
            ).clamp(0.0, 1.0)
            denominator = reliability.sum(dim=0, keepdim=True)
            routing = reliability / denominator.clamp_min(1e-6)
            routing = torch.where(denominator > 0, routing, uniform)
            weights = _residual_weights(routing, config.alpha)
        fused = (
            chunk.float().mean(dim=0)
            if exact_mean
            else (chunk.float() * weights[..., None]).sum(dim=0)
        )
        fused_chunks.append(fused.to(features.dtype))
        weight_chunks.append(weights)
        if scores is not None:
            score_chunks.append(scores)

    diagnostics = ProjectionAggregationDiagnostics(
        scores=torch.cat(score_chunks, dim=1) if score_chunks else None,
        weights=torch.cat(weight_chunks, dim=1),
        projected_corruption=projected_corruption,
    )
    return torch.cat(fused_chunks, dim=0), diagnostics


def aggregate_global_features(
    features: torch.Tensor,
    projection_weights: torch.Tensor,
    *,
    mode: GlobalAggregationMode,
) -> torch.Tensor:
    if features.ndim != 3:
        raise ValueError("global features must have shape [K, T, D]")
    if projection_weights.shape[0] != features.shape[0]:
        raise ValueError("global and projected view counts must match")
    if mode == "mean":
        fused = features.float().mean(dim=0, keepdim=True)
    else:
        view_weights = projection_weights.float().mean(dim=1)
        denominator = view_weights.sum()
        if not torch.isfinite(denominator) or denominator <= 0:
            view_weights = torch.full_like(
                view_weights, 1.0 / view_weights.numel()
            )
        else:
            view_weights = view_weights / denominator
        fused = (
            features.float() * view_weights[:, None, None]
        ).sum(dim=0, keepdim=True)
    return fused.to(features.dtype)
