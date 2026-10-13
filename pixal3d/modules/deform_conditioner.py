"""Role-aware source transport for DeformNeXt fine-tuning ablations.

Omega descriptors select addresses.  Native Pixal features remain the values
consumed by the denoiser.  The six public arms intentionally have distinct
schemas so that disabling deformation context cannot be represented by a
valid all-zero observation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import torch
from torch import nn
from torch.nn import functional as F


Arm = Literal["v1", "b1", "b2", "b3", "c1", "c2"]
Control = Literal["normal", "source-off", "wrong-corr", "shuffled-deform"]


@dataclass
class DeformationCondition:
    evidence: torch.Tensor
    candidate_indices: Optional[torch.Tensor]
    matching_logits: Optional[torch.Tensor]
    candidate_weights: Optional[torch.Tensor]
    view_weights: Optional[torch.Tensor]
    retarget_residual: Optional[torch.Tensor]


class _Retargeter(nn.Module):
    def __init__(self, input_dim: int, value_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, value_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.gate_logit = nn.Parameter(torch.zeros(()))

    def forward(self, inputs: torch.Tensor, g_max: float) -> torch.Tensor:
        return g_max * torch.tanh(self.gate_logit) * torch.tanh(self.net(inputs))


class DeformationAwareConditioner(nn.Module):
    """Fuse source Pixal values using spatial or frozen-Omega addresses.

    Shapes:
      target_keys: ``[B,Q,Dk]``
      source_keys/source_values: ``[B,V,S,Dk|Dv]``
      source_valid: ``[B,V,S]``
      spatial_indices (V1): ``[B,V,Q]``
      target_query (C1/C2): ``[B,Q,Dq]``
      deformation (C2 only): ``[B,Q,Ddef]``
    """

    ARMS = {"v1", "b1", "b2", "b3", "c1", "c2"}
    CONTROLS = {"normal", "source-off", "wrong-corr", "shuffled-deform"}

    def __init__(
        self,
        *,
        arm: Arm,
        key_dim: int,
        value_dim: int,
        query_dim: int,
        deformation_dim: Optional[int] = None,
        hidden_dim: int = 256,
        topk: int = 64,
        g_max: float = 0.25,
        q2c_checkpoint: Optional[str] = None,
        freeze_q2c: bool = False,
    ) -> None:
        super().__init__()
        if arm not in self.ARMS:
            raise ValueError(f"unknown deformation conditioner arm: {arm}")
        if topk <= 0:
            raise ValueError("topk must be positive")
        if arm == "c2" and (deformation_dim is None or deformation_dim <= 0):
            raise ValueError("c2 requires a positive deformation_dim")
        if arm != "c2" and deformation_dim is not None:
            raise ValueError("deformation channels exist only in c2")
        if freeze_q2c and q2c_checkpoint is None:
            raise ValueError("freeze_q2c requires q2c_checkpoint")
        if q2c_checkpoint is not None and arm not in {"b2", "b3", "c1", "c2"}:
            raise ValueError("q2c_checkpoint is only valid for Q2C arms")
        self.arm = arm
        self.key_dim = int(key_dim)
        self.value_dim = int(value_dim)
        self.query_dim = int(query_dim)
        self.topk = int(topk)
        self.g_max = float(g_max)

        self.q2c = None
        if arm in {"b2", "b3", "c1", "c2"}:
            self.q2c = nn.Sequential(
                nn.Linear(2 * key_dim + 1, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )
            if q2c_checkpoint is not None:
                checkpoint = torch.load(
                    Path(q2c_checkpoint), map_location="cpu", weights_only=True
                )
                if isinstance(checkpoint, dict) and "q2c" in checkpoint:
                    checkpoint = checkpoint["q2c"]
                if not isinstance(checkpoint, dict):
                    raise TypeError("Q2C checkpoint must contain a state dict")
                self.q2c.load_state_dict(checkpoint, strict=True)
            if freeze_q2c:
                self.q2c.requires_grad_(False)

        self.retargeter = None
        if arm == "c1":
            self.retargeter = _Retargeter(value_dim + query_dim, value_dim, hidden_dim)
        elif arm == "c2":
            self.retargeter = _Retargeter(
                value_dim + query_dim + int(deformation_dim), value_dim, hidden_dim
            )

    def _validate(
        self,
        target_keys: torch.Tensor,
        source_keys: torch.Tensor,
        source_values: torch.Tensor,
        source_valid: torch.Tensor,
    ) -> tuple[int, int, int, int]:
        if target_keys.ndim != 3 or target_keys.shape[-1] != self.key_dim:
            raise ValueError("target_keys must have shape [B,Q,key_dim]")
        if source_keys.ndim != 4 or source_keys.shape[-1] != self.key_dim:
            raise ValueError("source_keys must have shape [B,V,S,key_dim]")
        if source_values.shape[:-1] != source_keys.shape[:-1] or source_values.shape[-1] != self.value_dim:
            raise ValueError("source_values must have shape [B,V,S,value_dim]")
        if source_valid.shape != source_keys.shape[:3] or source_valid.dtype != torch.bool:
            raise ValueError("source_valid must be bool [B,V,S]")
        batch, views, sites, _ = source_keys.shape
        if target_keys.shape[0] != batch:
            raise ValueError("target/source batch differs")
        if views * sites < self.topk and self.arm != "v1":
            raise ValueError("source candidate count is smaller than topk")
        return batch, target_keys.shape[1], views, sites

    @staticmethod
    def _gather(flat: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        width = flat.shape[-1]
        return torch.gather(
            flat[:, None].expand(-1, indices.shape[1], -1, -1),
            2,
            indices[..., None].expand(-1, -1, -1, width),
        )

    def _spatial(
        self,
        source_values: torch.Tensor,
        source_valid: torch.Tensor,
        spatial_indices: Optional[torch.Tensor],
        queries: int,
    ) -> DeformationCondition:
        batch, views, sites, channels = source_values.shape
        if spatial_indices is None or spatial_indices.shape != (batch, views, queries):
            raise ValueError("v1 requires spatial_indices [B,V,Q]")
        if torch.any((spatial_indices < 0) | (spatial_indices >= sites)):
            raise ValueError("spatial index is out of range")
        by_view = torch.gather(
            source_values,
            2,
            spatial_indices[..., None].expand(-1, -1, -1, channels),
        )
        valid = torch.gather(source_valid, 2, spatial_indices).to(by_view.dtype)
        denominator = valid.sum(dim=1).clamp_min(1.0)
        evidence = (by_view * valid[..., None]).sum(dim=1) / denominator[..., None]
        return DeformationCondition(evidence, None, None, None, None, None)

    def _q2c_logits(
        self,
        target_keys: torch.Tensor,
        candidate_keys: torch.Tensor,
        cosine: torch.Tensor,
    ) -> torch.Tensor:
        target = target_keys[:, :, None].expand_as(candidate_keys)
        return self.q2c(torch.cat([target, candidate_keys, cosine[..., None]], dim=-1)).squeeze(-1)

    def _attention(
        self,
        logits: torch.Tensor,
        values: torch.Tensor,
        valid: torch.Tensor,
        view_index: torch.Tensor,
        views: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        masked = logits.masked_fill(~valid, -torch.inf)
        stable = masked - masked.amax(dim=-1, keepdim=True).clamp(min=-1e4)
        exponent = torch.exp(stable).masked_fill(~valid, 0.0)
        batch, queries, candidates, channels = values.shape
        by_view_den = values.new_zeros(batch, queries, views)
        by_view_den.scatter_add_(2, view_index, exponent)
        by_view_num = values.new_zeros(batch, queries, views, channels)
        by_view_num.scatter_add_(
            2,
            view_index[..., None].expand(-1, -1, -1, channels),
            exponent[..., None] * values,
        )
        by_view = by_view_num / by_view_den.clamp_min(1e-8)[..., None]

        confidence = values.new_zeros(batch, queries, views)
        confidence.scatter_reduce_(
            2, view_index, torch.sigmoid(masked).masked_fill(~valid, 0.0),
            reduce="amax", include_self=True,
        )
        confidence = confidence.masked_fill(by_view_den == 0, 0.0)
        view_weights = confidence / confidence.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        evidence = (by_view * view_weights[..., None]).sum(dim=2)
        candidate_weights = exponent / by_view_den.gather(2, view_index).clamp_min(1e-8)
        return evidence, candidate_weights, view_weights

    def forward(
        self,
        *,
        target_keys: torch.Tensor,
        source_keys: torch.Tensor,
        source_values: torch.Tensor,
        source_valid: torch.Tensor,
        spatial_indices: Optional[torch.Tensor] = None,
        target_query: Optional[torch.Tensor] = None,
        deformation: Optional[torch.Tensor] = None,
        control: Control = "normal",
    ) -> DeformationCondition:
        if control not in self.CONTROLS:
            raise ValueError(f"unknown causal control: {control}")
        batch, queries, views, sites = self._validate(
            target_keys, source_keys, source_values, source_valid
        )
        if control == "source-off":
            zero = source_values.new_zeros(batch, queries, self.value_dim)
            return DeformationCondition(zero, None, None, None, None, None)
        if control == "wrong-corr":
            source_values = torch.roll(source_values, shifts=1, dims=2)
        if self.arm == "v1":
            return self._spatial(source_values, source_valid, spatial_indices, queries)

        flat_keys = source_keys.reshape(batch, views * sites, self.key_dim)
        flat_values = source_values.reshape(batch, views * sites, self.value_dim)
        flat_valid = source_valid.reshape(batch, views * sites)
        similarity = torch.einsum(
            "bqd,bnd->bqn", F.normalize(target_keys, dim=-1), F.normalize(flat_keys, dim=-1)
        ).masked_fill(~flat_valid[:, None], -torch.inf)
        cosine, indices = similarity.topk(self.topk, dim=-1)
        valid = torch.isfinite(cosine)
        candidate_keys = self._gather(flat_keys, indices)
        candidate_values = self._gather(flat_values, indices)
        view_index = torch.div(indices, sites, rounding_mode="floor")

        if self.arm == "b1":
            evidence = candidate_values[:, :, 0]
            return DeformationCondition(evidence, indices[:, :, :1], None, None, None, None)

        logits = self._q2c_logits(target_keys, candidate_keys, cosine.nan_to_num(-1.0))
        logits = logits.masked_fill(~valid, -torch.inf)
        if self.arm == "b2":
            selected = logits.argmax(dim=-1, keepdim=True)
            evidence = torch.gather(
                candidate_values, 2,
                selected[..., None].expand(-1, -1, -1, self.value_dim),
            ).squeeze(2)
            chosen = torch.gather(indices, 2, selected)
            return DeformationCondition(evidence, chosen, logits, None, None, None)

        evidence, candidate_weights, view_weights = self._attention(
            logits, candidate_values, valid, view_index, views
        )
        residual = None
        if self.retargeter is not None:
            if target_query is None or target_query.shape != (batch, queries, self.query_dim):
                raise ValueError("retargeting arms require target_query [B,Q,query_dim]")
            parts = [evidence, target_query]
            if self.arm == "c2":
                if deformation is None:
                    expected = self.retargeter.input_dim - self.value_dim - self.query_dim
                    if expected != self.key_dim:
                        raise ValueError(
                            "inference-derived deformation requires deformation_dim=key_dim"
                        )
                    combined = candidate_weights * torch.gather(
                        view_weights, 2, view_index
                    )
                    source_context = (
                        candidate_keys * combined[..., None]
                    ).sum(dim=2)
                    deformation = target_keys - source_context
                if control == "shuffled-deform":
                    deformation = torch.roll(deformation, shifts=1, dims=0)
                expected = self.retargeter.input_dim - self.value_dim - self.query_dim
                if deformation.shape != (batch, queries, expected):
                    raise ValueError("deformation feature shape does not match c2 schema")
                parts.append(deformation)
            elif deformation is not None:
                raise ValueError("c1 has no deformation input channel")
            residual = self.retargeter(torch.cat(parts, dim=-1), self.g_max)
            evidence = evidence + residual
        return DeformationCondition(
            evidence, indices, logits, candidate_weights, view_weights, residual
        )
