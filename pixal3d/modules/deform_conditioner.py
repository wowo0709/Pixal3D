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


Arm = Literal["b1", "b2", "b3", "c1", "c2"]
Control = Literal["normal", "source-off", "wrong-corr", "shuffled-deform"]


@dataclass
class DeformationCondition:
    evidence: torch.Tensor
    candidate_indices: Optional[torch.Tensor]
    matching_logits: Optional[torch.Tensor]
    candidate_weights: Optional[torch.Tensor]
    view_weights: Optional[torch.Tensor]
    retarget_residual: Optional[torch.Tensor]
    query_valid: Optional[torch.Tensor] = None
    fusion_scale: Optional[torch.Tensor] = None
    usage_scale: Optional[torch.Tensor] = None
    base_evidence: Optional[torch.Tensor] = None


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
        self.gate_logit = nn.Parameter(torch.ones(()))

    def forward(self, inputs: torch.Tensor, g_max: float) -> torch.Tensor:
        return g_max * torch.tanh(self.gate_logit) * torch.tanh(self.net(inputs))


class DeformationAwareConditioner(nn.Module):
    """Fuse source Pixal values using spatial or frozen-Omega addresses.

    Shapes:
      target_keys: ``[B,Q,Dk]``
      source_keys/source_values: ``[B,V,S,Dk|Dv]``
      source_valid: ``[B,V,S]``
      target_query (C1/C2): ``[B,Q,Dq]``
      deformation (C2 only): ``[B,Q,Ddef]``
    """

    ARMS = {"b1", "b2", "b3", "c1", "c2"}
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
        implementation: str = "r9-native-v5",
        score_temperature: Optional[float] = None,
        projection_mode: str = "nearest",
        task_attention: Optional[dict] = None,
        correspondence_mode: str = 'semantic',
        candidate_aggregation: str = 'weighted',
    ) -> None:
        super().__init__()
        if candidate_aggregation not in {'weighted','top1'}:
            raise ValueError('unknown candidate aggregation')
        if candidate_aggregation=='top1' and (arm!='c1' or implementation!='r9-native-v5' or task_attention is not None):
            raise ValueError('matched Top1 retargeter control requires static R9 C1')
        self.candidate_aggregation=candidate_aggregation
        if implementation not in {"legacy-v4", "r9-native-v5"}:
            raise ValueError("unknown conditioner implementation")
        self.implementation = implementation
        if projection_mode not in {'nearest','native-bilinear'}:
            raise ValueError('unknown transport projection mode')
        if projection_mode=='native-bilinear' and (implementation!='r9-native-v5' or value_dim!=1024):
            raise ValueError('native-bilinear transport currently supports native SS only')
        self.projection_mode=projection_mode
        self.score_temperature = (0.005 if implementation=='r9-native-v5' else 1.) if score_temperature is None else float(score_temperature)
        if self.score_temperature<=0:raise ValueError('matching temperature must be positive')
        if implementation == "r9-native-v5" and (key_dim != 1024 or topk != 64):
            raise ValueError("r9-native-v5 requires FRAME 1024-D and Top-64")
        if arm == "v1":
            raise ValueError("V1 must use native multi-view conditioning without a transport module")
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
        if implementation == "r9-native-v5":
            # A nonzero fixed semantic branch and a zero scalar gate retain a
            # learning signal without changing the native condition at step 0.
            self.fusion_gate = nn.Parameter(torch.zeros(()))

        self.q2c = None
        if arm in {"b2", "b3", "c1", "c2"}:
            if implementation == "r9-native-v5":
                from .r9_matching import PromotedR9Matcher
                self.q2c = PromotedR9Matcher()
            else:
                self.q2c = nn.Sequential(
                    nn.Linear(2 * key_dim + 1, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, 1),
                )
            if q2c_checkpoint is not None:
                checkpoint = torch.load(
                    Path(q2c_checkpoint), map_location="cpu", weights_only=True
                )
                if implementation == "r9-native-v5":
                    self.q2c.load_checkpoint(checkpoint)
                elif isinstance(checkpoint, dict) and "q2c" in checkpoint:
                    checkpoint = checkpoint["q2c"]
                if not isinstance(checkpoint, dict):
                    raise TypeError("Q2C checkpoint must contain a state dict")
                if implementation == "legacy-v4":
                    self.q2c.load_state_dict(checkpoint, strict=True)
            if freeze_q2c:
                self.q2c.requires_grad_(False)
            if candidate_aggregation=='top1':
                self.q2c.requires_grad_(False)

        self.retargeter = None
        if arm == "c1":
            self.retargeter = _Retargeter(value_dim + query_dim, value_dim, hidden_dim)
        elif arm == "c2":
            self.retargeter = _Retargeter(
                value_dim + query_dim + int(deformation_dim), value_dim, hidden_dim
            )

        self.task_attention = None
        self.task_attention_config = None
        self.correspondence_mode = correspondence_mode
        if correspondence_mode not in {'semantic','native-spatial'} or (correspondence_mode=='native-spatial' and task_attention is None):
            raise ValueError('native-spatial candidate attention requires an explicit task module')
        if task_attention is not None:
            if implementation != 'r9-native-v5' or arm != 'c1' or value_dim != 1024 or query_dim != 1024:
                raise ValueError('task candidate attention supports explicit SS C1 only')
            required_fields = {'decoder', 'score_prior', 'chunk_size'}
            optional_fields = {'train_decoder', 'train_score', 'content_mode', 'initial_content_scale'}
            if not required_fields <= set(task_attention) or set(task_attention)-required_fields-optional_fields:
                raise ValueError('task attention requires explicit decoder/score/chunk contract')
            if (not isinstance(task_attention['decoder'],bool) or not isinstance(task_attention['score_prior'],bool)
                    or not isinstance(task_attention['chunk_size'],int) or task_attention['chunk_size']<1):
                raise ValueError('invalid task attention switches or chunk size')
            train_decoder = task_attention.get('train_decoder', task_attention['decoder'])
            train_score = task_attention.get('train_score', task_attention['score_prior'])
            if (not isinstance(train_decoder, bool) or not isinstance(train_score, bool)
                or (train_decoder and not task_attention['decoder'])
                or (train_score and not task_attention['score_prior'])):
                raise ValueError('cannot train a removed decoder or unused score prior')
            from .task_matching_attention import TaskCandidateAttention
            from .r9_matching import RegionDecoder
            self.task_attention_config = dict(task_attention)
            if not task_attention['decoder']:
                self.q2c.decoder = RegionDecoder('D0-PAIRWISE')
                self.q2c.decoder_arm = 'D0-PAIRWISE'
            self.q2c.decoder.requires_grad_(train_decoder)
            self.q2c.fusion_head.requires_grad_(train_score)
            # Equal task-layer initialization across physical decoder ablations;
            # do not alter the caller's denoiser/retargeter RNG consumption.
            with torch.random.fork_rng(devices=[]):
                # torch.manual_seed also resets every visible CUDA generator;
                # fork_rng(devices=[]) restores only CPU state. Seed the CPU
                # generator directly so DDP ranks retain their own RNG.
                torch.random.default_generator.manual_seed(20261018)
                self.task_attention = TaskCandidateAttention(content_mode=task_attention.get('content_mode','cosine'),
                    initial_content_scale=task_attention.get('initial_content_scale', 4.))

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
        if views * sites < (1 if self.arm == "b1" else self.topk):
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

    def _q2c_logits(
        self,
        target_keys: torch.Tensor,
        candidate_keys: torch.Tensor,
        cosine: torch.Tensor,
    ) -> torch.Tensor:
        if self.implementation == "r9-native-v5":
            with torch.autocast(device_type=target_keys.device.type, enabled=False):
                return self.q2c.score_candidates(target_keys.float(), candidate_keys.float(), cosine.float())
        # Decompose the first linear layer instead of materializing
        # [target, candidate, cosine].  At PBR resolution that concatenation
        # is more than 10 GiB per 10k-query sample for K=64, even though the
        # first hidden layer is only 256-D.
        first, activation, second = self.q2c
        split = self.key_dim
        target = F.linear(target_keys, first.weight[:, :split], bias=None)[:, :, None]
        candidate = F.linear(
            candidate_keys, first.weight[:, split:2 * split], bias=None
        )
        score = cosine[..., None] * first.weight[:, 2 * split].reshape(1, 1, 1, -1)
        hidden = target + candidate + score
        if first.bias is not None:
            hidden = hidden + first.bias
        return second(activation(hidden)).squeeze(-1)

    def _attention(
        self,
        logits: torch.Tensor,
        values: torch.Tensor,
        valid: torch.Tensor,
        view_index: torch.Tensor,
        views: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch,queries,candidates,channels=values.shape
        if self.implementation == "r9-native-v5":
            from .r9_matching import categorical_weights
            weights=categorical_weights(logits,valid,self.score_temperature)
            view_weights=values.new_zeros(batch,queries,views)
            view_weights.scatter_add_(2,view_index,weights)
            return (weights[...,None]*values).sum(2),weights,view_weights
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
        target_keys: Optional[torch.Tensor] = None,
        source_keys: Optional[torch.Tensor] = None,
        source_values: torch.Tensor,
        source_valid: torch.Tensor,
        target_query: Optional[torch.Tensor] = None,
        deformation: Optional[torch.Tensor] = None,
        semantic_indices: Optional[torch.Tensor] = None,
        matching_logits: Optional[torch.Tensor] = None,
        candidate_valid: Optional[torch.Tensor] = None,
        target_valid: Optional[torch.Tensor] = None,
        target_context: Optional[torch.Tensor] = None,
        source_context: Optional[torch.Tensor] = None,
        native_candidate_keys: Optional[torch.Tensor] = None,
        native_candidate_values: Optional[torch.Tensor] = None,
        control: Control = "normal",
    ) -> DeformationCondition:
        if control not in self.CONTROLS:
            raise ValueError(f"unknown causal control: {control}")
        if native_candidate_values is not None:
            if self.correspondence_mode!='native-spatial' or self.task_attention is None or control!='normal':
                raise ValueError('native spatial candidates require the explicit unperturbed spatial task arm')
            from .task_matching_attention import native_spatial_evidence
            return native_spatial_evidence(self,target_keys=target_keys,target_query=target_query,
                keys=native_candidate_keys,values=native_candidate_values,target_valid=target_valid)
        if self.correspondence_mode=='native-spatial':
            raise ValueError('native-spatial arm must receive original projected camera candidates')
        if self.arm != 'c2' and (target_context is not None or source_context is not None):
            raise ValueError('raw deformation context channels exist only in c2')
        if source_values.ndim != 4 or source_values.shape[-1] != self.value_dim:
            raise ValueError("source_values must have shape [B,V,S,value_dim]")
        if source_valid.shape != source_values.shape[:3] or source_valid.dtype != torch.bool:
            raise ValueError("source_valid must be bool [B,V,S]")
        source_values = source_values.float()
        batch, views, sites, _ = source_values.shape
        candidate_keys = None
        if semantic_indices is None:
            if target_keys is None or source_keys is None:
                raise ValueError(
                    f"{self.arm} requires semantic_indices or target/source keys"
                )
            batch, queries, views, sites = self._validate(
                target_keys, source_keys, source_values, source_valid
            )
        else:
            if semantic_indices.ndim != 3 or semantic_indices.shape[0] != batch:
                raise ValueError("semantic_indices must have shape [B,Q,K]")
            queries = semantic_indices.shape[1]
        if self.implementation == "r9-native-v5":
            if target_valid is None or target_valid.shape != (batch, queries) or target_valid.dtype != torch.bool:
                raise ValueError("r9-native-v5 requires inference-safe target_valid [B,Q]")
        fusion_scale = torch.tanh(self.fusion_gate) if self.implementation == "r9-native-v5" else None
        if control == "source-off":
            zero = source_values.new_zeros(batch, queries, self.value_dim)
            return DeformationCondition(zero, None, None, None, None, None,
                                        torch.zeros(batch, queries, dtype=torch.bool, device=zero.device), fusion_scale)
        if control == "wrong-corr":
            source_values = torch.roll(source_values, shifts=1, dims=2)

        flat_values = source_values.reshape(batch, views * sites, self.value_dim)
        flat_valid = source_valid.reshape(batch, views * sites)
        if semantic_indices is None:
            flat_keys = source_keys.reshape(batch, views * sites, self.key_dim)
            if self.implementation == 'r9-native-v5':
                from .r9_matching import frame_topk
                target_keys, flat_keys = target_keys.float(), flat_keys.float()
                cosine, indices = frame_topk(target_keys, source_keys, source_valid,
                                             topk=1 if self.arm == 'b1' else self.topk)
            else:
                similarity = torch.einsum('bqd,bnd->bqn', F.normalize(target_keys, dim=-1), F.normalize(flat_keys, dim=-1))
                similarity = similarity.masked_fill(~flat_valid[:, None], -torch.inf)
                candidates = 1 if self.arm == "b1" else self.topk
                cosine, indices = similarity.topk(candidates, dim=-1)
            valid = torch.isfinite(cosine)
        else:
            indices = semantic_indices.long()
            if self.arm == "b1":
                indices = indices[:, :, :1]
            elif indices.shape[2] != self.topk:
                raise ValueError("semantic match cache has the wrong topk")
            if torch.any((indices < 0) | (indices >= views * sites)):
                raise ValueError("semantic match index is out of range")
            valid = torch.gather(
                flat_valid[:, None].expand(-1, queries, -1), 2, indices
            )
            if candidate_valid is not None:
                cached_valid = candidate_valid[:, :, :indices.shape[2]].bool()
                if cached_valid.shape != indices.shape:
                    raise ValueError("candidate_valid shape differs from semantic indices")
                valid = valid & cached_valid
        if self.implementation == "r9-native-v5":
            complete_support = source_valid.flatten(1).sum(-1) >= 64
            valid = valid & target_valid[..., None] & complete_support[:, None, None]
        query_valid = valid.any(-1)
        view_index = torch.div(indices, sites, rounding_mode="floor")

        if self.task_attention is not None:
            from .task_matching_attention import task_evidence
            return task_evidence(self,target_keys=target_keys,source_keys=source_keys,
                source_values=source_values,target_query=target_query,indices=indices,
                valid=valid,view_index=view_index,views=views)

        candidate_values = self._gather(flat_values, indices)

        if self.arm == "b1":
            evidence = candidate_values[:, :, 0] * valid[:, :, 0, None]
            return DeformationCondition(evidence, indices[:, :, :1], None, None, None, None, query_valid, fusion_scale)

        if matching_logits is None:
            candidate_keys = self._gather(flat_keys, indices)
            logits = self._q2c_logits(
                target_keys, candidate_keys, cosine.nan_to_num(nan=-1.0,posinf=-1.0,neginf=-1.0)
            )
        else:
            logits = matching_logits
            if logits.shape != indices.shape:
                raise ValueError("matching_logits shape differs from semantic indices")
        logits = logits.to(candidate_values.dtype).masked_fill(~valid, -torch.inf)
        if self.arm == "b2":
            selected = logits.argmax(dim=-1, keepdim=True)
            evidence = torch.gather(
                candidate_values, 2,
                selected[..., None].expand(-1, -1, -1, self.value_dim),
            ).squeeze(2)
            evidence = evidence * valid.any(dim=-1, keepdim=True)
            chosen = torch.gather(indices, 2, selected)
            return DeformationCondition(evidence, chosen, logits, None, None, None, query_valid, fusion_scale)

        if self.candidate_aggregation=='top1':
            # Keep the same C1 retargeter and original matching-score argmax.
            # No uniform weighting, extra query channels or learned soft path.
            selected=logits.argmax(-1,keepdim=True)
            candidate_weights=torch.zeros_like(logits).scatter(2,selected,1.)*query_valid[...,None]
            evidence=(candidate_weights[...,None]*candidate_values).sum(2)
            view_weights=logits.new_zeros(batch,queries,views).scatter_add(2,view_index,candidate_weights)
        else:
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
                    if target_context is not None and source_context is not None:
                        if (target_context.shape != (batch, queries, expected)
                                or source_context.shape != (batch, views, sites, expected)):
                            raise ValueError('raw deformation context shape differs')
                        context_keys = self._gather(source_context.reshape(batch, views*sites, expected).float(), indices)
                        query_context = target_context.float()
                    elif expected == self.key_dim and candidate_keys is not None and target_keys is not None:
                        context_keys, query_context = candidate_keys, target_keys
                    else:
                        raise ValueError(
                            "c2 requires cached deformation or aligned raw deformation context"
                        )
                    combined = (candidate_weights if self.implementation == 'r9-native-v5'
                                else candidate_weights * torch.gather(view_weights, 2, view_index))
                    source_summary = (
                        context_keys * combined[..., None]
                    ).sum(dim=2)
                    deformation = query_context - source_summary
                    if self.implementation == 'r9-native-v5':
                        # The immutable C2 sidecar uses FP16 RAW context delta.
                        deformation = deformation.half()
                if control == "shuffled-deform":
                    deformation = torch.roll(deformation, shifts=1, dims=0)
                expected = self.retargeter.input_dim - self.value_dim - self.query_dim
                if deformation.shape != (batch, queries, expected):
                    raise ValueError("deformation feature shape does not match c2 schema")
                parts.append(deformation)
            elif deformation is not None:
                raise ValueError("c1 has no deformation input channel")
            residual = self.retargeter(torch.cat(parts, dim=-1), self.g_max)
            residual = residual * query_valid[..., None]
            evidence = evidence + residual
        return DeformationCondition(
            evidence, indices, logits, candidate_weights, view_weights, residual, query_valid, fusion_scale
        )
