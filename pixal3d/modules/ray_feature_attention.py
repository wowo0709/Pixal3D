"""Target-ray feature composition with an immediately learnable zero residual."""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .deform_conditioner import DeformationAwareConditioner, DeformationCondition
from .task_matching_attention import TaskCandidateAttention, contextual_query


class RayFeatureReadout(nn.Module):
    """Distinct head weights and value/output projections; no scalar zero gate."""
    def __init__(self, value_width=1024, heads=4, head_width=64):
        super().__init__()
        if value_width not in (1024, 2048) or heads != 4 or head_width != 64:
            raise ValueError('declared SS/PBR widths and four64-wide heads required')
        self.value_width, self.heads, self.head_width = value_width, heads, head_width
        hidden = heads * head_width
        self.score_selector = TaskCandidateAttention(content_mode='residual-zero', initial_content_scale=16.)
        self.query = nn.Linear(2048, hidden, bias=False)
        self.key = nn.Linear(1024, hidden, bias=False)
        self.value = nn.Linear(value_width, hidden, bias=False)
        self.output = nn.Linear(hidden, value_width, bias=False)
        for layer in [self.query, self.key, self.value]:
            nn.init.xavier_uniform_(layer.weight)
        nn.init.zeros_(self.output.weight)

    def forward(self, native, context, keys, values, valid, prior):
        m, k = valid.shape
        if (native.shape != (m, 1024) or context.shape != native.shape
                or keys.shape != (m, k, 1024) or values.shape != (m, k, self.value_width)
                or prior.shape != valid.shape or valid.dtype != torch.bool):
            raise ValueError('aligned RGB/Omega/native candidate features and score prior required')
        keys = torch.where(valid[..., None], keys, torch.zeros_like(keys))
        values = torch.where(valid[..., None], values, torch.zeros_like(values))
        _, _, score = self.score_selector(native, context, values[..., :1024], valid, prior)
        query_input = torch.cat([F.layer_norm(native, (1024,)), F.layer_norm(context, (1024,))], -1)
        q = self.query(query_input).reshape(m, self.heads, self.head_width)
        kk = self.key(F.layer_norm(keys, (1024,))).reshape(m, k, self.heads, self.head_width).transpose(1, 2)
        # Keep DINO and NAF scales separate in the PBR value projection.
        normalized = torch.cat([F.layer_norm(chunk, (1024,))
                                for chunk in values.split(1024, dim=-1)], dim=-1)
        vv = self.value(normalized).reshape(m, k, self.heads, self.head_width).transpose(1, 2)
        with torch.autocast(device_type=native.device.type, enabled=False):
            logits = (q.float()[:, :, None] * kk.float()).sum(-1) / math.sqrt(self.head_width)
            logits = (logits + score[:, None].float()).masked_fill(~valid[:, None], -torch.inf)
            maximum = logits.amax(-1, keepdim=True)
            maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
            probability = torch.exp(logits-maximum).masked_fill(~valid[:, None], 0.)
            probability = probability / probability.sum(-1, keepdim=True).clamp_min(1e-8)
            combined = (probability[..., None] * vv.float()).sum(-2).reshape(m, -1)
        return self.output(combined).float(), probability


def _readout(head, native, context, values, keys, batch, indices, valid, prior):
    return head(native, context, keys[batch[:, None], indices].float(),
                values[batch[:, None], indices].float(), valid, prior)


class RayFeatureAttentionConditioner(DeformationAwareConditioner):
    """Keep old Top1+R and nearest target-ray delivery; learn an additive feature."""
    def __init__(self, *, feature_chunk_size=256, **kwargs):
        super().__init__(**kwargs)
        if (self.arm != 'c1' or self.implementation != 'r9-native-v5'
                or self.projection_mode != 'nearest' or self.candidate_aggregation != 'top1'
                or self.task_attention is not None or self.correspondence_mode != 'semantic'
                or (self.key_dim, self.query_dim, self.topk) != (1024, 1024, 64)
                or self.value_dim not in (1024, 2048)):
            raise ValueError('original nearest target-ray C1 Top1+R carrier required')
        if isinstance(feature_chunk_size, bool) or not isinstance(feature_chunk_size, int) or feature_chunk_size < 1:
            raise ValueError('positive feature query chunk required')
        self.requires_grad_(False)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(20261021)
            self.ray_readout = RayFeatureReadout(self.value_dim)
        self.feature_chunk_size = feature_chunk_size

    def reference_state(self):
        return {name: value for name,value in self.state_dict().items()
                if not name.startswith('ray_readout.')}

    def load_reference(self, reference):
        if set(reference) != set(self.reference_state()) or torch.count_nonzero(self.ray_readout.output.weight):
            raise ValueError('complete reference and untouched zero-output feature branch required')
        merged = dict(self.state_dict()); merged.update(reference)
        self.load_state_dict(merged, strict=True)
        if any(not torch.equal(value.detach().cpu(), self.state_dict()[name].detach().cpu())
               for name,value in reference.items()):
            raise ValueError('original matcher/R/fusion reference changed')

    def forward(self, *, source_values, source_valid, target_query, semantic_indices,
                matching_logits, candidate_valid, target_valid, target_keys, source_keys,
                control='normal', **unsupported):
        if unsupported or control != 'normal':
            raise ValueError('only declared inference RGB/Omega inputs and normal transport supported')
        source_values = source_values.float()
        b, views, sites, width = source_values.shape
        q = target_query.shape[1]
        if (views != 3 or sites != 1024 or width != self.value_dim
                or target_query.shape != (b, q, 1024) or target_keys.shape != target_query.shape
                or source_keys.shape != (b, 3, 1024, 1024) or source_valid.shape != (b, 3, 1024)
                or semantic_indices.shape != (b, q, 64) or matching_logits.shape != semantic_indices.shape
                or candidate_valid.shape != semantic_indices.shape or target_valid.shape != (b, q)
                or source_valid.dtype != torch.bool or target_valid.dtype != torch.bool):
            raise ValueError('same target-patch rays and complete native64 candidate cache required')
        indices = semantic_indices.long()
        if ((indices < 0) | (indices >= 3072)).any():
            raise ValueError('candidate source address outside the three RGB views')
        flat_values, flat_keys = source_values.flatten(1, 2), source_keys.flatten(1, 2)
        flat_valid = source_valid.flatten(1)
        valid = flat_valid[:, None].expand(-1, q, -1).gather(2, indices) & candidate_valid.bool()
        valid &= target_valid[..., None] & (flat_valid.sum(-1) >= 64)[:, None, None]
        active = valid.any(-1)
        logits = matching_logits.float().masked_fill(~valid, -torch.inf)
        chosen = logits.argmax(-1, keepdim=True)
        hard = torch.zeros_like(logits).scatter(2, chosen, 1.) * active[..., None]
        address = indices.gather(2, chosen).squeeze(2)
        top1 = flat_values.gather(1, address[..., None].expand(-1, -1, width)) * active[..., None]
        selected = active.flatten().nonzero().flatten()
        residual_parts, weight_parts = [], []
        delta = source_values.new_zeros(b*q, width)
        for start in range(0, len(selected), self.feature_chunk_size):
            take = selected[start:start+self.feature_chunk_size];batch = take // q
            ids = indices.reshape(-1, 64)[take];mask = valid.reshape(-1, 64)[take]
            with torch.no_grad(), torch.autocast(device_type=source_values.device.type, enabled=False):
                keys = flat_keys[batch[:, None], ids].float()
                context = contextual_query(self.q2c.decoder, target_keys.reshape(-1, 1024)[take].float(),
                    torch.where(mask[..., None], keys, torch.zeros_like(keys)), mask)
            args = (self.ray_readout, target_query.reshape(-1, 1024)[take].float(), context,
                flat_values, flat_keys, batch, ids, mask, logits.reshape(-1, 64)[take]/self.score_temperature)
            change, weights = (checkpoint(_readout, *args, use_reentrant=False)
                               if torch.is_grad_enabled() else _readout(*args))
            residual_parts.append(change);weight_parts.append(weights)
        if len(selected):
            delta = delta.index_copy(0, selected, torch.cat(residual_parts))
            weights = torch.cat(weight_parts).detach()
            entropy = -(weights * weights.clamp_min(1e-30).log()).sum(-1)
            effective = float(entropy.exp().mean())
        else:
            delta = delta + sum(parameter.reshape(-1)[0]*0 for parameter in self.ray_readout.parameters())
            effective = 0.
        mixed = top1 + delta.reshape(b, q, width)
        retargeted = self.retargeter(torch.cat([mixed, target_query.float()], -1), self.g_max)
        retargeted = retargeted * active[..., None]
        self.last_feature_attention_stats = dict(effective_K_per_head=effective,
            feature_residual_RMS=float(delta.detach().square().mean().sqrt()),
            active_rays=int(active.sum()), ray_query='edited-target-patch', scalar_mix_gate=False)
        # Hard weights describe the unchanged anchor carrier. Feature-head
        # weights belong to the separate transformed residual, not one raw sum.
        view_weights = logits.new_zeros(b, q, 3).scatter_add(2, indices//1024, hard)
        return DeformationCondition(mixed+retargeted, indices, logits, hard, view_weights,
            retargeted, active, torch.tanh(self.fusion_gate))
