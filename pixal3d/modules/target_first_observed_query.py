"""Observed semantic queries with unmasked native utility and ray delivery.

The frozen RayGroup implementation is retained for earlier experiments. This
version separates its three query pools: only the matching descriptor uses
actual target-image observation support. Native utility/retarget context and
the downstream ProjGrid keep the complete ray. No surface visibility is used.
"""
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .deform_conditioner import DeformationCondition
from .ray_group_matching import semantic_ray_proposals, attend_semantic_chunk
from .ray_group_query import pool_ray_queries
from .target_first_ray_v2 import TargetFirstRayConditionerV2, TargetFirstRayEvidence
from .task_matching_attention import contextual_query


class ObservedQueryTargetFirstRayConditioner(TargetFirstRayConditionerV2):
    def forward(self, *, native_voxels, target_omega_voxels, target_native_voxels,
                ray_ids, observed_query, source_keys, source_values, source_valid,
                control='normal'):
        if control != 'normal':
            raise ValueError('use declared saved-weight diagnostic contexts')
        if (native_voxels.shape != target_omega_voxels.shape
                or native_voxels.shape != target_native_voxels.shape
                or native_voxels.shape[-1] != 1024 or ray_ids.shape[1] != 3
                or observed_query.shape != native_voxels.shape[:2]
                or observed_query.dtype != torch.bool):
            raise ValueError('aligned native SS inputs and explicit target-image query support required')
        batch, views, sites, width = source_values.shape
        if views != 3 or width != 1024:
            raise ValueError('three native source feature maps required')
        matching_ids = torch.where(observed_query[:, None], ray_ids, torch.full_like(ray_ids, -1))
        utility_query = pool_ray_queries(native_voxels, ray_ids, num_rays=sites)
        omega_query = pool_ray_queries(target_omega_voxels, matching_ids, num_rays=sites)
        retarget_query = pool_ray_queries(target_native_voxels, ray_ids, num_rays=sites)
        omega = omega_query.values.flatten(1, 2)
        self._validate(omega, source_keys, source_values, source_valid)
        available = (utility_query.valid & omega_query.valid).flatten(1)
        available &= (source_valid.flatten(1).sum(-1) >= 64)[:, None]
        indices, valid = semantic_ray_proposals(omega, source_keys, source_valid,
            available, self.ray_chunk_size)
        rays, count = views * sites, indices.shape[-1]
        selected = valid.any(-1).flatten().nonzero().flatten()
        context = source_values.new_zeros(batch * rays, 1024, dtype=torch.float32)
        score = source_values.new_full((batch * rays, count), -torch.inf, dtype=torch.float32)
        flat_keys, flat_values = source_keys.flatten(1, 2), source_values.flatten(1, 2).float()
        # Same RAW64/D1/score math as frozen RayGroupConditioner, with a clean
        # observed matching descriptor. GT labels never reach this function.
        with torch.no_grad(), torch.autocast(device_type=omega.device.type, enabled=False):
            for start in range(0, len(selected), self.ray_chunk_size):
                ix = selected[start:start + self.ray_chunk_size]
                b = ix // rays
                ids, mask = indices.reshape(-1, count)[ix], valid.reshape(-1, count)[ix]
                keys = flat_keys[b[:, None], ids].float()
                keys = torch.where(mask[..., None], keys, torch.zeros_like(keys))
                ctx = contextual_query(self.q2c.decoder, omega.reshape(-1, 1024)[ix].float(), keys, mask)
                context[ix] = ctx
                nq, nk = F.normalize(ctx, dim=-1), F.normalize(keys, dim=-1)
                cosine = (nq[:, None] * nk).sum(-1)
                score[ix] = self.q2c.fusion_head(nq, nk, cosine[..., None], cosine)
        evidence = source_values.new_zeros(batch * rays, 1024, dtype=torch.float32)
        probabilities = source_values.new_zeros(batch * rays, count, dtype=torch.float32)
        logits = source_values.new_full((batch * rays, count), -torch.inf, dtype=torch.float32)
        parts, weights, scores = [], [], []
        native_query = utility_query.values.reshape(-1, 1024)
        for start in range(0, len(selected), self.ray_chunk_size):
            ix = selected[start:start + self.ray_chunk_size]
            b = ix // rays
            ids, mask = indices.reshape(-1, count)[ix], valid.reshape(-1, count)[ix]
            prior = score[ix] / self.score_temperature
            if self.task_attention is None:
                values = flat_values[b[:, None], ids]
                e, p, _ = self._attention(prior[None], values[None], mask[None],
                    torch.div(ids, sites, rounding_mode='floor')[None], 3)
                result = e[0], p[0], prior
            else:
                args = (self.task_attention, native_query[ix], context[ix], flat_values, b, ids, mask,
                        prior if self.task_attention_config['score_prior'] else None)
                result = (checkpoint(attend_semantic_chunk, *args, use_reentrant=False)
                          if torch.is_grad_enabled() else attend_semantic_chunk(*args))
            e, p, l = result
            parts.append(e); weights.append(p); scores.append(l)
        if len(selected):
            evidence = evidence.index_copy(0, selected, torch.cat(parts))
            probabilities = probabilities.index_copy(0, selected, torch.cat(weights))
            logits = logits.index_copy(0, selected, torch.cat(scores))
        else:
            # Keep zero gradient edges for DDP ranks without observed queries.
            evidence = evidence + sum(p.reshape(-1)[0] * 0 for p in self.parameters() if p.requires_grad)
        residual = evidence.new_zeros(batch * rays, 1024)
        gate_query = retarget_query.values.reshape(batch, rays, 1024)
        if len(selected):
            changed = self.retargeter(torch.cat((evidence[selected], gate_query.flatten(0, 1)[selected]), -1), self.g_max)
            residual = residual.index_copy(0, selected, changed.float())
        evidence = evidence.reshape(batch, rays, 1024)
        residual = residual.reshape(batch, rays, 1024)
        probabilities = probabilities.reshape(batch, rays, count)
        query_valid = valid.any(-1)
        view_weights = evidence.new_zeros(batch, rays, 3).scatter_add(2,
            torch.div(indices, sites, rounding_mode='floor'), probabilities)
        scale = torch.tanh(self.fusion_gate)
        attention = DeformationCondition(evidence + residual, indices, logits.reshape(batch, rays, count),
            probabilities, view_weights, residual, query_valid, scale, base_evidence=evidence)
        use = self.evidence_use(gate_query, evidence, query_valid)
        mass = use[..., None] * scale
        numerator = mass * evidence + use[..., None] * residual
        return TargetFirstRayEvidence(numerator.reshape(batch, 3, sites, 1024),
            mass.reshape(batch, 3, sites, 1), attention, use)
