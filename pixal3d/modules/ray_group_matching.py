"""Ray-group source retrieval and retargeting before native per-view projection."""
from dataclasses import dataclass

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .deform_conditioner import DeformationAwareConditioner, DeformationCondition
from .r9_matching import categorical_weights, frame_topk
from .ray_group_query import pool_ray_queries
from .task_matching_attention import contextual_query


@dataclass
class RayGroupCondition:
    map_delta: torch.Tensor  # [B,3,S,1024]; ONE feature delta per camera ray
    attention: DeformationCondition


@torch.no_grad()
def semantic_ray_proposals(query, source, source_valid, query_valid, chunk_size):
    """Fresh frozen RAW64 support for each ray; never reuse anchor-patch IDs."""
    batch, rays, _ = query.shape
    indices = torch.zeros(batch, rays, 64, dtype=torch.long, device=query.device)
    valid = torch.zeros_like(indices, dtype=torch.bool)
    for b in range(batch):
        selected = query_valid[b].nonzero().flatten()
        for start in range(0, len(selected), chunk_size):
            take = selected[start:start + chunk_size]
            scores, ids = frame_topk(query[b:b + 1, take], source[b:b + 1], source_valid[b:b + 1], 64)
            indices[b, take] = ids[0]
            valid[b, take] = torch.isfinite(scores[0])
    return indices, valid


def attend_semantic_chunk(attention, query, context, source_values, batch, ids, valid, prior):
    # Gather inside checkpoint: retain shared source maps, not all ray/K values.
    values = source_values[batch[:, None], ids]
    return attention(query, context, values, valid, prior)


class RayGroupConditioner(DeformationAwareConditioner):
    MODES = {'semantic-top1', 'semantic-weighted', 'semantic-attention', 'native-spatial-attention'}

    def __init__(self, *, ray_mode, ray_chunk_size=64, **kwargs):
        if (ray_mode not in self.MODES or not isinstance(ray_chunk_size, int)
                or isinstance(ray_chunk_size, bool) or ray_chunk_size < 1):
            raise ValueError('declared ray matching mode and positive chunk size required')
        super().__init__(**kwargs)
        if (self.arm != 'c1' or self.implementation != 'r9-native-v5' or self.value_dim != 1024
                or self.query_dim != 1024 or self.projection_mode != 'native-bilinear'):
            raise ValueError('ray grouping requires native-bilinear SS C1 without deformation channels')
        if (ray_mode.endswith('attention')) != (self.task_attention is not None):
            raise ValueError('only attention modes contain a utility attention module')
        if (ray_mode == 'semantic-top1') != (self.candidate_aggregation == 'top1'):
            raise ValueError('No-score ray control must physically use Top1 feature aggregation')
        if (ray_mode == 'native-spatial-attention') != (self.correspondence_mode == 'native-spatial'):
            raise ValueError('No-corr ray control must use original native spatial evidence')
        if self.task_attention_config is not None and (
                self.task_attention_config.get('train_decoder', self.task_attention_config['decoder'])
                or self.task_attention_config.get('train_score', self.task_attention_config['score_prior'])):
            raise ValueError('initial ray study holds the selected matcher frozen')
        self.q2c.requires_grad_(False)
        self.ray_mode, self.ray_chunk_size = ray_mode, ray_chunk_size

    def forward(self, *, native_voxels, target_omega_voxels, target_native_voxels, ray_ids,
                source_keys, source_values, source_valid, spatial_keys=None, spatial_values=None,
                control='normal'):
        if control != 'normal':
            raise ValueError('ray controls must be explicit matched modes or value-only diagnostic hooks')
        if (native_voxels.shape != target_omega_voxels.shape
                or native_voxels.shape != target_native_voxels.shape or native_voxels.shape[-1] != 1024
                or ray_ids.shape[1] != 3):
            raise ValueError('three source-camera ray groups and aligned1024-D voxel query inputs required')
        batch, views, sites, _ = source_values.shape
        if views != 3:
            raise ValueError('exactly three observed M0 source maps required')
        utility_query = pool_ray_queries(native_voxels, ray_ids, num_rays=sites)
        omega_query = pool_ray_queries(target_omega_voxels, ray_ids, num_rays=sites)
        retarget_query = pool_ray_queries(target_native_voxels, ray_ids, num_rays=sites)
        omega = omega_query.values.flatten(1, 2)
        self._validate(omega, source_keys, source_values, source_valid)
        query_valid = utility_query.valid.flatten(1) & (source_valid.flatten(1).sum(-1) >= 64)[:, None]
        rays = views * sites
        native_spatial = self.ray_mode == 'native-spatial-attention'
        if native_spatial:
            shape = (batch, rays, 3, 1024)
            if spatial_keys is None or spatial_values is None or spatial_keys.shape != shape or spatial_values.shape != shape:
                raise ValueError('native-spatial control needs original projected/pooled3-view keys and values')
            indices = torch.arange(3, device=omega.device).reshape(1, 1, 3).expand(batch, rays, -1)
            valid = query_valid[..., None].expand_as(indices)
        else:
            if spatial_keys is not None or spatial_values is not None:
                raise ValueError('semantic ray arms cannot consume a spatial evidence side channel')
            indices, valid = semantic_ray_proposals(omega, source_keys, source_valid, query_valid, self.ray_chunk_size)
        selected = valid.any(-1).flatten().nonzero().flatten()
        width = indices.shape[-1]
        context = source_values.new_zeros(batch * rays, 1024, dtype=torch.float32)
        score = source_values.new_full((batch * rays, width), -torch.inf, dtype=torch.float32)
        flat_keys, flat_values = source_keys.flatten(1, 2), source_values.flatten(1, 2).float()
        # Same selected D1/scalar matcher, with ray-level Omega descriptors.
        with torch.no_grad(), torch.autocast(device_type=omega.device.type, enabled=False):
            for start in range(0, len(selected), self.ray_chunk_size):
                ix = selected[start:start + self.ray_chunk_size]
                b = ix // rays
                ids, mask = indices.reshape(-1, width)[ix], valid.reshape(-1, width)[ix]
                keys = (spatial_keys.reshape(-1, 3, 1024)[ix] if native_spatial
                        else flat_keys[b[:, None], ids]).float()
                keys = torch.where(mask[..., None], keys, torch.zeros_like(keys))
                ctx = contextual_query(self.q2c.decoder, omega.reshape(-1, 1024)[ix].float(), keys, mask)
                context[ix] = ctx
                nq, nk = F.normalize(ctx, dim=-1), F.normalize(keys, dim=-1)
                cosine = (nq[:, None] * nk).sum(-1)
                score[ix] = self.q2c.fusion_head(nq, nk, cosine[..., None], cosine)
        evidence = source_values.new_zeros(batch * rays, 1024, dtype=torch.float32)
        probabilities = source_values.new_zeros(batch * rays, width, dtype=torch.float32)
        logits = source_values.new_full((batch * rays, width), -torch.inf, dtype=torch.float32)
        outputs, weights, scores = [], [], []
        native_query = utility_query.values.reshape(-1, 1024)
        for start in range(0, len(selected), self.ray_chunk_size):
            ix = selected[start:start + self.ray_chunk_size]
            b = ix // rays
            ids, mask = indices.reshape(-1, width)[ix], valid.reshape(-1, width)[ix]
            prior = score[ix] / self.score_temperature
            if self.task_attention is None:
                p = categorical_weights(score[ix], mask, self.score_temperature)
                if self.ray_mode == 'semantic-top1':
                    p = F.one_hot(score[ix].argmax(-1), width).float() * mask.any(-1)[:, None]
                values = flat_values[b[:, None], ids]
                result = ((p[..., None] * values).sum(1), p, prior)
            elif native_spatial:
                values = spatial_values.reshape(-1, 3, 1024)[ix].float()
                result = self.task_attention(native_query[ix], context[ix], values, mask,
                    prior if self.task_attention_config['score_prior'] else None)
            else:
                args = (self.task_attention, native_query[ix], context[ix], flat_values, b, ids, mask,
                        prior if self.task_attention_config['score_prior'] else None)
                result = (checkpoint(attend_semantic_chunk, *args, use_reentrant=False)
                          if torch.is_grad_enabled() else attend_semantic_chunk(*args))
            e, w, s = result
            outputs.append(e); weights.append(w); scores.append(s)
        if len(selected):
            evidence = evidence.index_copy(0, selected, torch.cat(outputs))
            probabilities = probabilities.index_copy(0, selected, torch.cat(weights))
            logits = logits.index_copy(0, selected, torch.cat(scores))
        else:
            evidence = evidence + sum(p.reshape(-1)[0] * 0 for p in self.parameters() if p.requires_grad)
        residual = evidence.new_zeros(batch * rays, 1024)
        if len(selected):
            value = self.retargeter(torch.cat((evidence[selected], retarget_query.values.reshape(-1, 1024)[selected]), -1), self.g_max)
            residual = residual.index_copy(0, selected, value.float())
        evidence, residual = evidence.reshape(batch, rays, 1024), residual.reshape(batch, rays, 1024)
        probabilities = probabilities.reshape(batch, rays, width)
        query_valid = valid.any(-1)
        native_maps = source_values.float().reshape(batch, rays, 1024)
        delta = torch.tanh(self.fusion_gate) * (evidence - native_maps) + residual
        delta = torch.where(query_valid[..., None], delta, torch.zeros_like(delta))
        view_ids = indices if native_spatial else torch.div(indices, sites, rounding_mode='floor')
        view_weights = evidence.new_zeros(batch, rays, 3).scatter_add(2, view_ids, probabilities)
        output = DeformationCondition(evidence + residual, indices, logits.reshape(batch, rays, width),
            probabilities, view_weights, residual, query_valid, torch.tanh(self.fusion_gate), base_evidence=evidence)
        return RayGroupCondition(delta.reshape(batch, 3, sites, 1024), output)
