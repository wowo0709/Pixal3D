"""Native voxel queries over the original frozen semantic patch candidate pool."""
from __future__ import annotations

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .deform_conditioner import DeformationAwareConditioner, DeformationCondition
from .r9_matching import frame_topk
from .task_matching_attention import contextual_query


def _attend_chunk(attention, query, context, source_values, batch, indices, valid, prior):
    # Gathering inside checkpoint keeps a separate [voxel,K,1024] activation
    # for each chunk out of the retained graph. The shared source map stays
    # unchanged. Calling the Module retains the normal value-intervention hook.
    values = source_values[batch[:, None], indices]
    return attention(query, context, values, valid, prior)


class VoxelQueryConditioner(DeformationAwareConditioner):
    """Change utility queries, retaining FRAME retrieval, D1/score and C1 channels.

    Each voxel consumes the SAME Top64 pool/prior as its original target patch.
    Its utility query is the native four-view projected DINO feature. C1 still
    uses the original 2D target query; no deformation channels are added.
    """

    def __init__(self, *, voxel_chunk_size=128, **kwargs):
        if (isinstance(voxel_chunk_size, bool) or not isinstance(voxel_chunk_size, int)
                or voxel_chunk_size < 1):
            raise ValueError('positive explicit voxel chunk size required')
        super().__init__(**kwargs)
        if (self.implementation != 'r9-native-v5' or self.arm != 'c1'
                or self.correspondence_mode != 'semantic' or self.projection_mode != 'nearest'
                or self.task_attention is None or self.task_attention.content_mode != 'residual-zero'
                or any(p.requires_grad for p in self.q2c.parameters())):
            raise ValueError('voxel query requires native SS C1, frozen matching and zero-output utility')
        self.voxel_chunk_size = voxel_chunk_size

    def forward(self, *, native_query, patch_index, target_keys, source_keys,
                source_values, source_valid, target_query, target_valid,
                semantic_indices=None, candidate_valid=None, matching_logits=None,
                control='normal', **unused):
        if control != 'normal' or unused:
            raise ValueError('voxel query requires explicit unperturbed semantic inputs')
        b, q, views, sites = self._validate(target_keys, source_keys, source_values, source_valid)
        if (native_query.ndim != 3 or native_query.shape[0] != b or native_query.shape[-1] != 1024
                or patch_index.shape != native_query.shape[:2] or patch_index.dtype != torch.long
                or target_query.shape != (b, q, 1024) or target_valid.shape != (b, q)
                or target_valid.dtype != torch.bool):
            raise ValueError('explicit native [B,voxels,1024] and original target patch mapping required')
        if bool(((patch_index < 0) | (patch_index >= q)).any()):
            raise ValueError('voxel query patch index outside target map')
        source_values = source_values.float()
        flat_values = source_values.flatten(1, 2)
        flat_keys = source_keys.flatten(1, 2)
        if semantic_indices is None:
            cosine, indices = frame_topk(target_keys.float(), source_keys.float(), source_valid, topk=64)
            valid = torch.isfinite(cosine)
        else:
            if semantic_indices.shape != (b, q, 64):
                raise ValueError('original semantic Top64 cache required')
            indices = semantic_indices.long()
            if bool(((indices < 0) | (indices >= views * sites)).any()):
                raise ValueError('semantic candidate outside source map')
            valid = source_valid.flatten(1).gather(1, indices.flatten(1)).reshape_as(indices)
            if candidate_valid is not None:
                if candidate_valid.shape != indices.shape or candidate_valid.dtype != torch.bool:
                    raise ValueError('explicit bool candidate support must match Top64')
                valid = valid & candidate_valid
        valid = valid & target_valid[..., None] & (source_valid.flatten(1).sum(-1) >= 64)[:, None, None]
        active_patches = valid.any(-1).reshape(-1).nonzero().flatten()
        contexts = source_values.new_zeros(b * q, 1024)
        priors = source_values.new_full((b * q, 64), -torch.inf)
        # D1 and the scalar scorer are frozen, exactly as in the patch-query
        # reference. Cached logits deliberately do not override that head.
        with torch.no_grad(), torch.autocast(device_type=source_values.device.type, enabled=False):
            for start in range(0, len(active_patches), self.task_attention_config['chunk_size']):
                ix = active_patches[start:start + self.task_attention_config['chunk_size']]
                batch = ix // q
                ids, mask = indices.reshape(-1, 64)[ix], valid.reshape(-1, 64)[ix]
                keys = flat_keys[batch[:, None], ids].float()
                keys = torch.where(mask[..., None], keys, torch.zeros_like(keys))
                context = contextual_query(self.q2c.decoder, target_keys.reshape(-1, 1024)[ix].float(), keys, mask)
                contexts[ix] = context
                if self.task_attention_config['score_prior']:
                    nq, nk = F.normalize(context, dim=-1), F.normalize(keys, dim=-1)
                    cosine = (nq[:, None] * nk).sum(-1)
                    priors[ix] = self.q2c.fusion_head(nq, nk, cosine[..., None], cosine) / self.score_temperature

        voxels = native_query.shape[1]
        flat_patch = (torch.arange(b, device=patch_index.device)[:, None] * q + patch_index).flatten()
        voxel_valid = valid.any(-1).reshape(-1)[flat_patch]
        selected = voxel_valid.nonzero().flatten()
        evidence = source_values.new_zeros(b * voxels, 1024)
        weights = source_values.new_zeros(b * voxels, 64)
        logits = source_values.new_full((b * voxels, 64), -torch.inf)
        outputs, probabilities, scores = [], [], []
        for start in range(0, len(selected), self.voxel_chunk_size):
            ix = selected[start:start + self.voxel_chunk_size]
            patches = flat_patch[ix]
            arguments = (self.task_attention, native_query.reshape(-1, 1024)[ix].float(),
                contexts[patches], flat_values, ix // voxels,
                indices.reshape(-1, 64)[patches], valid.reshape(-1, 64)[patches],
                priors[patches] if self.task_attention_config['score_prior'] else None)
            if torch.is_grad_enabled():
                e, w, l = checkpoint(_attend_chunk, *arguments, use_reentrant=False)
            else:
                e, w, l = _attend_chunk(*arguments)
            outputs.append(e); probabilities.append(w); scores.append(l)
        if len(selected):
            evidence = evidence.index_copy(0, selected, torch.cat(outputs))
            weights = weights.index_copy(0, selected, torch.cat(probabilities))
            logits = logits.index_copy(0, selected, torch.cat(scores))
        else:
            evidence = evidence + sum(p.reshape(-1)[0] * 0 for p in self.parameters() if p.requires_grad)

        original_target = target_query.reshape(-1, 1024)[flat_patch].float()
        residual = source_values.new_zeros(b * voxels, 1024)
        if len(selected):
            active_residual = self.retargeter(torch.cat([evidence[selected], original_target[selected]], -1), self.g_max)
            residual = active_residual.new_zeros(b * voxels, 1024)
            residual = residual.index_copy(0, selected, active_residual)
        evidence = evidence.reshape(b, voxels, 1024)
        residual = residual.reshape_as(evidence)
        weights = weights.reshape(b, voxels, 64)
        candidate_indices = indices.reshape(-1, 64)[flat_patch].reshape(b, voxels, 64)
        view_index = torch.div(candidate_indices, sites, rounding_mode='floor')
        view_weights = evidence.new_zeros(b, voxels, views).scatter_add(2, view_index, weights)
        return DeformationCondition(evidence + residual, candidate_indices,
            logits.reshape(b, voxels, 64), weights, view_weights, residual,
            voxel_valid.reshape(b, voxels), torch.tanh(self.fusion_gate), base_evidence=evidence)


def fuse_voxel_condition(condition, anchor, output, source_views):
    """Original zero-gated native fusion, with evidence already at voxel sites."""
    if source_views != 3:
        raise ValueError('native voxel query retains three M0 source views')
    key = 'proj_semantic' if 'proj_semantic' in condition else 'proj'
    native = condition[key]
    if native.shape != output.evidence.shape or native.shape != anchor[key].shape:
        raise ValueError('native query, target anchor and evidence grid must agree')
    semantic = (anchor[key] + 3 * output.base_evidence.to(native.dtype)) / 4
    update = output.fusion_scale.to(native.dtype) * (semantic - native)
    update = update + .75 * output.retarget_residual.to(native.dtype)
    condition[key] = native + torch.where(output.query_valid[..., None], update, torch.zeros_like(update))
    return condition
