"""Matched physical Top1 and original-projection controls for four-head rays.

The semantic Top1 control keeps exactly the value/output transform and R/use
weights, with no Q/K/score-selector parameters. The spatial control keeps all
nine head tensors and replaces semantic retrieval with the three native
projected source views. Both retain the native MV4 carrier at initialization.
"""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .deform_conditioner import DeformationCondition
from .member_feature_source_ray import MemberFeatureSourceRayConditioner
from .member_ray_posterior import MemberRayPosterior
from .r9_matching import categorical_weights
from .ray_group_query import pool_ray_queries
from .target_first_ray_v2 import TargetFirstRayEvidence
from .task_matching_attention import contextual_query


class Top1FeatureReadout(nn.Module):
    """One selected correspondence; only the shared V and output weights exist."""
    def __init__(self, full_head):
        super().__init__()
        self.value, self.output = full_head.value, full_head.output
        self.value_width, self.heads, self.head_width = 1024, 4, 64
        self.value_control = 'normal'

    def forward(self, native, context, keys, values, valid, prior):
        rows, candidates = valid.shape
        if (values.shape != (rows, candidates, 1024) or prior.shape != valid.shape
                or valid.dtype != torch.bool or not valid.any(-1).all()):
            raise ValueError('active native matching candidates and prior required')
        order = prior.masked_fill(~valid, -torch.inf).argsort(dim=-1, descending=True, stable=True)
        chosen = order[:, :1]
        if self.value_control == 'normal':
            value_id = chosen
        elif self.value_control == 'wrong-feature-values-fixed-use':
            if self.training or torch.is_grad_enabled():
                raise ValueError('saved-weight Top1 value control requires eval/no_grad')
            value_id = order.gather(1, (valid.sum(-1, keepdim=True) // 2).clamp_max(candidates - 1))
        else:
            raise ValueError('unknown physical Top1 feature control')
        selected = values.gather(1, value_id[..., None].expand(-1, -1, 1024))[:, 0].float()
        # Exactly the same1024->256 V and256->1024 output as the four-head arm.
        transformed = self.value(F.layer_norm(selected, (1024,)))
        hard = torch.zeros_like(prior).scatter(1, chosen, 1.)
        return self.output(transformed).float(), hard[:, None].expand(-1, 4, -1)


class Top1FeatureSourceRayConditioner(MemberFeatureSourceRayConditioner):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ray_readout = Top1FeatureReadout(self.ray_readout)

    def load_utility(self, state):
        raise ValueError('physical Top1 has no utility-score module to load')

    def forward(self, *, control='normal', **kwargs):
        previous = self.ray_readout.value_control
        if control not in {'normal', 'wrong-feature-values-fixed-use'}:
            raise ValueError('explicit physical Top1 value control required')
        self.ray_readout.value_control = control
        try:
            # The original feature-chunk intervention assumes a K-wide V map.
            # This head projects just one feature, with its own value control.
            return super().forward(control='normal', **kwargs)
        finally:
            self.ray_readout.value_control = previous


@torch.no_grad()
def spatial_member_posteriors(matcher, *, target_keys, source_valid, ray_ids,
                              patch_ids, observed_query, spatial_keys,
                              temperature, chunk_size):
    """Same distinct target-patch members, D1 decoder and score; native3 keys.

    Candidate keys were projected through the original three input cameras
    and pooled along the complete source ray. They are not semantic patch
    indices, nearest neighbors in world GT, or a uniform confidence control.
    """
    batch = target_keys.shape[0]
    if (target_keys.shape != (batch, 1024, 1024)
            or spatial_keys.shape != (batch, 3072, 3, 1024)
            or source_valid.shape != (batch, 3, 1024) or source_valid.dtype != torch.bool
            or ray_ids.shape[:2] != (batch, 3) or ray_ids.dtype != torch.long
            or patch_ids.shape != (batch, ray_ids.shape[-1]) or patch_ids.dtype != torch.long
            or observed_query.shape != patch_ids.shape or observed_query.dtype != torch.bool
            or ((ray_ids < -1) | (ray_ids >= 1024)).any()
            or ((patch_ids < 0) | (patch_ids >= 1024)).any()
            or not 0 < temperature or not isinstance(chunk_size, int) or chunk_size < 1
            or any(parameter.requires_grad for parameter in matcher.parameters())):
        raise ValueError('native spatial keys and frozen distinct-member matching protocol required')
    device = target_keys.device
    indices = torch.arange(3, device=device).reshape(1, 1, 3).expand(batch, 3072, -1)
    valid = torch.zeros(batch, 3072, 3, dtype=torch.bool, device=device)
    probability = torch.zeros(batch, 3072, 3, device=device, dtype=torch.float32)
    context = torch.zeros(batch, 3072, 1024, device=device, dtype=torch.float32)
    counts = torch.zeros(batch, 3072, device=device, dtype=torch.long)
    for b in range(batch):
        pairs = []
        for view in range(3):
            seen = observed_query[b] & (ray_ids[b, view] >= 0)
            pairs.append((ray_ids[b, view, seen] + view * 1024) * 1024 + patch_ids[b, seen])
        encoded = torch.cat(pairs).unique(sorted=True)
        ray, patch = encoded // 1024, encoded % 1024
        counts[b] = torch.bincount(ray, minlength=3072)
        if not len(encoded) or int(source_valid[b].sum()) < 64:
            continue
        with torch.autocast(device_type=device.type, enabled=False):
            for start in range(0, len(encoded), chunk_size):
                take = slice(start, start + chunk_size)
                ids = ray[take]; keys = spatial_keys[b, ids].float()
                mask = torch.ones(len(ids), 3, dtype=torch.bool, device=device)
                ctx = contextual_query(matcher.decoder, target_keys[b, patch[take]].float(), keys, mask)
                nq, nk = F.normalize(ctx, dim=-1), F.normalize(keys, dim=-1)
                cosine = (nq[:, None] * nk).sum(-1)
                score = matcher.fusion_head(nq, nk, cosine[..., None], cosine)
                context[b].index_add_(0, ids, ctx)
                probability[b].index_add_(0, ids, categorical_weights(score / temperature, mask, 1.))
        active = counts[b] > 0
        context[b] /= counts[b].clamp_min(1)[:, None]
        probability[b] /= counts[b].clamp_min(1)[:, None]
        valid[b, active] = True
    retained = valid.any(-1).float()
    return MemberRayPosterior(indices, valid, probability, context, retained, counts)


def _spatial_feature_chunk(head, native, context, keys, values, indices, valid, prior, control):
    """Gather bounded active rows inside checkpoint; perturb projected V only."""
    handle = None
    if control == 'wrong-feature-values-fixed-use':
        if head.training or torch.is_grad_enabled():
            raise ValueError('saved-weight spatial value control requires eval/no_grad')
        handle = head.value.register_forward_hook(lambda layer, args, output: output.roll(1, dims=1))
    elif control != 'normal':
        raise ValueError('unknown spatial feature control')
    try:
        return head(native, context, keys[indices].float(), values[indices].float(), valid, prior)
    finally:
        if handle is not None:
            handle.remove()


class SpatialFeatureSourceRayConditioner(MemberFeatureSourceRayConditioner):
    """Same19 trainable tensors and full feature/R formula; original spatial3."""
    def __init__(self, **kwargs):
        if kwargs.pop('correspondence_mode', None) != 'native-spatial':
            raise ValueError('NoCorr requires original native spatial correspondence')
        super().__init__(correspondence_mode='semantic', **kwargs)
        self.correspondence_mode = 'native-spatial'

    def forward(self, *, native_voxels, target_omega_voxels, target_native_voxels,
                frame_target_keys, target_patch_ids, ray_ids, observed_query,
                source_keys, source_values, source_valid, spatial_keys, spatial_values,
                control='normal'):
        if (control not in {'normal', 'wrong-feature-values-fixed-use'}
                or native_voxels.shape != target_native_voxels.shape
                or native_voxels.shape != target_omega_voxels.shape
                or native_voxels.shape[-1] != 1024
                or source_values.shape != (native_voxels.shape[0], 3, 1024, 1024)
                or spatial_values.shape != spatial_keys.shape or not self.g_max > 0):
            raise ValueError('aligned native MV4 source rays and original spatial keys/values required')
        if control != 'normal' and (self.training or torch.is_grad_enabled()):
            raise ValueError('saved spatial feature control requires eval/no_grad')
        batch, rays, count = source_values.shape[0], 3072, 3
        utility = pool_ray_queries(native_voxels, ray_ids, num_rays=1024)
        target = pool_ray_queries(target_native_voxels, ray_ids, num_rays=1024)
        posterior = spatial_member_posteriors(self.q2c, target_keys=frame_target_keys,
            source_valid=source_valid, ray_ids=ray_ids, patch_ids=target_patch_ids,
            observed_query=observed_query, spatial_keys=spatial_keys,
            temperature=self.score_temperature, chunk_size=self.ray_chunk_size)
        indices, valid = posterior.candidate_indices, posterior.candidate_valid
        selected = valid.any(-1).flatten().nonzero().flatten()
        prior = posterior.probabilities.log().reshape(-1, count)
        keys, values = spatial_keys.reshape(-1, count, 1024), spatial_values.reshape(-1, count, 1024)
        query = utility.values.reshape(-1, 1024); context = posterior.contextual_query.reshape(-1, 1024)
        top1 = source_values.new_zeros(batch * rays, 1024, dtype=torch.float32)
        hard = source_values.new_zeros(batch * rays, count, dtype=torch.float32)
        appearances, changes, hard_weights, feature_weights = [], [], [], []
        for start in range(0, len(selected), self.ray_chunk_size):
            take = selected[start:start + self.ray_chunk_size]; mask = valid.reshape(-1, count)[take]
            value, weight, _ = self._attention(prior[take][None], values[take][None], mask[None], indices.reshape(-1, count)[take][None], 3)
            appearances.append(value[0]); hard_weights.append(weight[0])
            args = (self.ray_readout, query[take], context[take], keys, values, take, mask, prior[take], control)
            change, probability = (checkpoint(_spatial_feature_chunk, *args, use_reentrant=False)
                if torch.is_grad_enabled() else _spatial_feature_chunk(*args))
            changes.append(change); feature_weights.append(probability.detach())
        residual = torch.zeros_like(top1); target_query = target.values.reshape(batch * rays, 1024)
        if len(selected):
            appearance, change = torch.cat(appearances), torch.cat(changes)
            top1 = top1.index_copy(0, selected, appearance)
            hard = hard.index_copy(0, selected, torch.cat(hard_weights))
            retargeted = self.retargeter(torch.cat((appearance + change, target_query[selected]), -1), self.g_max)
            direct = self.g_max * torch.tanh(change / self.g_max)
            residual = residual.index_copy(0, selected, retargeted.float() + direct)
            self.last_direct_feature_correction = direct.detach()
        else:
            top1 = top1 + sum(p.reshape(-1)[0] * 0 for p in self.parameters() if p.requires_grad)
            self.last_direct_feature_correction = source_values.new_empty((0, 1024), dtype=torch.float32)
        top1 = top1.reshape(batch, rays, 1024); residual = residual.reshape(batch, rays, 1024)
        hard = hard.reshape(batch, rays, count); query_valid = valid.any(-1); scale = torch.tanh(self.fusion_gate)
        attention = DeformationCondition(top1 + residual, indices, prior.reshape(batch, rays, count),
            hard, hard, residual, query_valid, scale, base_evidence=top1)
        use = self.evidence_use(target_query.reshape(batch, rays, 1024), top1, query_valid)
        mass = use[..., None] * scale; numerator = mass * top1 + use[..., None] * residual
        self.last_feature_probabilities = (selected.detach(), torch.cat(feature_weights)
            if feature_weights else source_values.new_empty((0, 4, count), dtype=torch.float32))
        self._last_member_retained_mass = posterior.retained_mass.detach()
        self._last_distinct_member_count = posterior.distinct_member_count.detach()
        return TargetFirstRayEvidence(numerator.reshape(batch, 3, 1024, 1024),
            mass.reshape(batch, 3, 1024, 1), attention, use)
