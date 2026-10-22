"""Source-ray Top-K information reaches the consumer only through R.

Retain the native target stream, semantic Top1 carrier and independent source
use decision. Compare learned feature input with an additional raw weighted
appearance input while removing the head-to-consumer correction bypass.
"""
import torch
from torch.utils.checkpoint import checkpoint

from .bounded_source_contrast_member_feature import BoundedSourceContrastMemberFeatureConditioner
from .deform_conditioner import DeformationCondition
from .member_feature_source_ray import _feature_chunk
from .member_ray_posterior import member_ray_posteriors
from .ray_group_query import pool_ray_queries
from .target_first_ray_v2 import TargetFirstRayEvidence


APPEARANCE_MODES = ('top1-plus-feature', 'raw-topk-plus-feature')


def _input_chunk(head, native, context, values, keys, batch, indices, valid, prior,
                 control, raw_topk):
    legacy_control = ('normal' if control == 'normal'
                      else 'wrong-feature-values-fixed-use')
    change, probability = _feature_chunk(head, native, context, values, keys, batch,
                                        indices, valid, prior, legacy_control)
    if not raw_topk:
        # No raw aggregation is added to the existing Top1+feature R input.
        return change, change.new_empty((len(change), 0)), probability
    appearance = values[batch[:, None], indices].float()
    if control != 'normal':
        count = valid.sum(-1, keepdim=True)
        positions = torch.arange(valid.shape[-1], device=valid.device).expand_as(valid)
        ordered = positions.masked_fill(~valid, valid.shape[-1]).sort(-1).values
        rank = valid.long().cumsum(-1)-1
        donor_rank = (rank+count//2) % count.clamp_min(1)
        donor = ordered.gather(1, donor_rank).clamp_max(valid.shape[-1]-1)
        appearance = appearance.gather(1, donor[..., None].expand_as(appearance))
    appearance = torch.where(valid[..., None], appearance, torch.zeros_like(appearance))
    # The raw values and learned transform follow the SAME four-head posterior.
    # Their unnormalized native feature magnitudes are never replaced by zeros.
    aggregate = (probability.mean(1)[..., None]*appearance).sum(1)
    return change, aggregate, probability


class RMediatedSourceContrastConditioner(BoundedSourceContrastMemberFeatureConditioner):
    def __init__(self, *, appearance_mode, **kwargs):
        if appearance_mode not in APPEARANCE_MODES:
            raise ValueError('explicit R-mediated appearance mode required')
        super().__init__(**kwargs)
        self.appearance_mode = appearance_mode
        marker = ('r_mediated_feature_schema' if appearance_mode == APPEARANCE_MODES[0]
                  else 'r_mediated_raw_topk_schema')
        # Strict loading cannot silently reinterpret either trained parent or
        # the other appearance mode, despite identical nineteen parameters.
        self.register_buffer(marker, torch.tensor([1], dtype=torch.uint8))

    def forward(self, *, native_voxels, target_omega_voxels, target_native_voxels,
                frame_target_keys, target_patch_ids, ray_ids, observed_query,
                source_keys, source_values, source_valid, control='normal'):
        if (control not in {'normal', 'wrong-retrieved-values-fixed-use'}
                or native_voxels.shape != target_native_voxels.shape
                or native_voxels.shape != target_omega_voxels.shape
                or native_voxels.shape[-1] != 1024
                or source_values.shape != (native_voxels.shape[0], 3, 1024, 1024)
                or not self.g_max > 0):
            raise ValueError('aligned nativeSS/source-camera rays and positive bound required')
        if control != 'normal' and (self.training or torch.is_grad_enabled()):
            raise ValueError('retrieved-value diagnostic requires eval and no_grad')
        batch, rays, count = source_values.shape[0], 3072, 64
        utility = pool_ray_queries(native_voxels, ray_ids, num_rays=1024)
        target = pool_ray_queries(target_native_voxels, ray_ids, num_rays=1024)
        posterior = member_ray_posteriors(self.q2c, target_keys=frame_target_keys,
            source_keys=source_keys, source_valid=source_valid, ray_ids=ray_ids,
            patch_ids=target_patch_ids, observed_query=observed_query,
            temperature=self.score_temperature, chunk_size=self.ray_chunk_size)
        indices, valid = posterior.candidate_indices, posterior.candidate_valid
        selected = valid.any(-1).flatten().nonzero().flatten()
        flat_values = source_values.flatten(1, 2).float()
        flat_keys = source_keys.flatten(1, 2)
        prior = posterior.probabilities.log().reshape(-1, count)
        query = utility.values.reshape(-1, 1024)
        context = posterior.contextual_query.reshape(-1, 1024)
        top1 = source_values.new_zeros(batch*rays, 1024, dtype=torch.float32)
        hard = source_values.new_zeros(batch*rays, count, dtype=torch.float32)
        appearances, changes, aggregates, hard_weights, feature_weights = [], [], [], [], []
        raw_topk = self.appearance_mode == 'raw-topk-plus-feature'
        for start in range(0, len(selected), self.ray_chunk_size):
            take = selected[start:start+self.ray_chunk_size]
            b = take//rays
            ids, mask = indices.reshape(-1, count)[take], valid.reshape(-1, count)[take]
            value, weight, _ = self._attention(prior[take][None],
                flat_values[b[:, None], ids][None], mask[None], (ids//1024)[None], 3)
            appearances.append(value[0]); hard_weights.append(weight[0])
            args = (self.ray_readout, query[take], context[take], flat_values,
                    flat_keys, b, ids, mask, prior[take], control, raw_topk)
            change, aggregate, probability = (checkpoint(_input_chunk, *args, use_reentrant=False)
                if torch.is_grad_enabled() else _input_chunk(*args))
            changes.append(change); aggregates.append(aggregate)
            feature_weights.append(probability.detach())
        residual = torch.zeros_like(top1)
        target_query = target.values.reshape(batch*rays, 1024)
        if len(selected):
            appearance, change = torch.cat(appearances), torch.cat(changes)
            top1 = top1.index_copy(0, selected, appearance)
            hard = hard.index_copy(0, selected, torch.cat(hard_weights))
            R_appearance = torch.cat(aggregates) if raw_topk else appearance
            own_source = flat_values.reshape(-1, 1024)[selected]
            with self.retargeter.native_source_context(own_source):
                retargeted = self.retargeter(torch.cat((R_appearance+change, target_query[selected]), -1), self.g_max)
            residual = residual.index_copy(0, selected, retargeted.float())
            self.last_R_only_output = retargeted.detach().float()
            self.last_R_input_appearance = R_appearance.detach()
            self.last_direct_feature_correction = torch.zeros_like(change.detach())
        else:
            top1 = top1+sum(p.reshape(-1)[0]*0 for p in self.parameters() if p.requires_grad)
            self.last_R_input_appearance = source_values.new_empty((0, 1024), dtype=torch.float32)
            self.last_R_only_output = source_values.new_empty((0, 1024), dtype=torch.float32)
            self.last_direct_feature_correction = source_values.new_empty((0, 1024), dtype=torch.float32)
        top1 = top1.reshape(batch, rays, 1024)
        residual = residual.reshape(batch, rays, 1024)
        hard = hard.reshape(batch, rays, count)
        query_valid = valid.any(-1)
        view_weights = top1.new_zeros(batch, rays, 3).scatter_add(2, indices//1024, hard)
        scale = torch.tanh(self.fusion_gate)
        attention = DeformationCondition(top1+residual, indices,
            prior.reshape(batch, rays, count), hard, view_weights, residual,
            query_valid, scale, base_evidence=top1)
        use = self.evidence_use(target_query.reshape(batch, rays, 1024), top1, query_valid)
        mass = use[..., None]*scale
        numerator = mass*top1+use[..., None]*residual
        self.last_feature_probabilities = (selected.detach(), torch.cat(feature_weights)
            if feature_weights else source_values.new_empty((0, 4, count), dtype=torch.float32))
        self.last_source_reference_rows = selected.detach()
        self._last_member_retained_mass = posterior.retained_mass.detach()
        self._last_distinct_member_count = posterior.distinct_member_count.detach()
        return TargetFirstRayEvidence(numerator.reshape(batch, 3, 1024, 1024),
            mass.reshape(batch, 3, 1024, 1), attention, use)
