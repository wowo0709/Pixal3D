"""One additive physical-DQ arm on the frozen four-head source-ray model.

Matching probabilities and the candidate support are shared by the feature
and RAW Omega displacement readouts. No GT displacement or geometry is input.
"""
import torch
from torch.utils.checkpoint import checkpoint

from .deform_conditioner import DeformationCondition
from .member_feature_source_ray import MemberFeatureSourceRayConditioner,_feature_chunk
from .member_ray_posterior import member_ray_posteriors
from .ray_deformation_query import DeformationQueryRetargeter
from .ray_group_query import pool_ray_queries
from .target_first_ray_v2 import TargetFirstRayEvidence


@torch.no_grad()
def selected_target_ray_context(target_context,patch_ids,ray_ids,selected):
    """Same unmasked native ray mean, bounded to one asset before gathering.

    Avoid materializing full[B,4096,2048] or[B,3072,2048] RAW tensors.
    Only actual active source-ray rows are retained, in their original order.
    """
    output=target_context.new_empty((len(selected),2048),dtype=torch.float32)
    assets=selected//3072
    for asset in range(target_context.shape[0]):
        positions=(assets==asset).nonzero().flatten()
        if not len(positions):continue
        patch=patch_ids[asset:asset+1]
        raw=target_context[asset:asset+1].gather(1,patch[...,None].expand(-1,-1,2048))
        pooled=pool_ray_queries(raw,ray_ids[asset:asset+1],num_rays=1024)
        values=pooled.values.reshape(3072,2048)[selected[positions]%3072]
        output.index_copy_(0,positions,values)
    return output


def geometry_feature_chunk(head,native,context,values,keys,raw_source,raw_target,
                           batch,indices,valid,prior,control):
    feature,probability=_feature_chunk(head,native,context,values,keys,batch,indices,valid,prior,control)
    raw=raw_source[batch[:,None],indices].float()
    raw=torch.where(valid[...,None],raw,torch.zeros_like(raw))
    if not torch.isfinite(raw).all():
        raise ValueError('selected RAW correspondence descriptors must be finite')
    source=(probability.mean(1)[...,None]*raw).sum(1)
    return feature,raw_target.float()-source,probability


class MemberFeatureDeformationSourceRayConditioner(MemberFeatureSourceRayConditioner):
    """Same19 weights/function plus one zero-initialized2048-D projection.

    The new2048 raw observations are nonzero data; only their projection starts
    at zero. The No-DQ class physically lacks these channels and this parameter.
    """
    def __init__(self,*,deformation_dim,**kwargs):
        if deformation_dim!=2048:
            raise ValueError('explicit2048 physical RAW deformation-query channels required')
        super().__init__(**kwargs)
        reference=self.retargeter
        with torch.random.fork_rng(devices=[]):
            extended=DeformationQueryRetargeter(2048,1024,kwargs.get('hidden_dim',256),2048)
        state=extended.state_dict();state.update(reference.state_dict())
        extended.load_state_dict(state,strict=True)
        self.retargeter=extended
        self.deformation_dim=2048

    def forward(self, *, native_voxels, target_omega_voxels, target_native_voxels,
                frame_target_keys, target_patch_ids, ray_ids, observed_query,
                source_keys, source_values, source_valid, target_context, source_context, control='normal'):
        if (control not in {'normal', 'wrong-feature-values-fixed-use'}
                or native_voxels.shape != target_native_voxels.shape
                or native_voxels.shape != target_omega_voxels.shape
                or native_voxels.shape[-1] != 1024
                or source_values.shape != (native_voxels.shape[0], 3, 1024, 1024)
                or not self.g_max > 0):
            raise ValueError('aligned nativeSS/source-camera rays and positive bound required')
        if control != 'normal' and (self.training or torch.is_grad_enabled()):
            raise ValueError('feature correspondence intervention requires eval and no_grad')
        batch, rays, count = source_values.shape[0], 3072, 64
        if (target_context.shape != (batch,1024,2048)
                or source_context.shape != (batch,3,1024,2048)
                or not torch.equal(target_context[...,:1024],frame_target_keys)
                or not torch.equal(source_context[...,:1024],source_keys)):
            raise ValueError('explicit RAW FRAME+GLOBAL contexts must match the original FRAME keys')
        utility = pool_ray_queries(native_voxels, ray_ids, num_rays=1024)
        target = pool_ray_queries(target_native_voxels, ray_ids, num_rays=1024)
        posterior = member_ray_posteriors(self.q2c, target_keys=frame_target_keys,
            source_keys=source_keys, source_valid=source_valid, ray_ids=ray_ids,
            patch_ids=target_patch_ids, observed_query=observed_query,
            temperature=self.score_temperature, chunk_size=self.ray_chunk_size)
        indices, valid = posterior.candidate_indices, posterior.candidate_valid
        selected = valid.any(-1).flatten().nonzero().flatten()
        raw_target = selected_target_ray_context(target_context,target_patch_ids,ray_ids,selected)
        raw_source = source_context.flatten(1,2)
        flat_values, flat_keys = source_values.flatten(1, 2).float(), source_keys.flatten(1, 2)
        prior = posterior.probabilities.log().reshape(-1, count)
        query = utility.values.reshape(-1, 1024)
        context = posterior.contextual_query.reshape(-1, 1024)
        top1 = source_values.new_zeros(batch*rays, 1024, dtype=torch.float32)
        hard = source_values.new_zeros(batch*rays, count, dtype=torch.float32)
        appearances, changes, hard_weights, feature_weights, geometries = [], [], [], [], []
        for start in range(0, len(selected), self.ray_chunk_size):
            take = selected[start:start+self.ray_chunk_size]
            b = take//rays
            ids, mask = indices.reshape(-1, count)[take], valid.reshape(-1, count)[take]
            value, weight, _ = self._attention(prior[take][None],
                flat_values[b[:, None], ids][None], mask[None], (ids//1024)[None], 3)
            appearances.append(value[0]); hard_weights.append(weight[0])
            args = (self.ray_readout, query[take], context[take], flat_values,
                    flat_keys, raw_source, raw_target[start:start+len(take)],
                    b, ids, mask, prior[take], control)
            change, geometry, probability = (checkpoint(geometry_feature_chunk,*args,use_reentrant=False)
                if torch.is_grad_enabled() else geometry_feature_chunk(*args))
            changes.append(change); feature_weights.append(probability.detach()); geometries.append(geometry)
        residual = torch.zeros_like(top1)
        target_query = target.values.reshape(batch*rays, 1024)
        if len(selected):
            appearance, change = torch.cat(appearances), torch.cat(changes)
            top1 = top1.index_copy(0, selected, appearance)
            hard = hard.index_copy(0, selected, torch.cat(hard_weights))
            geometry = torch.cat(geometries)
            retargeted = self.retargeter(torch.cat((appearance+change, target_query[selected],geometry), -1), self.g_max)
            self.last_deformation_query = geometry.detach()
            direct = self.g_max*torch.tanh(change/self.g_max)
            residual = residual.index_copy(0, selected, retargeted.float()+direct)
            self.last_direct_feature_correction = direct.detach()
        else:
            # All trainable tensors remain DDP-connected on an empty rank.
            top1 = top1+sum(p.reshape(-1)[0]*0 for p in self.parameters() if p.requires_grad)
            self.last_direct_feature_correction = source_values.new_empty((0, 1024), dtype=torch.float32)
            self.last_deformation_query = source_values.new_empty((0,2048),dtype=torch.float32)
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
        # Only active four-head rows are retained, detached; no dense[B,3072,4,64] graph.
        self.last_feature_probabilities = (selected.detach(), torch.cat(feature_weights)
            if feature_weights else source_values.new_empty((0, 4, count), dtype=torch.float32))
        self._last_member_retained_mass = posterior.retained_mass.detach()
        self._last_distinct_member_count = posterior.distinct_member_count.detach()
        return TargetFirstRayEvidence(numerator.reshape(batch, 3, 1024, 1024),
            mass.reshape(batch, 3, 1024, 1), attention, use)
