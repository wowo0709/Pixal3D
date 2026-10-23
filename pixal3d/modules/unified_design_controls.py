"""Four single-design controls on the frozen common native R-only carrier.

New versions leave all running bound implementations unchanged. Voxel queries
share ray candidates/context and pool BOTH readouts before the original R.
"""
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .bounded_member_feature_source_ray import BoundedPriorRayFeatureReadout
from .deform_conditioner import DeformationCondition
from .member_ray_posterior import MemberRayPosterior, member_ray_posteriors
from .r9_matching import categorical_weights
from .r_mediated_source_contrast import RMediatedSourceContrastConditioner, _input_chunk
from .r_mediated_top1_source import RMediatedTop1SourceConditioner
from .ray_group_query import pool_ray_queries
from .target_first_ray_v2 import TargetFirstRayEvidence
from .task_matching_attention import contextual_query


VARIANTS = ('FULL_QK_ONLY', 'FULL_VOXEL_Q', 'FULL_MEAN_MATCH', 'C_DECODER_NO_R')
SOURCE_COUNTS = {v: 5 if v == 'C_DECODER_NO_R' else 19 for v in VARIANTS}


class NoPriorBoundedReadout(BoundedPriorRayFeatureReadout):
    """Remove only the additive attention prior, including its frozen anchor.

    The matcher still supplies candidates, contextual descriptors and the
    Top1 carrier. Finite zero logits are neutral offsets, not observed feature
    channels. They also prevent inf-minus-inf on zero matching mass keys.
    """
    def forward(self, native, context, keys, values, valid, prior):
        return super().forward(native, context, keys, values, valid,
                               torch.zeros_like(prior))


def distinct_ray_patch_pairs(ray_ids, patch_ids, observed_query):
    """Exact once-per-distinct-target-patch membership; no NN evaluation."""
    if (ray_ids.ndim != 2 or ray_ids.shape[0] != 3
            or patch_ids.shape != ray_ids.shape[1:] or observed_query.shape != patch_ids.shape
            or ray_ids.dtype != torch.long or patch_ids.dtype != torch.long
            or observed_query.dtype != torch.bool
            or ((ray_ids < -1) | (ray_ids >= 1024)).any()
            or ((patch_ids < 0) | (patch_ids >= 1024)).any()):
        raise ValueError('three native camera ray maps and observed target-patch addresses required')
    pairs=[]
    for view in range(3):
        seen=observed_query & (ray_ids[view] >= 0)
        pairs.append((ray_ids[view,seen] + view*1024)*1024 + patch_ids[seen])
    encoded=torch.cat(pairs).unique(sorted=True)
    return encoded//1024, encoded%1024


@torch.no_grad()
def mean_ray_posteriors(matcher, *, target_keys, source_keys, source_valid,
                        ray_ids, patch_ids, observed_query,
                        temperature=.05, chunk_size=64):
    """One mean descriptor before RAW64/D1 ranking, keeping the same members.

    Full uses a mixture of distinct members' posteriors. This control moves
    averaging before proposal retrieval/decoding, while preserving ray queries,
    observation support, all delivery operations and the same frozen matcher.
    """
    batch=len(target_keys)
    if (target_keys.shape!=(batch,1024,1024) or source_keys.shape!=(batch,3,1024,1024)
            or source_valid.shape!=(batch,3,1024) or source_valid.dtype!=torch.bool
            or ray_ids.ndim!=3 or ray_ids.shape[:2]!=(batch,3)
            or patch_ids.shape!=(batch,ray_ids.shape[-1])
            or observed_query.shape!=patch_ids.shape or not temperature > 0
            or not isinstance(chunk_size,int) or isinstance(chunk_size,bool) or chunk_size<1
            or any(p.requires_grad for p in matcher.parameters())):
        raise ValueError('same frozen matcher/native member observation contract required')
    device=target_keys.device
    indices=torch.zeros(batch,3072,64,dtype=torch.long,device=device)
    valid=torch.zeros_like(indices,dtype=torch.bool)
    probability=torch.zeros(batch,3072,64,device=device)
    context=torch.zeros(batch,3072,1024,device=device)
    retained=torch.zeros(batch,3072,device=device)
    counts=torch.zeros(batch,3072,dtype=torch.long,device=device)
    for b in range(batch):
        ray,patch=distinct_ray_patch_pairs(ray_ids[b],patch_ids[b],observed_query[b])
        counts[b]=torch.bincount(ray,minlength=3072)
        if not len(ray) or int(source_valid[b].sum())<64:continue
        used,local=ray.unique(sorted=True,return_inverse=True)
        with torch.autocast(device_type=device.type,enabled=False):
            mean=torch.zeros(len(used),1024,device=device)
            mean.index_add_(0,local,target_keys[b,patch].float())
            mean=mean/counts[b,used,None]
            source=source_keys[b].flatten(0,1).float()
            for start in range(0,len(used),chunk_size):
                take=used[start:start+chunk_size];query=mean[start:start+chunk_size]
                raw=F.normalize(query,dim=-1)@F.normalize(source,dim=-1).T
                ids=raw.masked_fill(~source_valid[b].flatten()[None],-torch.inf).topk(64,dim=-1).indices
                mask=source_valid[b].flatten()[ids];keys=source[ids]
                ctx=contextual_query(matcher.decoder,query,keys,mask)
                nq,nk=F.normalize(ctx,dim=-1),F.normalize(keys,dim=-1)
                cosine=(nq[:,None]*nk).sum(-1)
                score=matcher.fusion_head(nq,nk,cosine[...,None],cosine)
                indices[b,take]=ids;valid[b,take]=mask
                probability[b,take]=categorical_weights(score/temperature,mask,1.)
                context[b,take]=ctx;retained[b,take]=1.
    return MemberRayPosterior(indices,valid,probability,context,retained,counts)


def voxel_members_by_ray(ray_ids):
    """CSR membership of all native voxels, independent of matching support."""
    if ray_ids.ndim!=3 or ray_ids.shape[1]!=3 or ray_ids.dtype!=torch.long:
        raise ValueError('native long[B,3,N] ray IDs required')
    b,v,n=(ray_ids>=0).nonzero(as_tuple=True)
    ray=(b*3+v)*1024+ray_ids[b,v,n]
    order=ray.argsort(stable=True)
    counts=torch.bincount(ray,minlength=len(ray_ids)*3072)
    offsets=torch.cat((counts.new_zeros(1),counts.cumsum(0)))
    return b[order],n[order],offsets


def gather_ray_voxel_members(take, membership):
    """Local ray ID, asset ID, voxel ID for only the current outer ray chunk."""
    batch,voxel,offsets=membership
    counts=offsets[take+1]-offsets[take]
    local=torch.arange(len(take),device=take.device).repeat_interleave(counts)
    prefix=torch.cat((counts.new_zeros(1),counts.cumsum(0)))
    within=torch.arange(len(local),device=take.device)-prefix[local]
    point=offsets[take[local]]+within
    if (counts<=0).any():raise ValueError('an active ray cannot have no native voxel members')
    return local,batch[point],voxel[point],counts


def voxel_readout_chunk(head, native, context, values, keys, take,
                        indices, valid, prior, local, batches, voxels, counts,
                        chunk_size):
    """Checkpointed point Q/K/V; pool raw and transformed values before R.

    Candidate/value gathers and all point-level activations live inside this
    checkpoint. The outer graph retains one output per ray as in Full.
    """
    change=native.new_zeros((len(take),1024),dtype=torch.float32)
    aggregate=torch.zeros_like(change)
    probability=native.new_zeros((len(take),4,64),dtype=torch.float32)
    for start in range(0,len(local),chunk_size):
        stop=start+chunk_size;group=local[start:stop];rays=take[group]
        b=batches[start:stop];n=voxels[start:stop]
        delta,raw,weights=_input_chunk(head,native[b,n],context[rays],values,keys,b,
            indices[rays],valid[rays],prior[rays],'normal',True)
        change=change.index_add(0,group,delta)
        aggregate=aggregate.index_add(0,group,raw)
        probability=probability.index_add(0,group,weights)
    denom=counts[:,None].float()
    return change/denom,aggregate/denom,probability/denom[...,None]


class UnifiedFullDesignConditioner(RMediatedSourceContrastConditioner):
    def __init__(self, *, design, **kwargs):
        if design not in VARIANTS[:3]:raise ValueError('declared full-pipeline design control required')
        super().__init__(appearance_mode='raw-topk-plus-feature',residual_logit_bound=1.,**kwargs)
        self.design=design
        if design=='FULL_QK_ONLY':
            original=self.ray_readout
            with torch.random.fork_rng(devices=[]):
                replacement=NoPriorBoundedReadout(1024,residual_logit_bound=1.)
            replacement.load_state_dict(original.state_dict(),strict=True)
            self.ray_readout=replacement
        self.register_buffer('unified_'+design.lower()+'_schema',torch.tensor([1],dtype=torch.uint8))
        if sum(p.requires_grad for p in self.parameters())!=19:
            raise ValueError('same full19 source parameters required')

    def train(self, mode=True):
        super().train(mode)
        self.q2c.eval()
        return self

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
        posterior_fn = mean_ray_posteriors if self.design == 'FULL_MEAN_MATCH' else member_ray_posteriors
        posterior = posterior_fn(self.q2c, target_keys=frame_target_keys,
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
        membership = voxel_members_by_ray(ray_ids) if self.design == 'FULL_VOXEL_Q' else None
        for start in range(0, len(selected), self.ray_chunk_size):
            take = selected[start:start+self.ray_chunk_size]
            b = take//rays
            ids, mask = indices.reshape(-1, count)[take], valid.reshape(-1, count)[take]
            value, weight, _ = self._attention(prior[take][None],
                flat_values[b[:, None], ids][None], mask[None], (ids//1024)[None], 3)
            appearances.append(value[0]); hard_weights.append(weight[0])
            if membership is None:
                function = _input_chunk
                args = (self.ray_readout, query[take], context[take], flat_values,
                        flat_keys, b, ids, mask, prior[take], control, raw_topk)
            else:
                if control != 'normal':raise ValueError('no intervention controls in matched voxel arm')
                function = voxel_readout_chunk
                local,point_batch,point_voxel,point_counts = gather_ray_voxel_members(take,membership)
                args = (self.ray_readout,native_voxels,context,flat_values,flat_keys,take,
                        indices.reshape(-1,count),valid.reshape(-1,count),prior,
                        local,point_batch,point_voxel,point_counts,self.ray_chunk_size)
            change, aggregate, probability = (checkpoint(function, *args, use_reentrant=False)
                if torch.is_grad_enabled() else function(*args))
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


class UnifiedTop1NoRetargetingConditioner(RMediatedTop1SourceConditioner):
    """Physically remove the learned R, retaining actual Top1/use/fusion only."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.retargeter=None
        self.register_buffer('unified_top1_no_R_schema',torch.tensor([1],dtype=torch.uint8))
        if sum(p.requires_grad for p in self.parameters())!=5:
            raise ValueError('only fusion and four source-use parameters may remain')

    def forward(self, *, native_voxels, target_omega_voxels, target_native_voxels,
                frame_target_keys, target_patch_ids, ray_ids, observed_query,
                source_keys, source_values, source_valid, control='normal'):
        if (control != 'normal' or native_voxels.shape != target_native_voxels.shape
                or native_voxels.shape != target_omega_voxels.shape or native_voxels.shape[-1] != 1024
                or source_values.shape != (native_voxels.shape[0], 3, 1024, 1024)):
            raise ValueError('aligned native SS/source-camera rays and unmodified Top1 feature control required')
        batch, rays, count = len(source_values), 3072, 64
        target = pool_ray_queries(target_native_voxels, ray_ids, num_rays=1024)
        posterior = member_ray_posteriors(self.q2c, target_keys=frame_target_keys,
            source_keys=source_keys, source_valid=source_valid, ray_ids=ray_ids,
            patch_ids=target_patch_ids, observed_query=observed_query,
            temperature=self.score_temperature, chunk_size=self.ray_chunk_size)
        indices, valid = posterior.candidate_indices, posterior.candidate_valid
        selected = valid.any(-1).flatten().nonzero().flatten()
        flat = source_values.flatten(1, 2).float()
        evidence = flat.new_zeros(batch * rays, 1024)
        hard = flat.new_zeros(batch * rays, count)
        prior = posterior.probabilities.log().reshape(-1, count)
        residual = torch.zeros_like(evidence)
        target_query = target.values.reshape(batch * rays, 1024)
        if len(selected):
            # Gather only the winning FEATURE, rather than building a Top-K
            # value tensor and assigning uniform or one-hot attention to it.
            slots = prior[selected].argmax(-1)
            winner = indices.reshape(-1, count)[selected].gather(1, slots[:, None])[:, 0]
            appearance = flat[selected // rays, winner]
            evidence = evidence.index_copy(0, selected, appearance)
            weights = hard.new_zeros(len(selected), count).scatter_(1, slots[:, None], 1.)
            hard = hard.index_copy(0, selected, weights)
            # There is no R module, transform or alternate correction route.
            # Residual is an absent architectural term, not a fake observation.
            self.last_R_input_appearance = appearance.detach()
            self.last_R_only_output = torch.zeros_like(appearance.detach())
        else:
            evidence = evidence + sum(p.reshape(-1)[0] * 0 for p in self.parameters() if p.requires_grad)
            self.last_R_input_appearance = flat.new_empty((0, 1024))
            self.last_R_only_output = flat.new_empty((0, 1024))
        evidence = evidence.reshape(batch, rays, 1024)
        residual = residual.reshape(batch, rays, 1024)
        hard = hard.reshape(batch, rays, count)
        query_valid = valid.any(-1)
        view_weights = evidence.new_zeros(batch, rays, 3).scatter_add(2, indices // 1024, hard)
        scale = torch.tanh(self.fusion_gate)
        attention = DeformationCondition(evidence + residual, indices, prior.reshape(batch, rays, count),
            hard, view_weights, residual, query_valid, scale, base_evidence=evidence)
        use = self.evidence_use(target_query.reshape(batch, rays, 1024), evidence, query_valid)
        mass = use[..., None] * scale
        numerator = mass * evidence + use[..., None] * residual
        self.last_source_reference_rows = selected.detach()
        self._last_member_retained_mass = posterior.retained_mass.detach()
        self._last_distinct_member_count = posterior.distinct_member_count.detach()
        return TargetFirstRayEvidence(numerator.reshape(batch, 3, 1024, 1024),
            mass.reshape(batch, 3, 1024, 1), attention, use)
