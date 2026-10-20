"""Learn bounded task attention around a frozen semantic ray distribution."""
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .deform_conditioner import DeformationCondition
from .ray_consistent_retargeting import RayConsistentRetargetingConditioner, paired_correction
from .ray_feature_attention import _readout
from .task_matching_attention import contextual_query


class BoundedTaskRayAttention(nn.Module):
    """Four ray-query heads; a fixed semantic prior cannot be overwritten."""
    def __init__(self, reference_key, *, logit_bound=1.):
        super().__init__()
        if reference_key.shape != (256,1024) or not 0 < logit_bound <= 2:
            raise ValueError('native four64-wide keys and an explicit bounded correction required')
        self.logit_bound = float(logit_bound)
        self.query = nn.Linear(4096,256,bias=False)
        self.key = nn.Linear(1024,256,bias=False)
        nn.init.zeros_(self.query.weight)
        with torch.no_grad():
            self.key.weight.copy_(reference_key)

    def forward(self, native, context, deformation, keys, valid, reference):
        m,k = valid.shape
        if (native.shape != (m,1024) or context.shape != native.shape
                or deformation.shape != (m,2048) or keys.shape != (m,k,1024)
                or reference.shape != (m,4,k) or valid.dtype != torch.bool):
            raise ValueError('aligned native/decoded/deformation ray features and semantic support required')
        query = torch.cat([F.layer_norm(value.float(),(1024,))
            for value in [native,context,*deformation.split(1024,-1)]],-1)
        keys = keys.float().masked_fill(~valid[...,None],0.)
        with torch.autocast(device_type=native.device.type,enabled=False):
            q = self.query(query).reshape(m,4,64)
            kk = self.key(F.layer_norm(keys,(1024,))).reshape(m,k,4,64).transpose(1,2)
            delta = self.logit_bound*torch.tanh((q[:,:,None]*kk).sum(-1)/math.sqrt(64))
            prior = reference.detach().float().clamp_min(1e-30)*valid[:,None]
            prior = prior/prior.sum(-1,keepdim=True).clamp_min(1e-30)
            weight = prior*torch.exp(delta-delta.amax(-1,keepdim=True))
            probability = weight/weight.sum(-1,keepdim=True).clamp_min(1e-30)
            kl = (probability*(probability.clamp_min(1e-30).log()
                  -prior.clamp_min(1e-30).log())).sum(-1).mean(1).clamp_min(0.)
        return probability,kl


class TaskRayRetargetingConditioner(RayConsistentRetargetingConditioner):
    """Keep Top1/R and paired geometry, learn only a bounded attention residual.

    The additional2048 query channels contain an actual RAW Omega difference.
    Every voxel on the same projected target patch ray shares this query.
    The semantic matcher and availability head remain frozen.
    """
    def __init__(self, *, task_logit_bound=1., **kwargs):
        super().__init__(**kwargs)
        self.task_attention = BoundedTaskRayAttention(
            self.ray_readout.key.weight.detach(),logit_bound=task_logit_bound)
        self.task_matching_KL = None

    def reference_state(self):
        return {name:value for name,value in super().reference_state().items()
                if not name.startswith('task_attention.')}

    def forward(self, *, source_values,source_valid,target_query,semantic_indices,
                matching_logits,candidate_valid,target_valid,target_keys,source_keys,
                target_context,source_context,control='normal',**unsupported):
        if unsupported or control != 'normal' or self.composition != 'paired':
            raise ValueError('only explicit paired inference observations are supported')
        if any(p.requires_grad for p in self.ray_readout.parameters()):
            raise ValueError('the complete semantic prior must stay frozen')
        source_values = source_values.float()
        b,views,sites,width = source_values.shape
        q = target_query.shape[1]
        if (views,sites,width) != (3,1024,self.value_dim) or target_query.shape != (b,q,1024):
            raise ValueError('native three-source ray appearance contract required')
        if (target_context.shape != (b,q,2048) or source_context.shape != (b,3,1024,2048)
                or target_keys.shape != (b,q,1024) or source_keys.shape != (b,3,1024,1024)
                or not torch.equal(target_context[...,:1024].float(),target_keys.float())
                or not torch.equal(source_context[...,:1024].float(),source_keys.float())):
            raise ValueError('actual FRAME/WORLD observations must agree with the semantic keys')
        if (source_valid.shape != (b,3,1024) or source_valid.dtype != torch.bool
                or target_valid.shape != (b,q) or target_valid.dtype != torch.bool
                or semantic_indices.shape != (b,q,64) or matching_logits.shape != (b,q,64)
                or candidate_valid.shape != (b,q,64)):
            raise ValueError('complete aligned Top64 support is required')
        indices = semantic_indices.long()
        if ((indices<0)|(indices>=3072)).any():
            raise ValueError('invalid semantic source address')
        values,keys,raw_source = source_values.flatten(1,2),source_keys.flatten(1,2),source_context.flatten(1,2)
        flat_valid = source_valid.flatten(1)
        valid = flat_valid[:,None].expand(-1,q,-1).gather(2,indices)&candidate_valid.bool()
        valid &= target_valid[...,None]&(flat_valid.sum(-1)>=64)[:,None,None]
        active = valid.any(-1)
        logits = matching_logits.float().masked_fill(~valid,-torch.inf)
        chosen = logits.argmax(-1,keepdim=True)
        hard = torch.zeros_like(logits).scatter(2,chosen,1.)*active[...,None]
        address = indices.gather(2,chosen).squeeze(2)
        top1 = values.gather(1,address[...,None].expand(-1,-1,width))*active[...,None]
        selected = active.flatten().nonzero().flatten()
        parts,confidences,probabilities,regularizers = [],[],[],[]
        for start in range(0,len(selected),self.feature_chunk_size):
            take = selected[start:start+self.feature_chunk_size]
            batch = take//q
            ids,mask = indices.reshape(-1,64)[take],valid.reshape(-1,64)[take]
            native = target_query.reshape(-1,1024)[take].float()
            prior = logits.reshape(-1,64)[take]/self.score_temperature
            raw_target = target_context.reshape(-1,2048)[take].float()
            with torch.no_grad(),torch.autocast(device_type=source_values.device.type,enabled=False):
                key = keys[batch[:,None],ids].float().masked_fill(~mask[...,None],0.)
                context = contextual_query(self.q2c.decoder,target_keys.reshape(-1,1024)[take].float(),key,mask)
                _,reference = _readout(self.ray_readout,native,context,values,keys,batch,ids,mask,prior)
                raw = raw_source[batch[:,None],ids].float().masked_fill(~mask[...,None],0.)
                difference = raw_target-(reference.mean(1)[...,None]*raw).sum(1)
                confidence = (self.reliability(native,context,prior,mask,ids)
                    if self.reliability is not None else native.new_ones(len(native)))
            attention,kl = self.task_attention(native,context,difference,key,mask,reference)
            probability = attention.mean(1)
            args = (self.retargeter,native,raw_target,values,raw_source,batch,
                    ids,mask,probability,self.g_max,self.candidate_chunk_size)
            correction = checkpoint(paired_correction,*args,use_reentrant=False) if torch.is_grad_enabled() else paired_correction(*args)
            parts.append(confidence[:,None]*correction)
            confidences.append(confidence)
            probabilities.append(probability)
            regularizers.append(kl)
        correction = source_values.new_zeros(b*q,width)
        if len(selected):
            correction = correction.index_copy(0,selected,torch.cat(parts))
            self.task_matching_KL = torch.cat(regularizers).mean()
        else:
            zero = self.retargeter.deformation_projection.weight.flatten()[0]*0
            for parameter in self.task_attention.parameters():
                zero = zero+parameter.flatten()[0]*0
            correction = correction+zero
            self.task_matching_KL = zero
        base = self.retargeter.appearance_forward(torch.cat([top1,target_query.float()],-1),self.g_max)
        retargeted = (base+correction.reshape(b,q,width).to(base.dtype))*active[...,None]
        self.last_feature_attention_stats = dict(ray_query='edited-target-patch',
            composition='paired',semantic_prior7_frozen=True,task_attention_tensors=2,
            task_query_channels=4096,additional_actual_deformation_query_channels=2048,
            task_logit_bound=self.task_attention.logit_bound,task_KL=float(self.task_matching_KL.detach()),
            reliability_predicted=self.reliability is not None,
            reliability_mean=float(torch.cat(confidences).mean()) if confidences else 0.,
            effective_K=float((-(torch.cat(probabilities)*torch.cat(probabilities).clamp_min(1e-30).log()).sum(-1)).exp().mean()) if probabilities else 0.,
            geometry_correction_RMS=float(correction.detach().square().mean().sqrt()),
            physical_deformation_channels=2048,appearance_VO_physically_absent=True,
            extra_path_R_weights_detached=True,active_rays=int(active.sum()))
        view_weights = logits.new_zeros(b,q,3).scatter_add(2,indices//1024,hard)
        return DeformationCondition(top1+retargeted,indices,logits,hard,view_weights,
                                    retargeted,active,torch.tanh(self.fusion_gate))
