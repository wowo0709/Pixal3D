"""Physical Omega deformation-query channels on the positive target-ray carrier.

The no-DQ model has no deformation input or projection. Here the actual
nonzero Omega difference is an additional2048-wide input. A separate zero
projection preserves the original R tensors and their Adam moments exactly.
"""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .deform_conditioner import _Retargeter,DeformationCondition
from .ray_feature_attention import RayFeatureAttentionConditioner,_readout
from .task_matching_attention import contextual_query


class DeformationQueryRetargeter(_Retargeter):
    def __init__(self,appearance_dim,value_dim,hidden_dim,deformation_dim=2048):
        if deformation_dim!=2048:
            raise ValueError('explicit RAW FRAME+GLOBAL2048 deformation channels required')
        super().__init__(appearance_dim,value_dim,hidden_dim)
        self.appearance_dim=appearance_dim
        self.deformation_dim=deformation_dim
        self.input_dim=appearance_dim+deformation_dim
        self.deformation_projection=nn.Linear(deformation_dim,hidden_dim,bias=False)
        nn.init.zeros_(self.deformation_projection.weight)

    def forward(self,inputs,g_max):
        if inputs.shape[-1]!=self.input_dim:
            raise ValueError('physical appearance and deformation inputs must both be present')
        appearance,deformation=inputs.split((self.appearance_dim,self.deformation_dim),dim=-1)
        hidden=self.net[0](appearance.contiguous())+self.deformation_projection(deformation)
        hidden=self.net[1](hidden)
        return g_max*torch.tanh(self.gate_logit)*torch.tanh(self.net[2](hidden))


def geometry_readout(head,native,context,values,keys,raw_source,raw_target,
                     batch,indices,valid,prior):
    feature,probability=_readout(head,native,context,values,keys,batch,indices,valid,prior)
    raw=raw_source[batch[:,None],indices].float()
    raw=torch.where(valid[...,None],raw,torch.zeros_like(raw))
    # Geometry follows exactly the same learned correspondence probabilities.
    # It is not a GT displacement and not a separately cached Top1 difference.
    weights=probability.mean(1)
    source=(weights[...,None]*raw).sum(1)
    return feature,raw_target.float()-source,probability


class RayDeformationQueryConditioner(RayFeatureAttentionConditioner):
    def __init__(self,*,deformation_dim,**kwargs):
        if (deformation_dim!=2048 or kwargs.get('arm')!='c2'
                or kwargs.get('value_dim') not in (1024,2048) or kwargs.get('query_dim')!=1024):
            raise ValueError('nativeSS/PBR C2 with physical RAW2048 deformation-query channels required')
        super().__init__(**dict(kwargs,arm='c1'))
        self.retargeter=DeformationQueryRetargeter(self.value_dim+self.query_dim,
            self.value_dim,kwargs.get('hidden_dim',256),2048)
        self.retargeter.requires_grad_(False)
        self.retargeter.deformation_projection.requires_grad_(True)
        self.arm='c2'
        self.deformation_dim=2048

    def reference_state(self):
        return {name:value for name,value in self.state_dict().items()
                if not name.startswith('ray_readout.')
                and name!='retargeter.deformation_projection.weight'}

    def load_reference(self,reference):
        if torch.count_nonzero(self.retargeter.deformation_projection.weight):
            raise ValueError('untouched zero-initialized deformation projection required')
        super().load_reference(reference)

    def forward(self,*,source_values,source_valid,target_query,semantic_indices,
                matching_logits,candidate_valid,target_valid,target_keys,source_keys,
                target_context,source_context,control='normal',**unsupported):
        if unsupported or control!='normal':
            raise ValueError('explicit inference-only feature inputs and normal carrier required')
        source_values=source_values.float()
        b,views,sites,width=source_values.shape;q=target_query.shape[1]
        if (views,sites,width)!=(3,1024,self.value_dim) or target_query.shape!=(b,q,1024):
            raise ValueError('nativeSS/PBR source values and target patch query required')
        if (target_context.shape!=(b,q,2048) or source_context.shape!=(b,3,1024,2048)
                or target_keys.shape!=(b,q,1024) or source_keys.shape!=(b,3,1024,1024)
                or not torch.equal(target_context[...,:1024].float(),target_keys.float())
                or not torch.equal(source_context[...,:1024].float(),source_keys.float())):
            raise ValueError('RAW2048 contexts must contain the same FRAME1024 descriptors')
        if (source_valid.shape!=(b,3,1024) or source_valid.dtype!=torch.bool
                or target_valid.shape!=(b,q) or target_valid.dtype!=torch.bool
                or semantic_indices.shape!=(b,q,64) or matching_logits.shape!=(b,q,64)
                or candidate_valid.shape!=(b,q,64)):
            raise ValueError('complete native64 candidate/mask contract required')
        indices=semantic_indices.long()
        if ((indices<0)|(indices>=3072)).any():raise ValueError('invalid source address')
        flat_values=source_values.flatten(1,2);flat_keys=source_keys.flatten(1,2)
        raw_source=source_context.flatten(1,2);flat_valid=source_valid.flatten(1)
        valid=flat_valid[:,None].expand(-1,q,-1).gather(2,indices)&candidate_valid.bool()
        valid &= target_valid[...,None]&(flat_valid.sum(-1)>=64)[:,None,None]
        active=valid.any(-1);logits=matching_logits.float().masked_fill(~valid,-torch.inf)
        chosen=logits.argmax(-1,keepdim=True)
        hard=torch.zeros_like(logits).scatter(2,chosen,1.)*active[...,None]
        address=indices.gather(2,chosen).squeeze(2)
        top1=flat_values.gather(1,address[...,None].expand(-1,-1,width))*active[...,None]
        selected=active.flatten().nonzero().flatten();parts=[];geometry=[];probabilities=[]
        delta=source_values.new_zeros(b*q,width)
        deformation=source_values.new_zeros(b*q,2048)
        for start in range(0,len(selected),self.feature_chunk_size):
            take=selected[start:start+self.feature_chunk_size];batch=take//q
            ids=indices.reshape(-1,64)[take];mask=valid.reshape(-1,64)[take]
            with torch.no_grad(),torch.autocast(device_type=source_values.device.type,enabled=False):
                keys=flat_keys[batch[:,None],ids].float()
                context=contextual_query(self.q2c.decoder,target_keys.reshape(-1,1024)[take].float(),
                    torch.where(mask[...,None],keys,torch.zeros_like(keys)),mask)
            args=(self.ray_readout,target_query.reshape(-1,1024)[take].float(),context,
                flat_values,flat_keys,raw_source,target_context.reshape(-1,2048)[take],
                batch,ids,mask,logits.reshape(-1,64)[take]/self.score_temperature)
            change,dq,weights=(checkpoint(geometry_readout,*args,use_reentrant=False)
                               if torch.is_grad_enabled() else geometry_readout(*args))
            parts.append(change);geometry.append(dq);probabilities.append(weights)
        if len(selected):
            delta=delta.index_copy(0,selected,torch.cat(parts))
            deformation=deformation.index_copy(0,selected,torch.cat(geometry))
            weights=torch.cat(probabilities).detach()
            effective=float((-(weights*weights.clamp_min(1e-30).log()).sum(-1)).exp().mean())
        else:
            delta=delta+sum(p.reshape(-1)[0]*0 for p in self.ray_readout.parameters())
            effective=0.
        mixed=top1+delta.reshape(b,q,width)
        query=torch.cat([mixed,target_query.float(),deformation.reshape(b,q,2048)],-1)
        retargeted=self.retargeter(query,self.g_max)*active[...,None]
        self.last_feature_attention_stats=dict(effective_K_per_head=effective,
            feature_residual_RMS=float(delta.detach().square().mean().sqrt()),
            deformation_query_RMS=float(deformation.detach().square().mean().sqrt()),
            physical_deformation_channels=2048,retargeter_input_channels=self.retargeter.input_dim,
            active_rays=int(active.sum()),ray_query='edited-target-patch',scalar_mix_gate=False)
        view_weights=logits.new_zeros(b,q,3).scatter_add(2,indices//1024,hard)
        return DeformationCondition(mixed+retargeted,indices,logits,hard,view_weights,
            retargeted,active,torch.tanh(self.fusion_gate))
