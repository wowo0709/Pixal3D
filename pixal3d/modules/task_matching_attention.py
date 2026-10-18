"""Task-trained candidate attention with an independently removable R9 score prior."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class TaskCandidateAttention(nn.Module):
    """Use both contextual Omega and native DINO query; values remain native DINO.

    The R9 matching decoder contextualizes Omega independently of the scalar
    matching-score prior. Removing that prior therefore retains a working
    contextual query path. No exact-match confidence or GT is consumed.
    """

    def __init__(self, width=1024, hidden=128):
        super().__init__()
        self.query = nn.Linear(width, hidden, bias=False)
        self.context = nn.Linear(width, hidden, bias=False)
        self.key = nn.Linear(width, hidden, bias=False)
        self.log_content_scale = nn.Parameter(torch.tensor(math.log(4.)))
        for layer in [self.query, self.context, self.key]:
            nn.init.xavier_uniform_(layer.weight)

    def forward(self, target, contextual_omega, candidates, valid, prior=None):
        candidates = torch.where(valid[...,None],candidates,torch.zeros_like(candidates))
        q = self.query(F.layer_norm(target, target.shape[-1:]))
        q = q + self.context(F.layer_norm(contextual_omega, contextual_omega.shape[-1:]))
        k = self.key(F.layer_norm(candidates, candidates.shape[-1:]))
        content = (F.normalize(q.float(), dim=-1)[:,None] * F.normalize(k.float(),dim=-1)).sum(-1)
        logits = self.log_content_scale.exp().clamp(max=20.) * content
        if prior is not None:
            if prior.shape != logits.shape:
                raise ValueError('matching-score prior shape differs')
            logits = logits + prior.float()
        logits = logits.masked_fill(~valid, -torch.inf)
        maximum = logits.amax(-1, keepdim=True)
        maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
        p = torch.exp(logits-maximum).masked_fill(~valid, 0.)
        p = p/p.sum(-1,keepdim=True).clamp_min(1e-8)
        return (p[...,None]*candidates.float()).sum(1), p, logits


def contextual_query(decoder, query, keys, valid):
    """Match original D1 math on full support, exclude masked tokens otherwise."""
    if keys.shape[1]==64 and bool(valid.all()):
        return decoder(query,keys)[0]
    keys=torch.where(valid[...,None],keys,torch.zeros_like(keys))
    for block in decoder.blocks:
        if not block.query_only:
            raise ValueError('task attention requires a query-only matching decoder')
        normalized=block.norm_attention(query[:,None]);context=block.norm_attention(keys)
        def heads(x):return x.reshape(x.shape[0],x.shape[1],4,32).transpose(1,2)
        attention=F.scaled_dot_product_attention(heads(block.query_projection(normalized)),
            heads(block.key_projection(context)),heads(block.value_projection(context)),
            attn_mask=valid[:,None,None,:],dropout_p=0.)
        attention=attention.transpose(1,2).reshape(len(query),1,128)
        token=query[:,None]+block.attention_gate*block.attention_output(attention)
        token=token+block.ffn_gate*block.ffn(block.norm_ffn(token))
        query=token[:,0]
    return query


def task_evidence(module, *, target_keys, source_keys, source_values, target_query,
                  indices, valid, view_index, views):
    """Gather only alpha-valid queries; avoid a full[B,1024,64,1024] graph."""
    from .deform_conditioner import DeformationCondition

    b,q,k = indices.shape
    if target_keys is None or source_keys is None or target_query is None:
        raise ValueError('task attention requires explicit Omega FRAME keys and native target query')
    if target_keys.shape != (b,q,1024) or target_query.shape != (b,q,1024):
        raise ValueError('task attention query shape differs')
    if source_keys.shape != source_values.shape:
        raise ValueError('task attention requires aligned 1024-D Omega and DINO source maps')
    selected = valid.any(-1).reshape(-1).nonzero().flatten()
    evidence = source_values.new_zeros(b*q,1024,dtype=torch.float32)
    all_weights = source_values.new_zeros(b*q,k,dtype=torch.float32)
    all_logits = source_values.new_full((b*q,k),-torch.inf,dtype=torch.float32)
    flat_values = source_values.flatten(1,2)
    flat_keys = source_keys.flatten(1,2)
    chunk = module.task_attention_config['chunk_size']
    evidences=[];weights=[];logits=[]
    for start in range(0,len(selected),chunk):
        ix = selected[start:start+chunk];batch = ix//q
        ids = indices.reshape(-1,k)[ix]
        active = valid.reshape(-1,k)[ix]
        keys = flat_keys[batch[:,None],ids].float()
        keys = torch.where(active[...,None],keys,torch.zeros_like(keys))
        values = flat_values[batch[:,None],ids]
        query = target_keys.reshape(-1,1024)[ix].float()
        native_query = target_query.reshape(-1,1024)[ix].float()
        # Canonical D1 query-only decoder never changes candidate values.
        with torch.autocast(device_type=query.device.type,enabled=False):
            context = contextual_query(module.q2c.decoder,query,keys,active)
            prior = None
            if module.task_attention_config['score_prior']:
                nq,nk = F.normalize(context,dim=-1), F.normalize(keys,dim=-1)
                cosine = (nq[:,None]*nk).sum(-1)
                score = module.q2c.fusion_head(nq,nk,cosine[...,None],cosine)
                prior = score/module.score_temperature
        e,w,l = module.task_attention(native_query,context,values,active,prior)
        evidences.append(e);weights.append(w);logits.append(l)
    if len(selected):
        evidence=evidence.index_copy(0,selected,torch.cat(evidences))
        all_weights=all_weights.index_copy(0,selected,torch.cat(weights))
        all_logits=all_logits.index_copy(0,selected,torch.cat(logits))
    else:
        # Preserve a zero gradient edge for DDP ranks with no valid source
        # query. Do not change find_unused_parameters for the entire trainer.
        link=sum(p.reshape(-1)[0]*0 for p in module.parameters() if p.requires_grad)
        evidence=evidence+link
    evidence=evidence.reshape(b,q,1024);all_weights=all_weights.reshape(b,q,k)
    view_weights=evidence.new_zeros(b,q,views).scatter_add(2,view_index,all_weights)
    query_valid=valid.any(-1)
    residual=module.retargeter(torch.cat([evidence,target_query.float()],-1),module.g_max)
    residual=residual*query_valid[...,None]
    return DeformationCondition(evidence+residual,indices,all_logits.reshape(b,q,k),all_weights,
        view_weights,residual,query_valid,torch.tanh(module.fusion_gate),base_evidence=evidence)


def native_spatial_evidence(module, *, target_keys, target_query, keys, values, target_valid):
    """Attend the three original native projected source-camera values per voxel.

    This is the natural native-spatial module control (3 geometric view
    samples versus64 semantic patch candidates), not a fixed-candidate-budget
    address-only intervention. It never uses cached GT-depth spatial_indices.
    """
    from .deform_conditioner import DeformationCondition
    b,q,k,d=values.shape
    if k!=3 or d!=1024 or keys.shape!=values.shape or target_keys.shape!=(b,q,d) or target_query.shape!=(b,q,d):
        raise ValueError('native spatial attention requires3 projected source-camera candidates')
    valid=target_valid.bool();selected=valid.reshape(-1).nonzero().flatten()
    e=values.new_zeros(b*q,d,dtype=torch.float32);w=values.new_zeros(b*q,k,dtype=torch.float32)
    logits=values.new_full((b*q,k),-torch.inf,dtype=torch.float32)
    out_e=[];out_w=[];out_l=[]
    # Bound candidate tensors by the same budget as64 semantic candidates.
    # Native spatial has only3 per voxel; processing64 queries at a time
    # would launch thousands of tiny CUDA kernels for one training pack.
    query_chunk=max(1,module.task_attention_config['chunk_size']*64//k)
    for start in range(0,len(selected),query_chunk):
        ix=selected[start:start+query_chunk]
        omega=target_keys.reshape(-1,d)[ix].float();ck=keys.reshape(-1,k,d)[ix].float()
        cv=values.reshape(-1,k,d)[ix];active=torch.ones(len(ix),k,dtype=torch.bool,device=values.device)
        with torch.autocast(device_type=values.device.type,enabled=False):
            context=contextual_query(module.q2c.decoder,omega,ck,active)
            prior=None
            if module.task_attention_config['score_prior']:
                nq,nk=F.normalize(context,dim=-1),F.normalize(ck,dim=-1)
                cosine=(nq[:,None]*nk).sum(-1)
                prior=module.q2c.fusion_head(nq,nk,cosine[...,None],cosine)/module.score_temperature
        a,p,l=module.task_attention(target_query.reshape(-1,d)[ix].float(),context,cv,active,prior)
        out_e.append(a);out_w.append(p);out_l.append(l)
    if len(selected):
        e=e.index_copy(0,selected,torch.cat(out_e));w=w.index_copy(0,selected,torch.cat(out_w));logits=logits.index_copy(0,selected,torch.cat(out_l))
    else:e=e+sum(p.reshape(-1)[0]*0 for p in module.parameters() if p.requires_grad)
    e=e.reshape(b,q,d);w=w.reshape(b,q,k)
    residual=module.retargeter(torch.cat([e,target_query.float()],-1),module.g_max)*valid[...,None]
    ids=torch.arange(k,device=values.device).reshape(1,1,k).expand(b,q,-1)
    return DeformationCondition(e+residual,ids,logits.reshape(b,q,k),w,w,residual,valid,
        torch.tanh(module.fusion_gate),base_evidence=e)


def inject_native_spatial_task(module, wrapped, extractor, condition, anchor, transport, camera):
    """Original camera/ProjGrid geometry; frozen RGB and Omega maps only."""
    from ..trainers.flow_matching.mixins.image_conditioned_proj import compute_multiview_projection_matrices
    projection,_=compute_multiview_projection_matrices(camera['transform_matrix'],camera['distance'],extractor.fixed_projection_transform)
    def project(patches,view):
        return extractor.proj_grid(patches.reshape(patches.shape[0],32,32,-1),
            camera['camera_angle_x'][:,view],camera['distance'][:,view],camera['mesh_scale'],projection[:,view])
    values=torch.stack([project(transport['source_values'][:,i].float(),i+1) for i in range(3)],2)
    keys=torch.stack([project(transport['source_keys'][:,i].float(),i+1) for i in range(3)],2)
    target_keys=project(transport['target_keys'].float(),0)
    # Alpha is the observed edited RGB support, not target GT occupancy/depth.
    valid=project(transport['target_valid'][...,None].float(),0).squeeze(-1)>0
    key='proj_semantic' if 'proj_semantic' in condition else 'proj'
    result=wrapped(source_values=transport['source_values'],source_valid=transport['source_valid'],
        target_keys=target_keys,target_query=anchor[key],target_valid=valid,
        native_candidate_keys=keys,native_candidate_values=values)
    native=condition[key];semantic=(anchor[key]+3*result.base_evidence.to(native.dtype))/4
    update=result.fusion_scale.to(native.dtype)*(semantic-native)+.75*result.retarget_residual.to(native.dtype)
    condition[key]=native+torch.where(valid[...,None],update,torch.zeros_like(update))
    return condition,result
