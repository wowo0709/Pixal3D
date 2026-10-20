"""CPU diagnostic proposals from a ray's distinct observed image-patch members.

Prototype only; this is not an installed training/inference conditioner.
The ray retains one final Top64 support/distribution. Inputs contain frozen
image descriptors, input alpha validity and camera-derived addresses only.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F

from .ray_patch_score_pool import ray_target_patch_relations
from pixal3d.modules.r9_matching import categorical_weights
from pixal3d.modules.task_matching_attention import contextual_query


def member_table(ray_ids, patch_ids, target_valid, *, rays_per_view=1024):
    if (ray_ids.device.type!='cpu' or patch_ids.device.type!='cpu'
            or target_valid.shape!=(1024,) or target_valid.dtype!=torch.bool):
        raise ValueError('CPU camera addresses and explicit input-alpha validity required')
    if patch_ids.dtype!=torch.long or ray_ids.dtype!=torch.long:
        raise ValueError('integer native ray and patch addresses required')
    groups=torch.where(target_valid[patch_ids][None],ray_ids,torch.full_like(ray_ids,-1))
    ray,patch=ray_target_patch_relations(groups,patch_ids,rays_per_view=rays_per_view)
    point_ids=patch.unique(sorted=True)
    lookup=torch.full((1024,),-1,dtype=torch.long);lookup[point_ids]=torch.arange(len(point_ids))
    membership=torch.zeros((3*rays_per_view,len(point_ids)),dtype=torch.bool)
    if len(ray):membership[ray,lookup[patch]]=True
    counts=membership.sum(-1)
    maximum=int(counts.max()) if len(counts) else 0
    padded=torch.zeros((len(counts),maximum),dtype=torch.long)
    valid=torch.zeros_like(padded,dtype=torch.bool)
    if len(ray):
        starts=counts.cumsum(0)-counts
        offsets=torch.arange(len(ray))-starts[ray]
        padded[ray,offsets]=lookup[patch];valid[ray,offsets]=True
    return dict(groups=groups,ray=ray,patch=patch,point_ids=point_ids,
                membership=membership,counts=counts,padded=padded,padded_valid=valid)


@torch.no_grad()
def score_candidate_rows(matcher, query, source, candidates, valid, *, temperature=.05,chunk_size=64):
    if (query.device.type!='cpu' or source.device.type!='cpu' or query.shape!=(len(candidates),1024)
            or source.shape!=(3072,1024) or candidates.shape!=(len(query),64)
            or valid.shape!=candidates.shape or valid.dtype!=torch.bool
            or not 0<temperature or chunk_size<1):
        raise ValueError('explicit CPU1024-D rows/source and64 candidate IDs required')
    scores=torch.full(candidates.shape,-torch.inf,dtype=torch.float32)
    contexts=torch.zeros_like(query,dtype=torch.float32)
    # Preserve the actual conditioner batching: skip inactive rays BEFORE
    # chunking. Padding them into D1 changes GEMM batch shapes and can change
    # FP32 score ties, which invalidates an exact frozen-baseline comparison.
    selected=valid.any(-1).nonzero().flatten()
    for start in range(0,len(selected),chunk_size):
        take=selected[start:start+chunk_size];q=query[take].float();mask=valid[take]
        keys=source[candidates[take]].float()
        keys=torch.where(mask[...,None],keys,torch.zeros_like(keys))
        context=contextual_query(matcher.decoder,q,keys,mask)
        nq,nk=F.normalize(context,dim=-1),F.normalize(keys,dim=-1)
        cosine=(nq[:,None]*nk).sum(-1)
        scores[take]=matcher.fusion_head(nq,nk,cosine[...,None],cosine)
        contexts[take]=context
    weights=categorical_weights(scores/temperature,valid,1.)
    return weights,contexts,scores


@torch.no_grad()
def observed_point_routes(matcher, *, target_keys, source_keys, source_valid, members,
                          temperature=.05,chunk_size=64):
    if (target_keys.shape!=(1024,1024) or source_keys.shape!=(3,1024,1024)
            or source_valid.shape!=(3,1024) or source_valid.dtype!=torch.bool
            or target_keys.device.type!='cpu' or source_keys.device.type!='cpu'):
        raise ValueError('explicit CPU frozen FRAME target/source fields required')
    point_ids=members['point_ids'];source=source_keys.flatten(0,1).float()
    complete=int(source_valid.sum())>=64
    if not complete or not len(point_ids):
        return dict(complete=complete,raw=torch.full((len(point_ids),3072),-torch.inf),
                    candidates=torch.zeros((len(point_ids),64),dtype=torch.long),
                    weights=torch.zeros(len(point_ids),64),context=torch.zeros(len(point_ids),1024))
    query=target_keys[point_ids].float()
    raw=F.normalize(query,dim=-1)@F.normalize(source,dim=-1).T
    raw=raw.masked_fill(~source_valid.flatten()[None],-torch.inf)
    _,candidates=raw.topk(64,dim=-1)
    valid=source_valid.flatten()[candidates]
    weights,context,_=score_candidate_rows(matcher,query,source,candidates,valid,
        temperature=temperature,chunk_size=chunk_size)
    return dict(complete=True,raw=raw,candidates=candidates,weights=weights,context=context)


def maximum_member_raw64(members, routes, source_valid, *, chunk_size=64):
    count=len(members['counts']);candidates=torch.zeros(count,64,dtype=torch.long)
    valid=torch.zeros_like(candidates,dtype=torch.bool)
    active=(members['counts']>0)&routes['complete']
    selected=active.nonzero().flatten()
    for start in range(0,len(selected),chunk_size):
        rows=selected[start:start+chunk_size]
        raw=routes['raw'][members['padded'][rows]]
        raw=raw.masked_fill(~members['padded_valid'][rows,:,None],-torch.inf).amax(1)
        candidates[rows]=raw.topk(64,dim=-1).indices
        valid[rows]=source_valid.flatten()[candidates[rows]]
    return candidates,valid,active


def mixture_member_top64(members, routes, source_valid):
    """Mix normalized point posteriors, not uncalibrated scores across queries.

    Equal mass per DISTINCT observed target patch prevents repeated voxels
    from multiplying one member's vote. Keep exactly64 alpha-valid sources;
    report mass lost by truncating the union rather than claiming no loss.
    """
    count=len(members['counts']);candidates=torch.zeros(count,64,dtype=torch.long)
    probability=torch.zeros(count,64);context=torch.zeros(count,1024)
    retained=torch.zeros(count);valid=torch.zeros_like(candidates,dtype=torch.bool)
    active=(members['counts']>0)&routes['complete']
    if not active.any():return dict(candidates=candidates,valid=valid,weights=probability,
        context=context,retained_mass=retained,active=active)
    point_weights=routes['weights']
    if (not torch.isfinite(point_weights).all() or (point_weights<0).any()
            or not torch.allclose(point_weights.sum(-1),torch.ones(len(point_weights)),atol=1e-5,rtol=0)):
        raise ValueError('normalized per-point distributions required')
    dense=torch.zeros(len(point_weights),3072)
    dense.scatter_(1,routes['candidates'],point_weights)
    membership=members['membership'][active].float()
    mass=membership@dense/members['counts'][active,None]
    top,ids=mass.masked_fill(~source_valid.flatten()[None],-torch.inf).topk(64,dim=-1)
    kept=top.sum(-1)
    if not torch.isfinite(top).all() or (kept<=0).any():
        raise ValueError('complete source64 support and positive retained mass required')
    candidates[active]=ids;probability[active]=top/kept[:,None]
    retained[active]=kept;valid[active]=source_valid.flatten()[ids]
    context[active]=membership@routes['context']/members['counts'][active,None]
    return dict(candidates=candidates,valid=valid,weights=probability,
                context=context,retained_mass=retained,active=active)
