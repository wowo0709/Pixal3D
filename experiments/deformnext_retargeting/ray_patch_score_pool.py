"""Frozen patch-trained matcher readout over a ray's constituent target tokens.

Prototype only: each camera ray retains one final source-attention distribution
and one evidence vector. Distinct target-patch scores are pooled as probability
distributions, avoiding loss of complementary modes before correspondence readout.
"""
from __future__ import annotations

import math

import torch
from torch.nn import functional as F

from pixal3d.modules.task_matching_attention import contextual_query


def ray_target_patch_relations(ray_ids, patch_ids, *, rays_per_view=1024):
    if ray_ids.device.type!='cpu' or patch_ids.device.type!='cpu':
        raise ValueError('address-only ray/target relations are constructed on CPU')
    if (ray_ids.ndim!=2 or ray_ids.shape[0]!=3 or patch_ids.shape!=(ray_ids.shape[1],)
            or ((ray_ids<-1)|(ray_ids>=rays_per_view)).any()
            or ((patch_ids<0)|(patch_ids>=1024)).any()):
        raise ValueError('explicit native ray and target-patch addresses required')
    pairs=[]
    for view in range(3):
        active=ray_ids[view]>=0
        pairs.append((ray_ids[view,active]+view*rays_per_view)*1024+patch_ids[active])
    encoded=torch.cat(pairs).unique(sorted=True)
    return encoded//1024,encoded%1024


@torch.no_grad()
def score_pool_ray_patches(matcher, *, target_keys, source_keys, source_valid,
        candidates, ray_ids, patch_ids, temperature=.05, chunk_size=64, rays_per_view=1024):
    """Use the same64 keys and frozen D1/scorer; pool unique target-patch weights.

    Arguments contain RGB/Omega features and camera-derived addresses only.
    No topology, depth, GT visibility, source-point labels or target SS are read.
    Repeated voxels referencing the same target patch do not repeat its vote.
    """
    if not math.isfinite(temperature) or temperature<=0 or chunk_size<1:
        raise ValueError('positive finite temperature and chunk size required')
    if (target_keys.shape!=(1024,1024) or source_keys.shape!=(3,1024,1024)
            or source_valid.shape!=(3,1024) or candidates.shape!=(3*rays_per_view,64)
            or candidates.device!=target_keys.device or source_keys.device!=target_keys.device):
        raise ValueError('explicit FRAME target/source features and64 candidates required')
    if ((candidates<0)|(candidates>=3072)).any():
        raise ValueError('source candidate address out of range')
    ray,patch=ray_target_patch_relations(ray_ids,patch_ids,rays_per_view=rays_per_view)
    device=target_keys.device;ray=ray.to(device);patch=patch.to(device)
    probability=target_keys.new_zeros(3*rays_per_view,64,dtype=torch.float32)
    counts=torch.bincount(ray,minlength=3*rays_per_view)
    flat=source_keys.flatten(0,1).float();support=source_valid.flatten().bool()
    for start in range(0,len(ray),chunk_size):
        r,p=ray[start:start+chunk_size],patch[start:start+chunk_size]
        ids=candidates[r];valid=support[ids]
        if not valid.all():
            raise ValueError('score pooling retains64 alpha-valid source candidates')
        with torch.autocast(device_type=device.type,enabled=False):
            keys=flat[ids];query=target_keys[p].float()
            context=contextual_query(matcher.decoder,query,keys,valid)
            nq,nk=F.normalize(context,dim=-1),F.normalize(keys,dim=-1)
            cosine=(nq[:,None]*nk).sum(-1)
            logits=matcher.fusion_head(nq,nk,cosine[...,None],cosine)/temperature
            weights=torch.softmax(logits.float(),dim=-1)
        probability.index_add_(0,r,weights)
    return probability/counts.clamp_min(1)[:,None],counts
