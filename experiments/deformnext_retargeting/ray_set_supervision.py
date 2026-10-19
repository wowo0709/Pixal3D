"""Loss-only multi-positive correspondence supervision for shared camera-ray queries."""
from __future__ import annotations

import torch

from .ray_patch_score_pool import ray_target_patch_relations


def ray_positive_sets(*, ray_ids, patch_ids, candidates, source_valid,
        positive_indices, positive_visible, target_foreground, rays_per_view=1024):
    """Produce CPU teacher sets after the GT-free model proposal forward.

    Different observable target patches sharing a ray retain separate positive
    sets. Missing proposals remain empty supervised sets; they do not become
    a label that the surface is invisible or that no correspondence exists.
    """
    values=[ray_ids,patch_ids,candidates,source_valid,positive_indices,positive_visible,target_foreground]
    if any(value.device.type!='cpu' for value in values):
        raise ValueError('correspondence teacher construction is explicitly CPU-only')
    if (candidates.shape!=(3*rays_per_view,64) or source_valid.shape!=(3072,)
            or positive_indices.shape!=(1024,3) or positive_visible.shape!=(1024,3)
            or target_foreground.shape!=(1024,)):
        raise ValueError('native source64 and current topology teacher shapes required')
    ray,patch=ray_target_patch_relations(ray_ids,patch_ids,rays_per_view=rays_per_view)
    if ((candidates<0)|(candidates>=3072)).any() or ((positive_indices<0)|(positive_indices>=3072)).any():
        raise ValueError('correspondence source address out of range')
    visible=positive_visible.bool()&source_valid.bool()[positive_indices]&target_foreground.bool()[:,None]
    observable=visible.any(-1)[patch];ray,patch=ray[observable],patch[observable]
    exact=[];near=[]
    for start in range(0,len(ray),512):
        r,p=ray[start:start+512],patch[start:start+512]
        chosen=candidates[r,:,None];truth=positive_indices[p,None,:];seen=visible[p,None,:]
        valid=source_valid.bool()[candidates[r]]
        exact.append(((chosen==truth)&seen).any(-1)&valid)
        nearby=(chosen//1024==truth//1024)&((chosen%32-truth%32).abs()<=1)&(((chosen%1024)//32-(truth%1024)//32).abs()<=1)
        near.append((nearby&seen).any(-1)&valid)
    empty=torch.zeros(0,64,dtype=torch.bool)
    return dict(ray_indices=ray,target_patch_indices=patch,
        positive_exact=torch.cat(exact) if exact else empty,
        positive_near1=torch.cat(near) if near else empty,
        unobservable_relations_excluded=int((~observable).sum()))


def multi_positive_ray_loss(probabilities, relation_queries, positive_sets):
    """Balance distinct target-patch mass inside each ray, then balance rays."""
    if (probabilities.ndim!=2 or positive_sets.ndim!=2 or positive_sets.shape[1]!=probabilities.shape[1]
            or relation_queries.shape!=(len(positive_sets),) or positive_sets.dtype!=torch.bool
            or relation_queries.dtype!=torch.long or positive_sets.device!=probabilities.device
            or relation_queries.device!=probabilities.device):
        raise ValueError('explicit ray distributions and separate boolean loss-only positive sets required')
    if (not torch.isfinite(probabilities).all() or (probabilities<0).any()
            or ((relation_queries<0)|(relation_queries>=len(probabilities))).any()):
        raise ValueError('finite probabilities and valid relation query indices required')
    covered=positive_sets.any(-1)
    stats=dict(observable_relations=len(positive_sets),covered_relations=int(covered.sum()),
        proposal_misses=int((~covered).sum()),supervised_queries=int(relation_queries[covered].unique().numel()))
    if not covered.any():return probabilities.sum()*0,stats
    queries=relation_queries[covered]
    mass=(probabilities[queries]*positive_sets[covered]).sum(-1)
    loss=-mass.clamp_min(1e-8).log()
    totals=probabilities.new_zeros(len(probabilities)).index_add(0,queries,loss)
    counts=torch.bincount(queries,minlength=len(probabilities))
    return (totals[counts>0]/counts[counts>0]).mean(),stats
