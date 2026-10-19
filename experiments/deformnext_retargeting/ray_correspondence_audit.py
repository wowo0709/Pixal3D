"""Evaluator-only coverage and attention mass for multi-patch camera-ray queries.

GT labels describe all observed target patches touched by each ray. They never
construct queries, proposals, attention scores or model conditioning.
"""
from __future__ import annotations

import torch


RANKS = (1, 4, 8, 16, 32, 64)


def audit_ray_correspondences(*, ray_ids, patch_ids, candidates, prior_weights,
        learned_weights, source_valid, positive_indices, positive_visible,
        target_foreground, rays_per_view=1024):
    """Audit one asset on CPU; deduplicate (source-camera ray,target patch).

    Coverage denominators are observable target-patch/ray pairs, rather than
    rays with any one hit. Multiple distinct targets on one ray all count.
    Exact topology labels and same-view one-patch neighborhoods are separate.
    """
    inputs = [ray_ids,patch_ids,candidates,prior_weights,learned_weights,source_valid,
              positive_indices,positive_visible,target_foreground]
    if any(x.device.type!='cpu' for x in inputs):
        raise ValueError('GT correspondence auditing is explicitly CPU-only')
    if (ray_ids.ndim!=2 or ray_ids.shape[0]!=3 or patch_ids.shape!=(ray_ids.shape[1],)
            or candidates.shape!=(3*rays_per_view,64)
            or positive_indices.shape!=(1024,3) or positive_visible.shape!=(1024,3)
            or source_valid.shape!=(3072,) or target_foreground.shape!=(1024,)):
        raise ValueError('explicit three-camera rays,semantic64 and topology labels required')
    if ((ray_ids < -1)|(ray_ids>=rays_per_view)).any() or ((patch_ids<0)|(patch_ids>=1024)).any():
        raise ValueError('ray or target patch address out of range')
    if ((candidates<0)|(candidates>=3072)).any() or ((positive_indices<0)|(positive_indices>=3072)).any():
        raise ValueError('source correspondence address out of range')
    active = torch.zeros(3*rays_per_view,dtype=torch.bool)
    relation=[]
    for view in range(3):
        valid=ray_ids[view]>=0
        encoded=(ray_ids[view,valid]+view*rays_per_view)*1024+patch_ids[valid]
        relation.append(encoded)
    encoded=torch.cat(relation).unique(sorted=True)
    rays,patches=encoded//1024,encoded%1024
    active[rays]=True
    candidate_valid=source_valid.bool()[candidates]
    distributions={'prior':prior_weights.float(),'learned':learned_weights.float()}
    for weights in distributions.values():
        if (weights.shape!=candidates.shape or not torch.isfinite(weights).all() or (weights<0).any()
                or not torch.allclose(weights[active].sum(-1),torch.ones(int(active.sum())),atol=1e-5,rtol=0)
                or (weights[~candidate_valid]>0).any()):
            raise ValueError('actual normalized candidate probabilities required')
    visible=positive_visible.bool()&source_valid.bool()[positive_indices]&target_foreground.bool()[:,None]
    observable=visible.any(-1)[patches]
    excluded=int((positive_visible.bool()[patches]&~source_valid.bool()[positive_indices[patches]]).sum())
    rays,patches=rays[observable],patches[observable]
    totals=dict(active_rays=int(active.sum()),distinct_ray_target_patch_pairs=len(encoded),
        observable_ray_target_patch_pairs=len(rays),unobservable_pairs_excluded=int((~observable).sum()),
        visible_GT_pairs_excluded_by_source_patch_alpha=excluded,
        rays_with_observable_target_patches=int(rays.unique().numel()),
        coverage={kind:{ranking:{str(k):0 for k in RANKS} for ranking in ['raw','prior','learned']} for kind in ['exact','near1']},
        scalar_top1_miss_top64_hit={kind:0 for kind in ['exact','near1']},
        per_target_patch_probability_mass={kind:{ranking:0. for ranking in ['prior','learned','uniform']} for kind in ['exact','near1']},
        ray_union_probability_mass={kind:{ranking:0. for ranking in ['prior','learned','uniform']} for kind in ['exact','near1']})
    union={kind:torch.zeros_like(candidates,dtype=torch.long) for kind in ['exact','near1']}
    orders={'raw':torch.arange(64)[None].expand(len(candidates),-1),
        **{key:weights.argsort(dim=-1,descending=True,stable=True) for key,weights in distributions.items()}}
    uniform=candidate_valid.float()/candidate_valid.sum(-1,keepdim=True).clamp_min(1)
    for start in range(0,len(rays),512):
        r,p=rays[start:start+512],patches[start:start+512]
        chosen=candidates[r,:,None];truth=positive_indices[p,None,:];seen=visible[p,None,:]
        same_view=chosen//1024==truth//1024
        near=same_view&((chosen%32-truth%32).abs()<=1)&(((chosen%1024)//32-(truth%1024)//32).abs()<=1)
        matches={'exact':((chosen==truth)&seen).any(-1), 'near1':(near&seen).any(-1)}
        for kind,hits in matches.items():
            hits=hits&candidate_valid[r]
            union[kind].index_add_(0,r,hits.long())
            for ranking,order in orders.items():
                ranked=hits.gather(1,order[r])
                for k in RANKS:totals['coverage'][kind][ranking][str(k)]+=int(ranked[:,:k].any(-1).sum())
            top1=hits.gather(1,orders['prior'][r,:1]).any(-1)
            totals['scalar_top1_miss_top64_hit'][kind]+=int((~top1&hits.any(-1)).sum())
            for ranking,weights in {**distributions,'uniform':uniform}.items():
                totals['per_target_patch_probability_mass'][kind][ranking]+=float((weights[r]*hits).sum())
    used=torch.zeros_like(active);used[rays]=True
    for kind,hits in union.items():
        for ranking,weights in {**distributions,'uniform':uniform}.items():
            totals['ray_union_probability_mass'][kind][ranking]=float((weights[used]*(hits[used]>0)).sum())
    totals['attention_sums']={}
    for ranking,weights in distributions.items():
        w=weights[used]
        totals['attention_sums'][ranking]=dict(entropy=float(-(w*w.clamp_min(1e-12).log()).sum()),
            effective_candidates=float((1/w.square().sum(-1).clamp_min(1e-12)).sum()),
            maximum_probability=float(w.max(-1).values.sum()))
    totals['learned_vs_prior_probability_L1_sum']=float((learned_weights[used]-prior_weights[used]).abs().sum())
    return totals
