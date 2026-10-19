"""GT-free prototype: retain constituent target-patch proposals inside each ray."""
from __future__ import annotations

import torch


def merge_ray_patch_proposals(ray_ids, patch_ids, patch_candidates, source_valid, *, rays_per_view=1024):
    """Keep64 distinct source addresses by round-robin constituent-patch ranks.

    The attention query remains one camera ray. Candidate proposal construction
    uses RGB/Omega tokens and native camera membership only. It does not average
    distinct target tokens before retrieval or read correspondence labels.
    """
    if any(value.device.type!='cpu' for value in [ray_ids,patch_ids,patch_candidates,source_valid]):
        raise ValueError('small candidate-address merge is CPU-only')
    if (ray_ids.ndim!=2 or ray_ids.shape[0]!=3 or patch_ids.shape!=(ray_ids.shape[1],)
            or patch_candidates.shape!=(1024,64) or source_valid.shape!=(3072,)):
        raise ValueError('explicit three-camera ray and patch proposal shapes required')
    if ((ray_ids<-1)|(ray_ids>=rays_per_view)).any() or ((patch_ids<0)|(patch_ids>=1024)).any():
        raise ValueError('native ray or target-patch address out of range')
    if ((patch_candidates<0)|(patch_candidates>=3072)).any():
        raise ValueError('source proposal address out of range')
    result=torch.zeros(3*rays_per_view,64,dtype=torch.long)
    active=torch.zeros(3*rays_per_view,dtype=torch.bool)
    for view in range(3):
        for ray in ray_ids[view].unique(sorted=True).tolist():
            if ray<0:continue
            patches=patch_ids[ray_ids[view]==ray].unique(sorted=True)
            proposed=patch_candidates[patches]
            if not source_valid.bool()[proposed].all():
                raise ValueError('source-alpha-invalid candidate cannot enter the union')
            chosen=[];seen=set()
            # Fixed row-major target-patch order breaks partial-round ties.
            for address in proposed.T.flatten().tolist():
                if address not in seen:
                    chosen.append(address);seen.add(address)
                    if len(chosen)==64:break
            if len(chosen)!=64:
                raise ValueError('constituent patch proposals do not supply64 distinct addresses')
            index=view*rays_per_view+ray
            result[index]=torch.tensor(chosen,dtype=torch.long);active[index]=True
    return result,active
