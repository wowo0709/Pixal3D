"""RAW cosine patch matching pooled into one posterior per source-camera ray.

Only input image features, input-alpha support and camera-derived addresses
construct the posterior. Distinct target patches vote once per ray. This
retains complementary D1 modes before truncating to one final64-key support.
"""

import torch
from torch.nn import functional as F

from .r9_matching import categorical_weights


from .member_ray_posterior import MemberRayPosterior


@torch.no_grad()
def raw_member_ray_posteriors(*, target_keys, source_keys, source_valid,
                          ray_ids, patch_ids, observed_query,
                          temperature=.05, chunk_size=64):
    """Mix normalized RAW64 cosine posteriors; no decoder, values or GT are inputs.

    Process assets separately and D1 point queries in bounded chunks. Native
    delivery remains one feature per source-camera ray; these point queries
    construct its support/prior rather than separate voxel write decisions.
    Incomplete64-key source support or empty observations abstain explicitly.
    CPU and CUDA use the same implementation, with FP32 frozen matching math.
    """
    batch=target_keys.shape[0]
    if (target_keys.shape!=(batch,1024,1024)
            or source_keys.shape!=(batch,3,1024,1024)
            or source_valid.shape!=(batch,3,1024) or source_valid.dtype!=torch.bool
            or ray_ids.ndim!=3 or ray_ids.shape[:2]!=(batch,3)
            or patch_ids.shape!=(batch,ray_ids.shape[-1]) or patch_ids.dtype!=torch.long
            or ray_ids.dtype!=torch.long or observed_query.shape!=patch_ids.shape
            or observed_query.dtype!=torch.bool
            or not isinstance(chunk_size,int) or isinstance(chunk_size,bool) or chunk_size<1
            or not 0<temperature or not torch.isfinite(torch.tensor(temperature))):
        raise ValueError('explicit three32patch image maps, native ray/patch IDs and observed queries required')
    device=target_keys.device
    if any(x.device!=device for x in [source_keys,source_valid,ray_ids,patch_ids,observed_query]):
        raise ValueError('image descriptors and camera addresses must share one device')
    if ((ray_ids<-1)|(ray_ids>=1024)).any() or ((patch_ids<0)|(patch_ids>=1024)).any():
        raise ValueError('native three-camera ray or target-patch address out of range')
    indices=torch.zeros(batch,3072,64,dtype=torch.long,device=device)
    valid=torch.zeros_like(indices,dtype=torch.bool)
    probability=torch.zeros(batch,3072,64,dtype=torch.float32,device=device)
    context=torch.zeros(batch,3072,1024,dtype=torch.float32,device=device)
    retained=torch.zeros(batch,3072,dtype=torch.float32,device=device)
    counts=torch.zeros(batch,3072,dtype=torch.long,device=device)
    for b in range(batch):
        pairs=[]
        for view in range(3):
            seen=observed_query[b]&(ray_ids[b,view]>=0)
            pairs.append((ray_ids[b,view,seen]+view*1024)*1024+patch_ids[b,seen])
        encoded=torch.cat(pairs).unique(sorted=True)
        ray,patch=encoded//1024,encoded%1024
        counts[b]=torch.bincount(ray,minlength=3072)
        if not len(encoded) or int(source_valid[b].sum())<64:
            continue
        points=patch.unique(sorted=True)
        used,ray_local=ray.unique(sorted=True,return_inverse=True)
        point_local=torch.searchsorted(points,patch)
        membership=torch.zeros(len(used),len(points),device=device,dtype=torch.float32)
        membership[ray_local,point_local]=1.
        with torch.autocast(device_type=device.type,enabled=False):
            query=target_keys[b,points].float()
            source=source_keys[b].flatten(0,1).float()
            raw=F.normalize(query,dim=-1)@F.normalize(source,dim=-1).T
            raw=raw.masked_fill(~source_valid[b].flatten()[None],-torch.inf)
            point_ids=raw.topk(64,dim=-1).indices
            point_valid=source_valid[b].flatten()[point_ids]
            point_score=raw.gather(1,point_ids)
            point_context=query
            point_probability=categorical_weights(point_score/temperature,point_valid,1.)
            dense=torch.zeros(len(points),3072,device=device,dtype=torch.float32)
            dense.scatter_(1,point_ids,point_probability)
            mass=membership@dense/counts[b,used,None]
            top,chosen=mass.masked_fill(~source_valid[b].flatten()[None],-torch.inf).topk(64,dim=-1)
            kept=top.sum(-1)
            if not torch.isfinite(top).all() or (kept<=0).any():
                raise ValueError('complete source64 support and positive retained posterior mass required')
            indices[b,used]=chosen
            valid[b,used]=source_valid[b].flatten()[chosen]
            probability[b,used]=top/kept[:,None]
            context[b,used]=membership@point_context/counts[b,used,None]
            retained[b,used]=kept
    return MemberRayPosterior(indices,valid,probability,context,retained,counts)
