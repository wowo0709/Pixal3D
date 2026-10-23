"""Original camera-projected three-view candidates with RAW cosine scoring."""
import torch
from torch.nn import functional as F

from .member_ray_posterior import MemberRayPosterior
from .r9_matching import categorical_weights


@torch.no_grad()
def raw_spatial_member_posteriors(*,target_keys,source_valid,ray_ids,patch_ids,
                                  observed_query,spatial_keys,temperature,chunk_size):
    batch=target_keys.shape[0]
    if (target_keys.shape!=(batch,1024,1024) or spatial_keys.shape!=(batch,3072,3,1024)
            or source_valid.shape!=(batch,3,1024) or source_valid.dtype!=torch.bool
            or ray_ids.shape[:2]!=(batch,3) or ray_ids.dtype!=torch.long
            or patch_ids.shape!=(batch,ray_ids.shape[-1]) or patch_ids.dtype!=torch.long
            or observed_query.shape!=patch_ids.shape or observed_query.dtype!=torch.bool
            or ((ray_ids < -1)|(ray_ids>=1024)).any()
            or ((patch_ids < 0)|(patch_ids>=1024)).any()
            or not 0<temperature or not isinstance(chunk_size,int) or chunk_size<1):
        raise ValueError('original spatial keys, member addresses and positive RAW temperature required')
    device=target_keys.device
    ids=torch.arange(3,device=device).reshape(1,1,3).expand(batch,3072,-1)
    valid=torch.zeros(batch,3072,3,dtype=torch.bool,device=device)
    probability=torch.zeros(batch,3072,3,device=device,dtype=torch.float32)
    context=torch.zeros(batch,3072,1024,device=device,dtype=torch.float32)
    counts=torch.zeros(batch,3072,device=device,dtype=torch.long)
    for b in range(batch):
        pairs=[]
        for view in range(3):
            seen=observed_query[b]&(ray_ids[b,view]>=0)
            pairs.append((ray_ids[b,view,seen]+view*1024)*1024+patch_ids[b,seen])
        encoded=torch.cat(pairs).unique(sorted=True);ray,patch=encoded//1024,encoded%1024
        counts[b]=torch.bincount(ray,minlength=3072)
        if not len(encoded) or int(source_valid[b].sum())<64:continue
        with torch.autocast(device_type=device.type,enabled=False):
            for start in range(0,len(encoded),chunk_size):
                take=slice(start,start+chunk_size);r=ray[take]
                query=target_keys[b,patch[take]].float();keys=spatial_keys[b,r].float()
                score=(F.normalize(query,dim=-1)[:,None]*F.normalize(keys,dim=-1)).sum(-1)
                mask=torch.ones_like(score,dtype=torch.bool)
                context[b].index_add_(0,r,query)
                probability[b].index_add_(0,r,categorical_weights(score/temperature,mask,1.))
        active=counts[b]>0;context[b]/=counts[b].clamp_min(1)[:,None]
        probability[b]/=counts[b].clamp_min(1)[:,None];valid[b,active]=True
    return MemberRayPosterior(ids,valid,probability,context,valid.any(-1).float(),counts)
