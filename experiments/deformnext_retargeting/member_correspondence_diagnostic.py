"""Set-valued topology labels for an existing source-camera ray query.

Evaluation only: GT never constructs a proposal or a model input. A ray's
positive set is the union of visible M0 correspondences of its distinct M1
target-patch members. This is not a unique surface point per voxel/ray.
"""
import numpy as np


def distinct_members(ray_ids, patch_ids, observed):
    rays=np.asarray(ray_ids);patches=np.asarray(patch_ids);seen=np.asarray(observed)
    if (rays.ndim!=2 or rays.shape!=(3,len(patches)) or patches.ndim!=1
            or seen.shape!=patches.shape or seen.dtype!=np.bool_
            or rays.dtype.kind not in 'iu' or patches.dtype.kind not in 'iu'
            or ((rays < -1)|(rays >= 1024)).any()
            or ((patches < 0)|(patches >= 1024)).any()):
        raise ValueError('explicit native three-camera ray and target32patch addresses required')
    encoded=np.concatenate([(rays[v,seen&(rays[v]>=0)]+v*1024)*1024
        +patches[seen&(rays[v]>=0)]for v in range(3)])
    unique=np.unique(encoded)
    return unique//1024,unique%1024


def topology_sets(ray_ids, patch_ids, observed, positive_indices, positive_visible,
                  target_foreground, candidate_indices, candidate_valid, source_valid):
    """Build all-view and own-camera GT sets after the normal forward."""
    ray,patch=distinct_members(ray_ids,patch_ids,observed)
    positive=np.asarray(positive_indices);visible=np.asarray(positive_visible)
    fg=np.asarray(target_foreground);ids=np.asarray(candidate_indices)
    valid=np.asarray(candidate_valid);source=np.asarray(source_valid)
    if (positive.shape!=(1024,3) or visible.shape!=positive.shape
            or visible.dtype!=np.bool_ or fg.shape!=(1024,) or fg.dtype!=np.bool_
            or ids.shape!=(3072,64) or valid.shape!=ids.shape or valid.dtype!=np.bool_
            or source.shape!=(3,1024) or source.dtype!=np.bool_
            or positive.dtype.kind not in 'iu' or ids.dtype.kind not in 'iu'
            or ((positive<0)|(positive>=3072)).any() or ((ids<0)|(ids>=3072)).any()
            or not (positive//1024==np.arange(3)[None]).all()
            or (valid&~source.reshape(-1)[ids]).any()):
        raise ValueError('certified three-view topology labels and actual native64 proposal required')
    label_visible=visible[patch]&fg[patch,None]
    r=np.repeat(ray,3)[label_visible.reshape(-1)]
    q=positive[patch].reshape(-1)[label_visible.reshape(-1)]
    view=np.tile(np.arange(3),len(ray))[label_visible.reshape(-1)]
    results={}
    for name,keep in [('all_source_views',np.ones(len(r),bool)),
                      ('same_source_camera',view==r//1024)]:
        labels=np.zeros((3072,3072),bool);labels[r[keep],q[keep]]=True
        exact=labels[np.arange(3072)[:,None],ids]&valid
        near=np.zeros_like(exact)
        yy,xx=(ids%1024)//32,ids%32;bank=ids//1024*1024
        for dy in [-1,0,1]:
            for dx in [-1,0,1]:
                inside=(yy+dy>=0)&(yy+dy<32)&(xx+dx>=0)&(xx+dx<32)
                neighbor=bank+np.clip(yy+dy,0,31)*32+np.clip(xx+dx,0,31)
                near|=labels[np.arange(3072)[:,None],neighbor]&inside&valid
        results[name]=dict(observable=labels.any(-1),
            eligible_GT_exists=(labels&source.reshape(1,-1)).any(-1),
            exact=exact,near1=near,GT_set_size=labels.sum(-1))
    return results,dict(ray=ray,patch=patch,relation_ray=r,relation_patch=np.repeat(patch,3)[label_visible.reshape(-1)],
        relation_source_view=view,relation_positive=q)


def summarize_probabilities(label_sets, prior, heads, candidate_valid):
    prior=np.asarray(prior);heads=np.asarray(heads);valid=np.asarray(candidate_valid)
    if (prior.shape!=(3072,64) or heads.shape!=(3072,4,64) or valid.shape!=prior.shape
            or valid.dtype!=np.bool_ or not np.isfinite(prior).all() or not np.isfinite(heads).all()
            or (prior<0).any() or (heads<0).any()):
        raise ValueError('normal native ray probabilities required')
    active=valid.any(-1)
    if (not np.allclose(prior.sum(-1),active.astype(float),atol=1e-5)
            or not np.allclose(heads.sum(-1),active[:,None].astype(float),atol=1e-5)):
        raise ValueError('active probabilities normalize and inactive rows stay zero')
    result={};mean=heads.mean(1)
    for name,labels in label_sets.items():
        observed=labels['observable'];count=int(observed.sum())
        metrics=dict(observable_rays=count,native_supported_rays=int((observed&active).sum()),
            GT_eligible_rays=int((observed&labels['eligible_GT_exists']).sum()),
            GT_set_size_sum=int(labels['GT_set_size'][observed].sum()))
        for kind in ['exact','near1']:
            positive=labels[kind]
            top_prior=positive[np.arange(3072),prior.argmax(-1)]&active
            top_head=positive[np.arange(3072),mean.argmax(-1)]&active
            per_head=positive[np.arange(3072)[:,None],heads.argmax(-1)]&active[:,None]
            metrics[kind]=dict(proposal_hit_rays=int(positive[observed].any(-1).sum()),
                prior_top1_hit_rays=int(top_prior[observed].sum()),mean_head_top1_hit_rays=int(top_head[observed].sum()),
                per_head_top1_hits=int(per_head[observed].sum()),
                prior_GT_mass_sum=float((prior[observed]*positive[observed]).sum()),
                mean_head_GT_mass_sum=float((mean[observed]*positive[observed]).sum()))
        result[name]=metrics
    return result
