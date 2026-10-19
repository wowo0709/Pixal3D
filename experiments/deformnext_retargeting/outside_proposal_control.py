"""Replace only added-path values with valid source patches outside Top-K.

A within-Top-K permutation tests slot weighting, not whether the semantic
proposal set excludes irrelevant object parts. This separate saved-weight
control preserves the cached keys, scores, probabilities and Top1 carrier.
"""
from contextlib import contextmanager

import torch
from torch.nn import functional as F


def outside_addresses(source_valid, batch, indices, slot_valid):
    if (source_valid.ndim!=2 or source_valid.dtype!=torch.bool
            or indices.shape!=slot_valid.shape or slot_valid.dtype!=torch.bool
            or batch.shape!=(len(indices),)):
        raise ValueError('aligned full source validity and candidate address/mask required')
    allowed = source_valid[batch].clone()
    allowed.scatter_(1,indices,False)
    total = source_valid.shape[1]
    positions = torch.arange(total,device=indices.device).expand(len(indices),-1)
    ordered = positions.masked_fill(~allowed,total).sort(-1).values
    count = allowed.sum(-1)
    ranks = torch.arange(indices.shape[1],device=indices.device).expand_as(indices)
    donor = ordered.gather(1,ranks % count[:,None].clamp_min(1)).clamp_max(total-1)
    changed = slot_valid & (count[:,None]>0)
    return torch.where(changed,donor,indices),changed,count


@contextmanager
def outside_proposal_control(conditioner, events):
    from pixal3d.modules import ray_deformation_query as module
    if not isinstance(conditioner,module.RayDeformationQueryConditioner):
        raise ValueError('physical DQ/direct conditioner required')
    current = {}

    def capture_source(owner,args,kwargs):
        current['source_valid'] = kwargs['source_valid'].flatten(1)

    handle = conditioner.register_forward_pre_hook(capture_source,with_kwargs=True)
    original = module.geometry_readout

    def substituted(head,native,context,values,keys,raw_source,raw_target,batch,indices,valid,prior):
        if torch.is_grad_enabled():
            raise ValueError('outside-proposal values are a read-only saved-weight diagnostic')
        addresses,changed,available = outside_addresses(current['source_valid'],batch,indices,valid)
        wrong = values[batch[:,None],addresses].float()
        wrong = torch.where(valid[...,None],wrong,torch.zeros_like(wrong))
        normalized = torch.cat([F.layer_norm(part,(1024,)) for part in wrong.split(1024,-1)],-1)

        def substitute_value(layer,args):
            if args[0].shape!=normalized.shape:
                raise ValueError('actual normalized value projection shape changed')
            return (normalized,)

        value_hook = head.value.register_forward_pre_hook(substitute_value)
        try:
            feature,_,probability = original(head,native,context,values,keys,
                raw_source,raw_target,batch,indices,valid,prior)
        finally:
            value_hook.remove()
        raw = raw_source[batch[:,None],addresses].float()
        raw = torch.where(valid[...,None],raw,torch.zeros_like(raw))
        geometry = raw_target.float()-(probability.mean(1)[...,None]*raw).sum(1)
        events.append(dict(control='outside-proposal-values',queries=len(indices),
            valid_slots=int(valid.sum()),reassigned_outside_valid_slots=int(changed.sum()),
            queries_without_outside_source_patches=int((available==0).sum()),
            outside_source_patches_mean=float(available.float().mean()),
            candidate_keys_and_cached_scores_and_attention_probabilities_retained=True,
            original_Top1_carrier_retained=True,appearance_and_RAW_share_donor_addresses=True,
            donors_are_same_asset_RGB_valid_source_patches=True,
            missing_outside_donors_preserve_original_values=True,GT_added=False))
        return feature,geometry,probability

    module.geometry_readout = substituted
    try:
        yield
    finally:
        module.geometry_readout = original
        handle.remove()
