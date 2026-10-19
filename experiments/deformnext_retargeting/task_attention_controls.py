"""Saved-weight value-address intervention, preserving learned keys and weights."""
from contextlib import contextmanager

import torch


def _reassign(values, valid, evidence, weights, logits, events):
    n,k,_=values.shape
    count=valid.sum(-1)
    position=torch.arange(k,device=values.device).expand(n,-1)
    ordered=position.masked_fill(~valid,k).sort(-1).values
    rank=valid.long().cumsum(-1)-1
    donor_rank=(rank+count[:,None]//2)%count[:,None].clamp_min(1)
    donor=ordered.gather(1,donor_rank).clamp_max(k-1)
    moved=values.gather(1,donor[...,None].expand_as(values))
    moved=torch.where(valid[...,None],moved,torch.zeros_like(moved))
    altered=(weights[...,None]*moved.float()).sum(1)
    events.append({'queries':n,'candidate_count':k,
        'valid_slots':int(valid.sum()),'reassigned_valid_slots':int(((donor!=position)&valid).sum()),
        'evidence_change_RMS':float((altered-evidence).detach().float().square().mean().sqrt()),
        'matching_logits_and_weights_unchanged':True})
    return altered,weights,logits


@contextmanager
def task_value_context(conditioner,control,events):
    if control=='normal':
        yield
        return
    if control!='wrong-address':
        raise ValueError('task value-address control requires a task attention model')
    if getattr(conditioner,'task_attention',None) is not None:
        def swap_values(module,args,output):
            _,_,values,valid,*_=args
            evidence,weights,logits=output
            return _reassign(values,valid,evidence,weights,logits,events)
        hook=conditioner.task_attention.register_forward_hook(swap_values)
        try:yield
        finally:hook.remove()
        return
    # Frozen soft attention has no task-attention Module hook. Intervene at
    # its aggregation boundary instead, after the original weights were
    # computed. This keeps the SAME per-query candidate value multiset, unlike
    # perturbing the full source feature map before retrieval.
    static_ray = getattr(conditioner, 'ray_mode', None) in {'semantic-top1', 'semantic-weighted'}
    if (getattr(conditioner,'implementation',None)!='r9-native-v5'
            or getattr(conditioner,'arm',None) not in {'b3','c1'}
            or (not static_ray and getattr(conditioner,'candidate_aggregation',None)!='weighted')):
        raise ValueError('value-address control requires a task or static soft attention model')
    previous=conditioner.__dict__.get('_attention')
    original=conditioner._attention
    def aggregate(logits,values,valid,view_index,views):
        evidence,weights,view_weights=original(logits,values,valid,view_index,views)
        b,q,k,d=values.shape
        changed,_,_=_reassign(values.reshape(b*q,k,d),valid.reshape(b*q,k),
            evidence.reshape(b*q,d),weights.reshape(b*q,k),logits.reshape(b*q,k),events)
        return changed.reshape(b,q,d),weights,view_weights
    conditioner._attention=aggregate
    try:yield
    finally:
        if previous is None:del conditioner._attention
        else:conditioner._attention=previous
