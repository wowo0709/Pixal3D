"""Saved-weight value-address intervention, preserving learned keys and weights."""
from contextlib import contextmanager

import torch


@contextmanager
def task_value_context(conditioner,control,events):
    if control=='normal':
        yield
        return
    if control!='wrong-address' or getattr(conditioner,'task_attention',None) is None:
        raise ValueError('task value-address control requires a task attention model')
    def swap_values(module,args,output):
        _,_,values,valid,*_=args
        evidence,weights,logits=output
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
    hook=conditioner.task_attention.register_forward_hook(swap_values)
    try:yield
    finally:hook.remove()
