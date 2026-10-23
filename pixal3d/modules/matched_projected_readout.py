"""Actual Top1 versus frozen-prior TopK with identical V/output projections.

Neither control contains trainable Q/K or utility-score projections. The full
matching-aware readout remains in bounded_member_feature_source_ray.py.
"""
import torch
from torch import nn
from torch.nn import functional as F

from .bounded_member_feature_source_ray import masked_probabilities


def selection_probabilities(prior, valid, mode):
    if (prior.ndim != 2 or prior.shape != valid.shape or valid.dtype != torch.bool
            or not valid.any(-1).all() or mode not in {'top1','fixed-topk'}):
        raise ValueError('active actual candidate rows and explicit selection required')
    if mode == 'top1':
        choice = prior.masked_fill(~valid, -torch.inf).argmax(-1, keepdim=True)
        return torch.zeros_like(prior).scatter(1, choice, 1.)[:, None].expand(-1,4,-1)
    return masked_probabilities(prior[:, None].expand(-1,4,-1), valid)


class MatchedProjectedReadout(nn.Module):
    def __init__(self, full_head, *, mode):
        super().__init__()
        if mode not in {'top1','fixed-topk'} or full_head.value_width not in (1024,2048):
            raise ValueError('physical Top1 or fixed-score TopK source readout required')
        self.value, self.output = full_head.value, full_head.output
        self.value_width, self.heads, self.head_width = full_head.value_width,4,64
        self.mode = mode
        self.register_buffer('matched_selection_schema',torch.tensor([1 if mode=='top1' else 2],dtype=torch.uint8))
        if set(dict(self.named_parameters())) != {'value.weight','output.weight'}:
            raise ValueError('only shared V and output parameters may exist in this control')

    def forward(self, native, context, keys, values, valid, prior):
        rows,count = valid.shape
        if (native.shape!=(rows,1024) or context.shape!=native.shape
                or keys.shape!=(rows,count,1024) or values.shape!=(rows,count,self.value_width)):
            raise ValueError('same native/context/key/value candidate contract required')
        probability = selection_probabilities(prior, valid, self.mode)
        if self.mode == 'top1':
            # Gather one FEATURE before V, rather than compute a K-wide value
            # transform and describe uniform or onehot attention as Top1.
            choice=probability[:,0].argmax(-1,keepdim=True)
            chosen=values.gather(1,choice[...,None].expand(-1,-1,self.value_width))[:,0].float()
            normalized=torch.cat([F.layer_norm(chunk,(1024,))
                                  for chunk in chosen.split(1024,dim=-1)],-1)
            combined=self.value(normalized)
        else:
            values=torch.where(valid[...,None],values,0.)
            normalized=torch.cat([F.layer_norm(chunk,(1024,))
                                  for chunk in values.split(1024,dim=-1)],-1)
            projected=self.value(normalized).reshape(rows,count,4,64).transpose(1,2)
            with torch.autocast(device_type=values.device.type,enabled=False):
                combined=(probability[...,None]*projected.float()).sum(-2).reshape(rows,-1)
        return self.output(combined).float(),probability
