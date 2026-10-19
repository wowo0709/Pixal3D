"""Preserve the calibrated correspondence prior before learning ray Q/K.

Only the additional Q projection starts at zero. RGB/Omega observations stay
unchanged. The inherited zero output still preserves the reconstruction
function. This module is separate from every frozen running experiment.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .ray_feature_attention import RayFeatureReadout


class PriorAnchoredRayFeatureReadout(RayFeatureReadout):
    def __init__(self, value_width=1024, heads=4, head_width=64):
        super().__init__(value_width, heads, head_width)
        nn.init.zeros_(self.query.weight)


def correspondence_probabilities(head, native, context, keys, values, valid, prior):
    """Actual native probability math without unused value/output allocations.

    Used only to warm up Q/K. The five pretrained score-selector tensors and
    the value/output projections must be frozen. No labels are accepted.
    """
    if (any(p.requires_grad for p in head.score_selector.parameters())
            or head.value.weight.requires_grad or head.output.weight.requires_grad):
        raise ValueError('Q/K-only supervision requires frozen utility/value/output')
    m, k = valid.shape
    if (native.shape != (m, 1024) or context.shape != native.shape
            or keys.shape != (m, k, 1024) or values.shape != (m, k, head.value_width)
            or prior.shape != valid.shape or valid.dtype != torch.bool):
        raise ValueError('aligned inference-only ray/candidate observations required')
    with torch.no_grad():
        keys = torch.where(valid[..., None], keys, torch.zeros_like(keys))
        values = torch.where(valid[..., None], values, torch.zeros_like(values))
        score = head.score_selector(native, context, values[..., :1024], valid, prior)[2]
    query_input = torch.cat([F.layer_norm(native, (1024,)),
                             F.layer_norm(context, (1024,))], -1)
    q = head.query(query_input).reshape(m, head.heads, head.head_width)
    kk = head.key(F.layer_norm(keys, (1024,))).reshape(
        m, k, head.heads, head.head_width).transpose(1, 2)
    with torch.autocast(device_type=native.device.type, enabled=False):
        logits = (q.float()[:, :, None] * kk.float()).sum(-1) / math.sqrt(head.head_width)
        logits = (logits + score[:, None].float()).masked_fill(~valid[:, None], -torch.inf)
        maximum = logits.amax(-1, keepdim=True)
        maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
        probability = torch.exp(logits - maximum).masked_fill(~valid[:, None], 0.)
        return probability / probability.sum(-1, keepdim=True).clamp_min(1e-8)
