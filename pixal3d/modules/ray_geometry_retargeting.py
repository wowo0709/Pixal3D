"""Use ray correspondence to retarget geometry without transforming appearance.

The native Top1 appearance and the learned R adapter remain the carrier.
Matching-aware attention combines RAW Omega observations for the physical
deformation query. There are no appearance value/output projections. A zero
Q projection preserves the pretrained matching prior at initialization; it
does not replace any RGB or Omega observation with zeros.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .ray_deformation_query import RayDeformationQueryConditioner


class RayGeometryReadout(nn.Module):
    def __init__(self, reference):
        super().__init__()
        self.value_width = reference.value_width
        self.heads, self.head_width = reference.heads, reference.head_width
        self.score_selector = reference.score_selector
        self.query, self.key = reference.query, reference.key
        nn.init.zeros_(self.query.weight)

    def forward(self, native, context, keys, values, valid, prior):
        m, k = valid.shape
        if (native.shape != (m, 1024) or context.shape != native.shape
                or keys.shape != (m, k, 1024) or values.shape != (m, k, self.value_width)
                or prior.shape != valid.shape or valid.dtype != torch.bool):
            raise ValueError('aligned native ray and candidate observations required')
        keys = torch.where(valid[..., None], keys, torch.zeros_like(keys))
        values = torch.where(valid[..., None], values, torch.zeros_like(values))
        score = self.score_selector(native, context, values[..., :1024], valid, prior)[2]
        query_input = torch.cat([F.layer_norm(native, (1024,)),
                                 F.layer_norm(context, (1024,))], -1)
        q = self.query(query_input).reshape(m, self.heads, self.head_width)
        kk = self.key(F.layer_norm(keys, (1024,))).reshape(
            m, k, self.heads, self.head_width).transpose(1, 2)
        with torch.autocast(device_type=native.device.type, enabled=False):
            logits = (q.float()[:, :, None] * kk.float()).sum(-1) / math.sqrt(self.head_width)
            logits = (logits + score[:, None].float()).masked_fill(~valid[:, None], -torch.inf)
            maximum = logits.amax(-1, keepdim=True)
            maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
            probability = torch.exp(logits - maximum).masked_fill(~valid[:, None], 0.)
            probability = probability / probability.sum(-1, keepdim=True).clamp_min(1e-8)
        # The parent geometry_readout consumes these probabilities to combine
        # real RAW source observations. This zero is the absent appearance
        # correction, rather than a fabricated deformation observation.
        return native.new_zeros(m, self.value_width), probability


class RayGeometryRetargetingConditioner(RayDeformationQueryConditioner):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ray_readout = RayGeometryReadout(self.ray_readout)

    def load_reference(self, reference):
        if (set(reference) != set(self.reference_state())
                or torch.count_nonzero(self.retargeter.deformation_projection.weight)
                or torch.count_nonzero(self.ray_readout.query.weight)):
            raise ValueError('complete common R state and untouched geometry initializer required')
        merged = dict(self.state_dict())
        merged.update(reference)
        self.load_state_dict(merged, strict=True)
        if any(not torch.equal(value.detach().cpu(), self.state_dict()[name].detach().cpu())
               for name, value in reference.items()):
            raise ValueError('native matcher, R, or fusion reference changed')

    def forward(self, **kwargs):
        result = super().forward(**kwargs)
        self.last_feature_attention_stats.update(
            appearance_value_projection_present=False,
            appearance_output_projection_present=False,
            matching_tensors=7, geometry_only_correspondence=True)
        return result
