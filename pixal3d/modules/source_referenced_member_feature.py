"""Give a source-ray retargeter its own native source feature explicitly.

The original matching, ray query, Top1 carrier, source-use and projection are
unchanged. A zero-initialized source projection preserves the complete parent
function, including an already trained parent, before the new branch learns.
"""
from contextlib import contextmanager

import torch
from torch import nn

from .member_feature_source_ray import MemberFeatureSourceRayConditioner
from .ray_group_query import pool_ray_queries


class SourceReferencedRetargeter(nn.Module):
    def __init__(self, reference):
        super().__init__()
        self.input_dim = reference.input_dim
        self.net = reference.net
        self.gate_logit = reference.gate_logit
        self.source_width = self.net[-1].out_features
        # Preserve the caller's initialization/sampling RNG. The original
        # network retains its input shape, weights and matrix multiplication.
        with torch.random.fork_rng(devices=[]):
            self.source_projection = nn.Linear(self.source_width, self.net[0].out_features,
                                               bias=False)
            nn.init.zeros_(self.source_projection.weight)
        self.source_projection.to(device=self.net[0].weight.device,
                                  dtype=self.net[0].weight.dtype)
        self._source_context = None

    @contextmanager
    def native_source_context(self, source):
        if self._source_context is not None:
            raise ValueError('source reference belongs to exactly one conditioner forward')
        self._source_context = source
        try:
            yield
        finally:
            self._source_context = None

    def forward(self, inputs, g_max):
        source = self._source_context
        if (source is None or source.shape != (len(inputs), self.source_width)
                or source.device != inputs.device or source.dtype != inputs.dtype):
            raise ValueError('actual own-ray native source features are required')
        hidden = self.net[0](inputs) + self.source_projection(source)
        output = self.net[2](self.net[1](hidden))
        return g_max * torch.tanh(self.gate_logit) * torch.tanh(output)


class SourceReferencedMemberFeatureConditioner(MemberFeatureSourceRayConditioner):
    """Same semantic64/NoDQ branch with one additional native-source tensor.

    Source context is the original image patch of this source-camera ray,
    including real background descriptors. It is never a retrieved candidate
    or a zero placeholder. This adds no GT or visibility input/write gate.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.retargeter = SourceReferencedRetargeter(self.retargeter)

    def forward(self, *, source_values, source_valid, ray_ids, observed_query, **kwargs):
        # Member posterior has support iff the ray has an input-observed
        # target member and at least64 source candidates exist. Reconstruct
        # only this support; do not repeat or alter matching/score selection.
        observations = pool_ray_queries(observed_query.float()[..., None], ray_ids,
                                        num_rays=1024)
        supported = observations.values[..., 0] > 0
        supported &= (source_valid.flatten(1).sum(-1) >= 64)[:, None, None]
        selected = supported.flatten().nonzero().flatten()
        own_source = source_values.float().reshape(-1, 1024)[selected]
        with self.retargeter.native_source_context(own_source):
            output = super().forward(source_values=source_values, source_valid=source_valid,
                ray_ids=ray_ids, observed_query=observed_query, **kwargs)
        if not torch.equal(output.attention.query_valid, supported.flatten(1)):
            raise ValueError('native source rows must exactly match the original posterior support')
        self.last_source_reference_rows = selected.detach()
        return output
