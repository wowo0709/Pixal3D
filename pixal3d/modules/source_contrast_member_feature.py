"""Retarget a feature displacement, keeping the original nineteen parameters.

This is a cold-start alternative to the warm-preserving zero source branch.
R receives [matched appearance + feature change - own native source, target].
Its original zero output preserves the native prior without a second learned
zero projection hiding the new source information from the first update.
"""
from contextlib import contextmanager

import torch
from torch import nn

from .member_feature_source_ray import MemberFeatureSourceRayConditioner
from .source_referenced_member_feature import SourceReferencedMemberFeatureConditioner


class SourceContrastRetargeter(nn.Module):
    def __init__(self, reference):
        super().__init__()
        if torch.count_nonzero(reference.net[-1].weight) or torch.count_nonzero(reference.net[-1].bias):
            raise ValueError('source contrast must initialize from a cold zero-output retargeter')
        self.input_dim = reference.input_dim
        self.net, self.gate_logit = reference.net, reference.gate_logit
        self.source_width = self.net[-1].out_features
        self._source_context = None
        # An ordinary19-tensor parent must fail strict loading of this model;
        # identical parameter names must not silently change inference meaning.
        self.register_buffer('source_contrast_schema', torch.tensor([1], dtype=torch.uint8))

    @contextmanager
    def native_source_context(self, source):
        if self._source_context is not None:
            raise ValueError('one real source reference per forward required')
        self._source_context = source
        try:
            yield
        finally:
            self._source_context = None

    def forward(self, inputs, g_max):
        source = self._source_context
        if (source is None or source.shape != (len(inputs), self.source_width)
                or source.device != inputs.device or source.dtype != inputs.dtype
                or inputs.shape[-1] != self.input_dim):
            raise ValueError('aligned real own-ray source feature and original input required')
        appearance, target = inputs.split((self.source_width, self.input_dim-self.source_width), dim=-1)
        contrast = torch.cat((appearance-source, target), dim=-1)
        return g_max*torch.tanh(self.gate_logit)*torch.tanh(self.net(contrast))


class SourceContrastMemberFeatureConditioner(SourceReferencedMemberFeatureConditioner):
    """Same source-ray forward, Top64/head/use/projection and nineteen weights.

    This changes the R input representation, not the attention query. It is
    neither the RAW2048 deformation-query arm nor a warm-parent continuation.
    """
    def __init__(self, **kwargs):
        # Reuse own-ray forwarding without constructing an unused source
        # projection. The actual original19-weight initialization is retained.
        MemberFeatureSourceRayConditioner.__init__(self, **kwargs)
        self.retargeter = SourceContrastRetargeter(self.retargeter)
