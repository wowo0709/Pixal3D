"""Source-relative retargeting with a function-anchored matching posterior.

Composes the existing two mechanisms. No ray, camera, matching candidate,
target stream or projection rule changes, and no trainable tensor is added.
Explicit new checkpoint identity prevents silent loading as either parent.
"""
import torch

from .bounded_member_feature_source_ray import BoundedPriorRayFeatureReadout
from .source_contrast_member_feature import SourceContrastMemberFeatureConditioner


class BoundedSourceContrastMemberFeatureConditioner(SourceContrastMemberFeatureConditioner):
    def __init__(self, *, residual_logit_bound, **kwargs):
        super().__init__(**kwargs)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(20261021)
            self.ray_readout=BoundedPriorRayFeatureReadout(
                1024,residual_logit_bound=residual_logit_bound)
        self.register_buffer('bounded_source_contrast_schema',torch.tensor([1],dtype=torch.uint8))

    def load_utility(self,state):
        super().load_utility(state)
        self.ray_readout.capture_initial_function()
