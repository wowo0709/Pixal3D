"""Matched frozen-score Top64 control for observed-query task attention.

This retains exactly the observed semantic query, D1 matcher, native ray
context, retargeter, use decision and target-first projection. Only candidate
readout is fixed to the matching-score weighted mean. The user's NoScore
control remains the separate actual Top1 feature arm.
"""
import torch

from .ray_group_matching import RayGroupConditioner
from .target_first_observed_query import ObservedQueryTargetFirstRayConditioner
from .target_first_ray import SourceEvidenceUse


class ObservedQueryWeightedTargetFirstRayConditioner(ObservedQueryTargetFirstRayConditioner):
    def __init__(self, **kwargs):
        if (kwargs.get('ray_mode') != 'semantic-weighted'
                or kwargs.get('candidate_aggregation', 'weighted') != 'weighted'
                or kwargs.get('task_attention') is not None
                or kwargs.get('correspondence_mode', 'semantic') != 'semantic'):
            raise ValueError('fixed matching-score weighted Top64 semantic control required')
        # The historical target-first constructor only admits Top1/attention.
        # Initialize their identical RayGroup base directly; reuse the entire
        # frozen observed-query forward without changing any science source.
        RayGroupConditioner.__init__(self, **kwargs)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(20261020)
            self.evidence_use = SourceEvidenceUse()
