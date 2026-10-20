"""Target carrier with optional source-camera ray corrections.

The learned use gate is trained by generation loss. It is a source-utility
gate, not calibrated visibility or proof of a surface correspondence.
"""
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .ray_group_matching import RayGroupConditioner
from .ray_group_query import pool_ray_queries
from .deform_conditioner import DeformationCondition


@dataclass
class TargetFirstRayCondition:
    map_delta: torch.Tensor
    attention: DeformationCondition
    use_probability: torch.Tensor


class SourceEvidenceUse(nn.Module):
    """An explicit option to reduce source evidence without zero observations."""
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(2048, 64), nn.SiLU(), nn.Linear(64, 1))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, target, evidence, valid):
        if (target.shape != evidence.shape or target.shape[-1] != 1024
                or valid.shape != target.shape[:-1] or valid.dtype != torch.bool):
            raise ValueError('one target/evidence vector and validity bit per ray required')
        # Invalid slots cannot influence this trainable network. Their zero
        # storage is explicitly missing data, never an observed zero feature.
        target = torch.where(valid[..., None], target, torch.zeros_like(target))
        evidence = torch.where(valid[..., None], evidence, torch.zeros_like(evidence))
        x = torch.cat((F.layer_norm(target.float(), (1024,)),
                       F.layer_norm(evidence.float(), (1024,))), -1)
        probability = self.network(x).float().squeeze(-1).sigmoid()
        return torch.where(valid, probability, torch.zeros_like(probability))


class TargetFirstRayConditioner(RayGroupConditioner):
    """Only source maps change; target features are not an input/output buffer.

The inherited ray correction is g*(semantic_value-native_source_value)+R.
Both branches start at zero, while g and R's output receive initial gradients.
The native source map has no direct path to the target conditioning.
"""
    def __init__(self, **kwargs):
        if kwargs.get('ray_mode') not in {'semantic-top1', 'semantic-attention'}:
            raise ValueError('matched Top1 or semantic Top64 attention arm required')
        super().__init__(**kwargs)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(20261020)
            self.evidence_use = SourceEvidenceUse()

    def forward(self, **kwargs):
        output = super().forward(**kwargs)
        query = pool_ray_queries(kwargs['target_native_voxels'], kwargs['ray_ids'],
                                 num_rays=kwargs['source_values'].shape[2])
        query = query.values.flatten(1, 2)
        probability = self.evidence_use(query, output.attention.base_evidence,
                                        output.attention.query_valid)
        delta = output.map_delta * probability.reshape(*output.map_delta.shape[:-1], 1)
        return TargetFirstRayCondition(delta, output.attention, probability)


def add_source_ray_correction(target_condition, projected_delta):
    """Keep native target global/negative state out of source fusion entirely."""
    if set(target_condition) != {'global', 'proj'}:
        raise ValueError('declared native SS target-only global/proj condition required')
    if projected_delta.shape != target_condition['proj'].shape:
        raise ValueError('native source ray projection must match the target SS grid')
    # Return a new mapping. The anchor's tensors, including its global token,
    # are never overwritten or rescaled by the source branch.
    return {
        'global': target_condition['global'],
        'proj': target_condition['proj'] + projected_delta.to(target_condition['proj'].dtype),
    }
