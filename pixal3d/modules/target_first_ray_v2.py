"""Source evidence anchored to the target that is actually in the carrier.

Do not subtract an old source map from a target-only carrier. Native projection
of the weighted source numerator and its scalar weight implements the original
target-first partial-write formula without a surface visibility mask.
"""
from dataclasses import dataclass

import torch

from .ray_group_matching import RayGroupConditioner
from .ray_group_query import pool_ray_queries
from .target_first_ray import TargetFirstRayConditioner
from .deform_conditioner import DeformationCondition


@dataclass
class TargetFirstRayEvidence:
    source_numerator: torch.Tensor
    semantic_weight: torch.Tensor
    attention: DeformationCondition
    use_probability: torch.Tensor


class TargetFirstRayConditionerV2(TargetFirstRayConditioner):
    def forward(self, **kwargs):
        # Reuse retrieval/R, not its delta relative to the native MV4 source.
        output = RayGroupConditioner.forward(self, **kwargs)
        query = pool_ray_queries(kwargs['target_native_voxels'],kwargs['ray_ids'],
            num_rays=kwargs['source_values'].shape[2]).values.flatten(1,2)
        attention = output.attention
        use = self.evidence_use(query,attention.base_evidence,attention.query_valid)
        weight = use[...,None]*attention.fusion_scale
        numerator = weight*attention.base_evidence + use[...,None]*attention.retarget_residual
        shape = output.map_delta.shape
        return TargetFirstRayEvidence(numerator.reshape(shape),
            weight.reshape(*shape[:-1],1),attention,use)


def project_target_first_evidence(grid, numerator, semantic_weight, camera, target):
    """T + sum_v[P_v(g*u*E+u*R)-P_v(g*u)*T]/4, using native ray projection.

    The original semantic gate g is signed and bounded by tanh. It is not an
    observed-validity probability. Retargeting R stays outside its zero gate.
    """
    from ..trainers.flow_matching.mixins.image_conditioned_proj import compute_multiview_projection_matrices
    if (numerator.ndim!=4 or numerator.shape[1:]!=(3,1024,1024)
            or semantic_weight.shape!=numerator.shape[:-1]+(1,)
            or target.shape!=(numerator.shape[0],4096,1024)):
        raise ValueError('nativeSS three32patch maps, scalar semantic weights and target16grid required')
    matrices,_ = compute_multiview_projection_matrices(camera['transform_matrix'],
        camera['distance'],grid.front_view_transform_matrix)
    delta = None
    for source in range(3):
        view = source+1
        def project(values):
            return grid(values[:,source].float().reshape(values.shape[0],32,32,-1),
                camera['camera_angle_x'][:,view],camera['distance'][:,view],
                camera['mesh_scale'],matrices[:,view]).float()
        change = project(numerator)-project(semantic_weight)*target.float()
        delta = change if delta is None else delta+change
    return delta/4
