"""Same-carrier NoCorr: original camera projection, three view candidates.

This replaces semantic retrieval with the native spatially projected source
features. It retains the observed matching-query pool, full-ray utility/R
context, D1/scalar matcher and learned attention. Three geometric view samples
versus64 semantic candidates is a natural module ablation, not a fixed-K
address-only intervention. No depth, mesh visibility or GT addresses are used.
"""
import torch
from torch.nn import functional as F

from .deform_conditioner import DeformationCondition
from .ray_group_matching import RayGroupConditioner
from .ray_group_query import pool_ray_queries
from .target_first_observed_query import ObservedQueryTargetFirstRayConditioner
from .target_first_ray import SourceEvidenceUse
from .target_first_ray_v2 import TargetFirstRayEvidence
from .task_matching_attention import contextual_query


class ObservedQuerySpatialTargetFirstRayConditioner(ObservedQueryTargetFirstRayConditioner):
    def __init__(self, **kwargs):
        if (kwargs.get('ray_mode') != 'native-spatial-attention'
                or kwargs.get('correspondence_mode') != 'native-spatial'
                or kwargs.get('task_attention') is None):
            raise ValueError('native spatial three-view attention NoCorr control required')
        RayGroupConditioner.__init__(self, **kwargs)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(20261020)
            self.evidence_use = SourceEvidenceUse()

    def forward(self, *, native_voxels, target_omega_voxels, target_native_voxels,
                ray_ids, observed_query, source_keys, source_values, source_valid,
                spatial_keys, spatial_values, control='normal'):
        if (control != 'normal' or native_voxels.shape != target_omega_voxels.shape
                or native_voxels.shape != target_native_voxels.shape
                or native_voxels.shape[-1] != 1024 or ray_ids.shape[1] != 3
                or observed_query.shape != native_voxels.shape[:2]
                or observed_query.dtype != torch.bool):
            raise ValueError('aligned native SS inputs and observed target-image query support required')
        batch, views, sites, width = source_values.shape
        rays = views * sites
        if (views != 3 or width != 1024 or spatial_keys.shape != (batch, rays, 3, 1024)
                or spatial_values.shape != spatial_keys.shape):
            raise ValueError('three original projected source-view candidates per ray required')
        matching_ids = torch.where(observed_query[:, None], ray_ids, torch.full_like(ray_ids, -1))
        utility = pool_ray_queries(native_voxels, ray_ids, num_rays=sites)
        omega = pool_ray_queries(target_omega_voxels, matching_ids, num_rays=sites)
        retarget = pool_ray_queries(target_native_voxels, ray_ids, num_rays=sites)
        omega_values = omega.values.flatten(1, 2)
        self._validate(omega_values, source_keys, source_values, source_valid)
        active = (utility.valid & omega.valid).flatten(1)
        active &= (source_valid.flatten(1).sum(-1) >= 64)[:, None]
        selected = active.flatten().nonzero().flatten()
        # These IDs denote the three geometric views, not semantic patch IDs.
        indices = torch.arange(3, device=source_values.device).reshape(1, 1, 3).expand(batch, rays, -1)
        evidence = source_values.new_zeros(batch * rays, 1024, dtype=torch.float32)
        probability = source_values.new_zeros(batch * rays, 3, dtype=torch.float32)
        logits = source_values.new_full((batch * rays, 3), -torch.inf, dtype=torch.float32)
        outputs, weights, scores = [], [], []
        query = utility.values.reshape(-1, 1024)
        for start in range(0, len(selected), self.ray_chunk_size):
            ix = selected[start:start + self.ray_chunk_size]
            keys = spatial_keys.reshape(-1, 3, 1024)[ix].float()
            values = spatial_values.reshape(-1, 3, 1024)[ix].float()
            valid = torch.ones(len(ix), 3, dtype=torch.bool, device=values.device)
            with torch.no_grad(), torch.autocast(device_type=values.device.type, enabled=False):
                context = contextual_query(self.q2c.decoder, omega_values.reshape(-1, 1024)[ix].float(), keys, valid)
                nq, nk = F.normalize(context, dim=-1), F.normalize(keys, dim=-1)
                cosine = (nq[:, None] * nk).sum(-1)
                score = self.q2c.fusion_head(nq, nk, cosine[..., None], cosine) / self.score_temperature
            result = self.task_attention(query[ix], context, values, valid,
                score if self.task_attention_config['score_prior'] else None)
            outputs.append(result[0]); weights.append(result[1]); scores.append(result[2])
        if len(selected):
            evidence = evidence.index_copy(0, selected, torch.cat(outputs))
            probability = probability.index_copy(0, selected, torch.cat(weights))
            logits = logits.index_copy(0, selected, torch.cat(scores))
        else:
            evidence = evidence + sum(p.reshape(-1)[0] * 0 for p in self.parameters() if p.requires_grad)
        residual = evidence.new_zeros(batch * rays, 1024)
        target = retarget.values.reshape(batch, rays, 1024)
        if len(selected):
            changed = self.retargeter(torch.cat((evidence[selected], target.flatten(0, 1)[selected]), -1), self.g_max)
            residual = residual.index_copy(0, selected, changed.float())
        evidence = evidence.reshape(batch, rays, 1024)
        residual = residual.reshape(batch, rays, 1024)
        probability = probability.reshape(batch, rays, 3)
        scale = torch.tanh(self.fusion_gate)
        attention = DeformationCondition(evidence + residual, indices,
            logits.reshape(batch, rays, 3), probability, probability, residual, active,
            scale, base_evidence=evidence)
        use = self.evidence_use(target, evidence, active)
        mass = use[..., None] * scale
        numerator = mass * evidence + use[..., None] * residual
        return TargetFirstRayEvidence(numerator.reshape(batch, 3, sites, 1024),
            mass.reshape(batch, 3, sites, 1), attention, use)


@torch.no_grad()
def native_spatial_ray_candidates(grid, source_keys, source_values, ray_ids, camera):
    """Original source-camera ProjGrid followed by complete-ray mean pooling."""
    from ..trainers.flow_matching.mixins.image_conditioned_proj import compute_multiview_projection_matrices
    if (source_keys.shape != source_values.shape or source_values.shape[1:] != (3, 1024, 1024)
            or ray_ids.shape != (source_values.shape[0], 3, 4096)):
        raise ValueError('native three32patch maps and unmasked SS16 camera-ray membership required')
    matrices, _ = compute_multiview_projection_matrices(camera['transform_matrix'],
        camera['distance'], grid.front_view_transform_matrix)
    batch = source_values.shape[0]
    keys = source_values.new_empty(batch, 3072, 3, 1024, dtype=torch.float32)
    values = torch.empty_like(keys)
    # Fill final buffers directly: avoid holding both six map lists and their
    # stacked copies at full microbatch96. Features and cameras are frozen.
    for source in range(3):
        view = source + 1
        for maps, output in [(source_keys, keys), (source_values, values)]:
            volume = grid(maps[:, source].float().reshape(batch, 32, 32, 1024),
                camera['camera_angle_x'][:, view], camera['distance'][:, view],
                camera['mesh_scale'], matrices[:, view])
            output[:, :, source] = pool_ray_queries(volume, ray_ids, num_rays=1024).values.flatten(1, 2)
            del volume
    return keys, values
