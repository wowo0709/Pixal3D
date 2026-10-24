"""Correspondence substitutions in the same shared Fixed evidence delivery.

RAW and spatial controls physically omit the matching network. D1 cosine
retains its frozen descriptor decoder and omits the scalar score head.
Spatial values are the original projected three-view ray candidates, rather
than semantic image-map gathers or padded fictitious 64-key candidates.
"""
import torch

from .bounded_member_feature_source_ray import masked_probabilities
from .consistent_ray_evidence import ConsistentRayEvidenceConditioner, _check_schema, raw_evidence
from .deform_conditioner import DeformationCondition
from .d1_cosine_member_posterior import d1_cosine_member_ray_posteriors
from .raw_member_ray_posterior import raw_member_ray_posteriors
from .raw_spatial_member_posterior import raw_spatial_member_posteriors
from .ray_group_query import pool_ray_queries
from .target_first_ray_v2 import TargetFirstRayEvidence


MODES = ('native-spatial', 'RAW-cosine', 'D1-cosine')


def candidate_feature_rows(flat, selected, indices, *, spatial_values=None):
    """Gather real candidate values, keeping the two address spaces distinct."""
    if (flat.ndim != 3 or flat.shape[1:] != (3072, 1024)
            or selected.ndim != 1 or selected.dtype != torch.long
            or indices.ndim != 2 or indices.shape[0] != len(selected)
            or indices.dtype != torch.long or any(x.device != flat.device for x in [selected, indices])
            or ((selected < 0) | (selected >= len(flat)*3072)).any()):
        raise ValueError('actual flattened three-camera values and selected source-ray rows required')
    if spatial_values is None:
        if ((indices < 0) | (indices >= 3072)).any():
            raise ValueError('semantic candidates address actual three image feature maps')
        return flat[selected[:, None]//3072, indices]
    if (spatial_values.shape != (len(flat), 3072, 3, 1024)
            or spatial_values.device != flat.device or indices.shape[1] != 3
            or not torch.equal(indices, torch.arange(3, device=flat.device)[None].expand(len(selected), -1))):
        raise ValueError('original three projected spatial values and ordered view-slot addresses required')
    return spatial_values.reshape(-1, 3, 1024)[selected].float()


def candidate_view_weights(indices, weights, *, spatial):
    if indices.shape != weights.shape or indices.dtype != torch.long:
        raise ValueError('aligned candidate addresses and posterior weights required')
    if spatial:
        if (indices.shape[-1] != 3 or not torch.equal(indices,
                torch.arange(3, device=indices.device).expand_as(indices))):
            raise ValueError('three actual spatial view slots required')
        return weights
    if ((indices < 0) | (indices >= 3072)).any():
        raise ValueError('semantic addresses must retain their real source-view identity')
    return weights.new_zeros(*weights.shape[:-1], 3).scatter_add(2, indices//1024, weights)


class FixedCorrespondenceEvidenceConditioner(ConsistentRayEvidenceConditioner):
    def __init__(self, *, correspondence, **kwargs):
        if correspondence not in MODES:
            raise ValueError('explicit original spatial, RAW cosine, or D1 cosine control required')
        super().__init__(evidence_mode='fixed-prior', **kwargs)
        self.correspondence_mode = correspondence
        if correspondence == 'D1-cosine':
            self.q2c.fusion_head = None
            self.q2c.requires_grad_(False).eval()
        else:
            self.q2c = None
        self.register_buffer('fixed_correspondence_evidence_schema',
            torch.tensor([MODES.index(correspondence)+1], dtype=torch.uint8))
        if (sum(p.requires_grad for p in self.parameters()) != 10
                or self.ray_readout is not None or self.retargeter.input_dim != 2048):
            raise ValueError('same physical source10 and Fixed identity-value delivery required')

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        _check_schema(state_dict, prefix, 'fixed_correspondence_evidence_schema',
                      self.fixed_correspondence_evidence_schema, error_msgs)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)

    def train(self, mode=True):
        # RAW/spatial have no q2c module to put into evaluation mode.
        torch.nn.Module.train(self, mode)
        if self.q2c is not None:
            self.q2c.eval()
        return self

    def _posterior(self, *, spatial_keys=None, **kwargs):
        if self.correspondence_mode == 'native-spatial':
            if self.q2c is not None or spatial_keys is None:
                raise ValueError('physical matcher-free original spatial candidates required')
            args = dict(kwargs)
            args.pop('source_keys')
            return raw_spatial_member_posteriors(spatial_keys=spatial_keys, **args)
        if spatial_keys is not None:
            raise ValueError('semantic retrieval must not consume spatial candidates')
        if self.correspondence_mode == 'RAW-cosine':
            if self.q2c is not None:
                raise ValueError('RAW matching decoder and scalar head must be physically absent')
            return raw_member_ray_posteriors(**kwargs)
        if self.q2c is None or self.q2c.fusion_head is not None:
            raise ValueError('actual frozen decoder with scalar head physically absent required')
        return d1_cosine_member_ray_posteriors(self.q2c, **kwargs)

    def forward(self, *, native_voxels, target_omega_voxels, target_native_voxels,
                frame_target_keys, target_patch_ids, ray_ids, observed_query,
                source_keys, source_values, source_valid, spatial_keys=None,
                spatial_values=None, control='normal'):
        if (control != 'normal' or native_voxels.shape != target_native_voxels.shape
                or native_voxels.shape != target_omega_voxels.shape or native_voxels.shape[1:] != (4096, 1024)
                or source_values.shape != (len(native_voxels), 3, 1024, 1024)):
            raise ValueError('native SS16 grid and original three source-camera feature maps required')
        spatial = self.correspondence_mode == 'native-spatial'
        if spatial != (spatial_keys is not None and spatial_values is not None):
            raise ValueError('only spatial mode consumes both real projected key and value candidates')
        if not spatial and (spatial_keys is not None or spatial_values is not None):
            raise ValueError('semantic retrieval must not receive spatial features')
        if spatial and (spatial_values.shape != spatial_keys.shape
                or spatial_values.shape != (len(source_values), 3072, 3, 1024)):
            raise ValueError('original projected three-key/three-value candidates required')
        batch, rays = len(source_values), 3072
        target = pool_ray_queries(target_native_voxels, ray_ids, num_rays=1024)
        posterior = self._posterior(target_keys=frame_target_keys,
            source_keys=source_keys, source_valid=source_valid, ray_ids=ray_ids,
            patch_ids=target_patch_ids, observed_query=observed_query,
            temperature=self.score_temperature, chunk_size=self.ray_chunk_size,
            spatial_keys=spatial_keys)
        indices, valid = posterior.candidate_indices, posterior.candidate_valid
        count = 3 if spatial else 64
        if indices.shape != (batch, rays, count) or valid.shape != indices.shape:
            raise ValueError('actual spatial3 or semantic64 support, without dummy candidates, required')
        selected = valid.any(-1).flatten().nonzero().flatten()
        flat = source_values.flatten(1, 2).float()
        prior = posterior.probabilities.log().reshape(-1, count)
        target_query = target.values.reshape(batch*rays, 1024)
        evidence = flat.new_zeros(batch*rays, 1024)
        weights = flat.new_zeros(batch*rays, count)
        pieces, probabilities = [], []
        for start in range(0, len(selected), self.ray_chunk_size):
            take = selected[start:start+self.ray_chunk_size]
            ids, mask = indices.reshape(-1, count)[take], valid.reshape(-1, count)[take]
            p = masked_probabilities(prior[take, None].expand(-1, 4, -1), mask)
            values = candidate_feature_rows(flat, take, ids, spatial_values=spatial_values)
            e, weight = raw_evidence(values, p, mask)
            pieces.append(e)
            probabilities.append(weight)
        residual = torch.zeros_like(evidence)
        if len(selected):
            chosen = torch.cat(pieces)
            evidence = evidence.index_copy(0, selected, chosen)
            weights = weights.index_copy(0, selected, torch.cat(probabilities))
            with self.retargeter.native_source_context(flat.reshape(-1, 1024)[selected]):
                retargeted = self.retargeter(torch.cat((chosen, target_query[selected]), -1), self.g_max)
            residual = residual.index_copy(0, selected, retargeted.float())
            self.last_R_input_appearance = chosen.detach()
            self.last_R_only_output = retargeted.detach().float()
        else:
            evidence = evidence+sum(p.reshape(-1)[0]*0 for p in self.parameters() if p.requires_grad)
            self.last_R_input_appearance = flat.new_empty((0, 1024))
            self.last_R_only_output = flat.new_empty((0, 1024))
        evidence = evidence.reshape(batch, rays, 1024)
        residual = residual.reshape(batch, rays, 1024)
        weights = weights.reshape(batch, rays, count)
        query_valid = valid.any(-1)
        view_weights = candidate_view_weights(indices, weights, spatial=spatial)
        scale = torch.tanh(self.fusion_gate)
        attention = DeformationCondition(evidence+residual, indices, prior.reshape(batch, rays, count),
            weights, view_weights, residual, query_valid, scale, base_evidence=evidence)
        use = self.evidence_use(target_query.reshape(batch, rays, 1024), evidence, query_valid)
        mass = use[..., None]*scale
        numerator = mass*evidence+use[..., None]*residual
        self.last_source_reference_rows = selected.detach()
        self.last_selected_evidence = evidence.detach()
        self.last_direct_feature_correction = torch.zeros_like(residual.detach())
        self._last_member_retained_mass = posterior.retained_mass.detach()
        self._last_distinct_member_count = posterior.distinct_member_count.detach()
        return TargetFirstRayEvidence(numerator.reshape(batch, 3, 1024, 1024),
            mass.reshape(batch, 3, 1024, 1), attention, use)
