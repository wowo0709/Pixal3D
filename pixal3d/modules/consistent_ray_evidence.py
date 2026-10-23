"""One raw ray evidence for the carrier, source-use decision and retargeter.

Keep native MV4 source subtraction, matching candidates and target conditioning.
The learned variant changes probabilities, with identity V and no extra feature
correction branch. Its initial function is exactly the frozen matching prior.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .bounded_member_feature_source_ray import BoundedPriorRayFeatureReadout, masked_probabilities
from .deform_conditioner import DeformationCondition
from .member_posterior_source_ray import MemberPosteriorSourceRayConditioner
from .member_ray_posterior import member_ray_posteriors
from .ray_group_query import pool_ray_queries
from .source_contrast_member_feature import SourceContrastRetargeter
from .target_first_ray_v2 import TargetFirstRayEvidence


MODES = ('fixed-prior', 'learned-anchored')


def _check_schema(state, prefix, name, expected, errors):
    value = state.get(prefix+name)
    if value is not None and (not isinstance(value, torch.Tensor)
            or value.dtype != expected.dtype or value.shape != expected.shape
            or not torch.equal(value.cpu(), expected.cpu())):
        errors.append('checkpoint and declared consistent-evidence schema differ: '+prefix+name)


def anchored_probabilities(current, initial, prior, valid):
    """Unclipped learned score difference; retain exact zero prior support."""
    if (current.shape != initial.shape or current.ndim != 3 or current.shape[1] != 4
            or current.shape[::2] != prior.shape or prior.shape != valid.shape
            or valid.dtype != torch.bool):
        raise ValueError('aligned four-head scores, original prior and validity required')
    with torch.autocast(device_type=prior.device.type, enabled=False):
        base = masked_probabilities(prior[:, None].expand(-1, 4, -1), valid)
        support = valid & torch.isfinite(prior)
        difference = torch.where(support[:, None], current.float()-initial.float(), 0.)
        if not torch.isfinite(difference).all():
            raise ValueError('finite learned relative matching scores required')
        centered = difference-(base*difference).sum(-1, keepdim=True)
        return masked_probabilities(prior[:, None].float()+centered, valid)


def raw_evidence(values, probability, valid):
    if (values.ndim != 3 or probability.shape != (len(values), 4, values.shape[1])
            or valid.shape != values.shape[:2] or valid.dtype != torch.bool):
        raise ValueError('raw candidate features and four-head posterior required')
    clean = torch.where(valid[..., None], values.float(), 0.)
    weights = probability.float().mean(1)
    return (weights[..., None]*clean).sum(1), weights


class AnchoredRawRayReadout(nn.Module):
    """Seven Q/K/utility tensors, with no trainable value/output projection."""
    ANCHORS = BoundedPriorRayFeatureReadout.ANCHORS
    reference_raw = BoundedPriorRayFeatureReadout.reference_raw

    def __init__(self):
        super().__init__()
        reference = BoundedPriorRayFeatureReadout(1024, residual_logit_bound=1.)
        self.query, self.key, self.score_selector = reference.query, reference.key, reference.score_selector
        for name in self.ANCHORS:
            self.register_buffer('initial_'+name, getattr(reference, 'initial_'+name).detach().clone())
        self.register_buffer('anchored_raw_readout_schema', torch.tensor([1], dtype=torch.uint8))
        self.first_forward_prior_exact = None
        self.last_reference_raw = None
        self.last_probability = self.last_prior = None
        if len(list(self.parameters())) != 7 or hasattr(self, 'value') or hasattr(self, 'output'):
            raise ValueError('physical Q/K/utility7 and identity values required')

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        _check_schema(state_dict, prefix, 'anchored_raw_readout_schema',
                      self.anchored_raw_readout_schema, error_msgs)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)

    def forward(self, native, context, keys, values, valid, prior):
        rows, count = valid.shape
        if (native.shape != (rows, 1024) or context.shape != native.shape
                or keys.shape != (rows, count, 1024) or values.shape != keys.shape
                or prior.shape != valid.shape):
            raise ValueError('native ray query, member context, Omega keys and raw DINO values required')
        keys = torch.where(valid[..., None], keys, 0.)
        values = torch.where(valid[..., None], values, 0.)
        _, _, score = self.score_selector(native, context, values, valid, prior)
        query = self.query(torch.cat((F.layer_norm(native, (1024,)), F.layer_norm(context, (1024,))), -1)).reshape(rows, 4, 64)
        key = self.key(F.layer_norm(keys, (1024,))).reshape(rows, count, 4, 64).transpose(1, 2)
        initial = self.reference_raw(native, context, keys, values, valid, prior)
        with torch.autocast(device_type=native.device.type, enabled=False):
            current = (query.float()[:, :, None]*key.float()).sum(-1)/math.sqrt(64)+score[:, None].float()
            probability = anchored_probabilities(current, initial, prior, valid)
        if self.first_forward_prior_exact is None:
            self.first_forward_prior_exact = bool(torch.equal(probability,
                masked_probabilities(prior[:, None].expand(-1, 4, -1), valid)))
        self.last_reference_raw = initial.detach()
        self.last_probability = probability.detach()
        self.last_prior = masked_probabilities(prior[:, None].expand(-1, 4, -1), valid).detach()
        evidence, weights = raw_evidence(values, probability, valid)
        return evidence, weights


def _chunk(head, native, context, keys, values, batch, indices, valid, prior):
    return head(native, context, keys[batch[:, None], indices].float(),
                values[batch[:, None], indices].float(), valid, prior)


class ConsistentRayEvidenceConditioner(MemberPosteriorSourceRayConditioner):
    def __init__(self, *, evidence_mode, **kwargs):
        if (evidence_mode not in MODES or kwargs.get('ray_mode') != 'semantic-weighted'
                or kwargs.get('candidate_aggregation') != 'weighted' or kwargs.get('task_attention') is not None):
            raise ValueError('explicit consistent raw TopK evidence mode without an extra attention branch required')
        super().__init__(**kwargs)
        self.retargeter = SourceContrastRetargeter(self.retargeter)
        self.evidence_mode = evidence_mode
        self.ray_readout = None
        if evidence_mode == 'learned-anchored':
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(20261021)
                self.ray_readout = AnchoredRawRayReadout()
        self.register_buffer('consistent_ray_evidence_schema', torch.tensor([MODES.index(evidence_mode)+1], dtype=torch.uint8))
        count = sum(p.requires_grad for p in self.parameters())
        if count != (10 if self.ray_readout is None else 17) or self.retargeter.input_dim != 2048:
            raise ValueError('own core10 or core10+QK/utility7 with physical NoDQ required')
        self.q2c.eval()

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        _check_schema(state_dict, prefix, 'consistent_ray_evidence_schema',
                      self.consistent_ray_evidence_schema, error_msgs)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)

    def train(self, mode=True):
        super().train(mode)
        self.q2c.eval()
        return self

    def forward(self, *, native_voxels, target_omega_voxels, target_native_voxels,
                frame_target_keys, target_patch_ids, ray_ids, observed_query,
                source_keys, source_values, source_valid, control='normal'):
        if (control != 'normal' or native_voxels.shape != target_native_voxels.shape
                or native_voxels.shape != target_omega_voxels.shape or native_voxels.shape[1:] != (4096, 1024)
                or source_values.shape != (len(native_voxels), 3, 1024, 1024)):
            raise ValueError('native SS16 grid and original three source-camera feature maps required')
        batch, rays, count = len(source_values), 3072, 64
        target = pool_ray_queries(target_native_voxels, ray_ids, num_rays=1024)
        utility = pool_ray_queries(native_voxels, ray_ids, num_rays=1024) if self.ray_readout is not None else None
        posterior = member_ray_posteriors(self.q2c, target_keys=frame_target_keys,
            source_keys=source_keys, source_valid=source_valid, ray_ids=ray_ids,
            patch_ids=target_patch_ids, observed_query=observed_query,
            temperature=self.score_temperature, chunk_size=self.ray_chunk_size)
        indices, valid = posterior.candidate_indices, posterior.candidate_valid
        selected = valid.any(-1).flatten().nonzero().flatten()
        flat = source_values.flatten(1, 2).float()
        keys = source_keys.flatten(1, 2)
        prior = posterior.probabilities.log().reshape(-1, count)
        target_query = target.values.reshape(batch*rays, 1024)
        evidence = flat.new_zeros(batch*rays, 1024)
        weights = flat.new_zeros(batch*rays, count)
        pieces, probabilities = [], []
        for start in range(0, len(selected), self.ray_chunk_size):
            take = selected[start:start+self.ray_chunk_size]
            b = take//rays
            ids, mask = indices.reshape(-1, count)[take], valid.reshape(-1, count)[take]
            if self.ray_readout is None:
                p = masked_probabilities(prior[take, None].expand(-1, 4, -1), mask)
                e, weight = raw_evidence(flat[b[:, None], ids], p, mask)
            else:
                args = (self.ray_readout, utility.values.reshape(-1, 1024)[take],
                        posterior.contextual_query.reshape(-1, 1024)[take], keys, flat, b, ids, mask, prior[take])
                e, weight = checkpoint(_chunk, *args, use_reentrant=False) if torch.is_grad_enabled() else _chunk(*args)
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
        view_weights = evidence.new_zeros(batch, rays, 3).scatter_add(2, indices//1024, weights)
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
