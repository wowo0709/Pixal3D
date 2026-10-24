"""Matched weighting and physical retargeter controls for shared raw evidence.

The running Fixed/Learned implementations remain unchanged. These controls
retain their posterior, raw values, source-use and native projection contract.
Uniform-K is a Fixed control; a learned head without its additive logit prior
is a separate control and is not called uniform attention.
"""
import math

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .consistent_ray_evidence import (
    AnchoredRawRayReadout, ConsistentRayEvidenceConditioner, _check_schema,
    _chunk, anchored_probabilities, masked_probabilities, raw_evidence,
)
from .deform_conditioner import DeformationCondition
from .member_ray_posterior import member_ray_posteriors
from .ray_group_query import pool_ray_queries
from .target_first_ray_v2 import TargetFirstRayEvidence


CONTROLS = ('reference', 'uniform-k', 'without-retargeting', 'without-logit-prior')


def uniform_probabilities(valid):
    """Equal weights over the same valid candidate addresses, including ties."""
    if valid.dtype != torch.bool or valid.ndim != 2:
        raise ValueError('explicit boolean [ray,candidate] validity required')
    return masked_probabilities(torch.zeros_like(valid, dtype=torch.float32)[:, None].expand(-1, 4, -1), valid)


def unprioritized_probabilities(current, initial, prior, valid):
    """Remove the additive matching logit while retaining its eligible support.

    The matcher still constructs candidates/context and supplies utility input.
    This isolates the additive logit term, not all uses of matching scores.
    """
    if torch.isnan(prior).any() or torch.isposinf(prior).any():
        raise ValueError('finite or negative-infinite original matching logits required')
    neutral = torch.zeros_like(prior).masked_fill(~torch.isfinite(prior), -torch.inf)
    return anchored_probabilities(current, initial, neutral, valid)


class UnprioritizedRawRayReadout(AnchoredRawRayReadout):
    def __init__(self):
        super().__init__()
        self.register_buffer('without_matching_logit_prior_schema', torch.tensor([1], dtype=torch.uint8))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        _check_schema(state_dict, prefix, 'without_matching_logit_prior_schema',
                      self.without_matching_logit_prior_schema, error_msgs)
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
            probability = unprioritized_probabilities(current, initial, prior, valid)
        if self.first_forward_prior_exact is None:
            self.first_forward_prior_exact = bool(torch.equal(probability,
                masked_probabilities(prior[:, None].expand(-1, 4, -1), valid)))
        self.last_reference_raw = initial.detach()
        self.last_probability = probability.detach()
        self.last_prior = masked_probabilities(prior[:, None].expand(-1, 4, -1), valid).detach()
        evidence, weights = raw_evidence(values, probability, valid)
        return evidence, weights


class SharedEvidenceAblationConditioner(ConsistentRayEvidenceConditioner):
    def __init__(self, *, ablation, evidence_mode, **kwargs):
        if (ablation not in CONTROLS
                or ablation == 'uniform-k' and evidence_mode != 'fixed-prior'
                or ablation == 'without-logit-prior' and evidence_mode != 'learned-anchored'):
            raise ValueError('Uniform-K needs Fixed; additive-prior control needs the same learned head')
        super().__init__(evidence_mode=evidence_mode, **kwargs)
        self.ablation = ablation
        if ablation == 'without-retargeting':
            self.retargeter = None
        elif ablation == 'without-logit-prior':
            original = self.ray_readout
            with torch.random.fork_rng(devices=[]):
                replacement = UnprioritizedRawRayReadout()
            state = dict(original.state_dict())
            state['without_matching_logit_prior_schema'] = replacement.without_matching_logit_prior_schema
            replacement.load_state_dict(state, strict=True)
            self.ray_readout = replacement
        self.register_buffer('shared_evidence_ablation_schema',
                             torch.tensor([CONTROLS.index(ablation)+1], dtype=torch.uint8))
        expected = (10 if evidence_mode == 'fixed-prior' else 17)-(5 if self.retargeter is None else 0)
        if sum(p.requires_grad for p in self.parameters()) != expected:
            raise ValueError('only the declared source parameters may remain')

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        _check_schema(state_dict, prefix, 'shared_evidence_ablation_schema',
                      self.shared_evidence_ablation_schema, error_msgs)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)

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
                p = (uniform_probabilities(mask) if self.ablation == 'uniform-k' else
                     masked_probabilities(prior[take, None].expand(-1, 4, -1), mask))
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
            if self.retargeter is not None:
                with self.retargeter.native_source_context(flat.reshape(-1, 1024)[selected]):
                    retargeted = self.retargeter(torch.cat((chosen, target_query[selected]), -1), self.g_max)
                residual = residual.index_copy(0, selected, retargeted.float())
            else:
                retargeted = torch.zeros_like(chosen)
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
