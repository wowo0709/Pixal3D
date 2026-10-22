"""R-only semantic source-ray transport for sparse PBR DINO+NAF values.

Matching/query/context remain1024-D. Values and residuals are2048-D; R has
3072 input channels and no deformation-query projection or placeholder input.
Physical Top1 has ten trainable tensors and no feature/utility attention head.
"""
import math
from pathlib import Path

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .bounded_member_feature_source_ray import BoundedPriorRayFeatureReadout
from .deform_conditioner import DeformationCondition, _Retargeter
from .member_ray_posterior import member_ray_posteriors
from .pbr_source_ray_geometry import pool_sparse_source_ray_queries
from .r9_matching import PromotedR9Matcher
from .r_mediated_source_contrast import _input_chunk
from .source_contrast_member_feature import SourceContrastRetargeter
from .target_first_ray import SourceEvidenceUse
from .target_first_ray_v2 import TargetFirstRayEvidence


MODES = {'top1-only': 'pbr_R_top1_schema',
    'top1-plus-feature': 'pbr_R_feature_schema',
    'raw-topk-plus-feature': 'pbr_R_raw_topk_schema'}


def packed_member_posteriors(matcher, *, target_keys, source_keys, source_valid,
        ray_ids, patch_ids, observed_query, batch_ids, temperature, chunk_size):
    """Pad only integer addresses/validity, never neural features or votes."""
    batch = len(target_keys)
    if (ray_ids.ndim != 2 or ray_ids.shape != (len(batch_ids), 3)
            or ray_ids.dtype != torch.long or patch_ids.shape != batch_ids.shape
            or patch_ids.dtype != torch.long or batch_ids.ndim != 1
            or batch_ids.dtype not in (torch.int32, torch.int64)
            or observed_query.shape != batch_ids.shape or observed_query.dtype != torch.bool
            or any(x.device != target_keys.device for x in (ray_ids, patch_ids, observed_query, batch_ids))
            or ((batch_ids < 0) | (batch_ids >= batch)).any()):
        raise ValueError('aligned packed native sparse ray/patch/observation/batch addresses required')
    count = torch.bincount(batch_ids.long(), minlength=batch)
    maximum = int(count.max())
    rays = ray_ids.new_full((batch, 3, maximum), -1)
    patches = patch_ids.new_zeros(batch, maximum)
    observed = observed_query.new_zeros(batch, maximum)
    for b in range(batch):
        take = (batch_ids == b).nonzero().flatten()
        rays[b, :, :len(take)] = ray_ids[take].transpose(0, 1)
        patches[b, :len(take)] = patch_ids[take]
        observed[b, :len(take)] = observed_query[take]
    return member_ray_posteriors(matcher, target_keys=target_keys, source_keys=source_keys,
        source_valid=source_valid, ray_ids=rays, patch_ids=patches, observed_query=observed,
        temperature=temperature, chunk_size=chunk_size)


class RMediatedPBRSourceConditioner(nn.Module):
    def __init__(self, *, appearance_mode, q2c_checkpoint=None, hidden_dim=256,
                 score_temperature=.05, ray_chunk_size=64, g_max=.25,
                 residual_logit_bound=None):
        super().__init__()
        if (appearance_mode not in MODES or hidden_dim != 256
                or isinstance(ray_chunk_size, bool) or not isinstance(ray_chunk_size, int) or ray_chunk_size < 1
                or not math.isfinite(score_temperature) or score_temperature <= 0
                or not math.isfinite(g_max) or g_max <= 0
                or (appearance_mode == 'top1-only' and residual_logit_bound is not None)
                or (appearance_mode != 'top1-only' and residual_logit_bound != 1.)):
            raise ValueError('explicit PBR R-only mode, native widths, NoDQ and declared bound1 required')
        self.appearance_mode = appearance_mode
        self.score_temperature, self.ray_chunk_size, self.g_max = score_temperature, ray_chunk_size, g_max
        self.value_dim, self.query_dim, self.key_dim = 2048, 1024, 1024
        self.implementation, self.projection_mode = 'r9-native-v5', 'native-bilinear'
        self.task_attention = None
        self.fusion_gate = nn.Parameter(torch.zeros(()))
        self.q2c = PromotedR9Matcher()
        if q2c_checkpoint is not None:
            self.q2c.load_checkpoint(torch.load(Path(q2c_checkpoint), map_location='cpu', weights_only=True))
        self.q2c.requires_grad_(False).eval()
        self.retargeter = SourceContrastRetargeter(_Retargeter(3072, 2048, hidden_dim))
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(20261020)
            self.evidence_use = SourceEvidenceUse()
        if appearance_mode != 'top1-only':
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(20261021)
                self.ray_readout = BoundedPriorRayFeatureReadout(2048, residual_logit_bound=1.)
        self.register_buffer(MODES[appearance_mode], torch.tensor([1], dtype=torch.uint8))
        expected = 10 if appearance_mode == 'top1-only' else 19
        if sum(p.requires_grad for p in self.parameters()) != expected:
            raise ValueError('physical PBR source10 or source19 required')

    def train(self, mode=True):
        super().train(mode)
        self.q2c.eval()
        return self

    def load_utility(self, state):
        if self.appearance_mode == 'top1-only':
            raise ValueError('physical Top1 has no score utility or attention head')
        expected = self.ray_readout.score_selector.state_dict()
        if (torch.count_nonzero(self.ray_readout.output.weight)
                or set(state) != set(expected)
                or any(state[key].shape != value.shape or not torch.isfinite(state[key]).all()
                    for key, value in expected.items())):
            raise ValueError('complete finite utility at cold zero-output initialization required')
        self.ray_readout.score_selector.load_state_dict(state, strict=True)
        self.ray_readout.capture_initial_function()

    def forward(self, *, native_query_rows, target_query, frame_target_keys,
                target_patch_ids, target_valid, ray_ids, batch_ids,
                source_keys, source_values, source_valid):
        batch, rays, count = len(source_values), 3072, 64
        rows = len(batch_ids)
        feature_head = self.appearance_mode != 'top1-only'
        if ((feature_head and (native_query_rows is None or native_query_rows.shape != (rows, 1024)))
                or (not feature_head and native_query_rows is not None)
                or target_query.shape != (batch, 1024, 1024)
                or frame_target_keys.shape != target_query.shape
                or target_valid.shape != (batch, 1024) or target_valid.dtype != torch.bool
                or source_values.shape != (batch, 3, 1024, 2048)
                or source_keys.shape != (batch, 3, 1024, 1024)
                or source_valid.shape != (batch, 3, 1024) or source_valid.dtype != torch.bool
                or batch_ids.shape != (rows,) or batch_ids.dtype not in (torch.int32, torch.int64)
                or target_patch_ids.shape != (rows,) or target_patch_ids.dtype != torch.long
                or ((batch_ids < 0) | (batch_ids >= batch)).any()
                or ((target_patch_ids < 0) | (target_patch_ids >= 1024)).any()
                or any(x.device != source_values.device for x in (target_query,
                    frame_target_keys, target_valid, source_keys, source_valid, batch_ids, target_patch_ids, ray_ids))):
            raise ValueError('real packed PBR rows, DINO query, Omega keys and DINO+NAF values required')
        if (feature_head and native_query_rows.device != source_values.device
                or any(not torch.isfinite(x).all() for x in (source_values, target_query, source_keys, frame_target_keys))):
            raise ValueError('finite real PBR RGB/Omega/value inputs on one device required')
        batch_ids = batch_ids.long()
        utility = (pool_sparse_source_ray_queries(native_query_rows, ray_ids, batch_ids, batch_size=batch)
            if feature_head else None)
        target_rows = target_query[batch_ids, target_patch_ids]
        target = pool_sparse_source_ray_queries(target_rows, ray_ids, batch_ids, batch_size=batch)
        posterior = packed_member_posteriors(self.q2c, target_keys=frame_target_keys,
            source_keys=source_keys, source_valid=source_valid, ray_ids=ray_ids,
            patch_ids=target_patch_ids, observed_query=target_valid[batch_ids, target_patch_ids],
            batch_ids=batch_ids, temperature=self.score_temperature, chunk_size=self.ray_chunk_size)
        indices, valid = posterior.candidate_indices, posterior.candidate_valid
        selected = valid.any(-1).flatten().nonzero().flatten()
        flat_values, flat_keys = source_values.flatten(1, 2).float(), source_keys.flatten(1, 2)
        prior = posterior.probabilities.log().reshape(-1, count)
        evidence = flat_values.new_zeros(batch*rays, 2048)
        hard = flat_values.new_zeros(batch*rays, count)
        residual = torch.zeros_like(evidence)
        target_context = target.values.reshape(-1, 1024)
        probabilities, corrections, aggregates = [], [], []
        if len(selected):
            slots = prior[selected].argmax(-1)
            winner = indices.reshape(-1, count)[selected].gather(1, slots[:, None])[:, 0]
            appearance = flat_values[selected // rays, winner]
            evidence = evidence.index_copy(0, selected, appearance)
            hard = hard.index_copy(0, selected, hard.new_zeros(len(selected), count).scatter_(1, slots[:, None], 1.))
            R_appearance = appearance
            if self.appearance_mode != 'top1-only':
                query = utility.values.reshape(-1, 1024)
                context = posterior.contextual_query.reshape(-1, 1024)
                raw = self.appearance_mode == 'raw-topk-plus-feature'
                for start in range(0, len(selected), self.ray_chunk_size):
                    take = selected[start:start+self.ray_chunk_size]
                    ids = indices.reshape(-1, count)[take]
                    mask = valid.reshape(-1, count)[take]
                    args = (self.ray_readout, query[take], context[take], flat_values,
                        flat_keys, take // rays, ids, mask, prior[take], 'normal', raw)
                    change, aggregate, probability = (checkpoint(_input_chunk, *args, use_reentrant=False)
                        if torch.is_grad_enabled() else _input_chunk(*args))
                    corrections.append(change); aggregates.append(aggregate)
                    probabilities.append(probability.detach())
                R_appearance = (torch.cat(aggregates) if raw else appearance) + torch.cat(corrections)
            own_source = flat_values.reshape(-1, 2048)[selected]
            with self.retargeter.native_source_context(own_source):
                value = self.retargeter(torch.cat((R_appearance, target_context[selected]), -1), self.g_max)
            residual = residual.index_copy(0, selected, value.float())
            self.last_R_input_appearance = R_appearance.detach()
            self.last_R_only_output = value.detach().float()
        else:
            evidence = evidence + sum(p.reshape(-1)[0]*0 for p in self.parameters() if p.requires_grad)
            self.last_R_input_appearance = flat_values.new_empty((0, 2048))
            self.last_R_only_output = flat_values.new_empty((0, 2048))
        evidence = evidence.reshape(batch, rays, 2048)
        residual = residual.reshape(batch, rays, 2048)
        hard = hard.reshape(batch, rays, count)
        query_valid = valid.any(-1)
        scale = torch.tanh(self.fusion_gate)
        view_weights = evidence.new_zeros(batch, rays, 3).scatter_add(2, indices // 1024, hard)
        attention = DeformationCondition(evidence+residual, indices,
            prior.reshape(batch, rays, count), hard, view_weights, residual,
            query_valid, scale, base_evidence=evidence)
        # The independent use decision retains the same DINO-only query role.
        # NAF remains present in both the learned value head and R residual.
        use = self.evidence_use(target_context.reshape(batch, rays, 1024), evidence[..., :1024], query_valid)
        mass = use[..., None] * scale
        numerator = mass * evidence + use[..., None] * residual
        self.last_source_reference_rows = selected.detach()
        self.last_feature_probabilities = (selected.detach(), torch.cat(probabilities)
            if probabilities else flat_values.new_empty((0, 4, count)))
        self._last_member_retained_mass = posterior.retained_mass.detach()
        self._last_distinct_member_count = posterior.distinct_member_count.detach()
        if not torch.equal(residual.reshape(-1, 2048)[selected], self.last_R_only_output):
            raise ValueError('PBR readout must have no direct-to-consumer bypass outside R')
        return TargetFirstRayEvidence(numerator.reshape(batch, 3, 1024, 2048),
            mass.reshape(batch, 3, 1024, 1), attention, use)
