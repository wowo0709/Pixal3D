"""Keep the matching posterior and learn a bounded, function-anchored residual.

The complete initial Q/K/utility function is a frozen buffer reference. Its
subtraction preserves the matching posterior exactly at initialization, without
zeroing Q/K or discarding the pretrained utility. Learned relative log-odds can
then change by at most two declared bound units between any two candidates.
"""
import math

import torch
from torch.nn import functional as F

from .ray_feature_attention import RayFeatureReadout
from .member_feature_source_ray import MemberFeatureSourceRayConditioner


def masked_probabilities(logits, valid):
    logits = logits.float().masked_fill(~valid[:, None], -torch.inf)
    maximum = logits.amax(-1, keepdim=True)
    maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
    mass = torch.exp(logits-maximum).masked_fill(~valid[:, None], 0.)
    return mass/mass.sum(-1, keepdim=True).clamp_min(1e-8)


def bounded_probabilities(current_raw, initial_raw, prior, valid, bound):
    if (current_raw.shape != initial_raw.shape or current_raw.ndim != 3
            or current_raw.shape[1] != 4 or current_raw.shape[::2] != prior.shape
            or prior.shape != valid.shape or valid.dtype != torch.bool
            or not math.isfinite(bound) or bound <= 0):
        raise ValueError('aligned four-head raw scores, original prior and positive residual bound required')
    with torch.autocast(device_type=prior.device.type, enabled=False):
        base = masked_probabilities(prior[:, None].expand(-1, 4, -1), valid)
        # Valid image candidates can still have zero posterior mass after
        # truncation/underflow. Keep those exact zero masses without inf-inf.
        support = valid & torch.isfinite(prior)
        difference = torch.where(support[:, None], current_raw.float()-initial_raw.float(), 0.)
        # A per-query constant score shift must not consume the residual bound.
        centered = difference-(base*difference).sum(-1, keepdim=True)
        residual = bound*torch.tanh(centered/bound)
        result = masked_probabilities(prior[:, None].float()+residual, valid)
    return result


class BoundedPriorRayFeatureReadout(RayFeatureReadout):
    ANCHORS = dict(
        query='query.weight', key='key.weight',
        utility_query='score_selector.query.weight', utility_context='score_selector.context.weight',
        utility_key='score_selector.key.weight', utility_output='score_selector.content_output.weight',
        utility_log_scale='score_selector.log_content_scale')

    def __init__(self, value_width=1024, *, residual_logit_bound=1.):
        if not isinstance(residual_logit_bound, (int, float)) or isinstance(residual_logit_bound, bool) or not math.isfinite(residual_logit_bound) or residual_logit_bound <= 0:
            raise ValueError('explicit finite positive residual logit bound required')
        super().__init__(value_width)
        self.bound_value = float(residual_logit_bound)
        self.register_buffer('residual_logit_bound', torch.tensor(self.bound_value))
        parameters = dict(self.named_parameters())
        for name, key in self.ANCHORS.items():
            self.register_buffer('initial_'+name, parameters[key].detach().clone())
        self.last_reference_raw = None
        self.executed_forwards = 0
        self.first_forward_prior_exact = None

    @torch.no_grad()
    def capture_initial_function(self):
        if torch.count_nonzero(self.output.weight):
            raise ValueError('capture only at untouched zero-output initialization')
        parameters = dict(self.named_parameters())
        for name, key in self.ANCHORS.items():
            getattr(self, 'initial_'+name).copy_(parameters[key])
        self.first_forward_prior_exact = None

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        field = prefix+'residual_logit_bound'
        if field in state_dict and float(state_dict[field]) != self.bound_value:
            error_msgs.append('checkpoint and declared residual logit bounds differ: '+field)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)

    def reference_raw(self, native, context, keys, values, valid, prior):
        nq = F.layer_norm(native, (1024,)); nc = F.layer_norm(context, (1024,))
        q = F.linear(nq, self.initial_utility_query)+F.linear(nc, self.initial_utility_context)
        k = F.linear(F.layer_norm(values[..., :1024], (1024,)), self.initial_utility_key)
        with torch.autocast(device_type=native.device.type, enabled=False):
            qn = F.layer_norm(q.float(), q.shape[-1:])[:, None]
            kn = F.layer_norm(k.float(), k.shape[-1:])
            pair = torch.cat((qn*kn, (qn-kn).abs()), -1)
            content = torch.tanh(F.linear(pair, self.initial_utility_output).squeeze(-1))
            score = self.initial_utility_log_scale.exp().clamp(max=20.)*content+prior.float()
        query = F.linear(torch.cat((nq, nc), -1), self.initial_query).reshape(len(native), 4, 64)
        key = F.linear(F.layer_norm(keys, (1024,)), self.initial_key).reshape(len(native), -1, 4, 64).transpose(1, 2)
        with torch.autocast(device_type=native.device.type, enabled=False):
            return (query.float()[:, :, None]*key.float()).sum(-1)/math.sqrt(64)+score[:, None]

    def forward(self, native, context, keys, values, valid, prior):
        rows, candidates = valid.shape
        if (native.shape != (rows, 1024) or context.shape != native.shape
                or keys.shape != (rows, candidates, 1024)
                or values.shape != (rows, candidates, self.value_width)
                or prior.shape != valid.shape or valid.dtype != torch.bool):
            raise ValueError('original RGB/Omega candidates and matching prior required')
        keys = torch.where(valid[..., None], keys, 0.)
        values = torch.where(valid[..., None], values, 0.)
        _, _, score = self.score_selector(native, context, values[..., :1024], valid, prior)
        query = self.query(torch.cat((F.layer_norm(native, (1024,)), F.layer_norm(context, (1024,))), -1)).reshape(rows, 4, 64)
        key = self.key(F.layer_norm(keys, (1024,))).reshape(rows, candidates, 4, 64).transpose(1, 2)
        normalized = torch.cat([F.layer_norm(chunk, (1024,)) for chunk in values.split(1024, dim=-1)], -1)
        value = self.value(normalized).reshape(rows, candidates, 4, 64).transpose(1, 2)
        initial = self.reference_raw(native, context, keys, values, valid, prior)
        with torch.autocast(device_type=native.device.type, enabled=False):
            current = (query.float()[:, :, None]*key.float()).sum(-1)/math.sqrt(64)+score[:, None].float()
            probability = bounded_probabilities(current, initial, prior, valid, self.bound_value)
            combined = (probability[..., None]*value.float()).sum(-2).reshape(rows, -1)
        if self.first_forward_prior_exact is None:
            self.first_forward_prior_exact = bool(torch.equal(probability, masked_probabilities(prior[:, None].expand(-1, 4, -1), valid)))
        self.executed_forwards += 1
        self.last_reference_raw = initial.detach()
        return self.output(combined).float(), probability


class BoundedPriorMemberFeatureSourceRayConditioner(MemberFeatureSourceRayConditioner):
    def __init__(self, *, residual_logit_bound, **kwargs):
        super().__init__(**kwargs)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(20261021)
            self.ray_readout = BoundedPriorRayFeatureReadout(1024, residual_logit_bound=residual_logit_bound)

    def load_utility(self, state):
        super().load_utility(state)
        self.ray_readout.capture_initial_function()
