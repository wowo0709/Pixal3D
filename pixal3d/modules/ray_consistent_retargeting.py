"""Keep candidate appearance/deformation paired before score-weighted retargeting.

The native Top1/R path remains the reference. A zero-initialized geometry
projection adds a correction with a frozen matching distribution and an
optional frozen, predicted availability confidence. No appearance V/O or
GT correspondence enters the model. Each query remains a target patch ray.
"""
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .deform_conditioner import _Retargeter, DeformationCondition
from .ray_deformation_query import DeformationQueryRetargeter
from .ray_geometry_retargeting import RayGeometryRetargetingConditioner
from .ray_correspondence_reliability import RayCorrespondenceReliability
from .ray_feature_attention import _readout
from .task_matching_attention import contextual_query


class CandidateGeometryRetargeter(DeformationQueryRetargeter):
    def appearance_forward(self, appearance, g_max):
        return _Retargeter.forward(self, appearance, g_max)

    def correction(self, appearance, deformation, g_max):
        if appearance.shape[-1] != self.appearance_dim or deformation.shape != (*appearance.shape[:-1], 2048):
            raise ValueError('paired candidate appearance and actual RAW2048 difference required')
        # Extra geometry cannot perturb the common R gradient at initialization.
        # Its Jacobian uses the current R weights as a fixed feature transform.
        hidden = F.linear(appearance, self.net[0].weight.detach(), self.net[0].bias.detach())
        geometry = self.deformation_projection(deformation)
        base = F.linear(F.silu(hidden), self.net[2].weight.detach(), self.net[2].bias.detach())
        changed = F.linear(F.silu(hidden+geometry), self.net[2].weight.detach(), self.net[2].bias.detach())
        return g_max*torch.tanh(self.gate_logit.detach())*(torch.tanh(changed)-torch.tanh(base))


def paired_correction(retargeter, target, raw_target, values, raw_source,
                      batch, indices, valid, probability, g_max, candidate_chunk):
    result = target.new_zeros(len(target), retargeter.net[2].out_features)
    for begin in range(0, 64, candidate_chunk):
        take = slice(begin, min(64, begin+candidate_chunk))
        ids, mask = indices[:, take], valid[:, take]
        source = values[batch[:, None], ids].float().masked_fill(~mask[..., None], 0.)
        difference = (raw_target[:, None].float()-raw_source[batch[:, None], ids].float())
        difference = difference.masked_fill(~mask[..., None], 0.)
        appearance = torch.cat([source, target[:, None].expand(-1, ids.shape[1], -1)], -1)
        correction = retargeter.correction(appearance, difference, g_max)
        result = result+(probability[:, take, None]*correction.float()).sum(1)
    return result


class RayConsistentRetargetingConditioner(RayGeometryRetargetingConditioner):
    def __init__(self, *, use_reliability=True, composition='paired', candidate_chunk_size=8, **kwargs):
        if (not isinstance(use_reliability, bool) or composition not in {'paired', 'mean'}
                or candidate_chunk_size not in {4, 8, 16, 32, 64}):
            raise ValueError('explicit reliability and bounded complete Top64 composition required')
        super().__init__(**kwargs)
        self.retargeter = CandidateGeometryRetargeter(self.value_dim+1024,
            self.value_dim, kwargs.get('hidden_dim', 256), 2048)
        self.ray_readout.requires_grad_(False)
        self.reliability = RayCorrespondenceReliability() if use_reliability else None
        self.composition = composition
        self.candidate_chunk_size = candidate_chunk_size

    def reference_state(self):
        return {name: value for name, value in self.state_dict().items()
            if not name.startswith(('ray_readout.', 'reliability.'))
            and name != 'retargeter.deformation_projection.weight'}

    def load_geometry_state(self, reference):
        current = self.state_dict()
        if set(reference) != {k for k in current if not k.startswith('reliability.')}:
            raise ValueError('complete, exact-shape native geometry state required')
        current.update(reference)
        self.load_state_dict(current, strict=True)
        self.ray_readout.requires_grad_(False)

    def forward(self, *, source_values, source_valid, target_query, semantic_indices,
                matching_logits, candidate_valid, target_valid, target_keys, source_keys,
                target_context, source_context, control='normal', **unsupported):
        if unsupported or control != 'normal':
            raise ValueError('inference RGB/features only; controlled composition is constructor-defined')
        if any(p.requires_grad for p in self.ray_readout.parameters()):
            raise ValueError('matching must remain frozen during reconstruction training')
        source_values = source_values.float()
        b, views, sites, width = source_values.shape
        q = target_query.shape[1]
        if (views, sites, width) != (3, 1024, self.value_dim) or target_query.shape != (b, q, 1024):
            raise ValueError('native three-source SS/PBR ray values required')
        if (target_context.shape != (b, q, 2048) or source_context.shape != (b, 3, 1024, 2048)
                or target_keys.shape != (b, q, 1024) or source_keys.shape != (b, 3, 1024, 1024)
                or not torch.equal(target_context[..., :1024].float(), target_keys.float())
                or not torch.equal(source_context[..., :1024].float(), source_keys.float())):
            raise ValueError('RAW contexts must contain actual FRAME descriptors')
        if (source_valid.shape != (b, 3, 1024) or source_valid.dtype != torch.bool
                or target_valid.shape != (b, q) or target_valid.dtype != torch.bool
                or semantic_indices.shape != (b, q, 64) or matching_logits.shape != (b, q, 64)
                or candidate_valid.shape != (b, q, 64)):
            raise ValueError('complete native Top64 address/mask contract required')
        indices = semantic_indices.long()
        if ((indices < 0) | (indices >= 3072)).any():
            raise ValueError('invalid source address')
        values, keys, raw_source = source_values.flatten(1, 2), source_keys.flatten(1, 2), source_context.flatten(1, 2)
        flat_valid = source_valid.flatten(1)
        valid = flat_valid[:, None].expand(-1, q, -1).gather(2, indices)&candidate_valid.bool()
        valid &= target_valid[..., None]&(flat_valid.sum(-1) >= 64)[:, None, None]
        active = valid.any(-1)
        logits = matching_logits.float().masked_fill(~valid, -torch.inf)
        chosen = logits.argmax(-1, keepdim=True)
        hard = torch.zeros_like(logits).scatter(2, chosen, 1.)*active[..., None]
        address = indices.gather(2, chosen).squeeze(2)
        top1 = values.gather(1, address[..., None].expand(-1, -1, width))*active[..., None]
        selected = active.flatten().nonzero().flatten()
        parts, confidences, all_probabilities = [], [], []
        for start in range(0, len(selected), self.feature_chunk_size):
            take = selected[start:start+self.feature_chunk_size]
            batch = take//q
            ids, mask = indices.reshape(-1, 64)[take], valid.reshape(-1, 64)[take]
            native = target_query.reshape(-1, 1024)[take].float()
            prior = logits.reshape(-1, 64)[take]/self.score_temperature
            with torch.no_grad(), torch.autocast(device_type=source_values.device.type, enabled=False):
                source_keys = keys[batch[:, None], ids].float().masked_fill(~mask[..., None], 0.)
                context = contextual_query(self.q2c.decoder,
                    target_keys.reshape(-1, 1024)[take].float(), source_keys, mask)
            with torch.no_grad():
                _, probability = _readout(self.ray_readout, native, context, values, keys, batch, ids, mask, prior)
                probability = probability.mean(1)
                confidence = (self.reliability(native, context, prior, mask, ids)
                    if self.reliability is not None else native.new_ones(len(native)))
            raw_target = target_context.reshape(-1, 2048)[take]
            if self.composition == 'paired':
                args = (self.retargeter, native, raw_target, values, raw_source,
                        batch, ids, mask, probability, self.g_max, self.candidate_chunk_size)
                correction = (checkpoint(paired_correction, *args, use_reentrant=False)
                    if torch.is_grad_enabled() else paired_correction(*args))
            else:
                raw = raw_source[batch[:, None], ids].float().masked_fill(~mask[..., None], 0.)
                difference = raw_target.float()-(probability[..., None]*raw).sum(1)
                appearance = torch.cat([top1.reshape(-1, width)[take], native], -1)
                correction = self.retargeter.correction(appearance, difference, self.g_max)
            parts.append(confidence[:, None]*correction)
            confidences.append(confidence)
            all_probabilities.append(probability)
        correction = source_values.new_zeros(b*q, width)
        if len(selected):
            correction = correction.index_copy(0, selected, torch.cat(parts))
        else:
            correction = correction+self.retargeter.deformation_projection.weight.flatten()[0]*0
        appearance = torch.cat([top1, target_query.float()], -1)
        base = self.retargeter.appearance_forward(appearance, self.g_max)
        retargeted = (base+correction.reshape(b, q, width).to(base.dtype))*active[..., None]
        self.last_feature_attention_stats = dict(ray_query='edited-target-patch',
            composition=self.composition, matching_frozen=True,
            reliability_predicted=self.reliability is not None,
            reliability_mean=float(torch.cat(confidences).mean()) if confidences else 0.,
            effective_K=float((-(torch.cat(all_probabilities)*torch.cat(all_probabilities).clamp_min(1e-30).log()).sum(-1)).exp().mean()) if all_probabilities else 0.,
            geometry_correction_RMS=float(correction.detach().square().mean().sqrt()),
            physical_deformation_channels=2048, appearance_VO_physically_absent=True,
            extra_path_R_weights_detached=True, active_rays=int(active.sum()))
        view_weights = logits.new_zeros(b, q, 3).scatter_add(2, indices//1024, hard)
        return DeformationCondition(top1+retargeted, indices, logits, hard, view_weights,
            retargeted, active, torch.tanh(self.fusion_gate))
