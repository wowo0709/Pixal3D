"""Inference-only QKV controls on the actual saved native-spatial3 SS model.

Keep the native3 proposals, frozen matcher, Top1 carrier, source-use and R
target context. Change the feature readout before its existing R/direct
consumer. These are conditional reliance tests, never retrained ablations.
"""
from contextlib import contextmanager
import math

import torch

from pixal3d.modules.member_feature_controls import SpatialFeatureSourceRayConditioner
from .member_feature_query_support import _digest
from .member_feature_readout_control import probabilities


CONTROLS = ('normal', 'matching-prior-only', 'top1-feature-only',
            'zero-transformed-value', 'wrong-projected-V')


def select_probabilities(baseline, prior, valid, control):
    if (control not in CONTROLS or valid.ndim != 2 or valid.shape[-1] != 3
            or valid.dtype != torch.bool or baseline.shape != (len(valid), 4, 3)
            or prior.shape != valid.shape or not valid.any(-1).all()
            or not torch.isfinite(baseline).all() or (baseline < 0).any()
            or not torch.allclose(baseline.sum(-1), torch.ones_like(baseline[..., 0]), rtol=0, atol=1e-6)
            or baseline.masked_select(~valid[:, None].expand_as(baseline)).count_nonzero()
            or not torch.isfinite(prior[valid]).all()):
        raise ValueError('finite normalized four-head probabilities on active native3 proposals required')
    if control == 'matching-prior-only':
        return probabilities(prior[:, None].expand(-1, 4, -1), valid)
    if control == 'top1-feature-only':
        chosen = prior.masked_fill(~valid, -torch.inf).argmax(-1, keepdim=True)
        return torch.zeros_like(prior).scatter(1, chosen, 1.)[:, None].expand(-1, 4, -1)
    return baseline


def combine_projected_values(projected, selected, control):
    if (control not in CONTROLS or projected.shape != (*selected.shape, 64)
            or selected.ndim != 3 or selected.shape[1:] != (4, 3)):
        raise ValueError('actual native3 four64-wide projected values required')
    values = projected.roll(1, dims=2) if control == 'wrong-projected-V' else projected
    return (selected[..., None] * values.float()).sum(-2).reshape(len(selected), 256)


@contextmanager
def spatial_readout_control(module, events, *, control):
    if type(module) is not SpatialFeatureSourceRayConditioner or control not in CONTROLS:
        raise ValueError('actual saved SpatialFeatureSourceRayConditioner and named control required')
    state, captured, handles = {}, {}, []
    versions = {n: p._version for n, p in module.named_parameters()}

    def begin(layer, args, kwargs):
        if (state or args or layer.training or torch.is_grad_enabled()
                or kwargs.get('control', 'normal') != 'normal'
                or versions != {n: p._version for n, p in module.named_parameters()}):
            raise ValueError('uncombined saved-weight spatial eval/no_grad control required')
        state.update(inputs=kwargs, rows=0, chunks=[], gates=[], target_context=[])

    def capture(name):
        def hook(layer, args, output):
            captured[name] = output[2] if name == 'score' else output
        return hook

    def readout(head, args, result):
        if not state or len(args) != 6 or set(captured) != {'query', 'key', 'value', 'score'}:
            raise ValueError('one complete same-forward Q/K/V/score event required')
        native, context, keys, values, valid, prior = args
        rows, candidates = valid.shape
        if candidates != 3 or values.shape != (rows, 3, 1024):
            raise ValueError('original spatial3 proposals and appearance width required')
        with torch.autocast(device_type=valid.device.type, enabled=False):
            query = captured['query'].reshape(rows, 4, 64).float()
            key = captured['key'].reshape(rows, 3, 4, 64).transpose(1, 2).float()
            content = (query[:, :, None] * key).sum(-1) / math.sqrt(64)
            baseline = probabilities(content + captured['score'].float()[:, None], valid)
            if not torch.equal(baseline, result[1]):
                raise ValueError('original actual QKV probabilities did not reproduce exactly')
            selected = select_probabilities(baseline, prior, valid, control)
            projected = captured['value'].reshape(rows, 3, 4, 64).transpose(1, 2)
            combined = combine_projected_values(projected, selected, control)
            prior_probability = probabilities(prior[:, None].expand(-1, 4, -1), valid)
        if control == 'normal':
            changed = result[0]
        elif control == 'zero-transformed-value':
            changed = torch.zeros_like(result[0])
        else:
            changed = head.output(combined).float()
        entropy = -(selected * selected.clamp_min(1e-30).log()).sum(-1)
        original_entropy = -(baseline * baseline.clamp_min(1e-30).log()).sum(-1)
        kl = (baseline * (baseline.clamp_min(1e-30).log()
                         - prior_probability.clamp_min(1e-30).log())).sum(-1)
        top1 = prior.argmax(-1)
        top1_mass = baseline.gather(-1, top1[:, None, None].expand(-1, 4, 1)).squeeze(-1)
        state['rows'] += rows
        state['chunks'].append(dict(queries=rows, candidate_count=3,
            original_four_head_probabilities_exact_same_forward=True,
            original_effective_K_mean=float(original_entropy.exp().mean()),
            actual_effective_K_mean=float(entropy.exp().mean()),
            original_KL_to_matching_prior_mean=float(kl.mean()),
            original_mass_outside_matching_Top1_mean=float((1-top1_mass).mean()),
            original_feature_correction_RMS=float(result[0].square().mean().sqrt()),
            actual_feature_correction_RMS=float(changed.square().mean().sqrt()),
            feature_correction_change_RMS=float((changed-result[0]).square().mean().sqrt()),
            probability_change_RMS=float((selected-baseline).square().mean().sqrt())))
        captured.clear()
        return changed, selected

    def gate(layer, args, output):
        state['gates'].append((args, output))

    def retarget(layer, args):
        state['target_context'].append(args[0][..., -1024:])

    def finish(layer, args, kwargs, result):
        if captured or len(state['gates']) != 1:
            raise ValueError('complete spatial readout and one original source-use event required')
        (target, top1, valid), use = state['gates'][0]
        attention = result.attention
        indices, logits = attention.candidate_indices, attention.matching_logits
        if indices.shape[-1] != 3 or not torch.equal(indices, torch.arange(3, device=indices.device).expand_as(indices)):
            raise ValueError('original native3 proposal identities changed')
        values = state['inputs']['spatial_values'].float()
        chosen = logits.argmax(-1, keepdim=True)
        expected = values.gather(2, chosen[..., None].expand(-1, -1, 1, 1024)).squeeze(2) * valid[..., None]
        selected = valid.flatten().nonzero().flatten()
        if (not torch.equal(top1, expected) or not torch.equal(top1, attention.base_evidence)
                or not torch.equal(valid, attention.query_valid) or not torch.equal(use, result.use_probability)
                or state['rows'] != len(selected)):
            raise ValueError('original spatial Top1/support/source-use changed')
        if len(selected):
            if (len(state['target_context']) != 1
                    or not torch.equal(state['target_context'][0], target.reshape(-1, 1024)[selected])):
                raise ValueError('R target context changed')
        elif state['target_context']:
            raise ValueError('empty ray support cannot enter R')
        if versions != {n: p._version for n, p in module.named_parameters()}:
            raise ValueError('saved source weights changed')
        events.append(dict(control=control, queries=state['rows'], chunks=state['chunks'],
            fixed_source_fields=dict(candidate_indices=_digest(indices), matching_prior=_digest(logits),
                native_spatial_values=_digest(values), Top1_feature=_digest(top1),
                R_target_context=_digest(target), source_use_probability=_digest(use),
                fusion_gate=_digest(attention.fusion_scale)),
            actual_computed_R_output_SHA256=_digest(attention.retarget_residual),
            actual_source_numerator_SHA256=_digest(result.source_numerator),
            original_candidates_Top1_R_target_and_source_use_exact=True,
            same_forward_weights_unchanged=True, inference_intervention_not_retrained_ablation=True))
        state.clear()

    try:
        handles.append(module.register_forward_pre_hook(begin, with_kwargs=True))
        for name, layer in [('query', module.ray_readout.query), ('key', module.ray_readout.key),
                            ('value', module.ray_readout.value), ('score', module.ray_readout.score_selector)]:
            handles.append(layer.register_forward_hook(capture(name)))
        handles.append(module.ray_readout.register_forward_hook(readout))
        handles.append(module.evidence_use.register_forward_hook(gate))
        handles.append(module.retargeter.register_forward_pre_hook(retarget))
        handles.append(module.register_forward_hook(finish, with_kwargs=True))
        yield
        if state or captured:
            raise ValueError('unfinished spatial readout diagnostic')
    finally:
        for handle in handles: handle.remove()
        state.clear(); captured.clear()
