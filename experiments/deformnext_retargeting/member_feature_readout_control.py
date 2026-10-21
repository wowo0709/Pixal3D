"""Frozen-weight attribution of matching prior, learned mixing and correction.

Every intervention keeps original candidates, Top1, source-use and R's target
context. These are evaluation diagnostics, never the user's retrained NoScore
(physical Top1) or NoCorr (native spatial correspondence) arms.
"""
from contextlib import contextmanager
import math

import torch

from pixal3d.modules.member_feature_source_ray import MemberFeatureSourceRayConditioner
from .member_feature_query_support import _digest


MODES = ('normal', 'matching-prior-only', 'without-readout-logprior', 'zero-feature-correction')


def probabilities(logits, valid):
    logits = logits.float().masked_fill(~valid[:, None], -torch.inf)
    maximum = logits.amax(-1, keepdim=True)
    maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
    values = torch.exp(logits-maximum).masked_fill(~valid[:, None], 0.)
    return values/values.sum(-1, keepdim=True).clamp_min(1e-8)


@contextmanager
def feature_readout_control(module, events, *, mode):
    if type(module) is not MemberFeatureSourceRayConditioner or mode not in MODES:
        raise ValueError('explicit frozen NoDQ four-head model and readout mode required')
    state, captured, handles = {}, {}, []

    def begin(layer, args, kwargs):
        if (state or args or layer.training or torch.is_grad_enabled()
                or kwargs.get('control', 'normal') != 'normal'):
            raise ValueError('uncombined readout diagnostic requires eval/no_grad')
        state.update(inputs=kwargs, rows=0, chunks=[], gates=[], target_context=[])

    def capture(name):
        def hook(layer, args, output):
            captured[name] = output[2] if name == 'score' else output
        return hook

    def readout(head, args, result):
        if not state or len(args) != 6 or set(captured) != {'query', 'key', 'value', 'score'}:
            raise ValueError('one complete original Q/K/V/score forward required')
        native, context, keys, values, valid, prior = args
        rows, candidates = valid.shape
        with torch.autocast(device_type=valid.device.type, enabled=False):
            q = captured['query'].reshape(rows, 4, 64)
            key = captured['key'].reshape(rows, candidates, 4, 64).transpose(1, 2)
            content = (q.float()[:, :, None]*key.float()).sum(-1)/math.sqrt(64)
            score = captured['score'].float()
            baseline = probabilities(content+score[:, None], valid)
            if not torch.equal(baseline, result[1]):
                raise ValueError('original same-forward four-head probability differs')
            if mode == 'matching-prior-only':
                selected = probabilities(prior[:, None].expand(-1, 4, -1), valid)
            elif mode == 'without-readout-logprior':
                # score_selector's logits are prior plus its learned residual.
                residual = torch.where(valid, score-prior.float(), torch.zeros_like(score))
                selected = probabilities(content+residual[:, None], valid)
            else:
                selected = baseline
            projected = captured['value'].reshape(rows, candidates, 4, 64).transpose(1, 2)
            combined = (selected[..., None]*projected.float()).sum(-2).reshape(rows, -1)
            entropy = -(selected*selected.clamp_min(1e-30).log()).sum(-1)
            prior_probability = probabilities(prior[:, None].expand(-1, 4, -1), valid)
            kl = (selected*(selected.clamp_min(1e-30).log()-prior_probability.clamp_min(1e-30).log())).sum(-1)
        if mode == 'zero-feature-correction':
            changed = torch.zeros_like(result[0])
        elif mode == 'normal':
            changed = result[0]
        else:
            # Reuse this forward's actual projected V and the original output
            # layer. There is no second retrieval, ray pooling or head pass.
            changed = head.output(combined).float()
        state['rows'] += rows
        state['chunks'].append(dict(queries=rows,
            original_four_head_probabilities_exact_same_forward=True,
            original_effective_K_per_head_mean=float((-(baseline*baseline.clamp_min(1e-30).log()).sum(-1)).exp().mean()),
            effective_K_per_head_mean=float(entropy.exp().mean()),
            KL_to_matching_prior_mean=float(kl.mean()),
            original_feature_correction_RMS=float(result[0].square().mean().sqrt()),
            actual_feature_correction_RMS=float(changed.square().mean().sqrt()),
            feature_correction_change_RMS=float((changed-result[0]).square().mean().sqrt()),
            probability_change_RMS=float((selected-baseline).square().mean().sqrt())))
        captured.clear()
        return changed, selected

    def gate(layer, args):
        state['gates'].append(args)

    def retarget(layer, args):
        state['target_context'].append(args[0][..., -1024:])

    def finish(layer, args, kwargs, result):
        if captured or len(state['gates']) != 1:
            raise ValueError('complete readout and exactly one source-use decision required')
        target, top1, valid = state['gates'][0]
        attention = result.attention
        address = attention.candidate_indices.gather(
            -1, attention.matching_logits.argmax(-1, keepdim=True)).squeeze(-1)
        expected = state['inputs']['source_values'].float().flatten(1, 2).gather(
            1, address[..., None].expand(-1, -1, 1024))*valid[..., None]
        selected = valid.flatten().nonzero().flatten()
        if (not torch.equal(top1, expected) or not torch.equal(top1, attention.base_evidence)
                or not torch.equal(valid, attention.query_valid)
                or not torch.equal(result.use_probability, module.evidence_use.forward(target, top1, valid))
                or state['rows'] != len(selected)):
            raise ValueError('original Top1, matching or same-forward source-use changed')
        if len(selected):
            if (len(state['target_context']) != 1
                    or not torch.equal(state['target_context'][0], target.reshape(-1, 1024)[selected])):
                raise ValueError('original R target context changed')
        elif state['target_context']:
            raise ValueError('empty ray support cannot enter retargeter')
        events.append(dict(mode=mode, queries=state['rows'], chunks=state['chunks'],
            fixed_context_SHA256=dict(candidate_indices=_digest(attention.candidate_indices),
                matching_prior=_digest(attention.matching_logits), Top1_feature=_digest(top1),
                R_and_source_use_target=_digest(target), source_use_probability=_digest(result.use_probability)),
            original_candidates_Top1_R_target_and_source_use_exact_same_forward=True,
            actual_matching_decoder_unchanged=True, native_projection_and_write_support_unchanged=True,
            retrained_NoScore_NoCorr_or_decoder_ablation=False))
        state.clear()

    try:
        handles.append(module.register_forward_pre_hook(begin, with_kwargs=True))
        for name, layer in [('query', module.ray_readout.query), ('key', module.ray_readout.key),
                            ('value', module.ray_readout.value), ('score', module.ray_readout.score_selector)]:
            handles.append(layer.register_forward_hook(capture(name)))
        handles.append(module.ray_readout.register_forward_hook(readout))
        handles.append(module.evidence_use.register_forward_pre_hook(gate))
        handles.append(module.retargeter.register_forward_pre_hook(retarget))
        handles.append(module.register_forward_hook(finish, with_kwargs=True))
        yield
        if state or captured:
            raise ValueError('unfinished frozen readout diagnostic')
    finally:
        for handle in handles:
            handle.remove()
        state.clear(); captured.clear()
