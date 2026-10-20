"""Read-only sensitivity controls for a trained candidate-paired checkpoint.

These are interventions on saved weights, not separately trained ablations.
Keep the original Top1/R carrier and inference inputs fixed. In particular,
score-prior-off is distinct from the user's Top1-payload NoScore definition.
"""
from contextlib import contextmanager

import torch


MODES = ('normal', 'geometry-off', 'gate-off', 'correction-x4', 'correction-x16', 'correction-x64', 'mean-inputs',
         'native-top1', 'attention-top1', 'score-prior-off', 'attention-decoder-bypass')


def top1_probability(probability, prior, valid, *, native):
    """Collapse to one valid payload; never replace matching by uniform weights."""
    score = prior if native else probability.mean(1)
    score = score.masked_fill(~valid, -torch.inf)
    chosen = score.argmax(-1, keepdim=True)
    result = torch.zeros_like(probability)
    result.scatter_(2, chosen[:, None].expand(-1, probability.shape[1], -1), 1.)
    return result*valid[:, None]*valid.any(-1)[:, None, None]


def mean_input_correction(retargeter, target, raw_target, values, raw_source,
                          batch, indices, valid, probability, g_max, candidate_chunk):
    """R(mean appearance, mean deformation), using the same candidate weights."""
    appearance = values.new_zeros(len(target), values.shape[-1], dtype=torch.float32)
    raw = raw_source.new_zeros(len(target), raw_source.shape[-1], dtype=torch.float32)
    for begin in range(0, indices.shape[1], candidate_chunk):
        take = slice(begin, min(indices.shape[1], begin+candidate_chunk))
        ids, mask = indices[:, take], valid[:, take]
        source = values[batch[:, None], ids].float().masked_fill(~mask[..., None], 0.)
        context = raw_source[batch[:, None], ids].float().masked_fill(~mask[..., None], 0.)
        weight = probability[:, take].float()*mask
        appearance = appearance+(weight[..., None]*source).sum(1)
        raw = raw+(weight[..., None]*context).sum(1)
    return retargeter.correction(torch.cat([appearance, target.float()], -1),
                                raw_target.float()-raw, g_max)


@contextmanager
def saved_weight_control(mode):
    if mode not in MODES:
        raise ValueError('unknown saved-weight control')
    from pixal3d.modules import ray_consistent_retargeting as module

    original_pair, original_readout = module.paired_correction, module._readout
    original_context = module.contextual_query
    reliability_class = module.RayCorrespondenceReliability
    original_reliability = reliability_class.forward
    state = dict(mode=mode, readout_calls=0, correction_calls=0, reliability_calls=0,
                 raw_query=None, attention_L1_sum=0., attention_rows=0)

    def context(decoder, query, keys, valid):
        state['raw_query'] = query
        return original_context(decoder, query, keys, valid)

    def readout(head, native, decoded, values, keys, batch, indices, valid, prior):
        state['readout_calls'] += 1
        replacement = state['raw_query'] if mode == 'attention-decoder-bypass' else decoded
        if mode == 'attention-decoder-bypass' and replacement.shape != decoded.shape:
            raise ValueError('the raw Omega query must match the D1 context batch')
        feature, probability = original_readout(head, native, replacement, values, keys,
            batch, indices, valid, torch.zeros_like(prior) if mode == 'score-prior-off' else prior)
        if mode in {'native-top1', 'attention-top1'}:
            probability = top1_probability(probability, prior, valid, native=mode == 'native-top1')
        state['attention_rows'] += len(probability)
        state['attention_L1_sum'] += float(probability.detach().abs().sum())
        return feature, probability

    def pair(retargeter, target, raw_target, values, raw_source,
             batch, indices, valid, probability, g_max, candidate_chunk):
        state['correction_calls'] += 1
        args = (retargeter, target, raw_target, values, raw_source, batch,
                indices, valid, probability, g_max, candidate_chunk)
        if mode == 'geometry-off':
            return target.new_zeros(len(target), retargeter.net[2].out_features)
        if mode == 'mean-inputs':
            return mean_input_correction(*args)
        result = original_pair(*args)
        multiplier = {'correction-x4': 4, 'correction-x16': 16, 'correction-x64': 64}.get(mode, 1)
        return result*multiplier if multiplier != 1 else result

    def reliability(self, native, decoded, prior, valid, indices):
        state['reliability_calls'] += 1
        result = original_reliability(self, native, decoded, prior, valid, indices)
        return torch.ones_like(result)*valid.any(-1) if mode == 'gate-off' else result

    module.contextual_query, module._readout = context, readout
    module.paired_correction, reliability_class.forward = pair, reliability
    try:
        yield state
    finally:
        module.contextual_query, module._readout = original_context, original_readout
        module.paired_correction, reliability_class.forward = original_pair, original_reliability
        state.pop('raw_query', None)

