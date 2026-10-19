"""Saved-weight interventions for feature attention; no training arm is renamed.

The Top1 carrier remains identical in every control. Score-prior-off tests only
the added attention's use of cached scores; it is NOT the user's NoScore arm.
Decoder-context-bypass retains the cached proposal/score bank, so it does not
claim to remove the matcher's entire contribution.
"""
from contextlib import contextmanager

import torch


DEFINITIONS = {
    'residual-off': 'Zero only the transformed feature residual;retain Top1 carrier and original R.',
    'wrong-value-address': 'Derange valid projected values within each candidate set;retain keys,weights,score prior and Top1 carrier.',
    'score-prior-off': 'Remove only cached score logits from added feature attention;retain content attention and matching-selected Top1 carrier.',
    'decoder-context-bypass': 'Use raw target Omega instead of D1 contextual query in added feature attention;retain cached proposals/scores and Top1 carrier.',
}


def valid_value_permutation(values, valid):
    if values.ndim != 3 or valid.shape != values.shape[:2] or valid.dtype != torch.bool:
        raise ValueError('candidate values and matching validity mask required')
    n, k = valid.shape
    count = valid.sum(-1)
    position = torch.arange(k,device=values.device).expand(n,-1)
    ordered = position.masked_fill(~valid,k).sort(-1).values
    rank = valid.long().cumsum(-1)-1
    donor_rank = (rank+count[:,None]//2) % count[:,None].clamp_min(1)
    donor = ordered.gather(1,donor_rank).clamp_max(k-1)
    moved = values.gather(1,donor[...,None].expand_as(values))
    moved = torch.where(valid[...,None],moved,torch.zeros_like(moved))
    return moved, int(((donor != position) & valid).sum())


@contextmanager
def readout_control(head, control, events):
    """Hooks act at Wv, after the selector reads original candidate features."""
    if control == 'normal':
        yield
        return
    if control not in DEFINITIONS or control == 'decoder-context-bypass':
        raise ValueError('declared readout control required')
    handles = []
    current = {}
    if control == 'wrong-value-address':
        def capture_valid(module,args):
            current['valid'] = args[4]
        def move_values(module,args):
            original, = args
            moved, reassigned = valid_value_permutation(original,current['valid'])
            events.append(dict(control=control,queries=len(original),
                valid_slots=int(current['valid'].sum()),reassigned_valid_slots=reassigned,
                value_change_RMS=float((moved-original).detach().float().square().mean().sqrt()),
                keys_scores_queries_and_attention_weights_unchanged=True,
                Top1_carrier_unchanged=True))
            return (moved,)
        handles += [head.register_forward_pre_hook(capture_valid),
                    head.value.register_forward_pre_hook(move_values)]
    elif control == 'score-prior-off':
        def remove_prior(module,args):
            target,context,values,valid,prior = args
            events.append(dict(control=control,queries=len(target),
                removed_valid_score_slots=int(valid.sum()),Top1_carrier_unchanged=True,
                content_attention_retained=True))
            return target,context,values,valid,torch.zeros_like(prior)
        handles.append(head.score_selector.register_forward_pre_hook(remove_prior))
    else:
        def remove_residual(module,args,output):
            residual,weights = output
            events.append(dict(control=control,queries=len(residual),
                removed_feature_residual_RMS=float(residual.detach().float().square().mean().sqrt()),
                Top1_carrier_unchanged=True))
            return torch.zeros_like(residual),weights
        handles.append(head.register_forward_hook(remove_residual))
    try:
        yield
    finally:
        for handle in reversed(handles):
            handle.remove()


@contextmanager
def feature_control(conditioner,control,events):
    from pixal3d.modules.ray_feature_attention import RayFeatureAttentionConditioner
    if not isinstance(conditioner,RayFeatureAttentionConditioner):
        raise ValueError('feature-specific control requires the declared conditioner')
    if control != 'decoder-context-bypass':
        with readout_control(conditioner.ray_readout,control,events):
            yield
        return
    # The module-local function is looked up only in this evaluator process.
    # Frozen source bytes and other training processes are untouched.
    from pixal3d.modules import ray_feature_attention as module
    original = module.contextual_query
    def raw_target(decoder,query,keys,valid):
        if decoder is not conditioner.q2c.decoder:
            raise ValueError('intervention encountered an undeclared decoder instance')
        events.append(dict(control=control,queries=len(query),raw_Omega_query=True,
            cached_proposal_and_score_bank_retained=True,Top1_carrier_unchanged=True))
        return query
    module.contextual_query = raw_target
    try:
        yield
    finally:
        module.contextual_query = original
