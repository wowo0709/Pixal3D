"""Saved-weight correspondence/score/decoder controls for physical DQ.

These are inference interventions, not retrained NoScore/NoCorr/NoDQ arms.
The positive Top1 carrier is preserved. Wrong-value-address moves both
appearance values and RAW Omega values with the same valid-slot permutation.
"""
from contextlib import contextmanager

import torch

from .ray_feature_controls import readout_control, valid_value_permutation


DEFINITIONS = {
    'wrong-value-address': 'Derange candidate appearance AND RAW Omega values; preserve keys, attention probabilities, cached scores and original Top1 carrier.',
    'score-prior-off': 'Remove only the added attention score prior; preserve content attention and the cached-score selected Top1 carrier.',
    'decoder-context-bypass': 'Use raw target Omega for the added attention query; preserve cached D1 proposal/score bank and original Top1 carrier.',
    'added-path-off': 'Mute the added appearance residual and DQ projection outputs; retain real nonzero query observations, original R and Top1 carrier.',
    'DQ-projection-off': 'Mute only the DQ projection output; retain real input channels and appearance attention. Not the physically absent-channel NoDQ training arm.',
}


@contextmanager
def deformation_control(conditioner, control, events):
    from pixal3d.modules import ray_deformation_query as module
    if not isinstance(conditioner, module.RayDeformationQueryConditioner):
        raise ValueError('physical DQ conditioner required')
    if control not in DEFINITIONS:
        raise ValueError('declared physical DQ path control required')
    if control == 'wrong-value-address':
        original = module.geometry_readout

        def wrong_geometry(head, native, context, values, keys, raw_source, raw_target,
                           batch, indices, valid, prior):
            feature, _, probability = original(head, native, context, values, keys,
                raw_source, raw_target, batch, indices, valid, prior)
            raw = raw_source[batch[:, None], indices].float()
            raw = torch.where(valid[..., None], raw, torch.zeros_like(raw))
            moved, reassigned = valid_value_permutation(raw, valid)
            weights = probability.mean(1)
            geometry = raw_target.float() - (weights[..., None] * moved).sum(1)
            events.append(dict(control=control, raw_Omega_values_reassigned=reassigned,
                RAW_value_change_RMS=float((moved-raw).float().square().mean().sqrt()),
                appearance_and_RAW_use_same_valid_slot_permutation=True,
                attention_probabilities_and_keys_and_score_prior_unchanged=True,
                original_Top1_carrier_unchanged=True))
            return feature, geometry, probability

        module.geometry_readout = wrong_geometry
        try:
            with readout_control(conditioner.ray_readout, control, events):
                yield
        finally:
            module.geometry_readout = original
    elif control == 'score-prior-off':
        with readout_control(conditioner.ray_readout, control, events):
            yield
    elif control == 'decoder-context-bypass':
        original = module.contextual_query

        def raw_query(decoder, query, keys, valid):
            if decoder is not conditioner.q2c.decoder:
                raise ValueError('undeclared decoder instance in DQ control')
            events.append(dict(control=control, queries=len(query),
                cached_D1_proposal_and_score_bank_retained=True,
                original_Top1_carrier_unchanged=True))
            return query

        module.contextual_query = raw_query
        try:
            yield
        finally:
            module.contextual_query = original
    else:
        def mute_projection(layer, inputs, output):
            events.append(dict(control=control,
                real_DQ_input_RMS=float(inputs[0].detach().float().square().mean().sqrt()),
                removed_DQ_projection_RMS=float(output.detach().float().square().mean().sqrt()),
                input_channels_physically_retained=True, fake_zero_input_observations=False,
                original_Top1_carrier_unchanged=True))
            return torch.zeros_like(output)

        handle = conditioner.retargeter.deformation_projection.register_forward_hook(mute_projection)
        try:
            if control == 'added-path-off':
                with readout_control(conditioner.ray_readout, 'residual-off', events):
                    yield
            else:
                yield
        finally:
            handle.remove()
