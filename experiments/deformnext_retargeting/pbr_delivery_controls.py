"""Inference-only interventions for the measured legacy PBR failure paths.

Prior replacement also updates the attention feature and its RAW aggregation.
Removing direct delivery retains the normal R inputs and physical DQ channels.
Neither intervention is a retrained NoScore, NoCorr, or no-DQ architecture.
"""
from contextlib import contextmanager
from dataclasses import replace

import torch

from pixal3d.modules.bounded_member_feature_source_ray import masked_probabilities
from pixal3d.modules.direct_ray_retargeting import DirectRayRetargetingConditioner


MODES = ('normal', 'matching-prior', 'no-direct-delivery', 'matching-prior-no-direct-delivery')


@contextmanager
def delivery_control(model, mode, events):
    if (mode not in MODES or not isinstance(model, DirectRayRetargetingConditioner)
            or model.value_dim != 2048 or model.retargeter.input_dim != 5120
            or model.retargeter.deformation_dim != 2048 or model.training
            or torch.is_grad_enabled()):
        raise ValueError('one actual physical PBR2048/RAW2048 module in eval/no_grad required')
    prior_mode = mode in ('matching-prior', 'matching-prior-no-direct-delivery')
    remove_direct = mode in ('no-direct-delivery', 'matching-prior-no-direct-delivery')
    handles = []
    state = {}

    def begin(module, args, kwargs):
        state.clear()
        state.update(head_events=[], R=None)

    def capture_value(module, args, output):
        state['projected_values'] = output

    def preserve_prior(module, args, output):
        native, context, keys, values, valid, prior = args
        old_feature, old_probability = output
        projected = state.pop('projected_values')
        rows, candidates = valid.shape
        if projected.shape != (rows, candidates, 256):
            raise ValueError('same native four64-wide projected values required')
        probability = masked_probabilities(prior[:, None].expand(-1, 4, -1), valid)
        value = projected.reshape(rows, candidates, 4, 64).transpose(1, 2)
        combined = (probability[..., None] * value.float()).sum(-2).reshape(rows, 256)
        feature = module.output(combined).float()
        if probability.shape != old_probability.shape or feature.shape != old_feature.shape:
            raise ValueError('unchanged native feature/probability shapes required')
        state['head_events'].append(dict(queries=rows, valid_slots=int(valid.sum()),
            mean_probability_L1_change=float((probability-old_probability).abs().sum(-1).mean()),
            feature_change_RMS=float((feature-old_feature).square().mean().sqrt()),
            four_head_probability_exact_to_matching_prior=True,
            feature_recomputed_from_same_projected_values=True))
        # geometry_readout consumes this same replacement probability for RAW
        # aggregation; returning only a new probability would be inconsistent.
        return feature, probability

    def capture_R(module, args, output):
        if state['R'] is not None:
            raise ValueError('exactly one normal R call per source forward required')
        state['R'] = output

    def finish(module, args, kwargs, output):
        event = dict(control=mode, active_queries=int(output.query_valid.sum()),
            physical_R_channels=module.retargeter.input_dim, physical_DQ_channels=2048,
            candidate_keys_cached_scores_and_Top1_carrier_retained=True,
            matching_prior_replacement=prior_mode, direct_delivery_removed=remove_direct,
            normal_R_input_and_DQ_retained=True, head_events=state['head_events'],
            saved_weight_diagnostic_not_retrained_ablation=True, GT_added=False)
        changed = output
        if remove_direct:
            if state['R'] is None or state['R'].shape != output.evidence.shape:
                raise ValueError('actual unmodified R output required')
            pure_R = state['R'].float() * output.query_valid[..., None]
            values = kwargs['source_values'].float().flatten(1, 2)
            chosen = output.candidate_weights.argmax(-1, keepdim=True)
            address = output.candidate_indices.gather(2, chosen).squeeze(2)
            top1 = values.gather(1, address[..., None].expand(-1, -1, 2048))
            top1 = top1 * output.query_valid[..., None]
            event['removed_direct_correction_RMS'] = float(
                (output.retarget_residual.float()-pure_R).square().mean().sqrt())
            changed = replace(output, evidence=top1+pure_R, retarget_residual=pure_R)
        events.append(event)
        return None if mode == 'normal' else changed

    try:
        handles.append(model.register_forward_pre_hook(begin, with_kwargs=True))
        if prior_mode:
            handles.append(model.ray_readout.value.register_forward_hook(capture_value))
            handles.append(model.ray_readout.register_forward_hook(preserve_prior))
        if remove_direct:
            handles.append(model.retargeter.register_forward_hook(capture_R))
        handles.append(model.register_forward_hook(finish, with_kwargs=True))
        yield
    finally:
        for handle in handles:
            handle.remove()
