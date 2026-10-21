"""Explicit new pilot identity; no migration from a trained parent arm.

The fixed bound is a declared design choice, not selected on validation GT.
Native RGB identity and GPU capacity must still be measured before training.
"""
import math

import torch

VARIANT = 'bounded-source-contrast-NoDQ'
BOUND = 1.


def validate_plan(plan):
    if (plan.get('variant') != VARIANT or plan.get('epochs') != 7
            or plan.get('global_batch') != 96 or plan.get('seed') != 101
            or type(plan.get('residual_logit_bound')) not in (float, int)
            or not math.isfinite(plan['residual_logit_bound'])
            or plan['residual_logit_bound'] != BOUND
            or plan.get('parent_teacher_checkpoint') is not None
            or plan.get('initialization') != 'fresh19-plus-complete-utility-frozen-function-anchor'):
        raise ValueError('explicit cold bounded-source-contrast7epoch96/m96/a1 pilot required')


def build_conditioner(config, *, bound):
    from pixal3d.modules.bounded_source_contrast_member_feature import BoundedSourceContrastMemberFeatureConditioner
    if type(bound) not in (float, int) or bound != BOUND:
        raise ValueError('declared fixed residual logit bound1 required')
    return BoundedSourceContrastMemberFeatureConditioner(
        residual_logit_bound=bound, **config)


def initial_trainable_state(module):
    """Compare actual values of all19 parameters, rather than buffer schemas."""
    state = {key: value.detach() for key, value in module.named_parameters() if value.requires_grad}
    if len(state) != 19:
        raise ValueError('nineteen source parameter tensors required')
    return state


def assert_initial_prior(module, output):
    """Zero source output alone cannot prove that cold attention preserves π."""
    from pixal3d.modules.bounded_member_feature_source_ray import masked_probabilities
    selected, actual = module.last_feature_probabilities
    if not len(selected) or module.ray_readout.first_forward_prior_exact is not True:
        raise ValueError('actual active source rays and exact cold matching prior required')
    prior = output.attention.matching_logits.reshape(-1, 64)[selected]
    expected = masked_probabilities(prior[:, None].expand(-1, 4, -1),
        torch.isfinite(prior))
    if not torch.equal(actual, expected):
        raise ValueError('actual cold four-head probabilities differ from matching prior')
    return dict(active_source_rays=len(selected), four_head_matching_prior_exact=True)


def validate_mechanics(capacity, identity, *, same_initial19):
    """Check actual reports; this authorizes only the finite source-only pilot."""
    prior=capacity.get('actual_cold_four_head_prior', {})
    events=identity.get('actual_tensor_identity_events', [])
    comparison=identity.get('comparison', {})
    if (capacity.get('status') != 'PASS_ACTUAL_SOURCE_FEATURE_TEACHER_CAPACITY'
            or capacity.get('variant') != VARIANT
            or (capacity.get('actual_source_parameters'),capacity.get('optimizer_updates'),
                capacity.get('global_batch'),capacity.get('microbatch'),capacity.get('accumulation'))
                != (19,0,96,96,1)
            or capacity.get('actual_cold_GPU_source_delta_exact_zero') is not True
            or capacity.get('frozen_function_anchors_unchanged') is not True
            or capacity.get('matcher_unchanged') is not True
            or capacity.get('residual_logit_bound') != BOUND
            or capacity.get('parent_teacher_warm_migration') is not False
            or prior.get('four_head_matching_prior_exact') is not True
            or prior.get('active_source_rays',0) <= 0
            or identity.get('status') != 'PASS_ACTUAL_SOURCE_TEACHER_PILOT_NATIVE_INITIAL_IDENTITY32'
            or identity.get('variant') != VARIANT
            or identity.get('actual_native_global_projection_negative_RNG_exact') is not True
            or comparison.get('assets') != 32 or comparison.get('gain') != 0
            or comparison.get('paired_target_hashes_verified') != 32
            or not events or any(e.get('four_head_matching_prior_exact') is not True
                or e.get('active_source_rays',0) <= 0 or not all(e.values()) for e in events)
            or same_initial19 is not True):
        raise ValueError('actual micro96 backward, native32 identity, cold prior and same19 weights required')
    return dict(status='PASS', scope='Only fresh TRAIN1536 source-only7epoch112-update pilot',
        native32_global_projection_negative_RNG_and_GT_count_identity=True,
        actual_GPU96_backward_and_four_head_prior_identity=True,
        actual_common_initial19_values_exact=True, model_adopted=False,
        actual_joint_SS_training_authorized_by_this_report=False, full_goal_complete=False)
