"""Matched cold19 source pilots for R-only delivery and raw Top-K R inputs."""
import torch

from .bounded_source_contrast_pilot import (
    assert_initial_prior, initial_trainable_state, validate_plan as validate_parent_plan,
    VARIANT as PARENT_VARIANT,
)


VARIANTS = {'top1-plus-feature': 'R-mediated-feature-NoDQ',
            'raw-topk-plus-feature': 'R-mediated-rawTopK-NoDQ'}
MARKERS = {'top1-plus-feature': 'r_mediated_feature_schema',
           'raw-topk-plus-feature': 'r_mediated_raw_topk_schema'}
BOUND = 1.


def validate_plan(plan):
    validate_parent_plan(dict(plan, variant=PARENT_VARIANT))
    if (plan.get('appearance_mode') not in VARIANTS
            or plan.get('variant') != VARIANTS[plan['appearance_mode']]
            or plan.get('feature_delivery') != 'R-only-no-head-bypass'):
        raise ValueError('explicit cold19 R-mediated appearance/delivery contract required')


def build_conditioner(config, *, bound, appearance_mode):
    from pixal3d.modules.r_mediated_source_contrast import RMediatedSourceContrastConditioner
    if type(bound) not in (float, int) or bound != BOUND or appearance_mode not in VARIANTS:
        raise ValueError('declared bound1 and explicit R appearance mode required')
    return RMediatedSourceContrastConditioner(residual_logit_bound=bound,
        appearance_mode=appearance_mode, **config)


def assert_R_only(module, output):
    selected = module.last_source_reference_rows
    residual = output.attention.retarget_residual.reshape(-1, 1024)[selected]
    if (not torch.equal(residual, module.last_R_only_output)
            or module.last_direct_feature_correction.count_nonzero()
            or module.retargeter.input_dim != 2048
            or any('deformation_projection' in name for name in module.state_dict())
            or MARKERS[module.appearance_mode] not in module.state_dict()):
        raise ValueError('actual R output alone, physical NoDQ and strict own schema required')
    return dict(direct_head_delivery_absent=True, actual_R_residual_exact=True,
                physical_NoDQ=True, own_schema_present=True)


def validate_mechanics(capacity, identity, *, appearance_mode, same_initial19):
    variant = VARIANTS[appearance_mode]
    event_keys = {'global', 'proj', 'negative_global', 'negative_proj',
                  'active_source_rays', 'four_head_matching_prior_exact',
                  'direct_head_delivery_absent', 'actual_R_residual_exact',
                  'physical_NoDQ', 'own_schema_present'}
    if (capacity.get('status') != 'PASS_ACTUAL_SOURCE_FEATURE_TEACHER_CAPACITY'
            or capacity.get('variant') != variant
            or capacity.get('appearance_mode') != appearance_mode
            or capacity.get('feature_delivery') != 'R-only-no-head-bypass'
            or capacity.get('checkpoint_schema') != MARKERS[appearance_mode]
            or capacity.get('residual_logit_bound') != BOUND
            or capacity.get('parent_teacher_warm_migration') is not False
            or (capacity.get('actual_source_parameters'), capacity.get('optimizer_updates'),
                capacity.get('global_batch'), capacity.get('microbatch'), capacity.get('accumulation'))
                != (19, 0, 96, 96, 1)
            or capacity.get('actual_cold_GPU_source_delta_exact_zero') is not True
            or capacity.get('actual_cold_four_head_prior', {}).get('four_head_matching_prior_exact') is not True
            or capacity.get('frozen_function_anchors_unchanged') is not True
            or capacity.get('matcher_unchanged') is not True
            or capacity.get('cold_head_gradient_zero_as_required_by_R_zero') is not True
            or capacity.get('disposable_gradient_probe', {}).get('separate_probe_updates') != 3
            or capacity.get('disposable_gradient_probe', {}).get('all19_nonzero_finite_at_third_backward') is not True
            or capacity.get('disposable_gradient_probe', {}).get('scientific_initial_state_unchanged') is not True
            or identity.get('status') != 'PASS_ACTUAL_SOURCE_TEACHER_PILOT_NATIVE_INITIAL_IDENTITY32'
            or identity.get('variant') != variant
            or identity.get('actual_native_global_projection_negative_RNG_exact') is not True
            or identity.get('comparison', {}).get('assets') != 32
            or identity.get('comparison', {}).get('gain') != 0
            or identity.get('comparison', {}).get('paired_target_hashes_verified') != 32
            or not identity.get('actual_tensor_identity_events')
            or any(not event_keys.issubset(event) or not all(event.values())
                   for event in identity['actual_tensor_identity_events'])
            or same_initial19 is not True):
        raise ValueError('actual micro96 cold+delayed-gradient and native32 identity proof required')
    return dict(status='PASS', variant=variant, appearance_mode=appearance_mode,
        source_only_TRAIN1536_teacher112_scope=True, common_initial19_exact=True,
        actual_native_identity32_and_micro96_backward=True,
        R_zero_first_head_gradient_zero_but_separate_probe_learns_all19=True,
        warm_parent_migration=False, main_joint_training_admitted=False,
        model_adopted=False, full_goal_complete=False)
