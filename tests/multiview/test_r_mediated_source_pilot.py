from copy import deepcopy

import pytest

from experiments.deformnext_retargeting.r_mediated_source_pilot import (
    MARKERS, VARIANTS, validate_mechanics, validate_plan,
)


def plan(mode):
    return dict(variant=VARIANTS[mode], appearance_mode=mode,
        epochs=7, global_batch=96, seed=101, residual_logit_bound=1.,
        parent_teacher_checkpoint=None,
        initialization='fresh19-plus-complete-utility-frozen-function-anchor',
        feature_delivery='R-only-no-head-bypass')


@pytest.mark.parametrize('mode', list(VARIANTS))
def test_explicit_variant_mode_and_cold_budget_are_enforced(mode):
    p = plan(mode)
    validate_plan(p)
    for field, value in [('parent_teacher_checkpoint', 'trained-parent.pt'),
                         ('feature_delivery', 'direct-head'), ('epochs', 3),
                         ('residual_logit_bound', True), ('variant', 'bounded-source-contrast-NoDQ')]:
        with pytest.raises(ValueError):
            validate_plan(dict(p, **{field:value}))


def reports(mode):
    cap = dict(status='PASS_ACTUAL_SOURCE_FEATURE_TEACHER_CAPACITY', variant=VARIANTS[mode],
        appearance_mode=mode, feature_delivery='R-only-no-head-bypass',
        checkpoint_schema=MARKERS[mode], residual_logit_bound=1., parent_teacher_warm_migration=False,
        actual_source_parameters=19, optimizer_updates=0, global_batch=96, microbatch=96, accumulation=1,
        actual_cold_GPU_source_delta_exact_zero=True,
        actual_cold_four_head_prior=dict(four_head_matching_prior_exact=True),
        frozen_function_anchors_unchanged=True, matcher_unchanged=True,
        cold_head_gradient_zero_as_required_by_R_zero=True,
        disposable_gradient_probe=dict(separate_probe_updates=3,
            all19_nonzero_finite_at_third_backward=True, scientific_initial_state_unchanged=True))
    identity = dict(status='PASS_ACTUAL_SOURCE_TEACHER_PILOT_NATIVE_INITIAL_IDENTITY32',variant=VARIANTS[mode],
        actual_native_global_projection_negative_RNG_exact=True,
        comparison=dict(assets=32, gain=0, paired_target_hashes_verified=32),
        actual_tensor_identity_events=[dict(global_=True, proj=True, negative_global=True, negative_proj=True,
            active_source_rays=10, four_head_matching_prior_exact=True, direct_head_delivery_absent=True,
            actual_R_residual_exact=True, physical_NoDQ=True, own_schema_present=True)])
    identity['actual_tensor_identity_events'][0]['global'] = identity['actual_tensor_identity_events'][0].pop('global_')
    return cap, identity


@pytest.mark.parametrize('mode', list(VARIANTS))
def test_old_bypass_or_dead_head_or_missing_native_proof_cannot_admit_training(mode):
    cap, identity = reports(mode)
    assert validate_mechanics(cap, identity, appearance_mode=mode, same_initial19=True)['status'] == 'PASS'
    for field, value in [('cold_head_gradient_zero_as_required_by_R_zero', False),
                         ('microbatch', 48), ('checkpoint_schema', 'bounded_source_contrast_schema')]:
        with pytest.raises(ValueError):
            validate_mechanics(dict(cap, **{field:value}),identity,appearance_mode=mode,same_initial19=True)
    bad = deepcopy(cap); bad['disposable_gradient_probe']['all19_nonzero_finite_at_third_backward'] = False
    with pytest.raises(ValueError):
        validate_mechanics(bad,identity,appearance_mode=mode,same_initial19=True)
    bad = deepcopy(identity); del bad['actual_tensor_identity_events'][0]['actual_R_residual_exact']
    with pytest.raises(ValueError):
        validate_mechanics(cap,bad,appearance_mode=mode,same_initial19=True)
    with pytest.raises(ValueError):
        validate_mechanics(cap,identity,appearance_mode=mode,same_initial19=False)
