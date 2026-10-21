from copy import deepcopy

import pytest
import torch

from experiments.deformnext_retargeting.bounded_source_contrast_pilot import (
    VARIANT, validate_plan, initial_trainable_state, assert_initial_prior, validate_mechanics,
)
from test_bounded_source_contrast_member_feature import pair


def valid_plan():
    return dict(variant=VARIANT, epochs=7, global_batch=96, seed=101,
        residual_logit_bound=1.,
        initialization='fresh19-plus-complete-utility-frozen-function-anchor')


@pytest.mark.parametrize('change', [
    dict(variant='source-contrast-NoDQ'), dict(parent_teacher_checkpoint='old112.pt'),
    dict(residual_logit_bound=True), dict(residual_logit_bound=float('nan')),
    dict(residual_logit_bound=2.), dict(global_batch=24), dict(epochs=3),
    dict(initialization='warm-parent'),
])
def test_pilot_rejects_alias_warm_migration_or_changed_comparison_budget(change):
    plan=valid_plan();plan.update(change)
    with pytest.raises(ValueError):validate_plan(plan)


def test_same_actual_nineteen_weights_despite_distinct_checkpoint_buffers():
    validate_plan(valid_plan());parent,model,_=pair()
    a,b=initial_trainable_state(parent),initial_trainable_state(model)
    assert len(a)==19 and a.keys()==b.keys()
    assert all(torch.equal(value,b[key])for key,value in a.items())
    assert not any(key.startswith('q2c.')for key in b)
    assert set(parent.state_dict())!=set(model.state_dict())


def test_initial_prior_check_detects_wrong_attention_even_with_zero_source_delta():
    _,model,data=pair();model.eval()
    with torch.no_grad():out=model(**data)
    event=assert_initial_prior(model,out)
    assert event['active_source_rays']>0 and event['four_head_matching_prior_exact']
    assert not out.source_numerator.any()
    selected,probabilities=model.last_feature_probabilities
    model.last_feature_probabilities=selected,probabilities.roll(1,dims=-1)
    assert not out.source_numerator.any()
    with pytest.raises(ValueError,match='probabilities differ'):
        assert_initial_prior(model,out)


def test_empty_support_does_not_masquerade_as_measured_prior_identity():
    _,model,data=pair();data['observed_query'].zero_()
    with torch.no_grad():out=model(**data)
    with pytest.raises(ValueError,match='active source rays'):
        assert_initial_prior(model,out)


def mechanics():
    prior=dict(active_source_rays=3,four_head_matching_prior_exact=True)
    capacity=dict(status='PASS_ACTUAL_SOURCE_FEATURE_TEACHER_CAPACITY',variant=VARIANT,
        actual_source_parameters=19,optimizer_updates=0,global_batch=96,microbatch=96,accumulation=1,
        actual_cold_GPU_source_delta_exact_zero=True,frozen_function_anchors_unchanged=True,
        matcher_unchanged=True,residual_logit_bound=1.,parent_teacher_warm_migration=False,
        actual_cold_four_head_prior=deepcopy(prior))
    identity=dict(status='PASS_ACTUAL_SOURCE_TEACHER_PILOT_NATIVE_INITIAL_IDENTITY32',variant=VARIANT,
        actual_native_global_projection_negative_RNG_exact=True,
        comparison=dict(assets=32,gain=0.,paired_target_hashes_verified=32),
        actual_tensor_identity_events=[dict(**prior,global_field=True,projection=True,negative=True)])
    return capacity,identity


def test_voxel_identity_cannot_release_training_when_attention_or_native_field_check_fails():
    cap,identity=mechanics()
    assert validate_mechanics(cap,identity,same_initial19=True)['status']=='PASS'
    identity['actual_tensor_identity_events'][0]['four_head_matching_prior_exact']=False
    with pytest.raises(ValueError):validate_mechanics(cap,identity,same_initial19=True)
    cap,identity=mechanics();identity['actual_tensor_identity_events'][0]['projection']=False
    with pytest.raises(ValueError):validate_mechanics(cap,identity,same_initial19=True)
    cap,identity=mechanics()
    with pytest.raises(ValueError):validate_mechanics(cap,identity,same_initial19=False)
