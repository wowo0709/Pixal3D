import copy
import json

import pytest
import torch

from experiments.deformnext_retargeting.verify_ss_conditioning_contract import (
    check_conditions, compare_trace_tensors, attention_weight_statistics,
    online_dataset_args,verify_saved_conditioner,check_native_tokens,
)


def test_attention_statistics_measure_actual_active_distribution_only():
    p=torch.tensor([[[.25,.25,.25,.25],[1.,0.,0.,0.]],[[0.,0.,0.,0.],[0.,0.,0.,0.]]])
    r=attention_weight_statistics(p,torch.tensor([[True,True],[False,False]]))
    assert r['active_queries']==2 and r['candidate_slots']==4
    assert r['mean_max_weight']==pytest.approx(.625)
    assert r['mean_entropy_nats']==pytest.approx(torch.log(torch.tensor(4.)).item()/2)
    assert r['mean_effective_candidates']==pytest.approx(2.5)
    assert r['per_case'][1]['active_queries']==0 and r['per_case'][1]['mean_entropy_nats'] is None
    assert attention_weight_statistics(None,None) is None


@pytest.mark.parametrize('weights,valid,reason',[
    ([[[.1,.1]]],[[True]],'normalized'),
    ([[[-.1,1.1]]],[[True]],'negative'),
    ([[[float('nan'),1.]]],[[True]],'nonfinite'),
    ([[[.5,.5]]],[[True,False]],'shape'),
])
def test_attention_statistics_reject_invalid_active_trace(weights,valid,reason):
    with pytest.raises(ValueError,match=reason):
        attention_weight_statistics(torch.tensor(weights),torch.tensor(valid))


def values():
    cond={'global':torch.ones(2,3), 'proj':torch.full((2,4,3),2.)}
    neg={k:torch.zeros_like(v) for k,v in cond.items()}
    cached={'cond':cond,'neg_cond':neg};online=copy.deepcopy(cached)
    native=copy.deepcopy(cached);native['cond']['proj']-=1
    mixed={k:v.clone() for k,v in cond.items()}
    for value in mixed.values():value[0]=0
    return [cached,online,copy.deepcopy(cond),copy.deepcopy(neg),mixed,native,[True,False]]


def test_active_cache_cfg_extremes_and_per_sample_mask_pass():
    result=check_conditions(*values())
    assert result['active_projection_update_rms']==1.
    assert result['p1_all_branches_exact_zero']
    assert result['mixed_global_and_projection_mask_verified']


def test_rejects_zero_initialization_as_active_path_proof():
    args=values();args[5]=copy.deepcopy(args[0])
    with pytest.raises(ValueError,match='inactive'):check_conditions(*args)


def test_rejects_semantic_condition_leaking_through_cfg_drop():
    args=values();args[3]['proj'][0,0,0]=1
    with pytest.raises(ValueError,match='leaked'):check_conditions(*args)


def test_rejects_online_cache_mismatch_under_amp_tolerance():
    args=values();args[1]['cond']['proj'][1,0,0]+=.002
    with pytest.raises(AssertionError):check_conditions(*args)


def test_rejects_mixed_mask_applied_only_to_global_features():
    args=values();args[4]['proj']=args[2]['proj'].clone()
    with pytest.raises(AssertionError):check_conditions(*args)


def test_masked_logit_error_is_finite_and_has_explicit_support_counts():
    logits=torch.tensor([1.,float('-inf'),2.])
    result=compare_trace_tensors(logits,logits.clone(),name='matching_logits')
    assert result=={'max_abs_error':0.,'finite_values':2,'masked_negative_infinity_values':1}
    json.dumps(result,allow_nan=False)


def test_rejects_changed_logit_invalid_mask():
    with pytest.raises(ValueError,match='masks differ'):
        compare_trace_tensors(torch.tensor([1.,float('-inf')]),torch.tensor([1.,0.]),name='matching_logits')


@pytest.mark.parametrize('value,reason', [(float('nan'),'NaN'),(float('inf'),'positive infinity')])
def test_rejects_nonfinite_matching_logits_even_on_both_sides(value,reason):
    logits=torch.tensor([1.,value])
    with pytest.raises(ValueError,match=reason):
        compare_trace_tensors(logits,logits.clone(),name='matching_logits')


def test_all_masked_logits_do_not_certify_an_active_trace():
    logits=torch.full((3,),float('-inf'))
    with pytest.raises(ValueError,match='no finite trace'):
        compare_trace_tensors(logits,logits.clone(),name='matching_logits')


def test_evidence_must_be_finite_without_a_mask_exception():
    evidence=torch.tensor([1.,float('-inf')])
    with pytest.raises(ValueError,match='nonfinite trace'):
        compare_trace_tensors(evidence,evidence.clone(),name='evidence')


def test_finite_logit_mismatch_still_fails_the_equality_assertion():
    with pytest.raises(AssertionError):
        compare_trace_tensors(torch.tensor([1.,float('-inf')]),torch.tensor([1.1,float('-inf')]),name='matching_logits')


def test_online_task_dataset_uses_normal_R9_key_schema_without_cache_only_flag():
    source={'match_root':'cached','task_matching_features':True,'transport_arm':'c1'}
    online=online_dataset_args(source)
    assert online=={'match_root':None,'transport_arm':'c1'} and source['task_matching_features']


def test_updated_task_head_is_allowed_while_frozen_scalar_prior_must_stay_exact():
    initial={'q2c.decoder.weight':torch.ones(2),'q2c.fusion_head.weight':torch.ones(2),
        'task_attention.query.weight':torch.ones(2),'fusion_gate':torch.zeros(())}
    saved={k:v.clone() for k,v in initial.items()};saved['q2c.decoder.weight']+=.1;saved['task_attention.query.weight']+=.2
    trainable=set(initial)-{'q2c.fusion_head.weight'}
    r=verify_saved_conditioner(initial,saved,trainable,{'score_prior':False})
    assert r['frozen_initializer_state_checked_keys']==['q2c.fusion_head.weight']
    assert r['actual_trainable_matcher_saved_state_changed_keys']==['q2c.decoder.weight']
    saved['q2c.fusion_head.weight']+=.1
    with pytest.raises(ValueError,match='frozen state'):verify_saved_conditioner(initial,saved,trainable,True)


def test_fully_trainable_task_does_not_falsely_require_a_frozen_head():
    initial={'q2c.fusion_head.weight':torch.ones(2),'task_attention.key.weight':torch.ones(2)}
    saved={k:v+.1 for k,v in initial.items()}
    r=verify_saved_conditioner(initial,saved,set(initial),True)
    assert r['frozen_initializer_state_exact'] is None
    with pytest.raises(ValueError,match='matcher weights'):verify_saved_conditioner(initial,initial,set(initial),True)


def test_native_token_reference_allows_FP16_storage_but_rejects_feature_contract_mismatch():
    torch.manual_seed(17);fresh=torch.randn(2,4,8,1024);cached=fresh.half()
    r=check_native_tokens(cached,fresh)
    assert r['relative_RMS']<.001 and r['mean_token_cosine']>.9999
    with pytest.raises(ValueError,match='differs from fresh'):check_native_tokens(cached,fresh+1.)
    bad=cached.clone();bad[0,0,0,0]=float('nan')
    with pytest.raises(ValueError,match='finiteness'):check_native_tokens(bad,fresh)
