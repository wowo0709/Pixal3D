import copy
import json

import pytest
import torch

from experiments.deformnext_retargeting.verify_ss_conditioning_contract import (
    check_conditions, compare_trace_tensors,
)


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
