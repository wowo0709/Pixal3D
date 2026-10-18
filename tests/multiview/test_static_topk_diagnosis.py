import numpy as np
import pytest

from experiments.deformnext_retargeting.diagnose_static_topk import evidence_statistics,summarize_statistics,rank_correlation


def test_broad_weights_can_mix_identical_values_without_changing_evidence():
    values=np.tile([[1.,2.,3.]],(4,1));ids=np.arange(4)[None];logits=np.zeros((1,4));valid=np.ones((1,4),bool)
    result=summarize_statistics([evidence_statistics(values,ids,logits,valid,np.array([0,0]),1.,.05)])
    assert result['query_mean_effective_K']==pytest.approx(4.)
    assert result['query_relative_RMS']==0 and result['voxel_relative_RMS']==0
    assert result['fixed_gate_projection_delta_RMS']==0


def test_single_valid_candidate_is_top1_even_with_invalid_infinite_scores():
    values=np.eye(3);ids=np.arange(3)[None];valid=np.array([[False,True,False]])
    s=evidence_statistics(values,ids,np.array([[-np.inf,0.,-np.inf]]),valid,np.array([0]),.02,.05)
    result=summarize_statistics([s]);assert result['query_mean_effective_K']==1 and result['query_relative_RMS']==0


def test_projection_multiplicity_changes_weighted_statistics_without_changing_patch_evidence():
    values=np.array([[1.,0.],[0.,1.]]);ids=np.array([[0,1],[0,0]]);valid=np.ones((2,2),bool);logits=np.zeros((2,2))
    first=summarize_statistics([evidence_statistics(values,ids,logits,valid,np.array([0,0,1]),1.,.1)])
    second=summarize_statistics([evidence_statistics(values,ids,logits,valid,np.array([0,1,1]),1.,.1)])
    assert first['query_relative_RMS']==second['query_relative_RMS']
    assert first['voxel_relative_RMS']>second['voxel_relative_RMS']
    assert first['fixed_gate_projection_delta_RMS']>second['fixed_gate_projection_delta_RMS']


def test_inactive_fallback_queries_do_not_create_entropy_or_divide_by_zero():
    s=evidence_statistics(np.ones((3,2)),np.array([[0,1],[1,2]]),np.zeros((2,2)),np.zeros((2,2),bool),np.array([0,1]),1.,.1)
    assert summarize_statistics([s])=={'queries':0,'projected_voxels':0,'native_voxels_evaluated':2,
        'active_native_voxel_fraction':0.,'fixed_gate_projection_delta_RMS':0.}


def test_projection_rms_includes_zero_update_voxels_outside_rgb_alpha_support():
    values=np.eye(2);ids=np.array([[0,1],[0,1]]);scores=np.zeros((2,2));valid=np.array([[True,True],[False,False]])
    a=summarize_statistics([evidence_statistics(values,ids,scores,valid,np.array([0,1]),1.,.1)])
    b=summarize_statistics([evidence_statistics(values,ids,scores,valid,np.array([0,1,1,1]),1.,.1)])
    assert a['query_relative_RMS']==b['query_relative_RMS'] and a['voxel_relative_RMS']==b['voxel_relative_RMS']
    assert b['fixed_gate_projection_delta_RMS']==pytest.approx(a['fixed_gate_projection_delta_RMS']/np.sqrt(2))
    assert b['active_native_voxel_fraction']==.25


def test_categorical_score_offsets_do_not_change_feature_statistics():
    values=np.eye(3);ids=np.arange(3)[None];scores=np.array([[.1,.2,.3]]);mask=np.ones((1,3),bool);grid=np.array([0])
    a=summarize_statistics([evidence_statistics(values,ids,scores,mask,grid,.02,.05)])
    b=summarize_statistics([evidence_statistics(values,ids,scores+10,mask,grid,.02,.05)])
    for key in a:assert a[key]==pytest.approx(b[key],rel=1e-10,abs=1e-10)


def test_rank_correlation_handles_ties_and_constant_data_without_false_significance():
    assert rank_correlation([1,1,3],[2,2,4])==pytest.approx(1.)
    assert rank_correlation([1,2,3],[3,2,1])==pytest.approx(-1.)
    assert rank_correlation([1,1,1],[1,2,3]) is None


def test_invalid_projection_index_or_native_foreground_zero_feature_is_rejected():
    with pytest.raises(ValueError,match='contract'):
        evidence_statistics(np.ones((2,2)),np.array([[0,1]]),np.zeros((1,2)),np.ones((1,2),bool),np.array([1]),1.,.1)
    with pytest.raises(ValueError,match='zero norm'):
        evidence_statistics(np.zeros((2,2)),np.array([[0,1]]),np.zeros((1,2)),np.ones((1,2),bool),np.array([0]),1.,.1)
