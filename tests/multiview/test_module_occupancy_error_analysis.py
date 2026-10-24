"""Nontrivial identities for the descriptive occupancy count decomposition."""
import numpy as np
import pytest

from experiments.deformnext_retargeting.module_occupancy_error_analysis import analyze,decompose_counts
from experiments.deformnext_retargeting.unified_module_summary import ALIASES,SOURCE_COUNTS


def values(tp,fp,gt):
    return np.asarray([[[t,f,g-t] for t,f,g in zip(tp,fp,gt)]]*3,dtype=np.int64)


@pytest.mark.parametrize('kind',['hits-only','false-positives-only','opposing-changes','both-improve'])
def test_symmetric_decomposition_sums_to_actual_IoU_and_reverses_sign(kind):
    base=values([4,8],[3,6],[10,20])
    if kind=='hits-only':candidate=values([5,9],[3,6],[10,20])
    elif kind=='false-positives-only':candidate=values([4,8],[2,5],[10,20])
    elif kind=='opposing-changes':candidate=values([5,9],[5,9],[10,20])
    else:candidate=values([5,9],[2,5],[10,20])
    out=decompose_counts(candidate,base);reverse=decompose_counts(base,candidate)
    delta=np.mean(candidate[:,:,0]/candidate.sum(2)-base[:,:,0]/base.sum(2))
    assert out['mean_case_IoU_delta']==pytest.approx(delta)
    assert out['exact_symmetric_TP_component']+out['exact_symmetric_FP_component']==pytest.approx(delta)
    for key in ['mean_case_IoU_delta','exact_symmetric_TP_component','exact_symmetric_FP_component']:
        assert reverse[key]==pytest.approx(-out[key])
    if kind=='hits-only':assert out['exact_symmetric_FP_component']==0
    if kind=='false-positives-only':assert out['exact_symmetric_TP_component']==0
    if kind=='opposing-changes':assert out['exact_symmetric_TP_component']>0 and out['exact_symmetric_FP_component']<0
    assert out['decomposition_is_algebraic_not_mechanistic']


def test_case_macro_and_global_micro_are_both_explicit_and_different():
    a=values([1,80],[0,10],[2,100]);b=values([1,70],[1,5],[2,100]);out=decompose_counts(a,b)
    assert out['candidate']['mean_case_IoU']==pytest.approx((.5+80/110)/2)
    assert out['candidate']['mean_noise_global_micro_IoU']==pytest.approx(81/112)
    assert out['candidate']['mean_case_IoU']!=out['candidate']['mean_noise_global_micro_IoU']


def test_tied_GT_quartiles_stay_empty_instead_of_fabricating_zero_deltas():
    a=values([5]*4,[2]*4,[10]*4);b=values([4]*4,[3]*4,[10]*4);out=decompose_counts(a,b)
    assert sum(row['cases'] for row in out['GT_occupancy_strata'])==4
    assert any(row['cases']==0 and row['mean_case_IoU_delta'] is None for row in out['GT_occupancy_strata'])


@pytest.mark.parametrize('damage',['other-GT','noise-GT','negative','fractional','boolean','wrong-shape','empty-GT'])
def test_invalid_counts_or_changed_GT_cannot_be_used_to_explain_a_module_effect(damage):
    a=values([4,8],[3,6],[10,20]);b=a.copy()
    if damage=='other-GT':b[0,0,2]+=1
    elif damage=='noise-GT':a[1,0,2]+=1;b[1,0,2]+=1
    elif damage=='negative':a[0,0,1]=-1
    elif damage=='fractional':a=a.astype(float)
    elif damage=='boolean':a=a.astype(bool)
    elif damage=='wrong-shape':a=a[:,:,0]
    else:a[0,0,[0,2]]=0;b[0,0,[0,2]]=0
    with pytest.raises(ValueError):decompose_counts(a,b)


def native_records():
    records={}
    for arm,tp,fp in [('V1',40,20),('C_FIXED_TOPK',42,18),('FULL',41,19)]:
        records[arm]=[dict(status='PASS',stage='ss64',step=480,native_variant=ALIASES.get(arm,arm),
            actual_source_tensors=SOURCE_COUNTS[arm],evaluation_mode='final128',optimizer_updates_during_evaluation=0,
            per_asset=[dict(sample_id=f'case{i}',source_cluster_id=f'source{i%93}',seed=100+j,
                target_voxels_sha256=f'target{i}',tp=tp,fp=fp,fn=100-tp,voxel_iou=tp/(100+fp)) for i in range(128)]) for j in range(3)]
    return records


def test_partial_actual_study_preserves_model_names_and_available_original_contrasts():
    data=native_records();result=analyze(data)
    assert not result['all_twelve_arms_complete'] and len(result['missing_original_arms'])==9
    c=result['comparisons']['learned_attention']
    assert c['candidate']=='FULL' and c['reference']=='C_FIXED_TOPK' and c['mean_case_IoU_delta']<0
    assert c['candidate_occupancy_summary']['mean_case_IoU']==pytest.approx(41/119)
    assert result['comparisons']['pipeline_V1']['mean_case_IoU_delta']>0
    assert c['prespecified_IoU_family_size']==10 and result['comparisons']['pipeline_V1']['prespecified_IoU_family_size']==2
    assert not result['new_significance_tests_performed'] and not result['heldout_test_evaluated']


@pytest.mark.parametrize('damage',['step','source-tensors','GT-hash','counts-vs-score','duplicate-ID','same-noise'])
def test_study_rejects_incomplete_or_mismatched_actual_saved_model_measurements(damage):
    data=native_records();row=data['FULL'][0]
    if damage=='step':row['step']=384
    elif damage=='source-tensors':row['actual_source_tensors']=12
    elif damage=='GT-hash':row['per_asset'][0]['target_voxels_sha256']='different-target'
    elif damage=='counts-vs-score':row['per_asset'][0]['voxel_iou']=.9
    elif damage=='duplicate-ID':data['V1'][0]['per_asset'][1]['sample_id']='case0'
    else:
        for arm in data:data[arm][1]['per_asset'][0]['seed']=100
    with pytest.raises(ValueError):analyze(data)
