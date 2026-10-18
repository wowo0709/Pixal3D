import copy

import pytest

from experiments.deformnext_retargeting.semantic_topk_selection import CANDIDATES,selection,supported


def contrast(gain=.01,low=.001,noises=None):
    return {'gain':gain,'paired_cluster_bootstrap_ci95':[low,gain+.01],
        'per_noise':[{'gain':x} for x in (noises if noises is not None else [gain]*3)]}


def reports():
    names=['top1-reference','static-top1-c1','previous-c1','previous-v1','task-native-spatial',*CANDIDATES]
    models={name:{'micro_voxel_iou':.43,'micro_voxel_f1':.6} for name in names}
    models['task-full']['micro_voxel_iou']=.45
    seven={'status':'PASS_SEVEN_ARM_FIXED_FINAL_DEVELOPMENT_SEARCH','models':copy.deepcopy(models),'contrasts':{}}
    matched={'status':'PASS_MATCHED_RETARGETER_TOP1_FIXED_FINAL','models':copy.deepcopy(models),'contrasts':{}}
    for name in CANDIDATES:
        for baseline in ['previous-v1','top1-reference']:
            seven['contrasts'][name+'-vs-'+baseline]=contrast()
        matched['contrasts'][name+'-vs-static-top1-c1']=contrast()
    return seven,matched


def test_selects_observed_best_only_after_both_complete_reports():
    a,b=reports();r=selection(a,b)
    assert r['selected_candidate']=='task-full' and r['full_goal_complete'] is False
    a['status']='RUNNING'
    with pytest.raises(ValueError,match='both complete'):selection(a,b)


@pytest.mark.parametrize('winner',['task-no-decoder','task-no-prior'])
def test_does_not_force_decoder_or_scalar_prior_when_ablation_wins(winner):
    a,b=reports()
    for report in [a,b]:report['models'][winner]['micro_voxel_iou']=.46
    assert selection(a,b)['selected_candidate']==winner


def test_negative_or_noise_unstable_search_cannot_be_reported_as_positive():
    a,b=reports()
    for name in CANDIDATES:a['contrasts'][name+'-vs-top1-reference']=contrast(low=-.001)
    result=selection(a,b)
    assert result['status']=='NO_PROMISING_COMBINATION_YET' and result['selected_candidate'] is None
    assert not supported(contrast(noises=[.02,.02,-.001]))
    assert not supported(contrast(gain=-.01,low=-.02))


def test_best_control_cannot_be_hidden_by_semantic_candidate_selection():
    a,b=reports()
    for report in [a,b]:report['models']['task-native-spatial']['micro_voxel_iou']=.5
    result=selection(a,b)
    assert result['selected_candidate']=='task-full'
    assert result['best_overall_development_model']=='task-native-spatial'


def test_inconsistent_fixed_final_reports_are_rejected():
    a,b=reports();b['models']['static-c1']['micro_voxel_iou']+=.01
    with pytest.raises(ValueError,match='disagree'):selection(a,b)
