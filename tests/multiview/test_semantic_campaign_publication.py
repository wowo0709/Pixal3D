import copy
from pathlib import Path

import pytest

from experiments.deformnext_retargeting.semantic_topk_campaign_wandb import (
    conditional_scope, validate_noise_identity, publication_jobs,
)


def records():
    es=[]
    for noise in range(3):
        rows=[{'sample_id':str(i),'source_cluster_id':str(i%93),'seed':1000*noise+i,
               'tp':2,'fp':1,'fn':2,'target_voxels_sha256':str(i).zfill(64),
               'target_voxel_shape':[64,64,64]} for i in range(128)]
        es.append({'per_asset':rows})
    return {'reference':es,'candidate':copy.deepcopy(es)}


def selected_reports():
    return [
        {'status':'PASS_FIXED_FINAL_SEED202_CONFIRMATION','selected_candidate':'task-full',
         'positive_CI_and_all_noise_direction_at_both_seeds':False},
        {'status':'PASS_ADDITIONAL121_FIXED_SELECTED_MODEL_VALIDATION','selected_candidate':'task-full',
         'positive_CI_and_all_noises_on_both_additional_training_seeds':False},
        {'status':'PASS_SELECTED_MODEL_SAME_POOL_VALUE_ADDRESS_DIAGNOSTIC','selected_candidate':'task-full'},
    ]


def test_actual_complete_same_panel_remains_same_when_model_predictions_differ():
    r=records();r['candidate'][1]['per_asset'][4].update(tp=3,fn=1,fp=0)
    assert validate_noise_identity(r,128,93)==[str(i) for i in range(128)]


@pytest.mark.parametrize('field,value',[
    ('source_cluster_id','other'),('seed',90000),
    ('target_voxels_sha256','b'*64),('target_voxel_shape',[32,32,32]),('fn',5),
])
def test_matching_metric_cannot_hide_different_population_noise_or_target(field,value):
    r=records();r['candidate'][2]['per_asset'][9][field]=value
    with pytest.raises(ValueError,match='identity'):validate_noise_identity(r,128,93)


def test_same_three_sampling_noise_files_cannot_be_repeated_to_claim_three_noises():
    r=records()
    for es in r.values():es[2]=copy.deepcopy(es[1])
    with pytest.raises(ValueError,match='distinct sampling'):validate_noise_identity(r,128,93)


def test_same_set_reordered_cannot_be_presented_as_same_case_panels():
    r=records();r['candidate'][1]['per_asset'].reverse()
    with pytest.raises(ValueError,match='order'):validate_noise_identity(r,128,93)


def test_source_population_is_checked_separately_from_asset_count():
    with pytest.raises(ValueError,match='cluster'):validate_noise_identity(records(),128,128)


@pytest.mark.parametrize('missing',[0,1,2])
def test_positive_selection_requires_all_selected_model_work_before_publication(missing):
    reports=selected_reports();reports[missing]=None
    with pytest.raises(ValueError,match='incomplete'):conditional_scope('task-full',*reports)


def test_pass_status_is_not_interpreted_as_positive_scientific_replication_or_adoption():
    scope=conditional_scope('task-full',*selected_reports())
    assert scope['seed_confirmation_completed'] and scope['additional121_completed']
    assert not scope['positive_in_both_training_seeds']
    assert not scope['positive_on_additional121_both_seeds']
    assert not scope['same_retargeter_Top1_replication_across_training_seeds']
    assert not scope['model_adopted'] and not scope['full_campaign_complete']


def test_a_different_selected_candidate_cannot_supply_confirmation():
    with pytest.raises(ValueError,match='differs'):conditional_scope('static-c1',*selected_reports())


def test_no_selected_candidate_is_preserved_as_negative_evidence_with_active_scope():
    scope=conditional_scope(None)
    assert scope['selected_candidate'] is None and not scope['model_adopted']
    assert not scope['seed_confirmation_completed'] and not scope['full_campaign_complete']
    with pytest.raises(ValueError,match='unselected'):conditional_scope(None,*selected_reports())


@pytest.mark.parametrize('candidate',[None,'task-full','static-c1'])
def test_reporting_waits_for_actual_models_and_auditors_without_allocating_a_gpu(tmp_path,candidate):
    jobs=publication_jobs(tmp_path,candidate,'select-model')
    assert [j['name'] for j in jobs]==['campaign-report-render','campaign-report-publish','campaign-report-readback']
    assert all(j['resource']=='cpu' and 'gpu' not in j and 'gpus' not in j for j in jobs)
    deps=jobs[0]['depends_on'];assert 'select-model' in deps
    assert 'functional-QA-s101-static-top1-c1' in deps
    assert 'functional-QA-s202-top1-reference' in deps
    assert 'summarize-firstwave-fusion-dose' in deps
    assert ('summarize-additional121-semantic-topk' in deps)==(candidate is not None)
    assert len(jobs[0]['requires'])==(9 if candidate is None else 10)*6
    assert jobs[1]['depends_on']==['campaign-report-render']
    assert jobs[2]['depends_on']==['campaign-report-publish']
    assert all(Path(output).is_relative_to(tmp_path) for j in jobs for output in j['outputs'])
