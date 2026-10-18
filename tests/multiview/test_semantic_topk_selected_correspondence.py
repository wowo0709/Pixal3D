import copy

import pytest

from experiments.deformnext_retargeting.semantic_topk_selected_correspondence import jobs_and_contract,summarize,SUMMARY_JOB


@pytest.mark.parametrize('candidate',['soft-topk','static-c1','task-full','task-no-decoder','task-no-prior'])
def test_actual_winner_gets_its_own_same_pool_control_without_duplicate_full_evaluations(tmp_path,candidate):
    jobs,contract=jobs_and_contract(tmp_path,tmp_path/'work',candidate,'selection')
    assert jobs[-1]['name']==SUMMARY_JOB and contract['candidate']==candidate
    if candidate=='task-full':
        assert len(jobs)==1 and 'task-value-address-v1' in contract['normal']
        assert 'summarize-task-value-address-diagnostic' in jobs[0]['depends_on']
    else:
        assert len(jobs)==3
        for job,control in zip(jobs[:2],['normal','wrong-address']):
            assert job['resource']=='evaluation' and 'gpu' not in job and 'gpus' not in job
            assert job['depends_on']==['selection']
            command=job['command']
            assert command[command.index('--arm-config')+1].endswith(candidate+'.json')
            assert command[command.index('--task-value-control')+1]==control
            assert command[command.index('--panel-file')+1].endswith('panel-32.json')


def records():
    rows=[{'sample_id':str(i),'source_cluster_id':str(i),'seed':i,'tp':2,'fp':1,'fn':2,
           'target_voxels_sha256':'a'*64,'target_voxel_shape':[64,64,64]} for i in range(128)]
    common={'status':'PASS','stage':'ss64','step':480,'weights':'raw','sampling_steps':12,'guidance':3.,
        'generation_batch_size':8,'optimizer_updates_during_evaluation':0,'target_modality':'rgb',
        'input_camera_ids':[8,9,10,11],'input_states':['M1','M0','M0','M0'],'conditioning_mode':'edited-rgb',
        'checkpoint':'SS480','conditioner_checkpoint':'C480','arm_config':'candidate.json','manifest_sha256':'b'*64,
        'arm':'candidate','transport_control':'normal','r9_path_control':'normal'}
    normal={**common,'diagnostic_only':False,'task_value_control':'normal','per_asset':copy.deepcopy(rows[:32])}
    wrong={**common,'diagnostic_only':True,'task_value_control':'wrong-address','per_asset':copy.deepcopy(rows[:32]),
        'task_value_control_events':[{'matching_logits_and_weights_unchanged':True,'reassigned_valid_slots':64}]}
    reference={**common,'diagnostic_only':False,'per_asset':copy.deepcopy(rows)}
    return normal,wrong,reference


def test_no_effect_is_a_valid_completed_diagnostic_without_a_false_quality_claim():
    result=summarize(*records())
    assert result['status']=='PASS_SELECTED_MODEL_SAME_POOL_VALUE_ADDRESS_DIAGNOSTIC'
    assert result['quality_effect_supported_at_this_panel'] is False
    assert result['normal_minus_wrong_address']['paired_target_hashes_verified']==32


@pytest.mark.parametrize('field',['checkpoint','conditioner_checkpoint','arm_config','manifest_sha256','generation_batch_size'])
def test_normal_replay_cannot_bind_a_different_model_or_generation_protocol(field):
    a,b,c=records();c[field]='changed'
    with pytest.raises(ValueError):summarize(a,b,c)


def test_unrelated_r9_source_map_intervention_cannot_be_called_same_candidate_pool():
    a,b,c=records();b['r9_path_control']='wrong-address'
    with pytest.raises(ValueError,match='do not combine'):summarize(a,b,c)
